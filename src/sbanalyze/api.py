"""Local API + static frontend on localhost."""

from __future__ import annotations

import io
import math
import re
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from .config import PROJECT_ROOT, load_config
from .contracts import PHASE_ORDER, Phase, PipelineError, Validity, ViewRole
from .jobs import JobQueue
from .metrics import METRIC_UNITS
from .metrics.validity import validity
from .store import Store

cfg = load_config()
store = Store(cfg.data_dir)
app = FastAPI(title="Snowboard Approach Analyzer")


@app.middleware("http")
async def no_stale_frontend(request: Request, call_next):
    """The frontend changes as the app evolves: make browsers revalidate the page, scripts and
    styles on every load, so an old cached script never runs against a new page."""
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.endswith((".html", ".js", ".css")):
        response.headers["Cache-Control"] = "no-cache"
    return response


def _clean(a) -> list:
    return [None if (v is None or (isinstance(v, float) and not math.isfinite(v))) else v
            for v in (a.tolist() if hasattr(a, "tolist") else a)]


def _detail_or_404(clip_id: str) -> dict:
    d = store.clip_detail(clip_id)
    if d is None:
        raise HTTPException(404, f"unknown clip {clip_id}")
    return d


# ---- sessions & import ----------------------------------------------------------------------

class SessionIn(BaseModel):
    id: str
    venue: str = "default"
    note: str = ""


@app.post("/sessions", status_code=201)
def create_session(body: SessionIn):
    # Calibration is derived automatically from the first clip imported into the session.
    store.put_session(body.id, body.venue, store.get_calibration(body.id), body.note)
    return {"id": body.id}


class ClipIn(BaseModel):
    path: str
    session_id: str | None = None
    start_frame: int | None = None
    end_frame: int | None = None


@app.post("/clips", status_code=201)
def import_clip(body: ClipIn):
    from .pipeline import process

    p = Path(body.path).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        raise HTTPException(400, f"no such file: {p}")
    manual = ((body.start_frame, body.end_frame)
              if body.start_frame is not None and body.end_frame is not None else None)
    try:
        cid = process(p, cfg, store, session_id=body.session_id, manual_window=manual)
    except PipelineError as e:
        return JSONResponse({"status": e.status, "stage": e.stage, "detail": str(e)}, 422)
    return {"clip_id": cid}


# ---- import from the UI: upload, browse, background jobs --------------------------------------

VIDEO_EXT = {".mov", ".mp4", ".m4v"}
jobs = JobQueue(cfg, store)


def _safe_name(name: str) -> str:
    name = Path(name).name  # never a path, only a file name
    if Path(name).suffix.lower() not in VIDEO_EXT:
        raise HTTPException(400, f"not a video file: {name}")
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


@app.put("/uploads/{name}")
async def upload(name: str, request: Request):
    """Receive one video as the raw request body (streamed to disk, no size limit)."""
    dest_dir = store.root / "videos"
    dest_dir.mkdir(exist_ok=True)
    dest = dest_dir / _safe_name(name)
    size = int(request.headers.get("content-length") or -1)
    if dest.exists() and dest.stat().st_size == size:
        return {"path": str(dest), "name": dest.name, "size": size, "reused": True}
    tmp = dest.with_suffix(dest.suffix + ".part")
    # file I/O off the event loop, so other requests (job polling) stay responsive
    f = await run_in_threadpool(tmp.open, "wb")
    try:
        async for chunk in request.stream():
            await run_in_threadpool(f.write, chunk)
    finally:
        await run_in_threadpool(f.close)
    tmp.replace(dest)
    return {"path": str(dest), "name": dest.name, "size": dest.stat().st_size, "reused": False}


@app.get("/browse")
def browse(path: str | None = None):
    """List sub-folders and videos of a folder on this computer (local, single-user app)."""
    p = Path(path).expanduser() if path else Path.home()
    if not p.is_dir():
        raise HTTPException(400, f"not a folder: {p}")
    dirs, videos = [], []
    try:
        entries = sorted(p.iterdir(), key=lambda e: e.name.lower())
    except PermissionError as e:
        raise HTTPException(403, f"no permission to read {p}") from e
    for e in entries:
        if e.name.startswith("."):
            continue
        if e.is_dir():
            dirs.append(e.name)
        elif e.suffix.lower() in VIDEO_EXT:
            videos.append({"name": e.name, "path": str(e), "size": e.stat().st_size})
    return {"path": str(p), "parent": str(p.parent) if p.parent != p else None,
            "dirs": dirs, "videos": videos}


class JobsIn(BaseModel):
    paths: list[str]
    session_id: str | None = None


@app.post("/jobs", status_code=202)
def submit_jobs(body: JobsIn):
    out = []
    for raw in body.paths:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if not p.is_file() or p.suffix.lower() not in VIDEO_EXT:
            raise HTTPException(400, f"not a video file: {p}")
        running = jobs.active(str(p))
        out.append((running or jobs.submit(p, body.session_id)).as_dict())
    return out


@app.get("/jobs")
def list_jobs():
    return jobs.list()


# ---- catalog ---------------------------------------------------------------------------------

@app.get("/clips")
def list_clips():
    return store.list_clips()


@app.delete("/clips/{clip_id}")
def delete_clip(clip_id: str):
    d = _detail_or_404(clip_id)
    if jobs.active(d["path"]):
        raise HTTPException(409, "this video is being processed; remove it when it has finished")
    return store.delete_clip(clip_id)


@app.get("/clips/{clip_id}")
def get_clip(clip_id: str):
    return _detail_or_404(clip_id)


def _series(df: pd.DataFrame) -> dict:
    frames = np.sort(df["frame_idx"].unique())
    base = df.drop_duplicates("frame_idx").set_index("frame_idx").reindex(frames)
    out = {
        "frames": frames.tolist(),
        "t": _clean(base["t_sec"].astype(float).values),
        "phase": base["phase"].astype(str).tolist(),
        "rider_px_h": _clean(base["rider_px_h"].astype(float).values),
        "units": METRIC_UNITS,
        "series": {},
    }
    for m, g in df.groupby("metric", observed=True):
        g = g.set_index("frame_idx").reindex(frames)
        out["series"][m] = {
            "value": _clean(g["value"].astype(float).values),
            "err": _clean(g["err_est"].astype(float).values),
            "conf": _clean(g["confidence"].astype(float).values),
            "validity": [v if isinstance(v, str) else "invalid" for v in g["validity"].astype(object)],
            "view": [v if isinstance(v, str) else None for v in g["view"].astype(object)],
        }
    return out


@app.get("/clips/{clip_id}/metrics")
def get_metrics(clip_id: str, metric: Annotated[list[str] | None, Query()] = None):
    _detail_or_404(clip_id)
    df = store.read_metrics(clip_id)
    if metric:
        df = df[df["metric"].isin(metric)]
    return _series(df)


@app.get("/clips/{clip_id}/events")
def get_events(clip_id: str):
    d = _detail_or_404(clip_id)
    return store.get_events(d["run_id"])


@app.get("/clips/{clip_id}/overlay")
def get_overlay(clip_id: str):
    p = store.overlay_path(clip_id)
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="application/json")


@app.get("/clips/{clip_id}/frames/{n}")
def get_frame(clip_id: str, n: int):
    p = store.frames_dir(clip_id) / f"{n:05d}.jpg"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=86400"})


@app.get("/clips/{clip_id}/follow/{n}")
def get_follow_frame(clip_id: str, n: int):
    p = store.follow_dir(clip_id) / f"{n:05d}.jpg"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=86400"})


@app.get("/clips/{clip_id}/export.csv")
def export_csv(clip_id: str):
    _detail_or_404(clip_id)
    buf = io.StringIO()
    store.read_metrics(clip_id).to_csv(buf, index=False)
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{clip_id}.csv"'})


@app.get("/clips/{clip_id}/export.parquet")
def export_parquet(clip_id: str):
    _detail_or_404(clip_id)
    return FileResponse(store.metrics_path(clip_id), filename=f"{clip_id}.parquet")


# ---- manual corrections ---------------------------------------------------------------------

class PhasesIn(BaseModel):
    transition_start: float
    takeoff_start: float


@app.patch("/runs/{run_id}/phases")
def override_phases(run_id: str, body: PhasesIn):
    """Hand-corrected phase boundaries; validity flags are re-applied to stored metrics."""
    clips = [c for c in store.list_clips() if c["run_id"] == run_id]
    if not clips:
        raise HTTPException(404)
    cid = clips[0]["clip_id"]
    df = store.read_metrics(cid)
    old = {p["phase"]: (p["t_start"], p["t_end"]) for p in store.clip_detail(cid)["phases"]}
    t0 = float(df["t_sec"].min())
    air = old.get(Phase.AIR.value)
    t1 = air[0] if air else float(df["t_sec"].max())
    a, b = sorted((body.transition_start, body.takeoff_start))
    spans = {Phase.DROP_IN: (t0, a), Phase.TRANSITION: (a, b), Phase.TAKEOFF_RUN: (b, t1)}
    if air:
        spans[Phase.AIR] = air
    phase = np.where(df["t_sec"] < a, Phase.DROP_IN.value,
                     np.where(df["t_sec"] < b, Phase.TRANSITION.value,
                              np.where((air is None) | (df["t_sec"] < t1),
                                       Phase.TAKEOFF_RUN.value, Phase.AIR.value)))
    val = [validity(m, Phase(p), ViewRole(v)).value if c >= 0.15 else Validity.INVALID.value
           for m, p, v, c in zip(df["metric"], phase, df["view"], df["confidence"])]
    df["phase"] = pd.Categorical(phase)
    df["validity"] = pd.Categorical(val)
    df["valid"] = df["validity"] != Validity.INVALID.value
    df.to_parquet(store.metrics_path(cid), index=False)
    store.put_phases(run_id, spans, "manual")  # events keep their original phase labels
    return {"run_id": run_id, "phases": {p.value: s for p, s in spans.items()}}


@app.post("/runs/{run_id}/views")
def attach_view(run_id: str):
    raise HTTPException(501, "second-view ingest is milestone M8 and not implemented yet; "
                             "fuse() already accepts a secondary sample list")


@app.patch("/runs/{run_id}/sync")
def override_sync(run_id: str):
    raise HTTPException(501, "view synchronisation is milestone M8 and not implemented yet")


# ---- comparison ------------------------------------------------------------------------------

def _phase_aligned(clip_id: str, n_per_phase: int) -> dict:
    d = _detail_or_404(clip_id)
    df = store.read_metrics(clip_id)
    spans = {p["phase"]: (p["t_start"], p["t_end"]) for p in d["phases"]}
    n_ph = len(PHASE_ORDER)
    u = np.linspace(0, n_ph, n_ph * n_per_phase + 1)
    # map common phase time u (0..4) back to this run's clock, piecewise linearly; a run with
    # no flight has no air segment (NaN there)
    t_of_u = np.full_like(u, np.nan)
    for k, ph in enumerate(PHASE_ORDER):
        if ph.value not in spans:
            continue
        a, b = spans[ph.value]
        sel = (u >= k) & (u <= k + 1)
        t_of_u[sel] = a + (u[sel] - k) * (b - a)
    out = {}
    for m, g in df.groupby("metric", observed=True):
        g = g.sort_values("t_sec")
        inside = np.isfinite(t_of_u) & (t_of_u >= g["t_sec"].min()) & (t_of_u <= g["t_sec"].max())
        tq = np.nan_to_num(t_of_u)
        v = np.where(inside, np.interp(tq, g["t_sec"], g["value"]), np.nan)
        e = np.where(inside, np.interp(tq, g["t_sec"], g["err_est"]), np.nan)
        ok = inside & (np.interp(tq, g["t_sec"], g["valid"].astype(float)) > 0.5)
        out[m] = {"value": _clean(v), "err": _clean(e), "valid": ok.tolist()}
    return {"u": u.tolist(), "t": _clean(t_of_u), "series": out}


@app.get("/compare")
def compare(a: str, b: str, n: int = 60):
    return {"a": _phase_aligned(a, n), "b": _phase_aligned(b, n), "units": METRIC_UNITS}


# ---- frontend -------------------------------------------------------------------------------

app.mount("/", StaticFiles(directory=PROJECT_ROOT / "web", html=True), name="web")
