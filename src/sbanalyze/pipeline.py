"""End-to-end processing of one clip: ingest -> trim -> detect -> calib -> phase -> metrics."""

from __future__ import annotations

import json
import logging
import sys
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from . import PIPELINE_VERSION
from .air import analyze as analyze_air
from .air import find_takeoff
from .boardedge import clean_series, detect_tail_edge, lowpass, roll_from_segment, smooth_anchor
from .calib import calibrate, estimate_pitch
from .config import Config, load_config
from .contracts import K, Phase, PipelineError, PoseFrame, RunWindow, Validity, ViewRole
from .events import detect, detect_edge_changes
from .fuse import fuse
from .ingest import iter_frames, probe
from .metrics import compute_signals, phase_at, to_samples
from .phase import segment
from .render import FollowWriter, FrameWriter, follow_rects, rider_boxes, write_overlay
from .store import Store
from .trim import build_background, find_run_window, motion_energy

log = logging.getLogger("sbanalyze")


def _setup_logging(store: Store) -> None:
    if log.handlers:
        return
    log.setLevel(logging.INFO)
    (store.root / "logs").mkdir(exist_ok=True)
    fh = logging.FileHandler(store.root / "logs" / "pipeline.jsonl")
    fh.setFormatter(logging.Formatter("%(message)s"))
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(fh)
    log.addHandler(sh)


@contextmanager
def _stage(clip_id: str, name: str, **extra):
    t0 = time.perf_counter()
    info: dict = {}
    yield info
    log.info(json.dumps({"clip": clip_id, "stage": name,
                         "seconds": round(time.perf_counter() - t0, 2), **extra, **info}))


def session_id_for(recorded_at) -> str:
    return recorded_at.strftime("%Y%m%d")


def _backend(cfg: Config):
    from .detect.rtm import RtmBackend
    d = cfg.detect
    return RtmBackend(mode=d.mode, crop_margin=d.crop_margin, min_crop_px=d.min_crop_px,
                      kpt_thr=cfg.metrics.kpt_conf_threshold)


def process(path: Path, cfg: Config | None = None, store: Store | None = None,
            session_id: str | None = None, venue: str = "default",
            manual_window: tuple[int, int] | None = None, backend=None,
            reuse_poses: bool = False,
            progress: Callable[[str, float], None] | None = None) -> str:
    """Process one clip end to end. `progress(stage, fraction)` is called as work advances."""
    report = progress or (lambda stage, frac: None)
    cfg = cfg or load_config()
    store = store or Store(cfg.data_dir)
    _setup_logging(store)
    t_start = time.perf_counter()

    meta = probe(Path(path), run_id="")
    run_id = f"run-{meta.clip_id}"
    meta = replace(meta, run_id=run_id)
    session_id = session_id or session_id_for(meta.recorded_at)
    cid = meta.clip_id

    # ---- trim at analysis resolution ---------------------------------------------------------
    tc = cfg.trim
    report("Finding the run", 0.02)
    with _stage(cid, "ingest+trim") as info:
        small = [img for _, _, img in iter_frames(meta, scale_width=tc.analysis_width)]
        scale = tc.analysis_width / meta.width
        bg = build_background(small, tc.background_samples, scale)
        if manual_window:
            window = RunWindow(manual_window[0], manual_window[1], "manual")
        else:
            energy = motion_energy(small, bg, tc.diff_threshold)
            try:
                window = find_run_window(energy, meta.fps, tc.min_blob_frac, tc.pad_s)
            except PipelineError as e:
                store.put_session(session_id, venue, store.get_calibration(session_id))
                store.put_run(run_id, session_id, meta, PIPELINE_VERSION, cfg.hash, e.status)
                store.put_clip(meta, None)
                raise
        info.update(frames=len(small), window=[window.start_frame, window.end_frame])
    del small

    # ---- calibration (session-scoped) ---------------------------------------------------------
    cal = store.get_calibration(session_id)
    if cal is None:
        report("Calibrating the venue", 0.08)
        with _stage(cid, "calibrate") as info:
            pitch = _pitch(meta, bg, cfg)
            cal = calibrate(bg.mat_mask, meta.height, scale, _focal(meta, cfg),
                            cfg.rider.stature_m, cfg.rider.chain_fraction, pitch=pitch)
            info.update(pitch_deg=pitch[0], pitch_sd=pitch[1])
        store.put_session(session_id, venue, cal, note="auto from first clip of session")
    pitch_deg = cal.pitch_deg if np.isfinite(cal.pitch_deg) else cfg.camera.pitch_deg_fallback

    # ---- detect + render the frame cache in one decode pass ------------------------------------
    # Detection continues past the run window: the rider is followed through the flight.
    fps_nominal = meta.fps
    det_end = min(meta.n_frames - 1,
                  window.end_frame + round(cfg.air.max_flight_s * fps_nominal))
    det_window = RunWindow(window.start_frame, det_end, window.method, window.confidence)
    backend_name = f"rtmlib-{cfg.detect.mode}"
    cached = store.get_poses(cid, backend_name, det_window) if reuse_poses else None
    writer = FrameWriter(store.frames_dir(cid), (meta.height, meta.width),
                         cfg.render.long_edge_px, cfg.render.jpeg_quality,
                         clear=cached is None)
    if cached is not None:
        idxs, times, poses = cached
        log.info(json.dumps({"clip": cid, "stage": "detect", "reused_poses": True}))
    else:
        mat_full = cv2.resize(bg.mat_mask, (meta.width, meta.height),
                              interpolation=cv2.INTER_NEAREST)
        roi = cv2.dilate(mat_full, np.ones((121, 121), np.uint8))
        air_roi = _air_roi(mat_full, cal.mat_top_y)
        backend = backend or _backend(cfg)
        backend_name = backend.name
        poses: list[PoseFrame | None] = []
        idxs, times = [], []
        report("Loading the pose model", 0.14)
        n_det = max(1, det_end - window.start_frame + 1)
        # scene cues for choosing the right person: the mat itself (a few px of tolerance) and,
        # per frame, what is moving on and around it
        mat_small = cv2.dilate(bg.mat_mask, np.ones((9, 9), np.uint8))
        near_mat = cv2.dilate(bg.mat_mask, np.ones((31, 31), np.uint8))
        bg_gray = cv2.GaussianBlur(cv2.cvtColor(bg.image, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        small_wh = (bg.image.shape[1], bg.image.shape[0])
        with _stage(cid, "detect", backend=backend.name) as info:
            first, above, airborne, lost = True, 0, False, 0
            for fi, t, img in iter_frames(meta, window.start_frame, det_end):
                if first:
                    backend.reset(img, roi, mat_small=mat_small, scale=scale)
                    first = False
                if hasattr(backend, "observe"):
                    g = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(img, small_wh,
                                                                 interpolation=cv2.INTER_AREA),
                                                      cv2.COLOR_BGR2GRAY), (5, 5), 0)
                    moving = (cv2.absdiff(g, bg_gray) > cfg.trim.diff_threshold) & (near_mat > 0)
                    backend.observe(moving.astype(np.uint8))
                p = backend.infer_frame(fi, t, img)
                poses.append(p)
                idxs.append(fi)
                times.append(t)
                writer.write(fi, img)
                report("Tracking the rider", 0.15 + 0.65 * (fi - window.start_frame) / n_det)
                if p is not None and not airborne:
                    above = above + 1 if _feet_y(p) < cal.mat_top_y - 10 else 0
                    if above >= 3:  # off the lip: the rider may now be anywhere above it
                        airborne = True
                        backend.set_roi(air_roi)
                lost = lost + 1 if p is None else 0
                if airborne and lost > 12:
                    break
                if not airborne and fi > window.end_frame and lost > 12:
                    break
            info.update(frames=len(poses), detected=sum(p is not None for p in poses),
                        airborne=airborne)
        store.put_poses(cid, backend.name, det_window, idxs, times, poses)

    # Keep from the first on-mat detection. The run ends at takeoff; the flight ends at
    # touchdown (plus a short settle so the landing is visible).
    fy_raw = np.array([_feet_y(p) if p is not None else np.nan for p in poses])
    on_mat = [i for i, p in enumerate(poses) if p is not None and fy_raw[i] >= cal.mat_top_y]
    if len(on_mat) < 10:
        store.put_session(session_id, venue, cal)
        store.put_run(run_id, session_id, meta, PIPELINE_VERSION, cfg.hash, "no_rider_detected")
        store.put_clip(meta, writer.size)
        raise PipelineError("detect", "no_rider_detected", "rider not tracked")
    a = on_mat[0]
    k_to = find_takeoff(fy_raw[a:], cal.mat_top_y)
    if k_to is not None and k_to >= 10:
        takeoff = a + k_to
        detected = [i for i, p in enumerate(poses) if p is not None]
        b = detected[-1]
    else:  # no flight seen: stop at the lip
        takeoff, b = None, on_mat[-1]

    def run_metrics(lo: int, hi: int):
        fidx, tt = np.array(idxs[lo:hi + 1]), np.array(times[lo:hi + 1])
        fps = float(1.0 / np.median(np.diff(tt)))  # phone clips are variable-rate
        air_from = None if takeoff is None else takeoff - lo
        sig = compute_signals(poses[lo:hi + 1], fidx, tt, fps, cal, cfg.metrics,
                              (meta.width, meta.height), cfg.rider.stance_width_m, air_from,
                              stance_override=cfg.rider.get("stance", "auto"))
        jump = None
        if air_from is not None:
            jump = analyze_air(tt, sig.kp, air_from, float(sig.values["range"][air_from - 1]),
                               sig.takeoff_speed, cal.mat_top_y, pitch_deg, cal.pitch_sd_deg,
                               cal.focal_px, meta.height / 2, cfg.metrics.keypoint_noise_px)
        return fidx, tt, fps, sig, jump

    report("Measuring", 0.82)
    with _stage(cid, "metrics") as info:
        fidx, tt, fps, sig, jump = run_metrics(a, b)
        if jump is not None:
            # cut shortly after touchdown and recompute so filters see only the kept frames
            settle = jump.touchdown + round(cfg.air.settle_s * fps)
            b = min(b, a + settle)
            fidx, tt, fps, sig, jump = run_metrics(a, b)
        n = len(fidx)
        na = n if jump is None else jump.takeoff
        if jump is not None:
            sig.values["air_height"] = jump.height
            sig.errors["air_height"] = jump.height_err
        else:
            sig.values["air_height"] = np.full(n, np.nan)
            sig.errors["air_height"] = np.full(n, np.nan)
        sig.confidence["air_height"] = sig.confidence["com_height"]

        spans, source = segment(tt[:na], sig.values["board_yaw"][:na],
                                sig.values["speed_along"][:na], fps, cfg.phase,
                                feet_y=sig.contact[:na, 1],
                                terrain=(cal.transition_y, cal.takeoff_y))
        if jump is not None:
            spans[Phase.AIR] = (float(tt[na]), float(tt[-1]))
        info.update(phase_source=source,
                    air_time=None if jump is None else round(jump.air_time_s, 3))

    poses, idxs, times = poses[a:b + 1], idxs[a:b + 1], times[a:b + 1]
    kept = set(idxs)
    for f in store.frames_dir(cid).glob("*.jpg"):
        if int(f.stem) not in kept:
            f.unlink()
    window = RunWindow(idxs[0], idxs[-1], window.method, window.confidence)

    # ---- full-resolution pass: follow cam + the board's own edge -------------------------------
    boxes = rider_boxes(sig.kp)
    fsize = tuple(cfg.render.follow_size)
    rects = follow_rects(boxes, tt, fsize[0] / fsize[1], (meta.width, meta.height))
    hits: list = [None] * len(idxs)
    with _stage(cid, "follow+board_edge") as info:
        fw = FollowWriter(store.follow_dir(cid), fsize, cfg.render.jpeg_quality)
        pos = {f: i for i, f in enumerate(idxs)}
        for fi, _, img in iter_frames(meta, idxs[0], idxs[-1]):
            if fi not in pos:
                continue
            i = pos[fi]
            fw.write(fi, img, rects[i])
            if i < na:  # on the mat only: in the air there is no edge to be on
                hits[i] = detect_tail_edge(img, sig.kp[i])
            report("Rendering the follow cam", 0.85 + 0.12 * (i + 1) / len(pos))
        info.update(frames=len(pos), board_edges=sum(h is not None for h in hits))
    edge_src, edge_anchor, edge_half = _fuse_board_edge(sig, hits, tt, na, pitch_deg, cal,
                                                         meta, cfg)

    with _stage(cid, "samples+events") as info:
        samples = fuse(to_samples(sig, spans, run_id, ViewRole.FALL_LINE), None)
        events = detect_events(sig, samples, spans, cfg)
        info.update(samples=len(samples), events=len(events))

    # ---- persist -----------------------------------------------------------------------------
    qa = {
        "residual_px": cal.residual_px,
        "detection_rate": float(np.mean([p is not None for p in poses])),
        "mean_kpt_conf": {
            ph.value: float(np.mean([p.scores.mean() for p, t in zip(poses, tt)
                                     if p is not None and ph in spans
                                     and spans[ph][0] <= t <= spans[ph][1]]
                                    or [0.0]))
            for ph in Phase if ph in spans
        },
        "rider_px_h": [float(np.nanmax(sig.rider_px_h)), float(np.nanmin(sig.rider_px_h))],
        "range_m": [float(np.nanmin(sig.values["range"][:na])),
                    float(np.nanmax(sig.values["range"][:na]))],
        "peak_speed_mps": max((s.value for s in samples
                               if s.metric == "speed_along" and s.validity is Validity.VALID),
                              default=float("nan")),
        "range_model": (None if sig.range_model is None else {
            "mad_log": sig.range_model.mad_log, "n_obs": sig.range_model.n_obs,
            "rows": [sig.range_model.y_lo, sig.range_model.y_hi]}),
        "terrain_rows": [cal.transition_y, cal.takeoff_y, cal.mat_top_y],
        "pitch": {"deg": pitch_deg, "sd": cal.pitch_sd_deg,
                  "source": "vanishing_point" if np.isfinite(cal.pitch_deg) else "config"},
        "jump": None if jump is None else jump.summary(),
        "stance": sig.stance_info,
        "board_edge_source": {s: int((edge_src == s).sum()) for s in ("board", "feet")},
        "events": {"total": len(events),
                   "strong": sum(e.severity == "strong" for e in events)},
        "backend": backend_name,
        "processing_s": round(time.perf_counter() - t_start, 1),
    }
    store.put_session(session_id, venue, cal)
    store.put_run(run_id, session_id, meta, PIPELINE_VERSION, cfg.hash, "ok",
                  sig.stance, sig.stance_width_m, qa)
    store.put_clip(meta, writer.size)
    store.put_window(cid, window)
    store.put_phases(run_id, spans, source)
    store.put_metrics(cid, samples)
    store.put_events(run_id, events)
    write_overlay(store.overlay_path(cid), sig, poses, cal, writer.scale, writer.size,
                  boxes, {"size": fsize, "rects": rects},
                  edges={"src": edge_src, "anchor": edge_anchor, "half": edge_half})
    log.info(json.dumps({"clip": cid, "stage": "done", "seconds": qa["processing_s"]}))
    report("Done", 1.0)
    return cid


def _fuse_board_edge(sig, hits, tt, na, pitch_deg, cal, meta, cfg) -> np.ndarray:
    """Board edge angle from the board's tail edge where it is seen; the foot-based estimate
    elsewhere. Updates sig in place and returns the per-frame source ("board" | "feet" | "")."""
    n = len(tt)
    toe_on_right = sig.stance == "left_lead"  # regular riders face image-right
    depression = np.radians(pitch_deg) + np.arctan((sig.contact[:, 1] - meta.height / 2)
                                                  / cal.focal_px)
    raw = np.full(n, np.nan)
    lengths = np.full(n, np.nan)
    for i, h in enumerate(hits):
        if h is not None:
            raw[i] = roll_from_segment(h.seg, toe_on_right, depression[i])
            lengths[i] = h.length
    board, err = clean_series(raw, lengths, tt, cfg.metrics.keypoint_noise_px)
    feet_val = sig.values["board_edge"].copy()
    feet_err = sig.errors["board_edge"].copy()
    use_board = np.isfinite(board)
    # The feet fallback has its own bias; shift it onto the board's level where both exist,
    # so switching source never makes the value jump.
    both = use_board & np.isfinite(feet_val)
    if both.sum() >= 5:
        feet_val = feet_val + float(np.median(board[both] - feet_val[both]))
    src = np.where(use_board, "board", np.where(np.isfinite(feet_val), "feet", ""))
    src[na:] = ""  # no edge angle in the air
    val = np.where(use_board, board, feet_val)
    e = np.where(use_board, err, feet_err * 1.5)  # the foot method is biased on the kicker
    # one more zero-phase pass over the whole stretch on the mat smooths any seam
    mat = np.isfinite(val[:na])
    if mat.sum() >= 15:
        idx = np.arange(na)
        fps = 1.0 / float(np.median(np.diff(tt)))
        seg = lowpass(np.interp(idx, idx[mat], val[:na][mat]), fps, 3.0)
        val[:na] = np.where(mat, seg, np.nan)
    val[na:] = np.nan
    e[na:] = np.nan
    sig.values["board_edge"], sig.errors["board_edge"] = val, e
    anchor, half = smooth_anchor(hits, sig.contact, sig.rider_px_h, tt, na)
    return src, anchor, half


def _focal(meta, cfg) -> float:
    """Focal length in pixels for this clip: the configured value is for a frame whose long
    edge is `focal_ref_long_edge` px (4K), so it scales with the clip's resolution."""
    return cfg.camera.focal_px * max(meta.width, meta.height) / cfg.camera.focal_ref_long_edge


def _feet_y(p: PoseFrame) -> float:
    return float(np.max(p.keypoints[[K.L_HEEL, K.R_HEEL, K.L_ANKLE, K.R_ANKLE], 1]))


def _air_roi(mat_full: np.ndarray, lip_y: int) -> np.ndarray:
    """Where a flying rider can be: above the lip, over the width of the jump."""
    h, w = mat_full.shape
    band = mat_full[lip_y:lip_y + 40]
    xs = np.flatnonzero(band.any(axis=0))
    x0, x1 = (int(xs[0]), int(xs[-1])) if xs.size else (0, w)
    grow = int(0.6 * (x1 - x0)) + 200
    roi = np.zeros_like(mat_full)
    roi[:min(h, lip_y + int(0.08 * h)), max(0, x0 - grow):min(w, x1 + grow)] = 255
    return roi


def _pitch(meta, bg, cfg) -> tuple[float, float]:
    """Camera pitch from a full-resolution median background (vertical vanishing point)."""
    n = cfg.trim.background_samples
    pick = set(np.linspace(0, max(0, meta.n_frames - 20), n).astype(int).tolist())
    frames = [img for fi, _, img in iter_frames(meta) if fi in pick]
    if len(frames) < 3:
        return float("nan"), float("nan")
    full = np.median(np.stack(frames), axis=0).astype(np.uint8)
    mat = cv2.resize(bg.mat_mask, (meta.width, meta.height), interpolation=cv2.INTER_NEAREST)
    mat = cv2.dilate(mat, np.ones((61, 61), np.uint8))
    deg, sd, n_in = estimate_pitch(full, mat, _focal(meta, cfg))
    log.info(json.dumps({"clip": meta.clip_id, "stage": "pitch", "deg": deg, "sd": sd,
                         "inliers": n_in}))
    return deg, sd


def detect_events(sig, samples, spans, cfg) -> list:
    """Assemble per-metric series from the samples and run the sudden-movement rules."""
    frames = np.asarray(sig.frame_idx)
    pos = {int(f): i for i, f in enumerate(frames)}
    n = len(frames)
    series: dict[str, tuple[np.ndarray, np.ndarray, list[str], np.ndarray]] = {}
    for s in samples:
        if s.metric not in series:
            series[s.metric] = (np.full(n, np.nan), np.full(n, np.nan), ["invalid"] * n,
                                np.zeros(n))
        v, e, val, conf = series[s.metric]
        i = pos[s.frame_idx]
        v[i], e[i], val[i], conf[i] = s.value, s.err_est, s.validity.value, s.confidence
    phases = [p.value for p in phase_at(sig.t_sec, spans)]
    if sig.board_bump is not None:
        on_mat = [p != Phase.AIR.value for p in phases]
        err = np.full(n, 0.01)
        series["board_bump"] = (sig.board_bump, err,
                                ["valid" if ok else "invalid" for ok in on_mat],
                                sig.confidence["com_height"])
    events = detect(frames, sig.t_sec, phases, series, k_mad=cfg.events.k_mad)
    if "com_toe_heel" in series:
        v, e, val, conf = series["com_toe_heel"]
        usable = (np.isfinite(v) & np.array([x != "invalid" for x in val]) & (conf >= 0.5))
        toe_sign = 1.0 if sig.stance == "left_lead" else -1.0
        t_to = spans[Phase.AIR][0] if Phase.AIR in spans else None
        be = series["board_edge"][0] if "board_edge" in series else None
        events += detect_edge_changes(frames, sig.t_sec, phases, v, e, usable, toe_sign,
                                      t_takeoff=t_to, board_edge=be)
        events.sort(key=lambda ev: ev.t_peak)
    return events
