"""Local store: SQLite catalog, long-format Parquet metrics, JPEG frame cache, overlay JSON."""

from __future__ import annotations

import json
import shutil
import sqlite3
import threading
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import Calibration, ClipMeta, MetricSample, Phase, PoseFrame, RunWindow

SCHEMA = """
CREATE TABLE IF NOT EXISTS session (
    id TEXT PRIMARY KEY, venue TEXT, camera_pose_note TEXT,
    focal_px REAL, stature_m REAL, chain_fraction REAL,
    left_edge_x BLOB, mat_top_y INTEGER, calib_method TEXT, residual_px REAL,
    transition_y REAL, takeoff_y REAL, pitch_deg REAL, pitch_sd_deg REAL
);
CREATE TABLE IF NOT EXISTS run (
    id TEXT PRIMARY KEY, session_id TEXT REFERENCES session(id), recorded_at TEXT,
    pipeline_version TEXT, config_hash TEXT, status TEXT, stance TEXT, stance_width_m REAL,
    qa_json TEXT
);
CREATE TABLE IF NOT EXISTS clip (
    id TEXT PRIMARY KEY, run_id TEXT REFERENCES run(id), view TEXT, path TEXT, fps REAL,
    width INTEGER, height INTEGER, n_frames INTEGER, rotation_deg INTEGER,
    t_offset_s REAL, sync_method TEXT, frame_width INTEGER, frame_height INTEGER
);
CREATE TABLE IF NOT EXISTS run_window (
    clip_id TEXT PRIMARY KEY REFERENCES clip(id), start_frame INTEGER, end_frame INTEGER,
    method TEXT, confidence REAL
);
CREATE TABLE IF NOT EXISTS event (
    run_id TEXT REFERENCES run(id), seq INTEGER, metric TEXT, kind TEXT, severity TEXT,
    phase TEXT, t_start REAL, t_end REAL, t_peak REAL, frame_start INTEGER, frame_end INTEGER,
    frame_peak INTEGER, delta REAL, peak_rate REAL, text TEXT,
    PRIMARY KEY (run_id, seq)
);
CREATE TABLE IF NOT EXISTS phase_span (
    run_id TEXT REFERENCES run(id), phase TEXT, t_start REAL, t_end REAL, source TEXT,
    PRIMARY KEY (run_id, phase)
);
"""


class Store:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        (root / "metrics").mkdir(exist_ok=True)
        self._path = root / "catalog.sqlite"
        self._local = threading.local()
        self.db.executescript(SCHEMA)

    @property
    def db(self) -> sqlite3.Connection:
        """One connection per thread: the API serves requests from a thread pool, and a single
        sqlite3 connection must not be used from several threads at once."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._path, timeout=10)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    # ---- paths -----------------------------------------------------------------------------
    def frames_dir(self, clip_id: str) -> Path:
        return self.root / "frames" / clip_id

    def follow_dir(self, clip_id: str) -> Path:
        return self.root / "follow" / clip_id

    def overlay_path(self, clip_id: str) -> Path:
        return self.root / "overlay" / f"{clip_id}.json"

    def metrics_path(self, clip_id: str) -> Path:
        return self.root / "metrics" / f"{clip_id}.parquet"

    def poses_path(self, clip_id: str) -> Path:
        return self.root / "poses" / f"{clip_id}.npz"

    # ---- raw poses (so metrics can be recomputed without re-running inference) ------------
    def put_poses(self, clip_id: str, backend: str, window: RunWindow, idxs: list[int],
                  times: list[float], poses: list[PoseFrame | None]) -> None:
        present = np.array([p is not None for p in poses])
        kp = np.stack([p.keypoints if p else np.full((26, 2), np.nan, np.float32) for p in poses])
        sc = np.stack([p.scores if p else np.zeros(26, np.float32) for p in poses])
        bb = np.array([p.bbox if p else (0, 0, 0, 0) for p in poses], np.int32)
        hh = np.array([p.rider_px_h if p else np.nan for p in poses], np.float32)
        self.poses_path(clip_id).parent.mkdir(exist_ok=True)
        np.savez_compressed(self.poses_path(clip_id), backend=backend,
                            window=[window.start_frame, window.end_frame], idxs=idxs, times=times,
                            present=present, kp=kp, sc=sc, bbox=bb, rider_px_h=hh)

    def get_poses(self, clip_id: str, backend: str, window: RunWindow
                  ) -> tuple[list[int], list[float], list[PoseFrame | None]] | None:
        path = self.poses_path(clip_id)
        if not path.exists():
            return None
        z = np.load(path)
        if str(z["backend"]) != backend or list(z["window"]) != [window.start_frame,
                                                                 window.end_frame]:
            return None
        poses = [PoseFrame(int(i), float(t), z["kp"][k], z["sc"][k],
                           tuple(int(v) for v in z["bbox"][k]), float(z["rider_px_h"][k]))
                 if z["present"][k] else None
                 for k, (i, t) in enumerate(zip(z["idxs"], z["times"], strict=True))]
        return [int(i) for i in z["idxs"]], [float(t) for t in z["times"]], poses

    # ---- sessions --------------------------------------------------------------------------
    def get_calibration(self, session_id: str) -> Calibration | None:
        r = self.db.execute("SELECT * FROM session WHERE id=?", (session_id,)).fetchone()
        if r is None or r["left_edge_x"] is None:
            return None
        return Calibration(
            focal_px=r["focal_px"], stature_m=r["stature_m"], chain_fraction=r["chain_fraction"],
            left_edge_x=np.frombuffer(r["left_edge_x"], np.float32).copy(),
            mat_top_y=r["mat_top_y"], method=r["calib_method"], residual_px=r["residual_px"],
            transition_y=r["transition_y"], takeoff_y=r["takeoff_y"],
            pitch_deg=r["pitch_deg"], pitch_sd_deg=r["pitch_sd_deg"],
        )

    def put_session(self, session_id: str, venue: str, cal: Calibration | None,
                    note: str = "") -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO session VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (session_id, venue, note,
             cal.focal_px if cal else None, cal.stature_m if cal else None,
             cal.chain_fraction if cal else None,
             cal.left_edge_x.astype(np.float32).tobytes() if cal else None,
             cal.mat_top_y if cal else None, cal.method if cal else None,
             cal.residual_px if cal else None, cal.transition_y if cal else None,
             cal.takeoff_y if cal else None, cal.pitch_deg if cal else None,
             cal.pitch_sd_deg if cal else None))
        self.db.commit()

    # ---- runs and clips --------------------------------------------------------------------
    def put_run(self, run_id: str, session_id: str, meta: ClipMeta, version: str,
                cfg_hash: str, status: str, stance: str = "", stance_w: float = 0.0,
                qa: dict | None = None) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO run VALUES (?,?,?,?,?,?,?,?,?)",
            (run_id, session_id, meta.recorded_at.isoformat(), version, cfg_hash, status,
             stance, stance_w, json.dumps(qa or {})))
        self.db.commit()

    def put_clip(self, meta: ClipMeta, frame_size: tuple[int, int] | None,
                 sync_method: str = "primary") -> None:
        fw, fh = frame_size or (None, None)
        self.db.execute(
            "INSERT OR REPLACE INTO clip VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (meta.clip_id, meta.run_id, meta.view.value, str(meta.path), meta.fps, meta.width,
             meta.height, meta.n_frames, meta.rotation_deg, meta.t_offset_s, sync_method,
             fw, fh))
        self.db.commit()

    def put_window(self, clip_id: str, w: RunWindow) -> None:
        self.db.execute("INSERT OR REPLACE INTO run_window VALUES (?,?,?,?,?)",
                        (clip_id, w.start_frame, w.end_frame, w.method, w.confidence))
        self.db.commit()

    def put_phases(self, run_id: str, spans: dict[Phase, tuple[float, float]],
                   source: str) -> None:
        self.db.execute("DELETE FROM phase_span WHERE run_id=?", (run_id,))
        self.db.executemany("INSERT INTO phase_span VALUES (?,?,?,?,?)",
                            [(run_id, p.value, a, b, source) for p, (a, b) in spans.items()])
        self.db.commit()

    def put_events(self, run_id: str, events: list) -> None:
        self.db.execute("DELETE FROM event WHERE run_id=?", (run_id,))
        self.db.executemany(
            "INSERT INTO event VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [(run_id, k, e.metric, e.kind, e.severity, e.phase, e.t_start, e.t_end, e.t_peak,
              e.frame_start, e.frame_end, e.frame_peak, e.delta, e.peak_rate, e.text)
             for k, e in enumerate(events)])
        self.db.commit()

    def get_events(self, run_id: str) -> list[dict]:
        return [dict(r) for r in self.db.execute(
            "SELECT * FROM event WHERE run_id=? ORDER BY t_peak", (run_id,))]

    def put_metrics(self, clip_id: str, samples: list[MetricSample]) -> Path:
        df = pd.DataFrame({
            "run_id": [s.run_id for s in samples],
            "view": pd.Categorical([s.view.value for s in samples]),
            "frame_idx": np.array([s.frame_idx for s in samples], np.int32),
            "t_sec": np.array([s.t_sec for s in samples], np.float32),
            "phase": pd.Categorical([s.phase.value for s in samples]),
            "metric": pd.Categorical([s.metric for s in samples]),
            "value": np.array([s.value for s in samples], np.float32),
            "confidence": np.array([s.confidence for s in samples], np.float32),
            "rider_px_h": np.array([s.rider_px_h for s in samples], np.float32),
            "err_est": np.array([s.err_est for s in samples], np.float32),
            "validity": pd.Categorical([s.validity.value for s in samples]),
            "valid": np.array([s.valid for s in samples], bool),
        })
        path = self.metrics_path(clip_id)
        df.to_parquet(path, index=False)
        return path

    # ---- reads -----------------------------------------------------------------------------
    def list_clips(self) -> list[dict]:
        q = """SELECT c.id AS clip_id, c.run_id, c.view, c.path, c.fps, c.frame_width,
                      c.frame_height, r.session_id, r.recorded_at, r.status, r.stance,
                      w.start_frame, w.end_frame, w.method AS window_method, r.qa_json
               FROM clip c JOIN run r ON r.id = c.run_id
               LEFT JOIN run_window w ON w.clip_id = c.id
               ORDER BY r.recorded_at"""
        out = []
        for row in self.db.execute(q):
            d = dict(row)
            qa = json.loads(d.pop("qa_json") or "{}")
            jump = qa.get("jump") or {}
            d["summary"] = {
                "air_time_s": jump.get("air_time_s"),
                "jump_height_m": jump.get("apex_height_m"),
                "peak_speed_mps": qa.get("peak_speed_mps"),
                "events": (qa.get("events") or {}).get("total"),
                "strong": (qa.get("events") or {}).get("strong"),
                "stance": (qa.get("stance") or {}).get("value"),
            }
            # a representative frame for the thumbnail: the approach, before the lip
            if d["start_frame"] is not None and d["end_frame"] is not None:
                d["thumb_frame"] = d["start_frame"] + int(0.55 * (d["end_frame"] - d["start_frame"]))
            d["file_name"] = Path(d["path"]).name
            d["uploaded_copy"] = self._is_upload(Path(d["path"]))
            out.append(d)
        return out

    def _is_upload(self, path: Path) -> bool:
        try:
            return path.resolve().is_relative_to((self.root / "videos").resolve())
        except OSError:
            return False

    def delete_clip(self, clip_id: str) -> dict:
        """Remove a run and everything derived from it. The source video is deleted only if it
        is the app's own uploaded copy; videos in the user's folders are never touched."""
        row = self.db.execute("SELECT run_id, path FROM clip WHERE id=?", (clip_id,)).fetchone()
        if row is None:
            raise KeyError(clip_id)
        run_id, src = row["run_id"], Path(row["path"])
        for sql, arg in (("DELETE FROM event WHERE run_id=?", run_id),
                         ("DELETE FROM phase_span WHERE run_id=?", run_id),
                         ("DELETE FROM run_window WHERE clip_id=?", clip_id),
                         ("DELETE FROM clip WHERE id=?", clip_id),
                         ("DELETE FROM run WHERE id=?", run_id)):
            self.db.execute(sql, (arg,))
        self.db.commit()
        for d in (self.frames_dir(clip_id), self.follow_dir(clip_id)):
            if d.exists():
                shutil.rmtree(d)
        for f in (self.metrics_path(clip_id), self.overlay_path(clip_id),
                  self.poses_path(clip_id)):
            f.unlink(missing_ok=True)
        removed_source = False
        if self._is_upload(src) and src.exists():
            src.unlink()
            removed_source = True
        return {"clip_id": clip_id, "removed_source": removed_source, "source": str(src)}

    def clip_detail(self, clip_id: str) -> dict | None:
        rows = [c for c in self.list_clips() if c["clip_id"] == clip_id]
        if not rows:
            return None
        d = rows[0]
        run = self.db.execute("SELECT * FROM run WHERE id=?", (d["run_id"],)).fetchone()
        d["qa"] = json.loads(run["qa_json"] or "{}")
        d["pipeline_version"] = run["pipeline_version"]
        d["config_hash"] = run["config_hash"]
        d["stance_width_m"] = run["stance_width_m"]
        d["phases"] = [dict(r) for r in self.db.execute(
            "SELECT phase, t_start, t_end, source FROM phase_span WHERE run_id=? "
            "ORDER BY t_start", (d["run_id"],))]
        ses = self.db.execute("SELECT venue, calib_method, residual_px, focal_px, stature_m, "
                              "transition_y, takeoff_y, mat_top_y, pitch_deg, pitch_sd_deg "
                              "FROM session WHERE id=?", (d["session_id"],)).fetchone()
        d["session"] = dict(ses) if ses else None
        return d

    def read_metrics(self, clip_id: str) -> pd.DataFrame:
        return pd.read_parquet(self.metrics_path(clip_id))
