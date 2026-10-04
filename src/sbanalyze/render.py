"""Frame cache and overlay geometry. Overlays are JSON, not burned in, so layers toggle freely.

Two frame caches are written:

* the full frame at display resolution, for context;
* a follow cam: a fixed-aspect crop that keeps the rider centred, cut from the full-resolution
  frame. At the lip the rider is ~200 px tall in 4K but only ~57 px in the display cache, so
  zooming the display cache would show mush. The follow cam keeps every pixel.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d

from .contracts import HALPE26_EDGES, Calibration, PoseFrame


class FrameWriter:
    def __init__(self, out_dir: Path, full_hw: tuple[int, int], long_edge: int, quality: int,
                 clear: bool = True):
        self.dir = out_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        if clear:
            for old in self.dir.glob("*.jpg"):
                old.unlink()
        h, w = full_hw
        self.scale = long_edge / max(h, w)
        self.size = (round(w * self.scale), round(h * self.scale))
        self.quality = quality

    def write(self, frame_idx: int, image: np.ndarray) -> None:
        small = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(self.dir / f"{frame_idx:05d}.jpg"), small,
                    [cv2.IMWRITE_JPEG_QUALITY, self.quality])


def rider_boxes(kp: np.ndarray, pad: float = 0.12) -> np.ndarray:
    """Tight per-frame rider box from the (filtered) keypoints, padded for board and helmet."""
    lo, hi = np.nanmin(kp, axis=1), np.nanmax(kp, axis=1)
    size = hi - lo
    return np.concatenate([lo - pad * size, hi + pad * size], axis=1)


def follow_rects(boxes: np.ndarray, t: np.ndarray, aspect: float, frame_wh: tuple[int, int],
                 margin: float = 1.9, min_h: float = 420.0, center_s: float = 0.2,
                 zoom_s: float = 0.6) -> np.ndarray:
    """Per-frame crop (x0, y0, w, h) in full-res px: rider centred, smoothly zoomed.

    The centre follows the rider closely and the zoom changes slowly, so the camera glides
    rather than pumps as the rider crouches and extends.
    """
    ok = np.isfinite(boxes).all(axis=1)
    idx = np.arange(len(boxes))
    b = boxes.copy()
    for c in range(4):
        b[:, c] = np.interp(idx, idx[ok], boxes[ok, c])
    dt = float(np.median(np.diff(t))) if t.size > 1 else 1 / 60
    cx = uniform_filter1d((b[:, 0] + b[:, 2]) / 2, max(1, round(center_s / dt)), mode="nearest")
    cy = uniform_filter1d((b[:, 1] + b[:, 3]) / 2, max(1, round(center_s / dt)), mode="nearest")
    need_h = np.maximum(b[:, 3] - b[:, 1], (b[:, 2] - b[:, 0]) / aspect) * margin
    h = uniform_filter1d(np.maximum(need_h, min_h), max(1, round(zoom_s / dt)), mode="nearest")
    fw, fh = frame_wh
    h = np.minimum(np.minimum(h, fh), fw / aspect)  # never wider or taller than the picture
    w = h * aspect
    # always centred on the rider; where the crop runs past the picture's edge (the rider is
    # at the border, e.g. dropping in under the camera) it is padded with dark background
    x0 = cx - w / 2
    y0 = cy - h / 2
    return np.stack([x0, y0, w, h], axis=1)


class FollowWriter:
    def __init__(self, out_dir: Path, size: tuple[int, int], quality: int):
        self.dir = out_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        for old in self.dir.glob("*.jpg"):
            old.unlink()
        self.size = size  # (w, h)
        self.quality = quality

    def write(self, frame_idx: int, image: np.ndarray, rect: np.ndarray) -> None:
        x0, y0, w, _ = rect
        s = self.size[0] / w
        m = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]], np.float32)
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
        crop = cv2.warpAffine(image, m, self.size, flags=interp,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(12, 15, 20))
        cv2.imwrite(str(self.dir / f"{frame_idx:05d}.jpg"), crop,
                    [cv2.IMWRITE_JPEG_QUALITY, self.quality])


def _r(a: np.ndarray, scale: float) -> list | None:
    if a is None or not np.isfinite(a).all():
        return None
    return np.round(np.asarray(a, np.float64) * scale, 1).tolist()


def write_overlay(path: Path, sig, poses: list[PoseFrame | None], cal: Calibration,
                  scale: float, size: tuple[int, int], rider_box: np.ndarray,
                  follow: dict | None = None, edges: dict | None = None) -> None:
    frames = []
    for i, fi in enumerate(sig.frame_idx):
        p = poses[i]
        frames.append({
            "f": int(fi),
            "detected": p is not None,
            "rider": _r(rider_box[i], scale),
            "edge": (None if not np.isfinite(sig.values["board_edge"][i])
                     else round(float(sig.values["board_edge"][i]), 1)),
            # where to draw the tilt bar: the board tail's centre and half-width, smoothed over
            # time (display px); and whether the angle came from the board or the feet
            "edge_anchor": (None if not edges or not np.isfinite(edges["anchor"][i]).all()
                            else _r(edges["anchor"][i], scale)),
            "edge_half": (None if not edges or not np.isfinite(edges["half"][i])
                          else round(float(edges["half"][i]) * scale, 1)),
            "edge_src": "" if not edges else str(edges["src"][i]),
            "kp": _r(sig.kp[i], scale),
            "sc": np.round(sig.raw_scores[i], 2).tolist(),
            "bbox": _r(np.array(p.bbox, np.float64), scale) if p is not None else None,
            "com": _r(sig.com[i], scale),
            "contact": _r(sig.contact[i], scale),
            "board": [_r(sig.trail_foot[i], scale), _r(sig.lead_foot[i], scale)],
        })
    rows = np.arange(0, cal.left_edge_x.size, 16)
    edge = [[round(float(cal.left_edge_x[y]) * scale, 1), round(y * scale, 1)]
            for y in rows if np.isfinite(cal.left_edge_x[y])]
    doc = {
        "version": int(time.time()),
        "width": size[0], "height": size[1], "scale": scale,
        "edges": [list(e) for e in HALPE26_EDGES],
        "mat_edge": edge,
        "frames": frames,
        # follow cam: rects are in display-frame px (same space as kp), one per frame
        "follow": None if follow is None else {
            "width": follow["size"][0], "height": follow["size"][1],
            "rects": np.round(follow["rects"] * scale, 1).tolist(),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, separators=(",", ":")))
