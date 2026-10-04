"""Locate the run: median background, motion energy on the mat, longest sustained window."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import cv2
import numpy as np

from .contracts import PipelineError, RunWindow


@dataclass
class Background:
    image: np.ndarray  # BGR, analysis resolution
    mat_mask: np.ndarray  # uint8 {0,255}, analysis resolution
    scale: float  # analysis px per full-res px


def build_background(frames: list[np.ndarray], n_samples: int, scale: float) -> Background:
    idx = np.linspace(0, len(frames) - 1, min(n_samples, len(frames))).astype(int)
    bg = np.median(np.stack([frames[i] for i in idx]), axis=0).astype(np.uint8)
    return Background(bg, mat_mask(bg), scale)


def mat_mask(bg: np.ndarray) -> np.ndarray:
    """The white dry-slope mat the rider rides on.

    Pale and unsaturated, and never grass-coloured: under an overcast sky the grass banks turn
    pale grey-green and would otherwise pass as mat. The camera stands at the top of the in-run
    looking down it, so the mat is the pale region that reaches the bottom centre of the frame.
    That rule rejects the platform, gravel and airbag areas beyond the lip.
    """
    hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # measured: mat saturation <= 25 (it can carry a faint green-yellow tint in daylight);
    # grass is typically 90-140, and shaded grass still above ~30
    greenish = (h >= 28) & (h <= 95) & (s > 30)
    raw = ((s < 45) & (v > 110) & ~greenish).astype(np.uint8) * 255
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(raw)
    if n <= 1:
        return raw
    # the component under the bottom centre of the frame (the camera stands on the in-run);
    # fall back to the largest one
    hh, ww = raw.shape
    ys = np.arange(hh - 1, int(hh * 0.75), -1)
    pick = 0
    for y in ys:
        row = labels[y, ww // 4: 3 * ww // 4]
        vals, counts = np.unique(row[row > 0], return_counts=True)
        if vals.size:
            pick = int(vals[np.argmax(counts)])
            break
    if pick == 0:
        pick = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == pick).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    return _cut_at_lip(mask)


def _cut_at_lip(mask: np.ndarray, jump: float = 1.2, sustain: int = 4) -> np.ndarray:
    """Remove whatever white lies beyond the kicker's lip (landing airbag, platform).

    The in-run is a lane whose width changes smoothly up the picture. Beyond the lip, a landing
    airbag or platform spans far wider, so the white area's width jumps. Scanning up from the
    camera, the first sustained jump marks the lip; everything above it is cut.
    """
    width = (mask > 0).sum(axis=1).astype(float)
    rows = np.flatnonzero(width > 0)
    if rows.size < 40:
        return mask
    top, bottom = int(rows[0]), int(rows[-1])
    run = 0
    for y in range(int(bottom - 0.3 * (bottom - top)), top, -1):
        below = width[y + 3:y + 20]
        below = below[below > 0]
        if below.size < 5:
            run = 0
            continue
        run = run + 1 if width[y] > jump * np.median(below) else 0
        if run >= sustain:
            lip = y + sustain
            out = mask.copy()
            out[:lip] = 0
            return _keep_camera_component(out)
    return mask


def _keep_camera_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the white region that reaches the bottom centre (where the camera stands)."""
    n, labels = cv2.connectedComponents((mask > 0).astype(np.uint8))
    if n <= 2:
        return mask
    hh, ww = mask.shape
    row = labels[hh - 1, ww // 4: 3 * ww // 4]
    vals, counts = np.unique(row[row > 0], return_counts=True)
    if not vals.size:
        return mask
    return (labels == int(vals[np.argmax(counts)])).astype(np.uint8) * 255


def motion_energy(frames: list[np.ndarray], bg: Background, diff_threshold: int) -> np.ndarray:
    """Largest moving blob per frame as a fraction of frame area, restricted to the mat."""
    gray_bg = cv2.GaussianBlur(cv2.cvtColor(bg.image, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    roi = cv2.dilate(bg.mat_mask, np.ones((31, 31), np.uint8))
    area = float(gray_bg.size)
    out = np.zeros(len(frames), np.float32)
    for i, f in enumerate(frames):
        g = cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        m = ((cv2.absdiff(g, gray_bg) > diff_threshold) & (roi > 0)).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(m)
        out[i] = stats[1:, cv2.CC_STAT_AREA].max() / area if n > 1 else 0.0
    return out


def find_run_window(energy: np.ndarray, fps: float, min_blob_frac: float, pad_s: float,
                    max_gap_s: float = 0.2, min_len_s: float = 1.0) -> RunWindow:
    present = energy > min_blob_frac
    # close short gaps (occlusion, a frame where the rider matches the background)
    gap = round(max_gap_s * fps)
    idx = np.flatnonzero(present)
    if idx.size == 0:
        raise PipelineError("trim", "needs_manual_window", "no motion on the mat")
    for a, b in itertools.pairwise(idx):
        if 1 < b - a <= gap:
            present[a:b] = True
    # longest run of presence
    best, cur_start, best_span = None, None, 0
    for i, p in enumerate(np.append(present, False)):
        if p and cur_start is None:
            cur_start = i
        elif not p and cur_start is not None:
            if i - cur_start > best_span:
                best, best_span = (cur_start, i - 1), i - cur_start
            cur_start = None
    if best is None or best_span < min_len_s * fps:
        raise PipelineError("trim", "needs_manual_window", "no sustained rider motion")
    pad = round(pad_s * fps)
    start, end = max(0, best[0] - pad), min(len(energy) - 1, best[1] + pad)
    conf = float(present[best[0]:best[1] + 1].mean())
    return RunWindow(start, end, "motion_energy", conf)
