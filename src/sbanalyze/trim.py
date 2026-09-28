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
    """Pale, desaturated, floodlit matting. Keep the largest such region only."""
    hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
    raw = ((hsv[..., 1] < 55) & (hsv[..., 2] > 110)).astype(np.uint8) * 255
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(raw)
    if n <= 1:
        return raw
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == biggest).astype(np.uint8) * 255
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))


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
