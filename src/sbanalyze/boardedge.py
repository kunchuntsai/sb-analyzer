"""Board edge angle from the board itself: its tail edge, seen from behind.

The camera looks at the rider from behind, so the board's tail (the end nearest the camera) shows
as a short, straight bottom edge just below the bindings. That edge runs across the board, so its
tilt in the image is the board's roll, i.e. the toe/heel edge angle.

Why not the feet: the heel->toe line of each foot is only roughly across the board. Bindings are
angled (e.g. +15 / -6 deg), so part of each foot line runs along the board. When the board pitches
nose-up on the kicker face, that part becomes a false tilt of several degrees. The board's own
edge has no such coupling.

Detection per frame (full resolution):
  1. search band just under the lowest foot keypoints, centred on the feet;
  2. line segments (LSD) within +-35 deg of horizontal;
  3. keep only board-sized ones: about as long as a boot is wide, not a long ground line such as
     the lip or a mat seam; centred under the feet;
  4. require a real boundary: the strips above and below the line must differ in brightness;
  5. score = length x contrast x closeness to the feet; best wins.
The series is then cleaned over time: outliers against a running median are dropped and short
gaps are bridged. Frames without a trustworthy edge fall back to the foot-based estimate, and the
source of every value is recorded.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt

from .contracts import K

FEET = [K.L_HEEL, K.R_HEEL, K.L_BIG_TOE, K.R_BIG_TOE, K.L_SMALL_TOE, K.R_SMALL_TOE]
_LSD = None


def _lsd():
    global _LSD
    if _LSD is None:
        _LSD = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    return _LSD


@dataclass(frozen=True)
class EdgeHit:
    seg: tuple[float, float, float, float]  # full-res image px (x1, y1, x2, y2)
    angle_img: float  # degrees, image angle of the segment in (-90, 90]
    length: float  # px
    contrast: float  # grey-level difference across the edge


def _strip_mean(gray: np.ndarray, p: np.ndarray, q: np.ndarray, off: float, n: int = 9) -> float:
    """Mean brightness along a line parallel to p-q, shifted by `off` px along its normal."""
    d = q - p
    nrm = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
    ts = np.linspace(0.15, 0.85, n)
    pts = p[None, :] + ts[:, None] * d[None, :] + off * nrm[None, :]
    h, w = gray.shape
    xs = np.clip(pts[:, 0].round().astype(int), 0, w - 1)
    ys = np.clip(pts[:, 1].round().astype(int), 0, h - 1)
    return float(gray[ys, xs].mean())


def detect_tail_edge(img: np.ndarray, kp: np.ndarray, min_contrast: float = 12.0
                     ) -> EdgeHit | None:
    """The board's tail edge in one full-resolution frame, or None."""
    feet = kp[FEET]
    if not np.isfinite(feet).all():
        return None
    boot = max(24.0, float(np.ptp(feet[:, 0])), float(np.ptp(feet[:, 1])))
    cx = float(np.median(feet[:, 0]))
    low = float(feet[:, 1].max())
    x0, x1 = int(cx - 1.4 * boot), int(cx + 1.4 * boot)
    y0, y1 = int(low - 0.45 * boot), int(low + 0.75 * boot)
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 - x0 < 16 or y1 - y0 < 8:
        return None
    roi = img[y0:y1, x0:x1]
    up = max(1.0, 260.0 / max(roi.shape[:2]))  # small far-away boards: upsample for LSD
    g = cv2.cvtColor(cv2.resize(roi, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC),
                     cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    segs = _lsd().detect(g)[0]
    if segs is None:
        return None
    best, best_score = None, 0.0
    for sx1, sy1, sx2, sy2 in segs.reshape(-1, 4):
        p, q = np.array([sx1, sy1]) / up, np.array([sx2, sy2]) / up
        length = float(np.linalg.norm(q - p))
        if not (0.45 * boot <= length <= 2.3 * boot):  # board-sized, not a ground line
            continue
        ang = float(np.degrees(np.arctan2(q[1] - p[1], q[0] - p[0])))
        ang = (ang + 90) % 180 - 90
        if abs(ang) > 35:
            continue
        mid = (p + q) / 2 + [x0, y0]
        if abs(mid[0] - cx) > 0.7 * boot:  # centred under the feet
            continue
        # and actually under them: most of the segment within the bindings' horizontal extent
        fx0, fx1 = feet[:, 0].min() - 0.25 * boot, feet[:, 0].max() + 0.25 * boot
        sx_lo, sx_hi = sorted((p[0] + x0, q[0] + x0))
        if min(sx_hi, fx1) - max(sx_lo, fx0) < 0.6 * (sx_hi - sx_lo):
            continue
        pu, qu = np.array([sx1, sy1]), np.array([sx2, sy2])
        off = max(2.0, 0.06 * boot * up)
        s1, s2 = _strip_mean(g, pu, qu, off), _strip_mean(g, pu, qu, -off)
        contrast = abs(s1 - s2)
        if contrast < min_contrast:
            continue
        # the side facing the feet must be board: it has to look like the surface between this
        # line and the bindings. A mat seam or the lip beside/behind the board fails this.
        gap = (mid[1] - low) * up  # px (upsampled) from the lowest foot point down to the line
        if gap > 0.12 * boot * up:
            # normal pointing toward the feet (up in the image)
            nrm = np.array([-(qu - pu)[1], (qu - pu)[0]]) / (np.linalg.norm(qu - pu) + 1e-9)
            toward_feet = off if nrm[1] < 0 else -off
            above = s1 if toward_feet == off else s2
            board = _strip_mean(g, pu, qu, np.sign(toward_feet) * gap * 0.5)
            if abs(above - board) > 0.5 * contrast:
                continue
        near = 1.0 / (1.0 + abs(mid[1] - (low + 0.1 * boot)) / boot)
        score = length * contrast * near
        if score > best_score:
            a, b = p + [x0, y0], q + [x0, y0]
            best, best_score = EdgeHit((a[0], a[1], b[0], b[1]), ang, length, contrast), score
    return best


def roll_from_segment(seg: tuple[float, float, float, float], toe_on_right: bool,
                      depression_rad: float, gravity_rad: float = 0.0) -> float:
    """Board roll in degrees, + toe edge (toe side lower) / - heel edge.

    `gravity_rad` is how far true down leans from image down at the segment (see
    calib.gravity_tilt); the segment is rotated into gravity's frame first, so a phone held
    slightly rolled does not show up as board tilt.
    """
    x1, y1, x2, y2 = seg
    if gravity_rad:
        c, s = np.cos(gravity_rad), np.sin(gravity_rad)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        hx, hy = (x2 - x1) / 2, (y2 - y1) / 2
        hx, hy = hx * c - hy * s, hx * s + hy * c
        x1, y1, x2, y2 = mx - hx, my - hy, mx + hx, my + hy
    # orient the segment from the heel side to the toe side of the picture
    if (x2 - x1 > 0) != toe_on_right:
        x1, y1, x2, y2 = x2, y2, x1, y1
    dx, dy = abs(x2 - x1), (y2 - y1) / max(np.cos(depression_rad), 0.2)
    return float(np.degrees(np.arctan2(dy, dx)))


def lowpass(x: np.ndarray, fps: float, cutoff_hz: float) -> np.ndarray:
    """Zero-phase Butterworth low-pass; no lag, no stair-steps."""
    if x.size < 15:
        return median_filter(x, size=5, mode="nearest")
    b, a = butter(2, min(0.99, cutoff_hz / (fps / 2)), btype="low")
    return filtfilt(b, a, x)


def clean_series(angles: np.ndarray, lengths: np.ndarray, t: np.ndarray, noise_px: float,
                 max_dev: float = 6.0, max_gap_s: float = 0.5, cutoff_hz: float = 3.0
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Board-edge angle over time: reject outliers, bridge gaps, smooth. Returns (value, err).

    A board rolls from edge to edge over a few tenths of a second, never within a frame or two,
    so the series is low-passed at `cutoff_hz` with a zero-phase filter. Single-frame jitter of
    the detected edge (a few degrees) is removed without lag or stair-steps. Gaps up to
    `max_gap_s` between detections are bridged; longer stretches stay empty for the caller.
    """
    a = angles.astype(float).copy()
    ok = np.isfinite(a)
    if ok.sum() < 5:
        return np.full_like(a, np.nan), np.full_like(a, np.nan)
    idx = np.arange(a.size)
    dt = float(np.median(np.diff(t))) if t.size > 1 else 1 / 60
    fps = 1.0 / dt
    # outliers against a robust running median
    med = median_filter(np.interp(idx, idx[ok], a[ok]), size=9, mode="nearest")
    ok &= np.abs(a - med) <= max_dev
    if ok.sum() < 5:
        return np.full_like(a, np.nan), np.full_like(a, np.nan)
    out = lowpass(np.interp(idx, idx[ok], a[ok]), fps, cutoff_hz)
    # uncertainty of the smoothed value: per-frame spread (segment endpoint noise + scatter of
    # the detections around the smooth curve) over the number of detections it averages
    seg_err = np.degrees(np.arctan2(np.sqrt(2) * noise_px, np.maximum(lengths, 1.0)))
    resid = np.abs(np.where(ok, a, np.nan) - out)
    scatter = 1.4826 * median_filter(np.nan_to_num(resid, nan=0.0), size=9, mode="nearest")
    per_frame = np.hypot(np.where(np.isfinite(seg_err), seg_err, 5.0), scatter)
    win = max(3, round(fps / (2 * cutoff_hz)))  # the filter's effective averaging window
    n_used = np.convolve(ok.astype(float), np.ones(win), mode="same")
    err = per_frame / np.sqrt(np.maximum(n_used, 1.0))
    # keep detections and gaps between them up to max_gap_s
    gap = max(1, round(max_gap_s / dt))
    keep = np.zeros_like(ok)
    last = None
    for i in range(a.size):
        if ok[i]:
            if last is not None and 0 < i - last <= gap:
                keep[last:i] = True
            keep[i] = True
            last = i
    out[~keep] = np.nan
    err[~keep] = np.nan
    return out, err


def smooth_anchor(hits: list, contact: np.ndarray, scale_px: np.ndarray, t: np.ndarray,
                  n_on_mat: int, cutoff_hz: float = 1.2) -> tuple[np.ndarray, np.ndarray]:
    """Where to draw the tilt bar: the board tail's centre and half-width, smoothed over time.

    Detected segments give the tail's offset from the contact point and its length, both in
    units of the rider's pixel height so they stay valid as the rider moves away. Those are
    interpolated across frames without a detection and low-passed, so the bar glides with the
    board instead of jumping. Returns (anchor (N, 2), half_length (N,)) in full-res px, NaN
    where unknown.
    """
    n = len(hits)
    off = np.full((n, 2), np.nan)
    half = np.full(n, np.nan)
    for i, h in enumerate(hits):
        if h is None or i >= n_on_mat or not np.isfinite(scale_px[i]):
            continue
        x1, y1, x2, y2 = h.seg
        off[i] = [((x1 + x2) / 2 - contact[i, 0]) / scale_px[i],
                  ((y1 + y2) / 2 - contact[i, 1]) / scale_px[i]]
        half[i] = h.length / 2 / scale_px[i]
    ok = np.isfinite(half)
    anchor = np.full((n, 2), np.nan)
    hl = np.full(n, np.nan)
    if ok.sum() < 3:
        return anchor, hl
    idx = np.arange(n)
    fps = 1.0 / float(np.median(np.diff(t))) if t.size > 1 else 60.0
    m = idx < n_on_mat
    for c in range(2):
        s = lowpass(np.interp(idx, idx[ok], off[ok, c]), fps, cutoff_hz)
        anchor[m, c] = contact[m, c] + s[m] * scale_px[m]
    hs = lowpass(np.interp(idx, idx[ok], half[ok]), fps, cutoff_hz)
    hl[m] = hs[m] * scale_px[m]
    return anchor, hl
