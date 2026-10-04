"""Session calibration: the mat's left boundary, terrain sections, and rider-scale range.

Why not a ground homography: the mat here is not a plane. It has a steep in-run, a curve, then the
kicker face up to the lip, so one homography fitted anywhere is wrong somewhere else. Instead:

* Mat edge. The left boundary is taken per image row from the static background. Lateral
  position is measured from it.
* Terrain sections. A straight 3D edge projects to a straight image line, so wherever the mat
  edge bends in the image, the terrain bends. The edge's image slope is constant down the straight
  in-run, relaxes through the curve, and is near-vertical on the kicker face. Those two changes
  are the drop-in -> transition and transition -> takeoff-run boundaries, as image rows.
* Range and local scale. Each frame gives a body-size range estimate, `focal_px * stature /
  rider_px_h`. The map from the feet's image row to range is a fixed property of the venue, so a
  robust smoothing spline is fitted over the run. Range is then read off the feet, which is
  immune to crouching, the pop at the lip, and limbs folding toward the camera. Local scale is
  `range / focal_px` metres per pixel: the architecture's "1750 / rider_px_h".
* Camera pitch. Vertical structures in the background (lamp posts, shelter posts, building
  edges) converge on the vertical vanishing point below the image, which gives the pitch. Pitch
  separates height from distance in flight. Gravity cannot supply it: a pitch error is exactly a
  tilt, and a ballistic arc (or friction on the in-run) absorbs a tilt without complaint.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.interpolate import make_smoothing_spline
from scipy.ndimage import median_filter

from .contracts import Calibration


def mat_left_edge(mat_mask: np.ndarray, full_h: int, scale: float
                  ) -> tuple[np.ndarray, int, float]:
    """Per full-res row: x of the mat's left boundary. Returns (edge_x, mat_top_y, residual)."""
    h, w = mat_mask.shape
    rows = np.full(h, np.nan, np.float32)
    min_width = 0.04 * w
    for y in range(h):
        xs = np.flatnonzero(mat_mask[y])
        if xs.size > min_width and xs[0] > 1:  # touching the frame edge = boundary not visible
            rows[y] = xs[0]
    valid = np.flatnonzero(np.isfinite(rows))
    if valid.size < 20:
        return np.full(full_h, np.nan, np.float32), 0, float("nan")
    top = int(valid[0])
    # smooth, measure residual, then extrapolate the straight near-field below the last row
    seg = rows[top:valid[-1] + 1]
    seg = np.interp(np.arange(seg.size), np.flatnonzero(np.isfinite(seg)), seg[np.isfinite(seg)])
    smooth = median_filter(seg, size=9, mode="nearest")
    residual = float(np.median(np.abs(seg - smooth)) / scale)
    edge = np.full(h, np.nan, np.float32)
    edge[top:valid[-1] + 1] = smooth
    tail = np.arange(max(top, valid[-1] - 60), valid[-1] + 1)
    if tail.size >= 10:
        a, b = np.polyfit(tail, edge[tail], 1)
        below = np.arange(valid[-1] + 1, h)
        edge[below] = a * below + b
    # upsample rows to full resolution
    full_rows = np.arange(full_h) * scale
    edge_full = np.interp(full_rows, np.arange(h), np.nan_to_num(edge, nan=-1e6)) / scale
    edge_full[full_rows < top] = np.nan
    # the first rows under the lip see the lip's own outline, not the side of the mat
    edge_full[: round(top / scale) + 80] = np.nan
    return edge_full.astype(np.float32), round(top / scale), residual


def terrain_rows(edge_x: np.ndarray, top: int, frame_w: int, enter: float = 0.8,
                 face: float = 0.2, win: int = 160) -> tuple[float, float]:
    """(transition_start_y, takeoff_start_y) from the bend in the mat edge; NaN if not found.

    The edge slope dx/dy is normalised by its value on the straight in-run, found in the lower
    part of the mat. Scanning up from the camera, the transition starts where the normalised
    slope drops below `enter`, and the kicker face starts where it drops below `face`.
    """
    # only rows where the edge is clearly inside the frame; near the frame corner it is
    # extrapolated or kinked where the side boarding meets the picture edge
    seen = np.flatnonzero(np.nan_to_num(edge_x, nan=-1) > 0.1 * frame_w)
    if seen.size < 2 * win:
        return float("nan"), float("nan")
    h = int(seen[-1]) + 1
    ys = np.arange(top + win, h - win)
    if ys.size < 10:
        return float("nan"), float("nan")
    slope = (edge_x[ys + win // 2] - edge_x[ys - win // 2]) / win
    slope = median_filter(np.nan_to_num(slope, nan=0.0), size=win // 2 + 1, mode="nearest")
    lower = ys > top + 0.6 * (h - top)
    s_near = float(np.median(slope[lower])) if lower.any() else float("nan")
    if not np.isfinite(s_near) or abs(s_near) < 0.1:
        return float("nan"), float("nan")
    k = slope / s_near
    # scan upward from the camera: first sustained drop below each threshold
    order = ys[::-1]
    kk = k[::-1]

    def first_below(thr: float) -> float:
        run = 0
        for y, v in zip(order, kk, strict=True):
            run = run + 1 if v < thr else 0
            if run >= win // 2:
                return float(y + win // 2)
        return float("nan")

    return first_below(enter), first_below(face)


def estimate_pitch(bg: np.ndarray, mat: np.ndarray, focal_px: float, min_len: int = 60,
                   max_tilt_deg: float = 10.0, min_inliers: int = 6, seed: int = 0
                   ) -> tuple[float, float, int, float, float]:
    """(pitch_deg, bootstrap_sd_deg, n_inliers, vp_x, vp_y) from near-vertical background
    lines, or NaNs. (vp_x, vp_y) is the vertical vanishing point: the image of the gravity
    direction. It gives the true vertical at every pixel, whatever the phone's roll.

    `bg` is the full-resolution median background (BGR) and `mat` a full-resolution mask to
    exclude: the mat's seams converge on the fall line, not on the vertical.
    """
    h, w = bg.shape[:2]
    gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    segs = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
    if segs is None:
        return float("nan"), float("nan"), 0, float("nan"), float("nan")
    lines = []
    for x1, y1, x2, y2 in segs.reshape(-1, 4):
        length = float(np.hypot(x2 - x1, y2 - y1))
        tilt = (np.degrees(np.arctan2(x2 - x1, y2 - y1)) + 90) % 180 - 90
        xm, ym = int((x1 + x2) / 2), int((y1 + y2) / 2)
        if length >= min_len and abs(tilt) <= max_tilt_deg and not mat[min(ym, h - 1),
                                                                      min(xm, w - 1)]:
            lines.append((x1, y1, x2, y2, length))
    if len(lines) < min_inliers:
        return float("nan"), float("nan"), len(lines), float("nan"), float("nan")
    seg = np.array(lines, np.float64)
    xm, ym = (seg[:, 0] + seg[:, 2]) / 2, (seg[:, 1] + seg[:, 3]) / 2
    tan = (seg[:, 2] - seg[:, 0]) / (seg[:, 3] - seg[:, 1])
    wt = seg[:, 4]
    # each line through (xm, ym) with slope dx/dy = tan passes the VP (xv, yv):
    # xv - tan * yv = xm - tan * ym   (linear in the VP)
    a = np.stack([np.ones_like(tan), -tan], 1)
    b = xm - tan * ym
    rng = np.random.default_rng(seed)
    best, best_score = None, -1.0
    for _ in range(2000):
        i = rng.choice(len(seg), 2, replace=False)
        try:
            vp = np.linalg.solve(a[i], b[i])
        except np.linalg.LinAlgError:
            continue
        if vp[1] <= h:  # pitched down: verticals converge below the picture
            continue
        pred = (vp[0] - xm) / (vp[1] - ym)
        inl = np.degrees(np.abs(np.arctan(pred) - np.arctan(tan))) < 0.5
        score = wt[inl].sum()
        if score > best_score:
            best, best_score = inl, score
    if best is None or best.sum() < min_inliers:
        n_in = 0 if best is None else int(best.sum())
        return float("nan"), float("nan"), n_in, float("nan"), float("nan")
    cy = h / 2

    def vp_of(idx: np.ndarray) -> tuple[float, float]:
        sw = np.sqrt(wt[idx])
        xv, yv = np.linalg.lstsq(a[idx] * sw[:, None], b[idx] * sw, rcond=None)[0]
        return float(xv), float(yv)

    def pitch_of(idx: np.ndarray) -> float:
        return float(np.degrees(np.arctan(focal_px / (vp_of(idx)[1] - cy))))

    idx = np.flatnonzero(best)
    boots = [pitch_of(rng.choice(idx, idx.size)) for _ in range(200)]
    xv, yv = vp_of(idx)
    return pitch_of(idx), float(np.std(boots)), int(idx.size), xv, yv


def gravity_tilt(vp_x: float, vp_y: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Angle (rad) by which true "down" leans from the image's down at each pixel.

    Gravity points at the vertical vanishing point, so at pixel p, down = (vp - p). Positive
    means down leans toward image-right. Zero where the vanishing point is unknown.
    """
    if not (np.isfinite(vp_x) and np.isfinite(vp_y)):
        return np.zeros_like(np.asarray(x, np.float64))
    return np.arctan2(vp_x - np.asarray(x, np.float64), vp_y - np.asarray(y, np.float64))


def to_gravity(v: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Rotate image vectors (..., 2) so that true down becomes image down (+y)."""
    c, s = np.cos(phi), np.sin(phi)
    x, y = v[..., 0], v[..., 1]
    return np.stack([x * c - y * s, x * s + y * c], axis=-1)


def calibrate(mat_mask: np.ndarray, full_h: int, scale: float, focal_px: float,
              stature_m: float, chain_fraction: float,
              pitch: tuple[float, float] = (float("nan"), float("nan")),
              vp: tuple[float, float] = (float("nan"), float("nan"))) -> Calibration:
    edge, top, residual = mat_left_edge(mat_mask, full_h, scale)
    y_tr, y_to = terrain_rows(edge, top, round(mat_mask.shape[1] / scale))
    # QA: straightness of the edge along the straight in-run (rows below the transition)
    if np.isfinite(y_tr):
        rows = np.arange(int(y_tr), full_h)
        rows = rows[np.isfinite(edge[rows]) & (edge[rows] > 0.1 * mat_mask.shape[1] / scale)]
        if rows.size > 50:
            fit = np.polyval(np.polyfit(rows, edge[rows], 1), rows)
            residual = float(np.sqrt(np.mean((edge[rows] - fit) ** 2)))
    return Calibration(
        focal_px=focal_px, stature_m=stature_m, chain_fraction=chain_fraction,
        left_edge_x=edge, mat_top_y=top, method="body_scale+mat_mask", residual_px=residual,
        transition_y=y_tr, takeoff_y=y_to, pitch_deg=pitch[0], pitch_sd_deg=pitch[1],
        vp_x=vp[0], vp_y=vp[1],
    )


@dataclass(frozen=True)
class RangeModel:
    """Range from the camera as a function of the contact point's image row (see module doc)."""

    spline: object  # scipy BSpline in log-range
    y_lo: float
    y_hi: float
    mad_log: float  # frame-to-frame noise of the body-size range estimate (relative)
    n_obs: int

    def _log(self, y: np.ndarray) -> np.ndarray:
        yc = np.clip(y, self.y_lo, self.y_hi)
        out = self.spline(yc)
        # linear extension in log-range beyond the observed rows
        d_lo = self.spline.derivative()(self.y_lo)
        d_hi = self.spline.derivative()(self.y_hi)
        out = np.where(y < self.y_lo, out + d_lo * (y - self.y_lo), out)
        return np.where(y > self.y_hi, out + d_hi * (y - self.y_hi), out)

    def __call__(self, y: np.ndarray) -> np.ndarray:
        return np.exp(self._log(np.asarray(y, np.float64)))

    def slope(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, np.float64)
        d = self.spline.derivative()(np.clip(y, self.y_lo, self.y_hi))
        return self(y) * d


def fit_range_model(y: np.ndarray, r_obs: np.ndarray, iters: int = 4) -> RangeModel | None:
    """Robust (Tukey-bisquare IRLS) smoothing spline of log-range against image row.

    Observations must be in time order. Their frame-to-frame differences give the noise level,
    and the smoothing is the strongest whose residuals stay near it (discrepancy principle).
    GCV undersmooths here because neighbouring frames' errors are correlated.
    """
    ok = np.isfinite(y) & np.isfinite(r_obs) & (r_obs > 0)
    y, r = y[ok], np.log(r_obs[ok])
    if y.size < 20 or np.ptp(y) < 200:
        return None
    dr = np.diff(r)
    sigma = float(1.4826 * np.median(np.abs(dr - np.median(dr))) / np.sqrt(2)) + 1e-6
    o = np.argsort(y)
    y, r = y[o], r[o]
    keep = np.concatenate([[True], np.diff(y) > 1e-6])  # the spline needs increasing x
    y, r = y[keep], r[keep]
    w = np.ones_like(y)
    lams = np.logspace(3, 9, 13)
    for _ in range(iters):
        sp = None
        for lam in lams:  # smoothest spline that still fits to within the noise
            cand = make_smoothing_spline(y, r, w=w, lam=lam)
            if 1.4826 * np.median(np.abs(r - cand(y))) <= 1.3 * sigma or sp is None:
                sp = cand
            else:
                break
        res = r - sp(y)
        mad = float(1.4826 * np.median(np.abs(res))) + 1e-6
        u = res / (4.685 * mad)
        w = np.where(np.abs(u) < 1, (1 - u**2) ** 2, 0.0) + 1e-3
    return RangeModel(sp, float(y[0]), float(y[-1]), sigma, int(y.size))


def m_per_px(cal: Calibration, rider_px_h: np.ndarray) -> np.ndarray:
    return cal.stature_m / rider_px_h


def range_m(cal: Calibration, rider_px_h: np.ndarray) -> np.ndarray:
    return cal.focal_px * cal.stature_m / rider_px_h


def edge_x_at(cal: Calibration, y: np.ndarray) -> np.ndarray:
    yi = np.clip(np.nan_to_num(y, nan=0).astype(int), 0, cal.left_edge_x.size - 1)
    out = cal.left_edge_x[yi].astype(np.float64)
    out[~np.isfinite(y)] = np.nan
    return out
