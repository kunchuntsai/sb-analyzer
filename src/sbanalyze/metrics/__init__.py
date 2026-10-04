"""Biomechanics. Order matters: gap-fill -> filter -> derive. Never differentiate a raw signal."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter

from ..calib import (
    RangeModel,
    edge_x_at,
    fit_range_model,
    gravity_tilt,
    m_per_px,
    range_m,
    to_gravity,
)
from ..contracts import (
    Calibration,
    K,
    MetricSample,
    Phase,
    PoseFrame,
    Validity,
    ViewRole,
)
from ..detect.base import scale_px
from ..detect.board import LEFT_FOOT, RIGHT_FOOT
from .angles import joint_flexion, lean_from_vertical
from .com import centre_of_mass
from .validity import validity

METRIC_UNITS = {
    "speed_along": "m/s", "range": "m", "line_offset": "m",
    "com_height": "m", "com_lateral": "m", "com_foreaft": "m", "com_toe_heel": "m",
    "board_edge": "deg",
    "knee_flex_l": "deg", "knee_flex_r": "deg", "trunk_lean": "deg", "board_yaw": "deg",
    "air_height": "m",
}
METRICS = tuple(METRIC_UNITS)


@dataclass
class Signals:
    frame_idx: np.ndarray  # (N,)
    t_sec: np.ndarray  # (N,)
    kp: np.ndarray  # (N, 26, 2) gap-filled, filtered
    raw_scores: np.ndarray  # (N, 26), 0 where the rider was not detected
    rider_px_h: np.ndarray  # (N,)
    com: np.ndarray  # (N, 2) image px
    contact: np.ndarray  # (N, 2) image px
    lead_foot: np.ndarray  # (N, 2)
    trail_foot: np.ndarray  # (N, 2)
    stance: str  # "left_lead" | "right_lead"
    stance_width_m: float
    values: dict[str, np.ndarray] = field(default_factory=dict)
    errors: dict[str, np.ndarray] = field(default_factory=dict)
    confidence: dict[str, np.ndarray] = field(default_factory=dict)
    geom_invalid: dict[str, np.ndarray] = field(default_factory=dict)
    range_model: RangeModel | None = None
    air_from: int | None = None  # index of the first airborne frame
    takeoff_speed: float = float("nan")  # horizontal speed carried off the lip, m/s
    board_bump: np.ndarray | None = None  # board vertical residual (m), for event detection
    stance_info: dict = field(default_factory=dict)


def _fill(x: np.ndarray, ok: np.ndarray, min_coverage: float = 0.15) -> np.ndarray:
    """Linear interpolation across gaps, nearest-value hold at the ends."""
    out = x.astype(np.float64).copy()
    if ok.mean() < min_coverage:
        out[:] = np.nan
        return out
    idx = np.arange(x.shape[0])
    out[~ok] = np.interp(idx[~ok], idx[ok], x[ok])
    return out


def _lowpass(x: np.ndarray, fps: float, cutoff: float, order: int) -> np.ndarray:
    if not np.isfinite(x).all() or x.shape[0] < 3 * (order + 1) * 2:
        return x
    b, a = butter(order, cutoff / (fps / 2), btype="low")
    return filtfilt(b, a, x, axis=0)


def _savgol(x: np.ndarray, fps: float, window_s: float, order: int, deriv: int = 0) -> np.ndarray:
    win = max(order + 2, round(window_s * fps) | 1)
    if x.shape[0] <= win or not np.isfinite(x).all():
        return np.full_like(x, np.nan) if deriv else x
    return savgol_filter(x, win, order, deriv=deriv, delta=1.0 / fps, axis=0)


def decide_stance(kp: np.ndarray, sc: np.ndarray, override: str = "auto",
                  conf_min: float = 0.5) -> tuple[bool, dict]:
    """Regular (left foot forward) or goofy, from two independent cues. Returns (left_leads, info).

    * Lead foot: the ankle further from the camera (higher in the image) is the front foot. This
      needs the pose model's left/right labels to be right.
    * Toe direction: each foot's heel->toe line points to the toe side. Seen from behind, a
      regular rider faces image-right and a goofy rider image-left. This uses no left/right
      labels at all, so it survives a model that swaps them.

    Votes are per frame. If the cues disagree, the label-free toe direction wins and the
    disagreement is reported. Riding switch shows up as the opposite stance for that run, which
    is what the toe/heel signs need. `override` ("regular"/"goofy") forces the answer.
    """
    ok_a = np.minimum(sc[:, K.L_ANKLE], sc[:, K.R_ANKLE]) >= conf_min
    dy = kp[:, K.L_ANKLE, 1] - kp[:, K.R_ANKLE, 1]
    lead = np.where(ok_a & (np.abs(dy) > 4), np.where(dy < 0, 1, -1), 0)  # +1 = regular
    toe = []
    for heel, big in ((K.L_HEEL, K.L_BIG_TOE), (K.R_HEEL, K.R_BIG_TOE)):
        ok = np.minimum(sc[:, heel], sc[:, big]) >= conf_min
        dx = kp[:, big, 0] - kp[:, heel, 0]
        toe.append(np.where(ok & (np.abs(dx) > 3), np.where(dx > 0, 1, -1), 0))
    toe = np.concatenate(toe)
    lv, tv = lead[lead != 0], toe[toe != 0]
    l_mean = float(lv.mean()) if lv.size else 0.0
    t_mean = float(tv.mean()) if tv.size else 0.0
    agree = (l_mean > 0) == (t_mean > 0) or not lv.size or not tv.size
    score = (l_mean * lv.size + t_mean * tv.size) / max(1, lv.size + tv.size) if agree else t_mean
    regular = score > 0
    votes = np.concatenate([lv, tv])
    confidence = float(np.mean((votes > 0) == regular)) if votes.size else 0.0
    info = {
        "value": "regular" if regular else "goofy", "source": "auto",
        "confidence": round(confidence, 3), "cues_agree": bool(agree),
        "lead_foot": {"value": "regular" if l_mean > 0 else "goofy",
                      "agreement": round((1 + abs(l_mean)) / 2, 3), "frames": int(lv.size)},
        "toe_direction": {"value": "regular" if t_mean > 0 else "goofy",
                          "agreement": round((1 + abs(t_mean)) / 2, 3), "frames": int(tv.size)},
    }
    if override in ("regular", "goofy"):
        regular = override == "regular"
        info.update(value=override, source="config")
    return regular, info


def compute_signals(poses: list[PoseFrame | None], frame_idx: np.ndarray, t_sec: np.ndarray,
                    fps: float, cal: Calibration, cfg, frame_wh: tuple[int, int],
                    stance_width_m: float, air_from: int | None = None,
                    stance_override: str = "auto") -> Signals:
    n = len(frame_idx)
    na = n if air_from is None else air_from  # frames on the mat
    thr = cfg.kpt_conf_threshold
    kp = np.full((n, 26, 2), np.nan)
    sc = np.zeros((n, 26))
    for i, p in enumerate(poses):
        if p is not None:
            kp[i], sc[i] = p.keypoints, p.scores

    # Keypoints at the frame border are guesses for body parts outside the picture (the rider
    # drops in right under the camera and is cut off). Treat them as unseen.
    w, h = frame_wh
    margin = 0.015 * max(w, h)
    near_border = ((kp[..., 0] < margin) | (kp[..., 0] > w - margin)
                   | (kp[..., 1] < margin) | (kp[..., 1] > h - margin))
    sc[near_border] = 0.0

    # 1. reject low-confidence keypoints, interpolate across gaps
    ok = sc >= thr
    for j in range(26):
        for c in range(2):
            kp[:, j, c] = _fill(kp[:, j, c], ok[:, j] & np.isfinite(kp[:, j, c]))

    # 2. zero-phase low-pass on positions
    for j in range(26):
        if np.isfinite(kp[:, j]).all():
            kp[:, j] = _lowpass(kp[:, j], fps, cfg.butter_cutoff_hz, cfg.butter_order)

    com = centre_of_mass(kp)
    lf = np.nanmean(kp[:, list(LEFT_FOOT)], axis=1)
    rf = np.nanmean(kp[:, list(RIGHT_FOOT)], axis=1)
    contact = (lf + rf) / 2
    # The phone is rarely held perfectly level. Gravity's direction in the image (from the
    # vertical vanishing point) gives the true vertical at the rider, so "up", "sideways" and
    # "level" below are measured against gravity, not against the picture's edges.
    phi = gravity_tilt(cal.vp_x, cal.vp_y, contact[:, 0], contact[:, 1])

    # 3. scale. Each confidently seen frame gives a body-size range estimate from the
    # crouch-invariant limb chain. Those estimates are too noisy to differentiate directly (the
    # pop at the lip folds the legs toward the camera). So fit the venue's row->range map once
    # and read range, and hence scale, off the feet's image row.
    h_obs = np.array([scale_px(kp[i], sc[i], thr) for i in range(n)])
    rmodel = fit_range_model(contact[:na, 1], range_m(cal, h_obs[:na]))
    rng = np.full(n, np.nan)
    if rmodel is not None:
        rng[:na] = rmodel(contact[:na, 1])
    else:  # too little of the run seen: smoothed body-size estimate
        hh = _fill(h_obs[:na], np.isfinite(h_obs[:na]))
        rng[:na] = range_m(cal, _savgol(hh, fps, max(0.3, cfg.savgol_window_s), 2))
    # In the air nothing pushes horizontally: range grows at the speed carried off the lip.
    takeoff_speed = float("nan")
    if na < n:
        # The last metres before the lip are where the range map is least reliable (curved
        # kicker face, the pop). Take the approach speed just before it, and never more than
        # the in-run's own peak.
        v_run = _savgol(rng[:na], fps, cfg.savgol_window_s, cfg.savgol_order, deriv=1)
        t_to = t_sec[na - 1]
        approach = (t_sec[:na] > t_to - 0.6) & (t_sec[:na] < t_to - 0.15)
        before = t_sec[:na] < t_to - 0.6
        cap = 1.1 * float(np.nanmax(v_run[before])) if before.any() else np.inf
        takeoff_speed = float(min(np.nanmedian(v_run[approach]) if approach.any()
                                  else v_run[-1], cap))
        rng[na:] = rng[na - 1] + takeoff_speed * (t_sec[na:] - t_sec[na - 1])
    h_px = cal.focal_px * cal.stature_m / rng
    s = m_per_px(cal, h_px)  # m per px at the rider

    left_leads, stance_info = decide_stance(kp[:na], sc[:na], stance_override)
    lead, trail = (lf, rf) if left_leads else (rf, lf)
    d = lead - trail

    noise = cfg.keypoint_noise_px
    err_pos = noise * s  # 3 px at the rider's scale = architecture's 3 * 1750 / rider_px_h mm

    vals: dict[str, np.ndarray] = {}
    errs: dict[str, np.ndarray] = {}
    conf: dict[str, np.ndarray] = {}

    def score(*joints: int) -> np.ndarray:
        return sc[:, list(joints)].mean(axis=1)

    for side, (hip, knee, ank) in {"l": (K.L_HIP, K.L_KNEE, K.L_ANKLE),
                                   "r": (K.R_HIP, K.R_KNEE, K.R_ANKLE)}.items():
        seg = np.minimum(np.linalg.norm(kp[:, hip] - kp[:, knee], axis=1),
                         np.linalg.norm(kp[:, knee] - kp[:, ank], axis=1))
        vals[f"knee_flex_{side}"] = joint_flexion(kp[:, hip], kp[:, knee], kp[:, ank])
        errs[f"knee_flex_{side}"] = np.degrees(np.sqrt(2) * noise / np.maximum(seg, 1))
        conf[f"knee_flex_{side}"] = score(hip, knee, ank)

    torso = np.linalg.norm(kp[:, K.NECK] - kp[:, K.HIP], axis=1)
    trunk_g = to_gravity(kp[:, K.NECK] - kp[:, K.HIP], phi)
    vals["trunk_lean"] = lean_from_vertical(np.zeros_like(trunk_g), trunk_g)
    errs["trunk_lean"] = np.degrees(np.sqrt(2) * noise / np.maximum(torso, 1))
    conf["trunk_lean"] = score(K.HIP, K.NECK)

    body = score(*range(17, 26), K.L_SHOULDER, K.R_SHOULDER, K.L_KNEE, K.R_KNEE)
    feet = score(*LEFT_FOOT, *RIGHT_FOOT)
    # Image x is the across-slope axis and is never foreshortened, so it carries the offsets.
    # Image y mixes height with depth, so only vertical extents at the rider's depth are used.
    rel_g = to_gravity(com - contact, phi)  # CoM relative to the board, gravity-aligned
    dx_com = rel_g[:, 0] * s
    vals["com_height"] = -rel_g[:, 1] * s
    vals["com_lateral"] = dx_com

    # Board angle to the fall line from the across-slope spread of the feet. The lateral
    # component of a stance of known width is width * sin(yaw).
    stance_w = stance_width_m
    r = np.clip(np.abs(d[:, 0]) * s / stance_w, 0.0, 1.0)
    vals["board_yaw"] = np.degrees(np.arcsin(r))
    dr = np.sqrt(2) * err_pos / stance_w
    errs["board_yaw"] = np.minimum(np.degrees(dr / np.sqrt(np.maximum(1 - r**2, 1e-3))), 30.0)
    conf["board_yaw"] = feet

    # Fore/aft is the across-slope CoM offset toward the nose, and is observable only while the
    # board has an across-slope component. The error grows as 1/sin(yaw), so it becomes useless
    # once the board points down the fall line (the validity matrix marks that invalid).
    sin_yaw = np.maximum(r, 0.05)
    vals["com_foreaft"] = dx_com * np.sign(np.where(d[:, 0] == 0, 1.0, d[:, 0])) / sin_yaw
    vals["com_foreaft"] = np.clip(vals["com_foreaft"], -0.6, 0.6)
    # Toe/heel: the CoM's offset across the board, signed by stance. Seen from behind, a regular
    # rider (left foot forward) faces image-right and a goofy rider faces image-left, so the toe
    # edge is on that side. A sign change is an edge change.
    toe_sign = 1.0 if left_leads else -1.0
    vals["com_toe_heel"] = dx_com * toe_sign
    # Board edge angle (roll about the board's long axis), + toe edge / - heel edge. Seen from
    # behind, each foot's heel->toe line runs across the board. On a flat board it is level in
    # the image (the cross-slope direction is horizontal). On edge, the downhill-biting side
    # drops. Vertical image extent is foreshortened by cos(depression), so it is corrected.
    pitch = cal.pitch_deg if np.isfinite(cal.pitch_deg) else 18.0
    depression = np.radians(pitch) + np.arctan((contact[:, 1] - h / 2) / cal.focal_px)
    angs, wts = [], []
    for heel, big, small in ((K.L_HEEL, K.L_BIG_TOE, K.L_SMALL_TOE),
                             (K.R_HEEL, K.R_BIG_TOE, K.R_SMALL_TOE)):
        v = to_gravity((kp[:, big] + kp[:, small]) / 2 - kp[:, heel], phi)
        across = v[:, 0] * toe_sign
        drop = v[:, 1] / np.cos(depression)
        ang = np.degrees(np.arctan2(drop, across))
        ok = across > 0  # toes on the wrong side = keypoint swap: no reading
        c = np.minimum.reduce([sc[:, heel], sc[:, big], sc[:, small]])
        angs.append(np.where(ok, ang, np.nan))
        wts.append(np.where(ok, np.hypot(*v.T) * c, 0.0))
    angs, wts = np.array(angs), np.array(wts)
    wsum = wts.sum(0)
    edge_raw = np.where(wsum > 0, np.nansum(np.nan_to_num(angs) * wts, 0) / np.maximum(wsum, 1e-9),
                        np.nan)
    edge_ok = np.isfinite(edge_raw) & (wsum > 0)
    board_edge = np.full(n, np.nan)
    if edge_ok.sum() > 10:
        board_edge = _savgol(_fill(edge_raw, edge_ok), fps, cfg.savgol_window_s, 2)
        board_edge[~edge_ok] = np.nan
    vals["board_edge"] = board_edge
    per_frame_len = np.nanmean([np.hypot(*(kp[:, b] - kp[:, hh]).T)
                                for hh, b in ((K.L_HEEL, K.L_BIG_TOE), (K.R_HEEL, K.R_BIG_TOE))],
                               axis=0)
    errs["board_edge"] = np.degrees(noise / np.maximum(per_frame_len, 1.0))  # 2 feet, smoothed
    conf["board_edge"] = feet

    for m in ("com_height", "com_lateral", "com_foreaft", "com_toe_heel"):
        errs[m] = err_pos
        conf[m] = np.minimum(body, feet)
    errs["com_foreaft"] = np.minimum(err_pos / sin_yaw, 0.6)

    edge = edge_x_at(cal, contact[:, 1])
    vals["line_offset"] = (contact[:, 0] - edge) * s
    errs["line_offset"] = err_pos + (cal.residual_px if np.isfinite(cal.residual_px) else 0) * s
    conf["line_offset"] = feet

    win_n = max(3, round(cfg.savgol_window_s * fps))
    vals["range"] = rng
    vals["speed_along"] = _savgol(rng, fps, cfg.savgol_window_s, cfg.savgol_order, deriv=1)
    if rmodel is not None:
        # per-frame spread of body-size range around the fitted venue map, plus feet-row noise
        dr_dy = np.abs(rmodel.slope(contact[:, 1]))
        errs["range"] = np.hypot(rng * rmodel.mad_log, dr_dy * noise)
        errs["speed_along"] = errs["range"] * np.sqrt(12.0 / win_n**3) * fps * np.sqrt(2)
    else:
        errs["range"] = rng * 2 * noise / h_px
        errs["speed_along"] = errs["range"] * np.sqrt(12.0 / win_n) * fps / win_n
    conf["range"] = feet
    conf["speed_along"] = feet

    # Board vertical residual: the feet's path minus its smooth trend, in metres. A bump or a hop
    # shows up as a short excursion; the terrain itself is in the trend.
    fy_m = contact[:, 1] * s
    bump = np.full(n, np.nan)
    if na > 15 and np.isfinite(fy_m[:na]).all():
        bump[:na] = fy_m[:na] - _savgol(fy_m[:na], fps, 0.5, 2)

    # Geometry that the phase matrix cannot know about: fore/aft needs the board to have an
    # across-slope component. Below ~20 deg to the fall line it is unobservable from this camera.
    # Board edge needs the heel->toe line across the image: board pointing down the fall line.
    geom_invalid = {"com_foreaft": vals["board_yaw"] < 20.0,
                    "board_edge": vals["board_yaw"] > 35.0}

    return Signals(
        frame_idx=frame_idx, t_sec=t_sec, kp=kp.astype(np.float32), raw_scores=sc,
        rider_px_h=h_px, com=com, contact=contact, lead_foot=lead, trail_foot=trail,
        stance="left_lead" if left_leads else "right_lead", stance_width_m=stance_w,
        stance_info=stance_info,
        values=vals, errors=errs, confidence=conf, geom_invalid=geom_invalid,
        range_model=rmodel, air_from=air_from, takeoff_speed=takeoff_speed,
        board_bump=bump,
    )


def phase_at(t: np.ndarray, spans: dict[Phase, tuple[float, float]]) -> list[Phase]:
    out = []
    for ti in t:
        ph = Phase.TAKEOFF_RUN
        for p, (a, b) in spans.items():
            if a <= ti <= b:
                ph = p
                break
        out.append(ph)
    return out


def to_samples(sig: Signals, spans: dict[Phase, tuple[float, float]], run_id: str,
               view: ViewRole = ViewRole.FALL_LINE) -> list[MetricSample]:
    phases = phase_at(sig.t_sec, spans)
    out: list[MetricSample] = []
    for m in METRICS:
        v, e, c = sig.values[m], sig.errors[m], sig.confidence[m]
        for i in range(len(sig.frame_idx)):
            if not np.isfinite(v[i]):
                continue
            val = validity(m, phases[i], view)
            if c[i] < 0.15 or (m in sig.geom_invalid and sig.geom_invalid[m][i]):
                val = Validity.INVALID  # not actually seen, or not observable from here
            out.append(MetricSample(
                run_id=run_id, view=view, frame_idx=int(sig.frame_idx[i]),
                t_sec=float(sig.t_sec[i]), phase=phases[i], metric=m, value=float(v[i]),
                confidence=float(c[i]), rider_px_h=float(sig.rider_px_h[i]),
                err_est=float(e[i]) if np.isfinite(e[i]) else float("nan"), validity=val,
            ))
    return out
