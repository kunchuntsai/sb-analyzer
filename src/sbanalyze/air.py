"""Flight: takeoff, touchdown, air time and board height above the lip.

Geometry. A point at range R whose image row is y lies at depression angle
`pitch + atan((y - cy) / f)` below the camera, so its height relative to the camera is
`-R * sin(depression)`. Range in the air comes from the takeoff range and the horizontal speed
carried off the lip. No horizontal force acts in flight, so that speed is constant, and it is
far steadier than body-size range on a rider who is grabbing or rotating.

Height is measured for the board (the lowest foot point) relative to the lip. The camera sees
the rider from behind, so image rise mixes height with moving away. Pitch is what separates
them, and its uncertainty is propagated into `err`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .contracts import K

FEET = [K.L_ANKLE, K.R_ANKLE, K.L_HEEL, K.R_HEEL, K.L_BIG_TOE, K.R_BIG_TOE]


def feet_low_y(kp: np.ndarray) -> np.ndarray:
    """Image row of the lowest foot point per frame (board underside proxy). kp: (N, 26, 2)."""
    return np.nanmax(kp[:, FEET, 1], axis=1)


def find_takeoff(feet_y: np.ndarray, lip_y: float, margin_px: float = 10.0,
                 sustain: int = 3) -> int | None:
    """First frame of a sustained stretch with the feet above the lip row."""
    above = np.nan_to_num(feet_y, nan=np.inf) < lip_y - margin_px
    run = 0
    for i, a in enumerate(above):
        run = run + 1 if a else 0
        if run >= sustain:
            return i - sustain + 1
    return None


@dataclass
class Jump:
    takeoff: int  # index of the first airborne frame
    touchdown: int  # index of the landing frame (last tracked frame if not seen)
    landing_seen: bool
    t_takeoff: float
    t_touchdown: float
    t_apex: float
    air_time_s: float
    apex_height_m: float
    apex_err_m: float
    takeoff_speed_mps: float
    pitch_deg: float
    g_fit: float  # gravity recovered from the descent: a check on the whole geometry
    height: np.ndarray = field(repr=False)  # (N,) board height above lip, NaN outside flight
    height_err: np.ndarray = field(repr=False)

    def summary(self) -> dict:
        return {
            "t_takeoff": self.t_takeoff, "t_touchdown": self.t_touchdown, "t_apex": self.t_apex,
            "air_time_s": self.air_time_s, "apex_height_m": self.apex_height_m,
            "apex_err_m": self.apex_err_m, "landing_seen": self.landing_seen,
            "takeoff_speed_mps": self.takeoff_speed_mps, "pitch_deg": self.pitch_deg,
            "g_fit": self.g_fit,
        }


def board_height(t: np.ndarray, feet_y: np.ndarray, t0: float, r0: float, vh: float,
                 lip_y: float, pitch_deg: float, focal_px: float, cy: float) -> np.ndarray:
    th = np.radians(pitch_deg)
    rng = r0 + vh * (t - t0)
    z = -rng * np.sin(th + np.arctan((feet_y - cy) / focal_px))
    z_lip = -r0 * np.sin(th + np.arctan((lip_y - cy) / focal_px))
    return z - z_lip


def find_touchdown(t: np.ndarray, h: np.ndarray, start: int, fall_frac: float = 0.3,
                   sustain: int = 3) -> tuple[int, bool]:
    """Landing = the descent is arrested: after the fastest fall, the sink rate collapses."""
    seg = np.arange(start, t.size)
    if seg.size < 8:
        return t.size - 1, False
    v = -np.gradient(h[seg], t[seg])  # positive = falling
    apex = int(np.nanargmax(h[seg]))
    if apex >= seg.size - 4:
        return t.size - 1, False
    k_fast = apex + int(np.nanargmax(v[apex:]))
    v_max = v[k_fast]
    if not np.isfinite(v_max) or v_max < 1.0:  # never really fell: tracking lost
        return t.size - 1, False
    run = 0
    for k in range(k_fast, seg.size):
        run = run + 1 if v[k] < fall_frac * v_max else 0
        if run >= sustain:
            return int(seg[k - sustain + 1]), True
    return t.size - 1, False


def analyze(t: np.ndarray, kp: np.ndarray, takeoff: int, r0: float, vh: float, lip_y: float,
            pitch_deg: float, pitch_sd_deg: float, focal_px: float, cy: float,
            noise_px: float) -> Jump:
    fy = feet_low_y(kp)
    h = np.full(t.size, np.nan)
    air = np.arange(takeoff, t.size)
    t0 = float(t[takeoff])
    h[air] = board_height(t[air], fy[air], t0, r0, vh, lip_y, pitch_deg, focal_px, cy)
    td, seen = find_touchdown(t, h, takeoff)
    # beyond touchdown the rider is on the landing, not in flight
    h[td + 1:] = np.nan
    sd = pitch_sd_deg if np.isfinite(pitch_sd_deg) else 0.0
    dp = max(2.0, sd)  # tripod placement varies; don't trust the bootstrap alone
    flight = np.arange(takeoff, td + 1)
    h_hi = board_height(t[flight], fy[flight], t0, r0, vh, lip_y, pitch_deg - dp, focal_px, cy)
    rng = r0 + vh * (t[flight] - t0)
    err = np.full(t.size, np.nan)
    err[flight] = np.hypot(np.abs(h_hi - h[flight]), noise_px * rng / focal_px)
    k = flight[int(np.nanargmax(h[flight]))]
    desc = np.arange(k, td + 1)
    g_fit = float("nan")
    if desc.size >= 8:
        g_fit = float(-2 * np.polyfit(t[desc] - t[k], h[desc], 2)[0])
    return Jump(
        takeoff=takeoff, touchdown=td, landing_seen=seen, t_takeoff=t0,
        t_touchdown=float(t[td]), t_apex=float(t[k]), air_time_s=float(t[td] - t0),
        apex_height_m=float(h[k]), apex_err_m=float(err[k]), takeoff_speed_mps=float(vh),
        pitch_deg=float(pitch_deg), g_fit=g_fit, height=h, height_err=err,
    )
