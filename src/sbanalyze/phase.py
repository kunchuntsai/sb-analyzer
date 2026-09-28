"""Segment the run into drop-in, transition and takeoff run.

Phases are sections of terrain, so the primary method is spatial. Calibration finds the image
rows where the mat edge bends (the in-run curving out, then the kicker face), and each frame's
phase is where the rider's feet are. Boundaries are therefore identical for every run in a
session, which is what makes runs comparable.

Without terrain rows, the fallback is the architecture's board-yaw state machine. Board yaw (angle to the fall line) and its rate drive a three-state machine:

    DropIn --(|yaw rate| > enter)--> Transition --(yaw < settle for settle_s)--> TakeoffRun

That needs a rider who starts across the fall line and turns onto it. When the yaw signal
shows no such turn (the riders in the sample clips point down the fall line from the start),
the terrain is read off the speed profile instead:

* drop-in: the steep in-run, where acceleration is high;
* transition: the in-run curving out. Acceleration decays and speed peaks at its bottom;
* takeoff run: the climb up the kicker, where the rider decelerates to the lip.

The source is recorded so the UI can show which method produced the boundaries.
"""

from __future__ import annotations

import numpy as np

from .contracts import Phase

Spans = dict[Phase, tuple[float, float]]


def _sustained(mask: np.ndarray, n: int, start: int = 0) -> int | None:
    run = 0
    for i in range(start, mask.size):
        run = run + 1 if mask[i] else 0
        if run >= n:
            return i - n + 1
    return None


def segment_by_yaw(t: np.ndarray, yaw_deg: np.ndarray, fps: float, enter_dps: float,
                   settle_deg: float, settle_s: float, min_phase_s: float) -> Spans | None:
    if not np.isfinite(yaw_deg).all() or t.size < 10:
        return None
    rate = np.gradient(yaw_deg, t)
    min_n = max(1, round(min_phase_s * fps))
    enter = _sustained(np.abs(rate) > enter_dps, 3, start=min_n)
    if enter is None or np.max(yaw_deg[:enter + 1]) < 45.0:  # never really across the slope
        return None
    settle = _sustained(yaw_deg < settle_deg, max(1, round(settle_s * fps)), start=enter + 1)
    if settle is None or settle - enter < min_n or t.size - settle < min_n:
        return None
    return {
        Phase.DROP_IN: (float(t[0]), float(t[enter])),
        Phase.TRANSITION: (float(t[enter]), float(t[settle])),
        Phase.TAKEOFF_RUN: (float(t[settle]), float(t[-1])),
    }


def segment_by_speed(t: np.ndarray, speed: np.ndarray, min_phase_s: float,
                     accel_drop: float = 0.4) -> Spans:
    dt = float(np.median(np.diff(t))) if t.size > 1 else 1 / 60
    min_n = max(1, round(min_phase_s / dt))
    v = np.nan_to_num(speed, nan=0.0)
    # heavier smoothing for segmentation only
    k = max(3, round(0.15 / dt))
    vs = np.convolve(np.pad(v, k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), "valid")
    lo, hi = min_n, max(min_n + 1, t.size - min_n)
    peak = lo + int(np.argmax(vs[lo:hi]))
    acc = np.gradient(vs, t)
    a_max_i = int(np.argmax(acc[:peak + 1]))
    below = np.flatnonzero(acc[a_max_i:peak + 1] < accel_drop * acc[a_max_i])
    a = a_max_i + int(below[0]) if below.size else (a_max_i + peak) // 2
    a = int(np.clip(a, min_n, t.size - 2 * min_n))
    b = int(np.clip(peak, a + min_n, t.size - min_n))
    return {
        Phase.DROP_IN: (float(t[0]), float(t[a])),
        Phase.TRANSITION: (float(t[a]), float(t[b])),
        Phase.TAKEOFF_RUN: (float(t[b]), float(t[-1])),
    }


def segment_by_terrain(t: np.ndarray, feet_y: np.ndarray, transition_y: float,
                       takeoff_y: float, min_phase_s: float) -> Spans | None:
    """The first frame whose feet are above (further than) each terrain row."""
    if not (np.isfinite(transition_y) and np.isfinite(takeoff_y)) or t.size < 10:
        return None
    y = np.minimum.accumulate(np.nan_to_num(feet_y, nan=np.inf))  # progress is monotone
    a = int(np.argmax(y < transition_y)) if (y < transition_y).any() else None
    b = int(np.argmax(y < takeoff_y)) if (y < takeoff_y).any() else None
    dt = float(np.median(np.diff(t)))
    min_n = max(1, round(min_phase_s / dt))
    if a is None or b is None or a < min_n or b - a < min_n or t.size - b < min_n:
        return None
    return {
        Phase.DROP_IN: (float(t[0]), float(t[a])),
        Phase.TRANSITION: (float(t[a]), float(t[b])),
        Phase.TAKEOFF_RUN: (float(t[b]), float(t[-1])),
    }


def segment(t: np.ndarray, yaw_deg: np.ndarray, speed: np.ndarray, fps: float, cfg,
            feet_y: np.ndarray | None = None, terrain: tuple[float, float] | None = None
            ) -> tuple[Spans, str]:
    if feet_y is not None and terrain is not None:
        spans = segment_by_terrain(t, feet_y, *terrain, cfg.min_phase_s)
        if spans is not None:
            return spans, "terrain"
    spans = segment_by_yaw(t, yaw_deg, fps, cfg.yaw_rate_enter_dps, cfg.yaw_settle_deg,
                           cfg.settle_s, cfg.min_phase_s)
    if spans is not None:
        return spans, "yaw_state_machine"
    return segment_by_speed(t, speed, cfg.min_phase_s), "speed_profile"
