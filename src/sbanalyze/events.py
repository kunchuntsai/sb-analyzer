"""Sudden-movement detection: flag jerky, non-smooth changes in the rider's motion.

A frame is flagged when a signal changes faster than both of these thresholds:

* an absolute rate (`abs_rate`), so that only physically meaningful jolts count;
* an adaptive rate (`k_mad` robust deviations above this run's own typical rate), so that a
  rider who is simply more dynamic overall does not get flagged everywhere.

Contiguous flagged frames form one event. An event must also move the signal by at least
`min_delta` and by more than twice its own error band, otherwise it is noise. Only valid or
degraded samples are considered.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import savgol_filter

from .contracts import Validity


@dataclass(frozen=True)
class Rule:
    metric: str
    abs_rate: float  # units per second
    min_delta: float  # units
    kind: str
    up: str  # wording for a positive change
    down: str  # wording for a negative change
    unit: str
    scale: float = 1.0  # display multiplier (m -> cm)
    unit_disp: str = ""
    max_delta: float = float("inf")  # beyond this the "movement" is a tracking error
    max_rate: float = float("inf")  # physically impossible rates are tracking errors too
    use_degraded: bool = True


RULES: tuple[Rule, ...] = (
    Rule("com_height", 0.5, 0.05, "CoM vertical", "CoM rises", "CoM drops", "m", 100, "cm",
         max_delta=1.0),
    Rule("com_lateral", 0.5, 0.05, "CoM sideways", "CoM shifts right", "CoM shifts left",
         "m", 100, "cm", max_delta=0.5, use_degraded=False),
    Rule("line_offset", 1.0, 0.12, "Board sideways", "Board slides right", "Board slides left",
         "m", 100, "cm", max_delta=1.0, use_degraded=False),
    Rule("board_bump", 0.8, 0.04, "Board vertical", "Board jolts down", "Board jolts up",
         "m", 100, "cm", max_delta=0.5),
    Rule("knee_flex_l", 150.0, 12.0, "Knee", "Left knee snaps into flexion",
         "Left knee snaps straight", "deg", 1, "°", max_delta=90.0, use_degraded=False),
    Rule("knee_flex_r", 150.0, 12.0, "Knee", "Right knee snaps into flexion",
         "Right knee snaps straight", "deg", 1, "°", max_delta=90.0, use_degraded=False),
    Rule("trunk_lean", 90.0, 8.0, "Trunk", "Trunk jerks right", "Trunk jerks left",
         "deg", 1, "°", max_delta=60.0, use_degraded=False),
    Rule("speed_along", 3.5, 0.5, "Speed", "Sudden acceleration", "Sudden deceleration",
         "m/s", 3.6, " km/h", max_delta=4.0, use_degraded=False,
         max_rate=9.81),  # a board on a slope cannot out-accelerate free fall
)


@dataclass
class Event:
    metric: str
    kind: str
    severity: str  # "moderate" | "strong"
    phase: str
    t_start: float
    t_end: float
    t_peak: float
    frame_start: int
    frame_end: int
    frame_peak: int
    delta: float
    peak_rate: float
    text: str

    def as_dict(self) -> dict:
        return asdict(self)


def detect_edge_changes(frames: np.ndarray, t: np.ndarray, phases: list[str], x: np.ndarray,
                        err: np.ndarray, usable: np.ndarray, lateral_sign: float,
                        side_m: float = 0.05, min_swing_m: float = 0.12,
                        max_dur_s: float = 1.2, t_takeoff: float | None = None,
                        board_edge: np.ndarray | None = None) -> list[Event]:
    """Heel <-> toe edge changes from the CoM's toe(+)/heel(-) offset across the board.

    This is a transfer over ~0.3-0.8 s rather than a jolt, so the rate rules miss it. Sides use
    hysteresis: the rider is on the toe side above +side_m and on the heel side below -side_m.
    An edge change is a switch of side, spanning from the old side's extreme to the new side's
    extreme. It is reported as strong if it is quick or happens on the takeoff run, where the
    board should be settled and flat-based for the pop.
    """
    side = np.where(x > side_m, 1, np.where(x < -side_m, -1, 0))
    side[~usable] = 0
    events: list[Event] = []
    dt = float(np.median(np.diff(t)))
    reach = max(3, round(0.6 / dt))
    last_side, last_idx = 0, -1
    for i in range(len(x)):
        if side[i] == 0:
            continue
        if last_side and side[i] != last_side:
            # old extreme: within the stretch before; new extreme: shortly after the switch
            a0 = max(0, last_idx - reach)
            old = np.arange(a0, last_idx + 1)
            old = old[usable[old]]
            new = np.arange(i, min(len(x), i + reach))
            new = new[usable[new]]
            if old.size and new.size:
                # start: the last moment still at the old extreme (within 1 cm);
                # end: the first moment at the new extreme
                x_old = np.max(last_side * x[old])
                k0 = old[np.flatnonzero(last_side * x[old] >= x_old - 0.01)[-1]]
                x_new = np.max(side[i] * x[new])
                k1 = new[np.flatnonzero(side[i] * x[new] >= x_new - 0.01)[0]]
                swing = float(side[i] * x_new - last_side * x_old)  # extreme to extreme
                dur = float(t[k1] - t[k0])
                noise = 2 * float(np.nanmedian(err[old]))
                if abs(swing) >= max(min_swing_m, noise) and 0 < dur <= max_dur_s:
                    mid = (k0 + k1) // 2
                    to_toe = side[i] > 0
                    late = (t_takeoff is not None and 0 <= t_takeoff - t[k1] <= 1.0)
                    strong = phases[mid] == "takeoff_run" or late or abs(swing) / dur > 0.8
                    text = (f"Edge change {'heel → toe' if to_toe else 'toe → heel'}: CoM crosses "
                            f"{abs(swing) * 100:.0f}cm in {dur:.2f} s")
                    if board_edge is not None:
                        n_ = max(3, round(0.15 / dt))
                        before = np.nanmedian(board_edge[max(0, k0 - n_):k0 + 1])
                        after = np.nanmedian(board_edge[k1:k1 + n_])
                        if np.isfinite(before) and np.isfinite(after):
                            before, after = round(before) + 0.0, round(after) + 0.0  # no "-0°"
                            text += f"; board edge {before:+.0f}° → {after:+.0f}°"
                    if late:
                        text += f", {t_takeoff - t[k1]:.2f} s before takeoff"
                    events.append(Event(
                        metric="edge_change", kind="Edge change",
                        severity="strong" if strong else "moderate", phase=phases[mid],
                        t_start=float(t[k0]), t_end=float(t[k1]), t_peak=float(t[mid]),
                        frame_start=int(frames[k0]), frame_end=int(frames[k1]),
                        frame_peak=int(frames[mid]),
                        # delta in image-lateral terms, so the viewer can draw the arrow
                        delta=swing * lateral_sign, peak_rate=swing / dur, text=text,
                    ))
        last_side, last_idx = side[i], i
    return events


def _rate(v: np.ndarray, t: np.ndarray) -> np.ndarray:
    dt = float(np.median(np.diff(t)))
    win = max(5, round(0.1 / dt) | 1)
    if v.size <= win:
        return np.gradient(v, t)
    return savgol_filter(v, win, 2, deriv=1, delta=dt)


def detect(frames: np.ndarray, t: np.ndarray, phases: list[str],
           series: dict[str, tuple[np.ndarray, np.ndarray, list[str], np.ndarray]],
           k_mad: float = 5.0, max_gap: int = 2, pad: int = 3, min_conf: float = 0.5,
           edge: int = 4) -> list[Event]:
    """`series[metric] = (value, err, validity, confidence)`, aligned with `frames`."""
    events: list[Event] = []
    for rule in RULES:
        if rule.metric not in series:
            continue
        v, e, val, conf = series[rule.metric]
        ok_val = {Validity.VALID.value} | ({Validity.DEGRADED.value} if rule.use_degraded
                                           else set())
        usable = (np.isfinite(v) & np.array([x in ok_val for x in val]) & (conf >= min_conf))
        # the first and last frames of each usable stretch carry filter edge effects
        core = usable.copy()
        for k in range(1, edge + 1):
            core[k:] &= usable[:-k]
            core[:-k] &= usable[k:]
        if usable.sum() < 15:
            continue
        # detect on contiguous usable stretches only
        vv = v.copy()
        idx = np.arange(v.size)
        vv[~usable] = np.interp(idx[~usable], idx[usable], v[usable])
        rate = _rate(vv, t)
        r_ok = rate[usable]
        mad = 1.4826 * np.median(np.abs(r_ok - np.median(r_ok)))
        thr = max(rule.abs_rate, abs(float(np.median(r_ok))) + k_mad * mad)
        hot = core & (np.abs(rate) > thr)
        # group, bridging tiny gaps
        hits = np.flatnonzero(hot)
        groups: list[list[int]] = []
        for i in hits:
            if groups and i - groups[-1][-1] <= max_gap + 1:
                groups[-1].append(i)
            else:
                groups.append([i])
        for g in groups:
            a, b = max(0, g[0] - pad), min(v.size - 1, g[-1] + pad)
            seg = np.arange(a, b + 1)
            seg = seg[usable[seg]]
            if seg.size < 3:
                continue
            k = g[int(np.argmax(np.abs(rate[g])))]
            sign = np.sign(rate[k])
            # change across the event, measured between its extremes in the rate's direction
            lo, hi = seg[int(np.argmin(v[seg]))], seg[int(np.argmax(v[seg]))]
            first, last = (lo, hi) if sign > 0 else (hi, lo)
            delta = float(v[last] - v[first])
            noise = 2 * float(np.nanmedian(e[seg])) if np.isfinite(e[seg]).any() else 0.0
            if (abs(delta) < max(rule.min_delta, noise) or abs(delta) > rule.max_delta
                    or abs(rate[k]) > rule.max_rate):
                continue
            t0, t1 = float(t[min(first, last)]), float(t[max(first, last)])
            dur = max(t1 - t0, float(np.median(np.diff(t))))
            strong = abs(rate[k]) > 2 * thr or abs(delta) > 2.5 * rule.min_delta
            mag = abs(delta) * rule.scale
            text = (f"{rule.up if delta > 0 else rule.down} "
                    f"{mag:.0f}{rule.unit_disp} in {dur:.2f} s")
            events.append(Event(
                metric=rule.metric, kind=rule.kind, severity="strong" if strong else "moderate",
                phase=phases[k], t_start=t0, t_end=t1, t_peak=float(t[k]),
                frame_start=int(frames[min(first, last)]), frame_end=int(frames[max(first, last)]),
                frame_peak=int(frames[k]), delta=delta, peak_rate=float(rate[k]), text=text,
            ))
    events.sort(key=lambda ev: ev.t_peak)
    return events


def check_edge_transitions(events: list[Event], samples, frames: np.ndarray, t: np.ndarray,
                           board_width_m: float, side_m: float = 0.05) -> list[dict]:
    """For each edge change: how far the board's line moved across the slope *during the
    transition itself*.

    The transition runs from the last moment the rider is still clearly on the old edge (CoM
    more than `side_m` to that side of the board) to the first moment clearly on the new edge.
    Ideal technique rolls the board from edge to edge in place, so the new line lies within one
    board width of the old one. The event is annotated, and becomes "strong" when the shift
    exceeds one board width by more than its uncertainty.
    """
    pos = {int(f): i for i, f in enumerate(frames)}
    n = len(frames)
    line, err, th = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    for s in samples:
        if s.validity is Validity.INVALID:
            continue
        i = pos[s.frame_idx]
        if s.metric == "line_offset":
            line[i], err[i] = s.value, s.err_est
        elif s.metric == "com_toe_heel":
            th[i] = s.value
    out = []
    for ev in events:
        if ev.metric != "edge_change":
            continue
        k0, k1 = pos.get(ev.frame_start), pos.get(ev.frame_end)
        if k0 is None or k1 is None:
            continue
        to_toe = "heel → toe" in ev.text
        old = -1 if to_toe else 1  # side of the old edge (+ toe / - heel)
        # tighten to the transition: last frame still on the old side .. first on the new side
        seg = np.arange(k0, k1 + 1)
        on_old = seg[np.isfinite(th[seg]) & (old * th[seg] > side_m)]
        on_new = seg[np.isfinite(th[seg]) & (-old * th[seg] > side_m)]
        a = int(on_old[-1]) if on_old.size else k0
        b_candidates = on_new[on_new > a]
        b = int(b_candidates[0]) if b_candidates.size else k1
        # line at each end: median of the nearest 3 valid frames, to steady single-frame noise
        def at(i: int, direction: int) -> tuple[float, float]:
            idx = [j for j in range(i, i + 6 * direction, direction) if 0 <= j < n
                   and np.isfinite(line[j])][:3]
            if not idx:
                return float("nan"), float("nan")
            return float(np.median(line[idx])), float(np.median(err[idx]))
        l0, e0 = at(a, -1)
        l1, e1 = at(b, +1)
        if not (np.isfinite(l0) and np.isfinite(l1)):
            ev.text += "; line shift: not measurable here"
            continue
        shift = l1 - l0
        e_shift = float(np.hypot(e0, e1))
        widths = abs(shift) / board_width_m
        over = abs(shift) - e_shift > board_width_m
        ok = abs(shift) + e_shift <= board_width_m
        verdict = ("within one board width" if ok else
                   "more than one board width: the board slid sideways" if over else
                   "about one board width (borderline)")
        dur = float(t[b] - t[a])
        ev.text += (f"; during the {dur:.2f} s transition the line shifts "
                    f"{abs(shift) * 100:.0f}±{e_shift * 100:.0f} cm = {widths:.1f} board widths,"
                    f" {verdict}")
        if over:
            ev.severity = "strong"
        out.append({
            "t_start": float(t[a]), "t_end": float(t[b]), "frame_start": int(frames[a]),
            "frame_end": int(frames[b]), "duration_s": dur,
            "direction": "heel → toe" if to_toe else "toe → heel", "shift_m": shift,
            "err_m": e_shift, "board_widths": widths,
            "verdict": "ok" if ok else "over" if over else "borderline",
        })
    return out
