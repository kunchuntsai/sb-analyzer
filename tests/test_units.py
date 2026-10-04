import numpy as np
import pytest

from sbanalyze.calib import mat_left_edge, terrain_rows
from sbanalyze.contracts import K, MetricSample, Phase, Validity, ViewRole
from sbanalyze.fuse import fuse
from sbanalyze.metrics.angles import joint_flexion, lean_from_vertical
from sbanalyze.metrics.com import SEGMENTS, centre_of_mass
from sbanalyze.metrics.validity import validity
from sbanalyze.phase import segment_by_speed, segment_by_yaw
from sbanalyze.trim import find_run_window


def _sample(frame, metric, view, val, err):
    return MetricSample("r", view, frame, frame / 60, Phase.TAKEOFF_RUN, metric, 1.0, 0.9,
                        300.0, err, val)


def test_fuse_without_secondary_is_identity():
    primary = [_sample(i, "com_height", ViewRole.FALL_LINE, Validity.VALID, 0.01) for i in range(5)]
    assert fuse(primary, None) is primary


def test_fuse_prefers_better_validity_then_lower_error():
    p = [_sample(0, "com_foreaft", ViewRole.FALL_LINE, Validity.INVALID, 0.01),
         _sample(0, "com_height", ViewRole.FALL_LINE, Validity.VALID, 0.02)]
    s = [_sample(0, "com_foreaft", ViewRole.LIP_SIDE, Validity.VALID, 0.05),
         _sample(0, "com_height", ViewRole.LIP_SIDE, Validity.VALID, 0.01)]
    out = {m.metric: m.view for m in fuse(p, s)}
    assert out == {"com_foreaft": ViewRole.LIP_SIDE, "com_height": ViewRole.LIP_SIDE}


def test_de_leva_masses_sum_to_one():
    assert sum(s[2] for s in SEGMENTS) == pytest.approx(1.0, abs=1e-3)


def test_com_of_collapsed_body_is_that_point():
    kp = np.tile(np.array([[10.0, 20.0]]), (26, 1))
    np.testing.assert_allclose(centre_of_mass(kp), [10.0, 20.0])


def test_com_upright_figure_sits_near_hips():
    kp = np.zeros((26, 2))
    x = 0.0
    heights = {K.HEAD: 1.0, K.NECK: 0.818, K.HIP: 0.53, K.L_HIP: 0.53, K.R_HIP: 0.53,
               K.L_KNEE: 0.285, K.R_KNEE: 0.285, K.L_ANKLE: 0.039, K.R_ANKLE: 0.039,
               K.L_HEEL: 0.0, K.R_HEEL: 0.0, K.L_BIG_TOE: 0.0, K.R_BIG_TOE: 0.0,
               K.L_SHOULDER: 0.818, K.R_SHOULDER: 0.818, K.L_ELBOW: 0.63, K.R_ELBOW: 0.63,
               K.L_WRIST: 0.485, K.R_WRIST: 0.485}
    for j, h in heights.items():
        kp[j] = [x, -h]  # image y grows downward
    com_h = -centre_of_mass(kp)[1]
    assert 0.53 < com_h < 0.60  # textbook whole-body CoM ~0.55-0.57 of stature


def test_joint_flexion_and_lean():
    a, b = np.array([[0.0, 0.0]]), np.array([[0.0, 1.0]])
    assert joint_flexion(a, b, np.array([[0.0, 2.0]]))[0] == pytest.approx(0.0, abs=1e-6)
    assert joint_flexion(a, b, np.array([[1.0, 1.0]]))[0] == pytest.approx(90.0)
    assert lean_from_vertical(np.array([[0.0, 1.0]]), np.array([[1.0, 0.0]]))[0] == pytest.approx(45.0)


def test_phase_state_machine_on_synthetic_turn():
    fps = 60
    t = np.arange(150) / fps
    yaw = np.where(t < 0.8, 85.0, np.where(t < 1.3, 85.0 - (t - 0.8) / 0.5 * 75.0, 10.0))
    spans = segment_by_yaw(t, yaw, fps, 90.0, 30.0, 0.15, 0.2)
    assert spans is not None
    assert spans[Phase.DROP_IN][1] == pytest.approx(0.8, abs=0.05)
    assert 1.0 < spans[Phase.TAKEOFF_RUN][0] < 1.35


def test_phase_state_machine_rejects_flat_yaw_and_speed_fallback_orders_phases():
    t = np.arange(150) / 60
    assert segment_by_yaw(t, np.full(150, 5.0), 60, 90.0, 30.0, 0.15, 0.2) is None
    spans = segment_by_speed(t, np.linspace(0, 12, 150), 0.2)
    assert spans[Phase.DROP_IN][1] <= spans[Phase.TRANSITION][1] <= spans[Phase.TAKEOFF_RUN][1]


def test_run_window_closes_short_gaps_and_pads():
    e = np.zeros(300)
    e[100:200] = 1e-3
    e[150:155] = 0  # a few frames where the rider matches the background
    w = find_run_window(e, 60.0, 1e-4, 0.25)
    assert (w.start_frame, w.end_frame) == (85, 214)


def test_mat_left_edge_recovers_a_straight_boundary():
    mask = np.zeros((200, 100), np.uint8)
    for y in range(40, 200):
        mask[y, 60 - y // 5:] = 255
    edge, top, _residual = mat_left_edge(mask, 400, 0.5)
    assert top == 80
    assert edge[300] == pytest.approx((60 - 150 // 5) / 0.5, abs=2)


def test_terrain_rows_find_the_bend():
    h, w, top = 4000, 2000, 1000
    y = np.arange(h, dtype=np.float64)
    # straight in-run below 2600 (slope -0.6), blending to vertical above 2000
    k = np.clip((y - 2000) / 600, 0, 1)
    edge = np.cumsum(-0.6 * k)  # x decreases as y grows on the straight part
    edge = edge - edge[h - 1] + 1000
    edge[:top] = np.nan
    y_tr, y_to = terrain_rows(edge.astype(np.float32), top, w)
    assert 2350 < y_tr < 2700
    assert 1950 < y_to < 2250


def test_validity_matrix_kills_foreaft_in_takeoff_run_for_fall_line_only():
    assert validity("com_foreaft", Phase.TAKEOFF_RUN) is Validity.INVALID
    assert validity("com_foreaft", Phase.TAKEOFF_RUN, ViewRole.LIP_SIDE) is Validity.VALID
    assert validity("com_height", Phase.TRANSITION) is Validity.VALID


def test_flight_takeoff_touchdown_and_ballistic_height():
    from sbanalyze.air import find_takeoff, find_touchdown

    t = np.arange(0, 1.6, 1 / 60)
    lip = 1600.0
    # feet row: on the kicker (below the lip row) until 0.4 s, then above it
    fy = np.where(t < 0.4, lip + 50 - 100 * t, lip - 30)
    assert abs(t[find_takeoff(fy, lip)] - 0.4) < 0.05
    # ballistic board height: vz = 3 m/s from 0.4 s, lands on a surface 1 m below the lip
    tf = np.clip(t - 0.4, 0, None)
    h = np.maximum(3 * tf - 4.905 * tf**2, -1.0)
    k0 = int(np.argmax(t >= 0.4))
    td, seen = find_touchdown(t, h, k0)
    t_land = 0.4 + (3 + np.sqrt(9 + 4 * 4.905)) / (2 * 4.905)
    assert seen and abs(t[td] - t_land) < 0.06


def test_events_catch_a_sudden_com_drop_and_ignore_smooth_motion():
    from sbanalyze.events import detect

    t = np.arange(0, 3, 1 / 60)
    frames = np.arange(t.size)
    smooth = 0.8 + 0.03 * np.sin(2 * np.pi * 0.5 * t)
    jerky = smooth - 0.15 / (1 + np.exp(-(t - 1.5) / 0.02))  # drops 15 cm in ~0.1 s
    phases = ["transition"] * t.size

    def series(v):
        return {"com_height": (v, np.full(t.size, 0.01), ["valid"] * t.size, np.ones(t.size))}

    assert detect(frames, t, phases, series(smooth)) == []
    ev = detect(frames, t, phases, series(jerky))
    assert len(ev) == 1 and ev[0].delta < -0.1 and abs(ev[0].t_peak - 1.5) < 0.05
    assert "CoM drops" in ev[0].text


def test_edge_change_heel_to_toe_is_detected_with_direction_and_timing():
    from sbanalyze.events import detect_edge_changes

    t = np.arange(0, 3, 1 / 60)
    frames = np.arange(t.size)
    # heel side (-12 cm) until 1.2 s, crosses over 0.6 s, toe side (+14 cm) after
    x = np.interp(t, [0, 1.2, 1.8, 3], [-0.12, -0.12, 0.14, 0.14])
    ev = detect_edge_changes(frames, t, ["transition"] * t.size, x, np.full(t.size, 0.01),
                             np.ones(t.size, bool), lateral_sign=-1.0, t_takeoff=2.5)
    assert len(ev) == 1
    e = ev[0]
    assert "heel → toe" in e.text and abs(e.delta) > 0.25
    assert 1.1 < e.t_start < 1.3 and 1.7 < e.t_end < 2.0
    assert e.severity == "strong"  # within 1 s of takeoff


def test_stance_from_lead_foot_and_toe_direction_and_label_swap():
    from sbanalyze.contracts import K
    from sbanalyze.metrics import decide_stance

    n = 30
    kp = np.zeros((n, 26, 2))
    sc = np.ones((n, 26))
    # goofy seen from behind: right foot further (higher in image), toes point image-left
    kp[:, K.R_ANKLE] = [100, 400]; kp[:, K.L_ANKLE] = [100, 460]
    for heel, big, y in ((K.L_HEEL, K.L_BIG_TOE, 470), (K.R_HEEL, K.R_BIG_TOE, 410)):
        kp[:, heel] = [110, y]; kp[:, big] = [85, y]
    left_leads, info = decide_stance(kp, sc)
    assert not left_leads and info["value"] == "goofy" and info["cues_agree"]
    # the pose model swaps left/right: the lead-foot cue flips, toe direction does not
    kp[:, [K.L_ANKLE, K.R_ANKLE]] = kp[:, [K.R_ANKLE, K.L_ANKLE]]
    left_leads, info = decide_stance(kp, sc)
    assert not left_leads and not info["cues_agree"]
    assert decide_stance(kp, sc, "regular")[0] and decide_stance(kp, sc, "regular")[1]["source"] == "config"


def test_board_edge_roll_sign_and_cleaning():
    from sbanalyze.boardedge import clean_series, roll_from_segment

    # goofy seen from behind: toe side is image-left; toe end lower (larger y) = toe edge (+)
    seg = (100.0, 210.0, 200.0, 200.0)  # left end 10 px lower
    assert roll_from_segment(seg, toe_on_right=False, depression_rad=0.0) > 5
    assert roll_from_segment(seg, toe_on_right=True, depression_rad=0.0) < -5
    # a flat run with one wild detection: the outlier is dropped, error stays small
    t = np.arange(40) / 60
    raw = np.full(40, 1.0) + np.random.default_rng(0).normal(0, 0.5, 40)
    raw[20] = 25.0
    val, err = clean_series(raw, np.full(40, 80.0), t, 3.0)
    assert abs(val[20] - 1.0) < 1.5 and np.nanmax(err) < 3


def test_gravity_alignment_removes_phone_roll():
    from sbanalyze.boardedge import roll_from_segment
    from sbanalyze.calib import gravity_tilt, to_gravity

    # a phone rolled by 3 deg: gravity's vanishing point sits off to the side
    roll = np.radians(3.0)
    p = np.array([500.0, 1000.0])
    vp = p + 20000 * np.array([np.sin(roll), np.cos(roll)])
    phi = float(gravity_tilt(vp[0], vp[1], p[0], p[1]))
    assert abs(np.degrees(phi) - 3.0) < 1e-6
    # a true vertical (along gravity) becomes image-vertical after correction
    up = -(vp - p) / np.linalg.norm(vp - p)
    g = to_gravity(up, phi)
    assert abs(g[0]) < 1e-9 and g[1] < 0
    # a level board edge, drawn rotated by the phone's roll, reads ~0 after correction
    d = to_gravity(np.array([1.0, 0.0]), -phi) * 100  # how a level line appears in the image
    seg = (p[0] - d[0] / 2, p[1] - d[1] / 2, p[0] + d[0] / 2, p[1] + d[1] / 2)
    assert abs(roll_from_segment(seg, True, 0.0, phi)) < 1e-6
    assert abs(roll_from_segment(seg, True, 0.0, 0.0)) > 2.5  # uncorrected: the roll leaks in


def test_edge_transition_line_shift_against_board_width():
    from sbanalyze.contracts import MetricSample, Phase, Validity, ViewRole
    from sbanalyze.events import Event, check_edge_transitions

    t = np.arange(120) / 60
    frames = np.arange(120)

    def run(shift_m):
        line = np.where(t < 0.8, 1.0, np.where(t > 1.2, 1.0 + shift_m, 1.0 + shift_m * (t - 0.8) / 0.4))
        samples = [MetricSample("r", ViewRole.FALL_LINE, int(f), float(tt), Phase.TRANSITION,
                                "line_offset", float(v), 0.9, 300.0, 0.01, Validity.VALID)
                   for f, tt, v in zip(frames, t, line)]
        ev = Event("edge_change", "Edge change", "moderate", "transition", 0.8, 1.2, 1.0,
                   48, 72, 60, shift_m, 0.0, "Edge change heel → toe: ...")
        return ev, check_edge_transitions([ev], samples, frames, t, 0.25)

    ev, out = run(0.10)
    assert out[0]["verdict"] == "ok" and "within one board width" in ev.text
    ev, out = run(0.60)
    assert out[0]["verdict"] == "over" and ev.severity == "strong"
    assert abs(out[0]["board_widths"] - 2.4) < 0.05


def test_edge_transition_measures_only_the_crossover():
    from sbanalyze.contracts import MetricSample, Phase, Validity, ViewRole
    from sbanalyze.events import Event, check_edge_transitions

    t = np.arange(120) / 60
    frames = np.arange(120)
    # CoM: heel side until 0.9 s, crosses to the toe side by 1.1 s
    th = np.interp(t, [0, 0.9, 1.1, 2], [-0.12, -0.06, 0.06, 0.12])
    # the line drifts 0.5 m before the crossover (riding across) but only 0.1 m during it
    line = np.interp(t, [0, 0.9, 1.1, 2], [1.0, 1.5, 1.6, 1.6])
    mk = lambda metric, vals: [MetricSample("r", ViewRole.FALL_LINE, int(f), float(tt),
                                            Phase.TRANSITION, metric, float(v), 0.9, 300.0,
                                            0.01, Validity.VALID)
                               for f, tt, v in zip(frames, t, vals)]
    ev = Event("edge_change", "Edge change", "moderate", "transition", 0.0, 1.9, 1.0,
               0, 115, 60, 0.24, 0.0, "Edge change heel → toe: ...")
    out = check_edge_transitions([ev], mk("line_offset", line) + mk("com_toe_heel", th),
                                 frames, t, 0.25)
    assert 0.85 < out[0]["t_start"] < 0.95 and 1.05 < out[0]["t_end"] < 1.15
    assert abs(out[0]["shift_m"] - 0.1) < 0.03 and out[0]["verdict"] == "ok"
