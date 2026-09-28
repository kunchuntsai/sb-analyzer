"""Phase validity matrix, keyed on (metric, phase, view). Versioned: bump on any change."""

from __future__ import annotations

from ..contracts import PHASE_ORDER, Phase, Validity, ViewRole

VALIDITY_VERSION = 5

V, D, X = Validity.VALID, Validity.DEGRADED, Validity.INVALID

# columns: drop-in, transition, takeoff run, air
_FALL_LINE: dict[str, tuple[Validity, Validity, Validity, Validity]] = {
    # In the air the rider grabs and rotates: 2D joint angles see foreshortening and the pose
    # model can swap left and right, so they are only indicative there.
    "knee_flex_l": (V, V, V, D),
    "knee_flex_r": (V, V, V, D),
    "trunk_lean": (V, V, V, D),
    # in the air the board is no longer a stable reference (grabs, rotation)
    "com_height": (V, V, V, D),
    "com_lateral": (V, V, V, D),
    "com_toe_heel": (V, V, V, X),  # no edge to be on in the air
    # far from the camera each foot is only ~25 px across, so the takeoff run is noisier
    "board_edge": (V, V, D, X),
    "line_offset": (V, V, V, X),
    # On the kicker face the path rises toward the camera's line of sight, and the pop confounds
    # body scale, so range rate is only an approximation of along-path speed there.
    "speed_along": (V, V, D, X),
    "range": (V, V, V, D),
    # along the board = along the optical axis once the rider points down the fall line
    "com_foreaft": (V, D, X, X),
    "board_yaw": (V, V, D, X),
    "air_height": (X, X, X, V),
}

# Rotated ninety degrees: strengths and weaknesses mirror the fall-line camera.
_LIP_SIDE: dict[str, tuple[Validity, Validity, Validity, Validity]] = {
    "com_foreaft": (D, V, V, V),
    "com_height": (V, V, V, V),
    "com_lateral": (X, D, D, X),
    "line_offset": (X, D, D, X),
    "knee_flex_l": (D, V, V, V),
    "knee_flex_r": (D, V, V, V),
    "air_height": (X, X, X, V),
}

MATRIX = {ViewRole.FALL_LINE: _FALL_LINE, ViewRole.LIP_SIDE: _LIP_SIDE}


def validity(metric: str, phase: Phase, view: ViewRole = ViewRole.FALL_LINE) -> Validity:
    row = MATRIX[view].get(metric)
    return row[PHASE_ORDER.index(phase)] if row else Validity.INVALID


RANK = {Validity.VALID: 2, Validity.DEGRADED: 1, Validity.INVALID: 0}
