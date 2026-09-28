"""Board axis from HALPE-26 foot keypoints (ADR-009): the feet are bound to the board."""

from __future__ import annotations

import numpy as np

from ..contracts import K

LEFT_FOOT = (K.L_ANKLE, K.L_HEEL, K.L_BIG_TOE, K.L_SMALL_TOE)
RIGHT_FOOT = (K.R_ANKLE, K.R_HEEL, K.R_BIG_TOE, K.R_SMALL_TOE)


def foot_centroid(kp: np.ndarray, sc: np.ndarray, idx: tuple[int, ...], thr: float
                  ) -> tuple[np.ndarray, float]:
    """Score-weighted centroid of one foot, and its mean confidence."""
    s = sc[list(idx)]
    w = np.where(s >= thr, s, 0.0)
    if w.sum() <= 0:
        return np.array([np.nan, np.nan], np.float32), float(s.mean())
    return (kp[list(idx)] * w[:, None]).sum(0) / w.sum(), float(s.mean())


def board_axis(kp: np.ndarray, sc: np.ndarray, thr: float
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """(left_foot, right_foot, contact_midpoint, confidence) in image pixels."""
    lf, lc = foot_centroid(kp, sc, LEFT_FOOT, thr)
    rf, rc = foot_centroid(kp, sc, RIGHT_FOOT, thr)
    return lf, rf, (lf + rf) / 2, min(lc, rc)
