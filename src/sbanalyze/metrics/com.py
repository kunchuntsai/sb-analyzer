"""Whole-body centre of mass from 2D keypoints using de Leva (1996) segment parameters."""

from __future__ import annotations

import numpy as np

from ..contracts import K

# (proximal, distal, mass fraction, CoM position from proximal as fraction of length)
# de Leva 1996, male. Trunk runs cervicale (neck) -> mid-hip; head runs vertex -> cervicale.
SEGMENTS: tuple[tuple[int, int, float, float], ...] = (
    (K.HEAD, K.NECK, 0.0694, 0.5002),
    (K.NECK, K.HIP, 0.4346, 0.4486),
    (K.L_SHOULDER, K.L_ELBOW, 0.0271, 0.5772),
    (K.R_SHOULDER, K.R_ELBOW, 0.0271, 0.5772),
    (K.L_ELBOW, K.L_WRIST, 0.0162, 0.4574),
    (K.R_ELBOW, K.R_WRIST, 0.0162, 0.4574),
    (K.L_WRIST, K.L_WRIST, 0.0061, 0.0),  # hand lumped at the wrist
    (K.R_WRIST, K.R_WRIST, 0.0061, 0.0),
    (K.L_HIP, K.L_KNEE, 0.1416, 0.4095),
    (K.R_HIP, K.R_KNEE, 0.1416, 0.4095),
    (K.L_KNEE, K.L_ANKLE, 0.0433, 0.4459),
    (K.R_KNEE, K.R_ANKLE, 0.0433, 0.4459),
    (K.L_HEEL, K.L_BIG_TOE, 0.0137, 0.4415),
    (K.R_HEEL, K.R_BIG_TOE, 0.0137, 0.4415),
)

assert abs(sum(s[2] for s in SEGMENTS) - 1.0) < 1e-3


def centre_of_mass(kp: np.ndarray) -> np.ndarray:
    """kp: (..., 26, 2) -> (..., 2). NaN if any contributing keypoint is NaN."""
    com = np.zeros(kp.shape[:-2] + (2,), np.float64)
    for prox, dist, mass, pos in SEGMENTS:
        com += mass * (kp[..., prox, :] + pos * (kp[..., dist, :] - kp[..., prox, :]))
    return com
