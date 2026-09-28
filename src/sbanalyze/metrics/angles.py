"""Joint and segment angles in the image plane."""

from __future__ import annotations

import numpy as np


def joint_flexion(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Flexion at b for the chain a-b-c: 0 deg = straight, positive = bent. Shapes (N, 2)."""
    u, v = a - b, c - b
    cos = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1))
    return 180.0 - np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def lean_from_vertical(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Signed angle of lower->upper from image-up; positive leans toward image right."""
    d = upper - lower
    return np.degrees(np.arctan2(d[..., 0], -d[..., 1]))
