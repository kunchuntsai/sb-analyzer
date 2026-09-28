"""Pose backend protocol. Anything that yields PoseFrames can replace the RTM backend (N-6)."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..contracts import K, PoseFrame

# Landmark heights as a fraction of stature (Winter / de Leva): ankle 0.039, knee 0.285,
# hip 0.530, neck (C7) 0.818, head top 1.0. The chain between them is crouch-invariant.
CHAIN_FRACTION = 0.961


class PoseBackend(Protocol):
    name: str

    def reset(self, first_frame: np.ndarray, roi_mask: np.ndarray | None) -> None:
        """Start a new track. `roi_mask` (full-res, uint8) marks where the rider may be."""

    def set_roi(self, roi_mask: np.ndarray | None) -> None:
        """Change where the rider may be, e.g. once airborne above the lip."""

    def observe(self, motion_small: np.ndarray | None) -> None:
        """Optional: this frame's moving-pixel mask (analysis resolution), used to tell the
        rider from people standing around."""

    def infer_frame(self, frame_idx: int, t_sec: float, image: np.ndarray) -> PoseFrame | None:
        """Pose for the tracked rider in one frame, or None when the rider is lost."""


# hip joint centre (0.530) -> head top (1.0): rigid enough not to fold when the legs do
TORSO_FRACTION = 0.470


def scale_px(kp: np.ndarray, sc: np.ndarray, thr: float) -> float:
    """Standing-equivalent rider height in px from the body, or NaN.

    Both the full chain and the torso+head chain only ever get shorter through foreshortening:
    a crouch folds the thighs away from the camera, and the pop at the lip folds the legs toward
    it. So the larger of the two estimates is the less biased one.
    """
    full = chain_px(kp, sc, thr) / CHAIN_FRACTION
    t1 = np.linalg.norm(kp[K.HIP] - kp[K.NECK]) if min(sc[K.HIP], sc[K.NECK]) >= thr else np.nan
    t2 = np.linalg.norm(kp[K.NECK] - kp[K.HEAD]) if min(sc[K.NECK], sc[K.HEAD]) >= thr else np.nan
    torso = (t1 + t2) / TORSO_FRACTION
    vals = [v for v in (full, torso) if np.isfinite(v)]
    return float(max(vals)) if vals else float("nan")


def chain_px(kp: np.ndarray, sc: np.ndarray, thr: float) -> float:
    """Ankle->knee->hip->neck->head-top length in pixels, averaged over confident legs."""

    def d(a: int, b: int) -> float:
        return float(np.linalg.norm(kp[a] - kp[b])) if min(sc[a], sc[b]) >= thr else np.nan

    legs = [d(K.L_ANKLE, K.L_KNEE) + d(K.L_KNEE, K.L_HIP),
            d(K.R_ANKLE, K.R_KNEE) + d(K.R_KNEE, K.R_HIP)]
    legs = [v for v in legs if np.isfinite(v)]
    torso = d(K.HIP, K.NECK) + d(K.NECK, K.HEAD)
    if not legs or not np.isfinite(torso):
        return np.nan
    # the longer leg is the less foreshortened one
    return max(legs) + torso
