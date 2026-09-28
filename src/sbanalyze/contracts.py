"""Data contracts: the only module every other module may import."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path

import numpy as np


class ViewRole(StrEnum):
    FALL_LINE = "fall_line"  # primary, always present
    LIP_SIDE = "lip_side"  # optional, perpendicular to travel


class Phase(StrEnum):
    DROP_IN = "drop_in"
    TRANSITION = "transition"
    TAKEOFF_RUN = "takeoff_run"
    AIR = "air"  # takeoff -> touchdown


PHASE_ORDER = (Phase.DROP_IN, Phase.TRANSITION, Phase.TAKEOFF_RUN, Phase.AIR)


class Validity(StrEnum):
    VALID = "valid"
    DEGRADED = "degraded"
    INVALID = "invalid"


@dataclass(frozen=True)
class ClipMeta:
    clip_id: str
    run_id: str
    view: ViewRole
    path: Path
    width: int  # after rotation applied
    height: int
    fps: float
    n_frames: int
    rotation_deg: int
    recorded_at: datetime
    t_offset_s: float  # clip time -> run time; 0.0 for the primary view


@dataclass(frozen=True)
class RunWindow:
    start_frame: int
    end_frame: int  # inclusive
    method: str  # "motion_energy" | "manual" | "full_clip"
    confidence: float = 1.0


# HALPE-26 keypoint indices
class K:
    NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
    L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 5, 6, 7, 8, 9, 10
    L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 11, 12, 13, 14, 15, 16
    HEAD, NECK, HIP = 17, 18, 19
    L_BIG_TOE, R_BIG_TOE, L_SMALL_TOE, R_SMALL_TOE, L_HEEL, R_HEEL = 20, 21, 22, 23, 24, 25


HALPE26_EDGES: tuple[tuple[int, int], ...] = (
    (K.HEAD, K.NECK), (K.NECK, K.HIP),
    (K.NECK, K.L_SHOULDER), (K.NECK, K.R_SHOULDER),
    (K.L_SHOULDER, K.L_ELBOW), (K.L_ELBOW, K.L_WRIST),
    (K.R_SHOULDER, K.R_ELBOW), (K.R_ELBOW, K.R_WRIST),
    (K.HIP, K.L_HIP), (K.HIP, K.R_HIP),
    (K.L_HIP, K.L_KNEE), (K.L_KNEE, K.L_ANKLE),
    (K.R_HIP, K.R_KNEE), (K.R_KNEE, K.R_ANKLE),
    (K.L_ANKLE, K.L_HEEL), (K.L_ANKLE, K.L_BIG_TOE), (K.L_BIG_TOE, K.L_SMALL_TOE),
    (K.R_ANKLE, K.R_HEEL), (K.R_ANKLE, K.R_BIG_TOE), (K.R_BIG_TOE, K.R_SMALL_TOE),
    (K.NOSE, K.L_EYE), (K.NOSE, K.R_EYE), (K.L_EYE, K.L_EAR), (K.R_EYE, K.R_EAR),
)


@dataclass(frozen=True)
class PoseFrame:
    frame_idx: int
    t_sec: float
    keypoints: np.ndarray  # (26, 2) float32, image pixels, HALPE-26
    scores: np.ndarray  # (26,) float32
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1
    rider_px_h: float  # standing-equivalent rider height; drives all uncertainty downstream


@dataclass(frozen=True)
class Calibration:
    """Session-scoped camera and venue calibration.

    The mat is curved (steep in-run, then the kicker), so it is not one plane, and a single
    ground homography would be wrong over most of the run. See calib.py for what replaces it.
    """

    focal_px: float
    stature_m: float
    chain_fraction: float
    left_edge_x: np.ndarray  # (H,) float32: mat left boundary x per image row, NaN where unknown
    mat_top_y: int  # image row of the lip / far end of the mat
    method: str  # "body_scale+mat_mask"
    residual_px: float  # left-edge smoothness residual, for QA
    transition_y: float = float("nan")  # image row where the in-run starts to curve
    takeoff_y: float = float("nan")  # image row where the kicker face starts
    pitch_deg: float = float("nan")  # camera pitch below horizontal, from the vertical VP
    pitch_sd_deg: float = float("nan")


@dataclass(frozen=True)
class MetricSample:
    run_id: str
    view: ViewRole  # which camera produced this value
    frame_idx: int
    t_sec: float  # run time, not clip time
    phase: Phase
    metric: str
    value: float
    confidence: float
    rider_px_h: float
    err_est: float
    validity: Validity

    @property
    def valid(self) -> bool:
        return self.validity is not Validity.INVALID


class PipelineError(Exception):
    """Typed stage failure. `status` is persisted on the clip."""

    def __init__(self, stage: str, status: str, message: str):
        super().__init__(f"[{stage}] {message}")
        self.stage = stage
        self.status = status
