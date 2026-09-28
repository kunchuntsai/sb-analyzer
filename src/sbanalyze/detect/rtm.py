"""RTMDet/YOLOX person detection + RTMPose HALPE-26 via rtmlib (ONNX Runtime, CPU).

Person detection, not background subtraction: the floodlit cast shadow on the pale matting is
rider-sized (ADR-005). Detection runs on a crop around the tracked box, upscaled to the detector
input, so a ~200 px rider at the lip is still found. RTMPose then samples its 192x256 input from
the full-resolution frame, so the far rider keeps every available pixel.
"""

from __future__ import annotations

import numpy as np

from ..contracts import PoseFrame
from .base import scale_px

_MODES = {
    "lightweight": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip",
        (416, 416),
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7-halpe26_700e-256x192-7f134165_20230605.zip",
        (192, 256),
    ),
    "balanced": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
        (640, 640),
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip",
        (192, 256),
    ),
    "performance": (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip",
        (640, 640),
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.zip",
        (288, 384),
    ),
}


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class RtmBackend:
    name = "rtmlib"

    def __init__(self, mode: str = "balanced", crop_margin: float = 1.8, min_crop_px: int = 480,
                 kpt_thr: float = 0.35, max_lost: int = 12):
        import onnxruntime as ort
        from rtmlib import YOLOX, RTMPose

        ort.set_default_logger_severity(3)
        det_url, det_in, pose_url, pose_in = _MODES[mode]
        self.det = YOLOX(det_url, model_input_size=det_in, score_thr=0.45, nms_thr=0.45)
        self.pose = RTMPose(pose_url, model_input_size=pose_in)
        self.name = f"rtmlib-{mode}"
        self.crop_margin = crop_margin
        self.min_crop_px = min_crop_px
        self.kpt_thr = kpt_thr
        self.max_lost = max_lost
        self.box: np.ndarray | None = None
        self.vel = np.zeros(4)
        self.lost = 0
        self.roi: np.ndarray | None = None
        # Scene cues at analysis resolution: the mat itself (not its surroundings) and what is
        # moving in the current frame. Only a moving person on the mat can be the rider.
        self.mat_small: np.ndarray | None = None
        self.motion: np.ndarray | None = None
        self.s = 1.0  # analysis px per full-res px
        self.history: list[tuple[float, np.ndarray]] = []
        self.airborne = False
        self.t = 0.0

    # ---- tracking -------------------------------------------------------------------------

    def reset(self, first_frame: np.ndarray, roi_mask: np.ndarray | None,
              mat_small: np.ndarray | None = None, scale: float = 1.0) -> None:
        self.box, self.vel, self.lost, self.roi = None, np.zeros(4), 0, roi_mask
        self.mat_small, self.s = mat_small, scale
        self.history, self.airborne, self.motion = [], False, None

    def set_roi(self, roi_mask: np.ndarray | None) -> None:
        self.roi = roi_mask
        self.airborne = True  # only switched once the rider has left the lip

    def observe(self, motion_small: np.ndarray | None) -> None:
        """This frame's foreground (moving) mask at analysis resolution."""
        self.motion = motion_small

    def _motion_frac(self, box: np.ndarray) -> float:
        if self.motion is None:
            return 1.0
        h, w = self.motion.shape
        x0, y0, x1, y1 = (np.array(box) * self.s).astype(int)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        return float((self.motion[y0:y1, x0:x1] > 0).mean())

    def _on_mat(self, box: np.ndarray) -> bool:
        """Feet on the white mat itself, not merely next to it."""
        if self.mat_small is None:
            return True
        h, w = self.mat_small.shape
        x = int(np.clip((box[0] + box[2]) / 2 * self.s, 0, w - 1))
        y = int(np.clip(box[3] * self.s, 0, h - 1))
        return bool(self.mat_small[y, x])

    def _is_rider(self, box: np.ndarray) -> bool:
        return self._on_mat(box) and self._motion_frac(box) >= 0.08

    def _on_roi(self, box: np.ndarray) -> bool:
        if self.roi is None:
            return True
        # feet (bottom centre) must be on or next to the mat
        x = int(np.clip((box[0] + box[2]) / 2, 0, self.roi.shape[1] - 1))
        y = int(np.clip(box[3], 0, self.roi.shape[0] - 1))
        return bool(self.roi[y, x])

    def _detect(self, image: np.ndarray, region: tuple[int, int, int, int] | None) -> np.ndarray:
        if region is None:
            boxes = self.det(image)
        else:
            x0, y0, x1, y1 = region
            boxes = self.det(image[y0:y1, x0:x1])
            if len(boxes):
                boxes = boxes + np.array([x0, y0, x0, y0], dtype=np.float32)
        return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)

    def _search_region(self, image: np.ndarray, box: np.ndarray) -> tuple[int, int, int, int]:
        h, w = image.shape[:2]
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        half = max(self.min_crop_px, self.crop_margin * max(box[2] - box[0], box[3] - box[1])) / 2
        return (int(max(0, cx - half)), int(max(0, cy - half)),
                int(min(w, cx + half)), int(min(h, cy + half)))

    def _pick(self, boxes: np.ndarray, pred: np.ndarray | None) -> np.ndarray | None:
        boxes = np.array([b for b in boxes if self._on_roi(b)]).reshape(-1, 4)
        if not len(boxes):
            return None
        if pred is None:
            # Acquire: only a person standing on the mat AND moving can be the rider. A
            # spectator beside the slope, or someone standing still, never qualifies; if nobody
            # does yet (the rider has not dropped in), keep waiting.
            boxes = np.array([b for b in boxes if self._is_rider(b)]).reshape(-1, 4)
            if not len(boxes):
                return None
            motion = np.array([self._motion_frac(b) for b in boxes])
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            return boxes[int(np.argmax(areas * motion))]
        ious = np.array([_iou(b, pred) for b in boxes])
        if ious.max() > 0.05:
            return boxes[int(np.argmax(ious))]
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        pc = (pred[:2] + pred[2:]) / 2
        dist = np.linalg.norm(centers - pc, axis=1)
        diag = np.linalg.norm(pred[2:] - pred[:2])
        i = int(np.argmin(dist))
        return boxes[i] if dist[i] < diag else None

    def _stationary(self, box: np.ndarray, t: float, window_s: float = 1.0) -> bool:
        if self.airborne:
            return False  # after takeoff the rider may stop on the landing: keep them
        c = np.array([(box[0] + box[2]) / 2, box[3]])
        self.history.append((t, c))
        self.history = [(ti, ci) for ti, ci in self.history if t - ti <= window_s]
        if t - self.history[0][0] < 0.9 * window_s:
            return False
        moved = max(np.linalg.norm(ci - c) for _, ci in self.history)
        return moved < 0.25 * (box[3] - box[1]) and self._motion_frac(box) < 0.03

    def infer_frame(self, frame_idx: int, t_sec: float, image: np.ndarray) -> PoseFrame | None:
        pred = None if self.box is None else self.box + self.vel
        box = None
        if pred is not None:
            box = self._pick(self._detect(image, self._search_region(image, pred)), pred)
        if box is None:
            box = self._pick(self._detect(image, None), pred)
        if box is None:
            self.lost += 1
            if self.lost > self.max_lost:
                self.box, self.vel = None, np.zeros(4)
            return None
        if self.box is not None:
            self.vel = 0.5 * self.vel + 0.5 * (box - self.box)
        self.box, self.lost = box, 0
        self.t = t_sec
        if self._stationary(box, t_sec):
            # a person who has not moved for a second while the rider would be moving is not
            # the rider: let go and re-acquire from the moving people on the mat
            self.box, self.vel, self.history = None, np.zeros(4), []
            return None

        kps, scores = self.pose(image, bboxes=[box.tolist()])
        kp = kps[0].astype(np.float32)
        sc = scores[0].astype(np.float32)
        return PoseFrame(
            frame_idx=frame_idx, t_sec=t_sec, keypoints=kp, scores=sc,
            bbox=tuple(int(v) for v in box), rider_px_h=scale_px(kp, sc, self.kpt_thr),
        )
