from __future__ import annotations

import numpy as np
from .types import ViolationEvent, ViolationType


class TripleRidingDetector:
    """Detects 3+ riders on a two-wheeler.

    Heuristic: if a motorcycle bounding box is unusually tall relative
    to its width, or if rider head detections exceed 2 in the crop area.
    """

    def __init__(self, aspect_ratio_threshold: float = 2.5):
        self.aspect_ratio_threshold = aspect_ratio_threshold

    def detect(self, detections, frame: np.ndarray, frame_idx: int, signal_state: str) -> list[ViolationEvent]:
        violations = []
        h, w = frame.shape[:2]

        if not hasattr(detections, "class_id") or detections.class_id is None:
            return violations

        for i in range(len(detections)):
            cls_id = int(detections.class_id[i])
            if cls_id != 3:
                continue

            bbox = detections.xyxy[i]
            bw = bbox[2] - bbox[0]
            bh = bbox[3] - bbox[1]

            if bw <= 0 or bh <= 0:
                continue

            aspect_ratio = bh / bw

            crop = self._safe_crop(frame, bbox)
            head_count = self._count_heads(crop) if crop is not None else 0

            if aspect_ratio > self.aspect_ratio_threshold or head_count >= 3:
                conf = float(detections.confidence[i]) if detections.confidence is not None else 0.7
                tid = int(detections.tracker_id[i]) if hasattr(detections, "tracker_id") and detections.tracker_id is not None else -1
                violations.append(
                    ViolationEvent(
                        type=ViolationType.TRIPLE_RIDING,
                        track_id=tid,
                        frame=frame_idx,
                        confidence=conf,
                        bbox=tuple(bbox.astype(int)),
                    )
                )

        return violations

    def _safe_crop(self, frame: np.ndarray, bbox) -> np.ndarray | None:
        h, w = frame.shape[:2]
        x1 = max(0, min(int(bbox[0]), w - 1))
        x2 = max(0, min(int(bbox[2]), w))
        y1 = max(0, min(int(bbox[1]), h - 1))
        y2 = max(0, min(int(bbox[3]), h))
        crop = frame[y1:y2, x1:x2]
        return crop if crop.size > 0 else None

    def _count_heads(self, crop: np.ndarray) -> int:
        if crop.size == 0 or len(crop.shape) < 3:
            return 0
        hsv = crop.copy()
        try:
            import cv2
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        except Exception:
            return 0

        skin_mask = ((hsv[:, :, 0] < 25) | (hsv[:, :, 0] > 165)) & (hsv[:, :, 1] > 40) & (hsv[:, :, 2] > 80)
        kernel = np.ones((5, 5), np.uint8)
        try:
            import cv2
            skin_mask_u8 = skin_mask.astype(np.uint8) * 255
            closed = cv2.morphologyEx(skin_mask_u8, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return len(contours)
        except Exception:
            return 0
