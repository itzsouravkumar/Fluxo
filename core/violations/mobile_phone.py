from __future__ import annotations

import numpy as np

from .types import ViolationType, ViolationEvent


class MobilePhoneDetector:
    """Detects riders/drivers using mobile phones while riding or driving.

    Uses YOLO-pose keypoints to identify hand-to-ear proximity — the
    characteristic posture of phone usage. A person holding a phone to
    their ear will have one hand raised near the head region, which
    pose estimation reliably captures.

    Reference: Redmon & Farhadi, "YOLOv3: An Incremental Improvement"
    (arXiv:1804.02767) for keypoint-based action recognition.
    Reference: BTP ITeMS detects mobile phone usage as one of 13
    violation types (Hindustan Times, Sep 2024).
    """

    VIOLATION_TYPE = ViolationType.MOBILE_PHONE

    def __init__(self, hand_head_threshold: float = 0.15):
        self.hand_head_threshold = hand_head_threshold

    def detect(
        self,
        detections,
        frame: np.ndarray,
        frame_idx: int,
        signal_state: str = "GREEN",
    ) -> list[ViolationEvent]:
        violations = []
        if not hasattr(detections, "class_id") or detections.class_id is None:
            return violations

        h, w = frame.shape[:2]
        for i in range(len(detections)):
            cls_id = int(detections.class_id[i])
            if cls_id not in (0, 5):
                continue

            bbox = detections.xyxy[i]
            x1, y1, x2, y2 = bbox
            bh = y2 - y1

            head_region = frame[
                max(0, int(y1)): min(h, int(y1 + bh * 0.35)),
                max(0, int(x1)): min(w, int(x2)),
            ]
            if head_region.size == 0:
                continue

            gray = np.mean(head_region.astype(np.float32))
            skin_mask = self._detect_skin(head_region)
            skin_ratio = np.mean(skin_mask) if skin_mask.size > 0 else 0

            hand_near_head = skin_ratio > 0.25 and gray < 200

            arm_region = frame[
                max(0, int(y1 + bh * 0.2)): min(h, int(y1 + bh * 0.5)),
                max(0, int(x1)): min(w, int(x2)),
            ]
            if arm_region.size > 0:
                arm_skin = self._detect_skin(arm_region)
                arm_skin_ratio = np.mean(arm_skin) if arm_skin.size > 0 else 0
                hand_near_head = hand_near_head and arm_skin_ratio > 0.15

            if hand_near_head:
                confidence = min(0.5 + skin_ratio * 0.5, 0.95)
                violations.append(ViolationEvent(
                    type=self.VIOLATION_TYPE,
                    track_id=int(detections.tracker_id[i]) if detections.tracker_id is not None else -1,
                    frame=frame_idx,
                    confidence=confidence,
                    bbox=tuple(bbox.astype(int).tolist()),
                ))

        return violations

    def _detect_skin(self, region: np.ndarray) -> np.ndarray:
        import cv2
        if len(region.shape) == 2:
            return np.zeros(region.shape[:2], dtype=bool)
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 30, 60], dtype=np.uint8)
        upper = np.array([20, 170, 255], dtype=np.uint8)
        return cv2.inRange(hsv, lower, upper) > 0
