from __future__ import annotations

import numpy as np

from .types import ViolationType, ViolationEvent


class SeatbeltDetector:
    """Detects front-seat occupants not wearing seat belts.

    Examines the diagonal chest region of detected LMV occupants.
    A seat belt creates a visible diagonal stripe across the torso —
    the detector looks for this high-contrast diagonal pattern.
    Missing seat belt means the chest region lacks this characteristic
    diagonal edge.

    Reference: BTP ITeMS data shows seat belt violations are 16% of
    all automated detections (Times of India, Oct 2025).
    Reference: Karnataka-Mysuru highway cameras specifically target
    seat belt non-compliance (Indian Express, Dec 2024).
    """

    VIOLATION_TYPE = ViolationType.NO_SEATBELT

    def __init__(self):
        pass

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
            if cls_id != 2:
                continue

            bbox = detections.xyxy[i]
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            bw, bh = x2 - x1, y2 - y1
            if bw < 30 or bh < 50:
                continue

            chest_y1 = max(0, y1 + int(bh * 0.35))
            chest_y2 = min(h, y1 + int(bh * 0.70))
            chest_x1 = max(0, x1 + int(bw * 0.15))
            chest_x2 = min(w, x2 - int(bw * 0.15))

            chest = frame[chest_y1:chest_y2, chest_x1:chest_x2]
            if chest.size == 0:
                continue

            has_seatbelt = self._check_diagonal_stripe(chest)

            if not has_seatbelt:
                violations.append(ViolationEvent(
                    type=self.VIOLATION_TYPE,
                    track_id=int(detections.tracker_id[i]) if detections.tracker_id is not None else -1,
                    frame=frame_idx,
                    confidence=0.65,
                    bbox=tuple(bbox.astype(int).tolist()),
                ))

        return violations

    def _check_diagonal_stripe(self, chest_region: np.ndarray) -> bool:
        import cv2
        gray = cv2.cvtColor(chest_region, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        h, w = edges.shape
        diag_len = int(min(h, w) * 0.7)
        if diag_len < 10:
            return False

        scores = []
        for offset in range(-w // 4, w // 4, max(w // 20, 1)):
            pts = []
            for k in range(diag_len):
                y = int(k * h / diag_len)
                x = int(w // 2 + offset + k * w / diag_len * 0.5)
                if 0 <= y < h and 0 <= x < w:
                    pts.append(edges[y, x])
            if pts:
                scores.append(np.mean(pts))

        return max(scores) > 20 if scores else False
