from __future__ import annotations

import cv2
import numpy as np

from .types import ViolationEvent, ViolationType


class MirrorDetector:
    """Detects missing rear-view mirrors on two-wheelers.

    A 2025 IEEE paper (arXiv:2511.12206) introduced detection of missing
    rear-view mirrors as a novel violation class, achieving mAP@50 = 0.843.
    This violation is legally enforceable under the Central Motor Vehicles
    Rules and completely absent from every other competitor's system.

    Heuristic approach: checks for expected mirror protrusions on the
    left and right sides of two-wheeler handlebars. If a mirror region
    is empty (no significant edge/color feature), flags the violation.
    """

    def __init__(self):
        pass

    def detect(self, detections, frame: np.ndarray, frame_idx: int, signal_state: str) -> list[ViolationEvent]:
        violations = []
        h, w = frame.shape[:2]

        if not hasattr(detections, "class_id") or detections.class_id is None:
            return violations

        for i in range(len(detections)):
            cls_id = int(detections.class_id[i])
            if cls_id != 0:
                continue

            bbox = detections.xyxy[i].astype(int)
            x1, y1, x2, y2 = bbox
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h))
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            has_mirror = self._check_mirrors(crop)
            if not has_mirror:
                conf = float(detections.confidence[i]) if detections.confidence is not None else 0.6
                tid = int(detections.tracker_id[i]) if hasattr(detections, "tracker_id") and detections.tracker_id is not None else -1
                violations.append(
                    ViolationEvent(
                        type=ViolationType.MISSING_MIRROR,
                        track_id=tid,
                        frame=frame_idx,
                        confidence=conf * 0.8,
                        bbox=tuple(bbox),
                    )
                )

        return violations

    def _check_mirrors(self, vehicle_crop: np.ndarray) -> bool:
        h, w = vehicle_crop.shape[:2]
        if h < 20 or w < 20:
            return True

        handlebar_y = int(h * 0.3)
        handlebar_region = vehicle_crop[max(0, handlebar_y - 10):min(h, handlebar_y + 10), :]
        if handlebar_region.size == 0:
            return True

        left_region = handlebar_region[:, :int(w * 0.25)]
        right_region = handlebar_region[:, int(w * 0.75):]

        left_mirror = self._has_mirror_feature(left_region)
        right_mirror = self._has_mirror_feature(right_region)

        return left_mirror or right_mirror

    def _has_mirror_feature(self, region: np.ndarray) -> bool:
        if region.size == 0:
            return True

        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / max(edges.size, 1)

        return edge_density > 0.08
