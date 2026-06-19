from __future__ import annotations

import numpy as np
from .types import ViolationEvent, ViolationType


class PedestrianRedLightDetector:
    """Detects pedestrians crossing against the signal.

    Extends the same homography stop-line infrastructure used for
    vehicle signal violations to pedestrian detection. Pedestrians
    are class_id=5 in the YOLO26 model.

    Reference: Docs Section 7.3 — pedestrian violations are a
    significant cause of fatalities in dense Indian urban environments
    and entirely unaddressed by existing automated systems.
    """

    PEDESTRIAN_CLASS = 5

    def __init__(self, stop_line_y: float = 0.5):
        self.stop_line_y = stop_line_y
        self._pedestrian_positions: dict[int, bool] = {}

    def detect(self, detections, frame: np.ndarray, frame_idx: int, signal_state: str) -> list[ViolationEvent]:
        violations = []
        is_red = signal_state.upper() == "RED"
        if not is_red:
            self._pedestrian_positions.clear()
            return violations

        h = frame.shape[0]
        stop_px = int(h * self.stop_line_y)

        if not hasattr(detections, "tracker_id") or detections.tracker_id is None:
            return violations
        if not hasattr(detections, "class_id") or detections.class_id is None:
            return violations

        for i in range(len(detections)):
            cls_id = int(detections.class_id[i])
            if cls_id != self.PEDESTRIAN_CLASS:
                continue

            tid = int(detections.tracker_id[i]) if detections.tracker_id is not None else -1
            if tid < 0:
                continue

            bbox = detections.xyxy[i]
            cy = (bbox[1] + bbox[3]) / 2.0
            is_past_stop = cy > stop_px

            was_before = self._pedestrian_positions.get(tid, False)

            if not was_before and is_past_stop:
                conf = float(detections.confidence[i]) if detections.confidence is not None else 0.7
                violations.append(ViolationEvent(
                    type=ViolationType.PEDESTRIAN_RED_LIGHT,
                    track_id=tid,
                    frame=frame_idx,
                    confidence=conf,
                    bbox=tuple(bbox.astype(int)),
                    signal_phase_at_detection="RED",
                ))

            self._pedestrian_positions[tid] = is_past_stop

        return violations
