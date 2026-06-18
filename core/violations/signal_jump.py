from __future__ import annotations

from collections import defaultdict

import numpy as np
from .types import ViolationEvent, ViolationType


class SignalJumpDetector:
    """Detects vehicles crossing stop line during red phase.

    Tracks whether a vehicle was above the stop line before red,
    and crosses below it during red.
    """

    def __init__(self, stop_line_y: float = 0.5):
        self.stop_line_y = stop_line_y
        self._was_above: dict[int, bool] = {}
        self._red_active = False

    def detect(self, detections, frame: np.ndarray, frame_idx: int, signal_state: str) -> list[ViolationEvent]:
        violations = []
        h = frame.shape[0]
        stop_px = int(h * self.stop_line_y)

        is_red = signal_state.upper() == "RED"

        if is_red and not self._red_active:
            self._red_active = True
            self._snapshot_positions(detections, stop_px)
        elif not is_red:
            self._red_active = False
            self._was_above.clear()

        if not is_red:
            return violations

        if not hasattr(detections, "tracker_id") or detections.tracker_id is None:
            return violations

        for i in range(len(detections)):
            tid = int(detections.tracker_id[i]) if detections.tracker_id is not None else -1
            if tid < 0:
                continue

            bbox = detections.xyxy[i]
            cy = (bbox[1] + bbox[3]) / 2.0
            is_above = cy < stop_px

            was_above = self._was_above.get(tid, True)

            if was_above and not is_above:
                conf = float(detections.confidence[i]) if detections.confidence is not None else 0.8
                violations.append(
                    ViolationEvent(
                        type=ViolationType.SIGNAL_JUMP,
                        track_id=tid,
                        frame=frame_idx,
                        confidence=conf,
                        bbox=tuple(bbox.astype(int)),
                    )
                )

            self._was_above[tid] = is_above

        return violations

    def _snapshot_positions(self, detections, stop_px: int):
        if not hasattr(detections, "tracker_id") or detections.tracker_id is None:
            return
        for i in range(len(detections)):
            tid = int(detections.tracker_id[i]) if detections.tracker_id is not None else -1
            if tid < 0:
                continue
            bbox = detections.xyxy[i]
            cy = (bbox[1] + bbox[3]) / 2.0
            self._was_above[tid] = cy < stop_px
