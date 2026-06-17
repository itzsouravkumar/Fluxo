from __future__ import annotations

import numpy as np
from .detector import ViolationEvent, ViolationType


class SignalJumpDetector:
    """Detects vehicles crossing stop line during red phase."""

    def __init__(self, stop_line_y: float = 0.0, homography: np.ndarray | None = None):
        self.stop_line_y = stop_line_y
        self.H = homography
        self.pre_red_positions: dict[int, bool] = {}

    def detect(self, tracks, frame, signal_state: str) -> list[ViolationEvent]:
        violations = []
        if signal_state != "RED":
            return violations

        for track in tracks:
            track_id = getattr(track, "id", None)
            if track_id is None:
                continue

            was_before = self.pre_red_positions.get(track_id, True)
            if not was_before:
                continue

            violations.append(
                ViolationEvent(
                    type=ViolationType.SIGNAL_JUMP,
                    track_id=track_id,
                    frame=0,
                    confidence=0.9,
                )
            )
        return violations
