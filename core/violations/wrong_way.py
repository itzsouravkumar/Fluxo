from __future__ import annotations

import numpy as np
from .detector import ViolationEvent, ViolationType


class WrongWayDetector:
    """Detects vehicles moving against expected lane direction."""

    def __init__(self, expected_directions: dict[int, float] | None = None):
        self.expected_directions = expected_directions or {}

    def detect(self, tracks, frame: np.ndarray, signal_state: str) -> list[ViolationEvent]:
        violations = []
        for track in tracks:
            positions = getattr(track, "positions", [])
            if len(positions) < 5:
                continue

            dx = positions[-1][0] - positions[-5][0]
            dy = positions[-1][1] - positions[-5][1]
            velocity_angle = np.arctan2(dy, dx)

            track_id = getattr(track, "id", -1)
            expected = self.expected_directions.get(track_id, 0.0)
            cosine_sim = np.cos(velocity_angle - expected)

            if cosine_sim < -0.5:
                violations.append(
                    ViolationEvent(
                        type=ViolationType.WRONG_WAY,
                        track_id=track_id,
                        frame=0,
                        confidence=abs(cosine_sim),
                    )
                )
        return violations
