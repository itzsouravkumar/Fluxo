from __future__ import annotations

import numpy as np


class SafetyConstraints:
    """Safety guardrails for RL signal control."""

    MIN_GREEN = 15
    MAX_GREEN = 120
    CLEARANCE_BUFFER = 2.0

    def apply(self, recommended_duration: float, lane_states: list[dict] | None = None) -> float:
        duration = np.clip(recommended_duration, self.MIN_GREEN, self.MAX_GREEN)

        if lane_states:
            starved = [l for l in lane_states if l.get("wait_time", 0) > 90]
            if starved:
                return float(self.MIN_GREEN)

        return float(duration)

    def compute_clearance_interval(self, max_vehicle_length: float = 6.0, avg_speed_ms: float = 5.0) -> float:
        return (max_vehicle_length / max(avg_speed_ms, 0.1)) + self.CLEARANCE_BUFFER
