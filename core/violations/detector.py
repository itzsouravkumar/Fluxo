from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ViolationType(str, Enum):
    NO_HELMET = "no_helmet"
    SIGNAL_JUMP = "signal_jump"
    WRONG_WAY = "wrong_way"
    TRIPLE_RIDING = "triple_riding"
    PEDESTRIAN_BLOCK = "pedestrian_block"
    OVERSPEEDING = "overspeeding"


@dataclass
class ViolationEvent:
    type: ViolationType
    track_id: int
    frame: int
    confidence: float
    plate_number: str | None = None
    clip_path: str | None = None


class ViolationDetector:
    """Orchestrates all violation detection sub-modules."""

    def __init__(self):
        self._detectors = []

    def check(self, tracks, frame, signal_state: str) -> list[ViolationEvent]:
        violations = []
        for detector in self._detectors:
            violations.extend(detector.detect(tracks, frame, signal_state))
        return violations
