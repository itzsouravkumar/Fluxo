from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ViolationType(str, Enum):
    NO_HELMET = "no_helmet"
    SIGNAL_JUMP = "signal_jump"
    WRONG_WAY = "wrong_way"
    TRIPLE_RIDING = "triple_riding"
    OVERSPEEDING = "overspeeding"
    FANCY_PLATE = "fancy_plate"
    MISSING_MIRROR = "missing_mirror"
    PEDESTRIAN_BLOCK = "pedestrian_block"


@dataclass
class ViolationEvent:
    type: ViolationType
    track_id: int
    frame: int
    confidence: float
    plate_number: str | None = None
    clip_path: str | None = None
    bbox: tuple = (0, 0, 0, 0)
    seat_positions: dict | None = None
    evidence_narration: str | None = None


@dataclass
class ViolationConfig:
    enable_signal_jump: bool = True
    enable_helmet: bool = True
    enable_wrong_way: bool = True
    enable_triple_riding: bool = True
    enable_fancy_plate: bool = True
    enable_missing_mirror: bool = True
    enable_anpr: bool = True
    enable_clip_extract: bool = True
    enable_vlm_narration: bool = False
    stop_line_y: float = 0.5
    overspeed_limit_kmh: float = 60.0
    clip_pre_seconds: int = 2
    clip_post_seconds: int = 3
