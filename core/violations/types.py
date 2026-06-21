from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
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
    PEDESTRIAN_RED_LIGHT = "pedestrian_red_light"
    MOBILE_PHONE = "mobile_phone"
    NO_SEATBELT = "no_seatbelt"
    OVERLOADING = "overloading"
    PLATE_OBSTRUCTION = "plate_obstruction"
    AMBER_VIOLATION = "amber_violation"
    JUNCTION_BLOCKING = "junction_blocking"


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
    evidence_frame: bytes | None = None
    evidence_hash: str | None = None
    partial_plate: bool = False
    requires_human_review: bool = False
    signal_phase_at_detection: str | None = None
    vehicle_speed_kmh: float | None = None

    def compute_evidence_hash(self) -> str:
        def _convert(obj):
            if hasattr(obj, 'item'):
                return obj.item()
            if isinstance(obj, (list, tuple)):
                return [_convert(x) for x in obj]
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            return obj

        payload = json.dumps(_convert({
            "type": self.type.value,
            "track_id": self.track_id,
            "frame": self.frame,
            "confidence": round(self.confidence, 4),
            "plate_number": self.plate_number,
            "bbox": list(self.bbox),
        }), sort_keys=True)
        self.evidence_hash = hashlib.sha256(payload.encode()).hexdigest()
        return self.evidence_hash


@dataclass
class TrackState:
    track_id: int
    first_seen_frame: int = 0
    consecutive_detections: int = 0
    last_seen_frame: int = 0
    plate_history: list[str] = field(default_factory=list)
    is_id_unstable: bool = False
    class_id: int = -1


@dataclass
class ViolationConfig:
    enable_signal_jump: bool = True
    enable_helmet: bool = True
    enable_wrong_way: bool = True
    enable_triple_riding: bool = True
    enable_fancy_plate: bool = True
    enable_missing_mirror: bool = True
    enable_mobile_phone: bool = True
    enable_seatbelt: bool = False
    enable_overloading: bool = True
    enable_junction_blocking: bool = True
    enable_anpr: bool = True
    enable_clip_extract: bool = False
    enable_pedestrian_red_light: bool = True
    enable_plate_obstruction: bool = False
    stop_line_y: float = 0.5
    overspeed_limit_kmh: float = 60.0
    clip_pre_seconds: int = 2
    clip_post_seconds: int = 3
    track_min_frames_helmet: int = 3
    track_min_frames_wrong_way: int = 5
    track_min_frames_signal_jump: int = 1
    track_min_frames_yellow: int = 3
    frame_completeness_threshold: float = 0.7
    evidence_hash_enabled: bool = True
    partial_plate_confidence_threshold: float = 0.6
    max_track_age_frames: int = 90
    ocr_consistency_window: int = 10
    camera_motion_compensation: bool = True
    min_violation_confidence: float = 0.5
    enable_fog_preprocessing: bool = True
    enable_plate_color_aware: bool = True
    enable_per_char_confidence: bool = True
    enable_adversarial_defense: bool = True
    enable_camera_health_monitor: bool = True
    enable_cross_camera_reid: bool = True
    enable_federated_learning: bool = True
    enable_ntp_validation: bool = True
