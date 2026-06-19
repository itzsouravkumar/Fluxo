"""FLUXO Vision Pipeline Configuration.

All tunable parameters in one place. No hardcoded values in scripts.
"""

from dataclasses import dataclass, field


@dataclass
class DetectionConfig:
    model_path: str = "yolo26n.pt"
    confidence: float = 0.55
    iou_threshold: float = 0.5
    device: str = "auto"


@dataclass
class VehicleClass:
    id: int
    name: str
    pce: float
    color: tuple = (255, 255, 255)


VEHICLE_CLASSES = {
    0: VehicleClass(0, "two_wheeler", 0.25, (255, 165, 0)),
    1: VehicleClass(1, "auto_rickshaw", 0.5, (0, 255, 255)),
    2: VehicleClass(2, "light_motor_vehicle", 1.0, (0, 255, 0)),
    3: VehicleClass(3, "bus", 3.0, (255, 0, 0)),
    4: VehicleClass(4, "heavy_vehicle", 3.5, (0, 0, 255)),
    5: VehicleClass(5, "pedestrian", 0.0, (255, 255, 0)),
    6: VehicleClass(6, "emergency_vehicle", 0.0, (255, 0, 255)),
}


@dataclass
class LaneConfig:
    names: list = field(default_factory=lambda: ["north", "south", "east", "west"])
    roi_ratio: float = 0.25


@dataclass
class DensityConfig:
    max_density: float = 10.0
    pixel_to_meter: float = 0.05
    roi_area_divisor: float = 1000.0


@dataclass
class SpeedConfig:
    min_positions: int = 5
    pixel_to_meter: float = 0.05


@dataclass
class PreprocessorConfig:
    clahe_clip_limit: float = 2.0
    clahe_grid_size: tuple = (8, 8)
    target_width: int = 640
    target_height: int = 640
    enable_weather: bool = True


@dataclass
class DensityLevels:
    clear: float = 0.2
    moderate: float = 0.4
    high: float = 0.7

    def get_level(self, density: float) -> str:
        if density > self.high:
            return "CRITICAL"
        elif density > self.moderate:
            return "HIGH"
        elif density > self.clear:
            return "MODERATE"
        return "CLEAR"


@dataclass
class TrackerConfig:
    use_bot_sort: bool = True
    track_buffer: int = 90
    track_thresh: float = 0.25
    match_thresh: float = 0.8
    min_frames_helmet: int = 3
    min_frames_wrong_way: int = 5
    min_frames_signal_jump: int = 1
    min_frames_yellow: int = 3
    ocr_consistency_window: int = 10


@dataclass
class EvidenceConfig:
    hash_enabled: bool = True
    jpeg_quality: int = 85
    clip_pre_seconds: int = 2
    clip_post_seconds: int = 3


@dataclass
class CameraHealthConfig:
    freeze_timeout_sec: float = 5.0
    blur_degradation_threshold: float = 0.3
    scene_change_threshold: float = 0.6


@dataclass
class FederatedConfig:
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    gradient_buffer_dir: str = "outputs/gradient_buffer"
    max_buffer_size: int = 100


@dataclass
class ReIDConfig:
    similarity_threshold: float = 0.85
    plate_match_required: bool = True
    gallery_ttl_sec: float = 3600.0


@dataclass
class AdversarialConfig:
    anomaly_threshold: float = 0.35
    max_missed_frames: int = 3


@dataclass
class VisionConfig:
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    lane: LaneConfig = field(default_factory=LaneConfig)
    density: DensityConfig = field(default_factory=DensityConfig)
    speed: SpeedConfig = field(default_factory=SpeedConfig)
    preprocessor: PreprocessorConfig = field(default_factory=PreprocessorConfig)
    density_levels: DensityLevels = field(default_factory=DensityLevels)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    camera_health: CameraHealthConfig = field(default_factory=CameraHealthConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    reid: ReIDConfig = field(default_factory=ReIDConfig)
    adversarial: AdversarialConfig = field(default_factory=AdversarialConfig)


DEFAULT_CONFIG = VisionConfig()
