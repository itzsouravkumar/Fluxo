"""FLUXO Vision Pipeline Configuration.

All tunable parameters in one place. No hardcoded values in scripts.
"""

from dataclasses import dataclass, field


@dataclass
class DetectionConfig:
    model_path: str = "yolo11n.pt"
    confidence: float = 0.4
    iou_threshold: float = 0.5
    device: str = "auto"


@dataclass
class VehicleClass:
    id: int
    name: str
    pce: float
    color: tuple = (255, 255, 255)


VEHICLE_CLASSES = {
    1: VehicleClass(1, "bicycle", 0.25, (0, 255, 255)),
    2: VehicleClass(2, "car", 1.0, (0, 255, 0)),
    3: VehicleClass(3, "motorcycle", 0.25, (255, 165, 0)),
    5: VehicleClass(5, "bus", 3.0, (255, 0, 0)),
    7: VehicleClass(7, "truck", 3.5, (0, 0, 255)),
}

INDIAN_VEHICLE_CLASSES = {
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
class VisionConfig:
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    lane: LaneConfig = field(default_factory=LaneConfig)
    density: DensityConfig = field(default_factory=DensityConfig)
    speed: SpeedConfig = field(default_factory=SpeedConfig)
    preprocessor: PreprocessorConfig = field(default_factory=PreprocessorConfig)
    density_levels: DensityLevels = field(default_factory=DensityLevels)


DEFAULT_CONFIG = VisionConfig()
