"""Tests for FLUXO vision module."""

import numpy as np


def test_config_defaults():
    from core.vision.config import DEFAULT_CONFIG, VisionConfig
    config = DEFAULT_CONFIG
    assert isinstance(config, VisionConfig)
    assert config.detection.confidence == 0.4
    assert config.detection.model_path == "yolo26n.pt"


def test_vehicle_classes():
    from core.vision.config import VEHICLE_CLASSES
    assert 0 in VEHICLE_CLASSES
    assert VEHICLE_CLASSES[0].name == "two_wheeler"
    assert VEHICLE_CLASSES[0].pce == 0.25
    assert 2 in VEHICLE_CLASSES
    assert VEHICLE_CLASSES[2].name == "light_motor_vehicle"
    assert VEHICLE_CLASSES[2].pce == 1.0


def test_density_levels():
    from core.vision.config import DensityLevels
    levels = DensityLevels()
    assert levels.get_level(0.0) == "CLEAR"
    assert levels.get_level(0.3) == "MODERATE"
    assert levels.get_level(0.5) == "HIGH"
    assert levels.get_level(0.8) == "CRITICAL"


def test_compute_lane_density_empty():
    from scripts.demo_vision import compute_lane_density
    from core.vision.config import DEFAULT_CONFIG
    import supervision as sv
    empty = sv.Detections(xyxy=np.empty((0, 4)), class_id=np.array([]))
    lanes, density, pce, count = compute_lane_density(empty, 640, 480, DEFAULT_CONFIG)
    assert count == 0
    assert pce == 0.0
    assert density == 0.0
    assert all(lane["count"] == 0 for lane in lanes.values())


def test_compute_lane_density_with_vehicles():
    from scripts.demo_vision import compute_lane_density
    from core.vision.config import DEFAULT_CONFIG
    import supervision as sv
    bboxes = np.array([[100, 100, 200, 200], [500, 400, 600, 500]])
    class_ids = np.array([2, 3])
    dets = sv.Detections(xyxy=bboxes, class_id=class_ids)
    lanes, density, pce, count = compute_lane_density(dets, 640, 480, DEFAULT_CONFIG)
    assert count == 2
    assert pce == 4.0


def test_estimate_speed_short_track():
    from scripts.demo_vision import estimate_speed
    track = [(100, 100), (110, 110)]
    speed = estimate_speed(track)
    assert speed == 0.0


def test_estimate_speed_with_movement():
    from scripts.demo_vision import estimate_speed
    track = [(0, 0), (10, 0), (20, 0), (30, 0), (40, 0)]
    speed = estimate_speed(track, fps=30.0)
    assert speed > 0


def test_clahe_preprocessing():
    from scripts.demo_vision import apply_clahe
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = apply_clahe(frame)
    assert result.shape == frame.shape
    assert result.dtype == frame.dtype


def test_validate_source_missing():
    from scripts.demo_vision import validate_source
    result = validate_source("nonexistent_video.mp4")
    assert result is None


def test_validate_source_valid():
    from scripts.demo_vision import validate_source
    result = validate_source("0")
    if result is not None:
        result.release()


def test_profiler():
    from scripts.demo_vision import Profiler
    p = Profiler()
    p.start()
    p.lap("detection")
    p.start()
    p.lap("tracking")
    summary = p.summary()
    assert "detection" in summary
    assert "tracking" in summary


def test_detector_init():
    from core.vision.detector import FluxoDetector
    det = FluxoDetector(model_path="yolo26n.pt")
    assert det.model_path == "yolo26n.pt"
    assert det.conf == 0.4


def test_tracker_init():
    from core.vision.tracker import FluxoTracker
    tracker = FluxoTracker()
    assert tracker._tracker is None
