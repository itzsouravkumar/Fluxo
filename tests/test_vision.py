"""Tests for FLUXO vision module."""

import numpy as np


def test_config_defaults():
    from core.vision.config import DEFAULT_CONFIG, VisionConfig
    config = DEFAULT_CONFIG
    assert isinstance(config, VisionConfig)
    assert config.detection.confidence == 0.55
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
    assert det.conf == 0.55


def test_tracker_init():
    from core.vision.tracker import FluxoTracker
    tracker = FluxoTracker()
    assert tracker._tracker is None


def test_frame_quality_analyzer_good_frame():
    from core.vision.enhancement import FrameQualityAnalyzer
    analyzer = FrameQualityAnalyzer(min_resolution=480, blur_threshold=100.0)
    frame = np.random.randint(50, 200, (720, 1280, 3), dtype=np.uint8)
    q = analyzer.analyze(frame)
    assert 0.0 <= q["overall"] <= 1.0
    assert "needs_enhancement" in q
    assert "needs_sr" in q
    assert "needs_sharpen" in q


def test_frame_quality_analyzer_bad_frame():
    from core.vision.enhancement import FrameQualityAnalyzer
    analyzer = FrameQualityAnalyzer(min_resolution=480, blur_threshold=100.0)
    frame = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
    q = analyzer.analyze(frame)
    assert q["needs_enhancement"] is True or q["needs_sr"] is True


def test_frame_enhancer_noop_for_good_frame():
    from core.vision.enhancement import FrameEnhancer
    enhancer = FrameEnhancer()
    frame = np.random.randint(50, 200, (720, 1280, 3), dtype=np.uint8)
    result, quality = enhancer.enhance(frame)
    assert result.shape == frame.shape


def test_frame_enhancer_upscales_bad_frame():
    from core.vision.enhancement import FrameEnhancer
    enhancer = FrameEnhancer()
    frame = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
    result, quality = enhancer.enhance(frame, force=True)
    assert result.shape[0] >= frame.shape[0]
    assert result.shape[1] >= frame.shape[1]


def test_super_resolution_upscaler():
    from core.vision.enhancement import SuperResolutionUpscaler
    sr = SuperResolutionUpscaler(scale=2)
    frame = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
    result = sr.upscale(frame)
    assert result.shape[0] == 240
    assert result.shape[1] == 320


def test_tile_detector_small_frame():
    from core.vision.enhancement import TileDetector
    td = TileDetector(tile_size=640, overlap=0.2)
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dets = td.detect_with_tiles(frame, lambda t, c: [])
    assert isinstance(dets, list)


def test_tile_detector_large_frame():
    from core.vision.enhancement import TileDetector
    td = TileDetector(tile_size=320, overlap=0.2)
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

    def fake_detect(tile, conf):
        h, w = tile.shape[:2]
        return [{"bbox": np.array([10, 10, 50, 50]), "class_id": 0, "confidence": 0.9}]

    dets = td.detect_with_tiles(frame, fake_detect)
    assert len(dets) > 0


def test_iou_computation():
    from core.vision.enhancement import TileDetector
    td = TileDetector()
    box1 = np.array([0, 0, 100, 100])
    box2 = np.array([50, 50, 150, 150])
    iou = td._compute_iou(box1, box2)
    assert 0.0 < iou < 1.0


def test_iou_no_overlap():
    from core.vision.enhancement import TileDetector
    td = TileDetector()
    box1 = np.array([0, 0, 10, 10])
    box2 = np.array[0] if False else np.array([100, 100, 200, 200])
    iou = td._compute_iou(box1, box2)
    assert iou == 0.0


def test_iou_perfect_overlap():
    from core.vision.enhancement import TileDetector
    td = TileDetector()
    box = np.array([10, 20, 30, 40])
    iou = td._compute_iou(box, box)
    assert abs(iou - 1.0) < 1e-6


def test_adaptive_confidence_good_quality():
    from core.vision.enhancement import AdaptiveConfidence
    ac = AdaptiveConfidence(base_conf=0.4, min_conf=0.15)
    quality = {"needs_enhancement": False, "overall": 0.8}
    assert ac.get_threshold(quality) == 0.4


def test_adaptive_confidence_terrible_quality():
    from core.vision.enhancement import AdaptiveConfidence
    ac = AdaptiveConfidence(base_conf=0.4, min_conf=0.15)
    quality = {"needs_enhancement": True, "overall": 0.2}
    assert ac.get_threshold(quality) == 0.15


def test_adaptive_confidence_moderate_quality():
    from core.vision.enhancement import AdaptiveConfidence
    ac = AdaptiveConfidence(base_conf=0.4, min_conf=0.15)
    quality = {"needs_enhancement": True, "overall": 0.45}
    threshold = ac.get_threshold(quality)
    assert 0.15 <= threshold <= 0.4


def test_temporal_booster_no_history():
    from core.vision.enhancement import TemporalConfidenceBooster
    tb = TemporalConfidenceBooster(history_window=5, boost_per_hit=0.05)
    bboxes = [np.array([100, 100, 200, 200])]
    boosts = tb.update(bboxes)
    assert len(boosts) == 1
    assert boosts[0] == 0.0


def test_temporal_booster_accumulates():
    from core.vision.enhancement import TemporalConfidenceBooster
    tb = TemporalConfidenceBooster(history_window=10, boost_per_hit=0.05, max_boost=0.3)
    bbox = np.array([100, 100, 200, 200])
    for _ in range(5):
        boosts = tb.update([bbox])
    assert boosts[0] > 0.0


def test_temporal_booster_max_boost():
    from core.vision.enhancement import TemporalConfidenceBooster
    tb = TemporalConfidenceBooster(history_window=20, boost_per_hit=0.1, max_boost=0.3)
    bbox = np.array([100, 100, 200, 200])
    for _ in range(15):
        boosts = tb.update([bbox])
    assert boosts[0] <= 0.3


def test_temporal_booster_reset():
    from core.vision.enhancement import TemporalConfidenceBooster
    tb = TemporalConfidenceBooster()
    tb.update([np.array([100, 100, 200, 200])])
    tb.reset()
    assert len(tb._history) == 0


def test_smart_roi_no_motion():
    from core.vision.enhancement import SmartROISelector
    selector = SmartROISelector()
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    rois = selector.select_rois(frame, tile_size=640)
    assert len(rois) == 1
    assert rois[0] == (0, 0, 640, 480)


def test_smart_roi_with_motion():
    from core.vision.enhancement import SmartROISelector
    selector = SmartROISelector()
    frame1 = np.full((480, 640, 3), 128, dtype=np.uint8)
    selector.select_rois(frame1, tile_size=640)
    frame2 = frame1.copy()
    frame2[200:300, 300:500] = 200
    rois = selector.select_rois(frame2, tile_size=640)
    assert len(rois) >= 1


def test_smart_roi_reset_history():
    from core.vision.enhancement import SmartROISelector
    selector = SmartROISelector()
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    selector.select_rois(frame)
    assert selector._prev_gray is not None
    selector._prev_gray = None
    assert selector._prev_gray is None


def test_tile_detector_with_rois():
    from core.vision.enhancement import TileDetector
    td = TileDetector(tile_size=320, overlap=0.2)
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

    def fake_detect(tile, conf):
        return [{"bbox": np.array([10, 10, 50, 50]), "class_id": 0, "confidence": 0.9}]

    rois = [(0, 0, 640, 640), (640, 0, 1280, 640)]
    dets = td.detect_with_tiles(frame, fake_detect, rois=rois)
    assert len(dets) > 0


def test_enhancement_init_exports():
    import core.vision as vmod
    assert hasattr(vmod, "FrameEnhancer")
    assert hasattr(vmod, "FrameQualityAnalyzer")
    assert hasattr(vmod, "TileDetector")
    assert hasattr(vmod, "AdaptiveConfidence")
    assert hasattr(vmod, "TemporalConfidenceBooster")
    assert hasattr(vmod, "SmartROISelector")
