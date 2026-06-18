"""Tests for FLUXO violation detection module."""

import numpy as np
import pytest


class MockDetections:
    """Minimal mock for sv.Detections."""

    def __init__(self, xyxy, class_id, tracker_id=None, confidence=None):
        self.xyxy = np.array(xyxy, dtype=np.float32)
        self.class_id = np.array(class_id, dtype=int)
        self.tracker_id = np.array(tracker_id, dtype=int) if tracker_id is not None else None
        self.confidence = np.array(confidence, dtype=float) if confidence is not None else None

    def __len__(self):
        return len(self.xyxy)


def make_frame(h=480, w=640):
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def test_violation_types():
    from core.violations.types import ViolationType
    assert ViolationType.NO_HELMET.value == "no_helmet"
    assert ViolationType.SIGNAL_JUMP.value == "signal_jump"
    assert ViolationType.WRONG_WAY.value == "wrong_way"
    assert ViolationType.TRIPLE_RIDING.value == "triple_riding"


def test_violation_event_defaults():
    from core.violations.types import ViolationEvent, ViolationType
    v = ViolationEvent(type=ViolationType.SIGNAL_JUMP, track_id=1, frame=0, confidence=0.9)
    assert v.plate_number is None
    assert v.clip_path is None
    assert v.bbox == (0, 0, 0, 0)


def test_violation_config_defaults():
    from core.violations.types import ViolationConfig
    config = ViolationConfig()
    assert config.enable_signal_jump is True
    assert config.enable_helmet is True
    assert config.enable_wrong_way is True
    assert config.enable_triple_riding is True
    assert config.enable_anpr is True
    assert config.stop_line_y == 0.5


def test_signal_jump_no_red():
    from core.violations.signal_jump import SignalJumpDetector
    det = SignalJumpDetector(stop_line_y=0.5)
    frame = make_frame()
    dets = MockDetections([[100, 200, 200, 300]], [2], [1], [0.9])
    violations = det.detect(dets, frame, 0, "GREEN")
    assert len(violations) == 0


def test_signal_jump_red_crosses_stop_line():
    from core.violations.signal_jump import SignalJumpDetector
    det = SignalJumpDetector(stop_line_y=0.5)
    frame = make_frame()
    dets_above = MockDetections([[100, 100, 200, 200]], [2], [1], [0.9])
    det.detect(dets_above, frame, 0, "RED")
    dets_below = MockDetections([[100, 300, 200, 400]], [2], [1], [0.9])
    violations = det.detect(dets_below, frame, 1, "RED")
    assert len(violations) == 1
    assert violations[0].track_id == 1
    assert violations[0].confidence > 0


def test_signal_jump_stays_above():
    from core.violations.signal_jump import SignalJumpDetector
    det = SignalJumpDetector(stop_line_y=0.5)
    frame = make_frame()
    dets = MockDetections([[100, 100, 200, 200]], [2], [1], [0.9])
    det.detect(dets, frame, 0, "RED")
    violations = det.detect(dets, frame, 1, "RED")
    assert len(violations) == 0


def test_wrong_way_no_violation():
    from core.violations.wrong_way import WrongWayDetector
    det = WrongWayDetector()
    frame = make_frame()
    dets = MockDetections([[300, 400, 400, 500]], [2], [1], [0.9])
    violations = det.detect(dets, frame, 0, "GREEN")
    assert len(violations) == 0


def test_triple_riding_tall_box():
    from core.violations.triple_riding import TripleRidingDetector
    det = TripleRidingDetector(aspect_ratio_threshold=2.0)
    frame = make_frame()
    dets = MockDetections([[300, 100, 350, 400]], [3], [1], [0.9])
    violations = det.detect(dets, frame, 0, "GREEN")
    assert len(violations) == 1
    assert violations[0].track_id == 1


def test_triple_riding_normal_box():
    from core.violations.triple_riding import TripleRidingDetector
    det = TripleRidingDetector(aspect_ratio_threshold=2.5)
    frame = make_frame()
    dets = MockDetections([[300, 300, 400, 350]], [3], [1], [0.9])
    violations = det.detect(dets, frame, 0, "GREEN")
    assert len(violations) == 0


def test_triple_riding_ignores_non_motorcycle():
    from core.violations.triple_riding import TripleRidingDetector
    det = TripleRidingDetector(aspect_ratio_threshold=2.0)
    frame = make_frame()
    dets = MockDetections([[300, 100, 350, 400]], [2], [1], [0.9])
    violations = det.detect(dets, frame, 0, "GREEN")
    assert len(violations) == 0


def test_anpr_reader_init():
    from core.violations.anpr import ANPRReader
    reader = ANPRReader()
    assert reader._reader is None


def test_clip_extractor_init():
    from core.violations.clip_extractor import ClipExtractor
    ext = ClipExtractor(fps=30, buffer_size=300)
    assert ext.fps == 30
    assert ext.buffer_size == 300


def test_violation_detector_init():
    from core.violations.detector import ViolationDetector
    from core.violations.types import ViolationConfig
    config = ViolationConfig(enable_anpr=False, enable_clip_extract=False)
    vd = ViolationDetector(config)
    assert len(vd._detectors) == 4


def test_violation_detector_check_empty():
    from core.violations.detector import ViolationDetector
    from core.violations.types import ViolationConfig
    config = ViolationConfig(enable_anpr=False, enable_clip_extract=False)
    vd = ViolationDetector(config)
    frame = make_frame()
    dets = MockDetections([[100, 200, 200, 300]], [2], [1], [0.9])
    violations = vd.check(dets, frame, 0, "GREEN")
    assert isinstance(violations, list)


def test_helmet_detector_no_model():
    from core.violations.helmet import HelmetDetector
    det = HelmetDetector()
    frame = make_frame()
    dets = MockDetections([[100, 100, 200, 200]], [3], [1], [0.9])
    violations = det.detect(dets, frame, 0, "GREEN")
    assert isinstance(violations, list)
