"""Tests for FLUXO violation detection module."""

import numpy as np


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
    assert ViolationType.FANCY_PLATE.value == "fancy_plate"
    assert ViolationType.MISSING_MIRROR.value == "missing_mirror"


def test_violation_event_defaults():
    from core.violations.types import ViolationEvent, ViolationType
    v = ViolationEvent(type=ViolationType.SIGNAL_JUMP, track_id=1, frame=0, confidence=0.9)
    assert v.plate_number is None
    assert v.clip_path is None
    assert v.bbox == (0, 0, 0, 0)
    assert v.seat_positions is None
    assert v.evidence_narration is None


def test_violation_config_defaults():
    from core.violations.types import ViolationConfig
    config = ViolationConfig()
    assert config.enable_signal_jump is True
    assert config.enable_helmet is True
    assert config.enable_wrong_way is True
    assert config.enable_triple_riding is True
    assert config.enable_fancy_plate is True
    assert config.enable_missing_mirror is True
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


def test_triple_riding_detects_three_heads():
    from core.violations.triple_riding import TripleRidingDetector
    det = TripleRidingDetector()
    frame = make_frame()
    dets = MockDetections([[300, 100, 400, 400]], [0], [1], [0.9])
    violations = det.detect(dets, frame, 0, "GREEN")
    assert isinstance(violations, list)


def test_triple_riding_ignores_non_two_wheeler():
    from core.violations.triple_riding import TripleRidingDetector
    det = TripleRidingDetector()
    frame = make_frame()
    dets = MockDetections([[300, 100, 350, 400]], [2], [1], [0.9])
    violations = det.detect(dets, frame, 0, "GREEN")
    assert len(violations) == 0


def test_anpr_reader_init():
    from core.violations.anpr import ANPRReader
    reader = ANPRReader()
    assert reader._reader is None


def test_anpr_validate_plate():
    from core.violations.anpr import ANPRReader
    reader = ANPRReader()
    result = reader.validate_plate("KA05HJ4392")
    assert result["valid"] is True
    assert result["state"] == "KA"
    assert result["is_hsrp"] is True


def test_anpr_validate_non_standard():
    from core.violations.anpr import ANPRReader
    reader = ANPRReader()
    result = reader.validate_plate("XYZ123")
    assert result["valid"] is False
    assert result["format"] == "non_standard"


def test_anpr_classify_plate_type():
    from core.violations.anpr import ANPRReader
    reader = ANPRReader()
    frame = np.zeros((30, 100, 3), dtype=np.uint8)
    frame[:, :] = [0, 0, 220]
    plate_type = reader.classify_plate_type(frame)
    assert plate_type in ("private_white", "commercial_yellow", "electric_green", "unknown")


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
    assert len(vd._detectors) == 9


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
    dets = MockDetections([[100, 100, 200, 200]], [0], [1], [0.9])
    violations = det.detect(dets, frame, 0, "GREEN")
    assert isinstance(violations, list)


def test_trapezium_computation():
    from core.violations.triple_riding import TripleRidingDetector
    det = TripleRidingDetector()
    bbox = np.array([100.0, 50.0, 300.0, 250.0])
    trap = det._compute_trapezium(bbox)
    assert trap.shape == (4, 2)
    assert trap[0][0] < trap[1][0]
    assert trap[3][0] < trap[2][0]


def test_fancy_plate_detector_init():
    from core.violations.fancy_plate import FancyPlateDetector
    det = FancyPlateDetector()
    assert det.ocr_confidence_threshold == 0.3


def test_fancy_plate_no_detections():
    from core.violations.fancy_plate import FancyPlateDetector
    det = FancyPlateDetector()
    frame = make_frame()
    dets = MockDetections([], [])
    violations = det.detect(dets, frame, 0)
    assert len(violations) == 0


def test_fancy_plate_with_invalid_ocr():
    from core.violations.fancy_plate import FancyPlateDetector
    det = FancyPlateDetector()
    frame = make_frame()
    dets = MockDetections([[100, 100, 300, 200]], [2], [1], [0.9])
    ocr_results = {1: ("XYZ123", 0.8)}
    violations = det.detect(dets, frame, 0, ocr_results)
    assert len(violations) == 1
    assert violations[0].type.value == "fancy_plate"


def test_mirror_detector_init():
    from core.violations.mirror import MirrorDetector
    det = MirrorDetector()
    assert det is not None


def test_mirror_detector_no_two_wheelers():
    from core.violations.mirror import MirrorDetector
    det = MirrorDetector()
    frame = make_frame()
    dets = MockDetections([[100, 100, 200, 200]], [2], [1], [0.9])
    violations = det.detect(dets, frame, 0, "GREEN")
    assert len(violations) == 0


def test_mirror_detector_two_wheeler():
    from core.violations.mirror import MirrorDetector
    det = MirrorDetector()
    frame = make_frame()
    dets = MockDetections([[100, 100, 300, 300]], [0], [1], [0.9])
    violations = det.detect(dets, frame, 0, "GREEN")
    assert isinstance(violations, list)


def test_vlm_evidence_template():
    from core.violations.vlm_evidence import VLMEvidenceLayer
    from core.violations.types import ViolationEvent, ViolationType
    vlm = VLMEvidenceLayer()
    v = ViolationEvent(type=ViolationType.NO_HELMET, track_id=1, frame=10, confidence=0.85, plate_number="KA05HJ4392")
    narration = vlm._template_narrate(v)
    assert "no helmet" in narration.lower()
    assert "KA05HJ4392" in narration


def test_vlm_evidence_unavailable():
    from core.violations.vlm_evidence import VLMEvidenceLayer
    from core.violations.types import ViolationEvent, ViolationType
    vlm = VLMEvidenceLayer()
    v = ViolationEvent(type=ViolationType.TRIPLE_RIDING, track_id=5, frame=20, confidence=0.7)
    narration = vlm.narrate(v)
    assert narration is not None
    assert "triple riding" in narration.lower()


def test_violation_types_new_entries():
    from core.violations.types import ViolationType
    assert ViolationType.MOBILE_PHONE.value == "mobile_phone"
    assert ViolationType.NO_SEATBELT.value == "no_seatbelt"
    assert ViolationType.OVERLOADING.value == "overloading"


def test_violation_config_new_defaults():
    from core.violations.types import ViolationConfig
    config = ViolationConfig()
    assert config.enable_mobile_phone is True
    assert config.enable_seatbelt is True
    assert config.enable_overloading is True


def test_mobile_phone_detector_init():
    from core.violations.mobile_phone import MobilePhoneDetector
    det = MobilePhoneDetector()
    assert det.hand_head_threshold == 0.15


def test_mobile_phone_no_detections():
    from core.violations.mobile_phone import MobilePhoneDetector
    det = MobilePhoneDetector()
    frame = make_frame()
    dets = MockDetections(xyxy=[], class_id=[])
    result = det.detect(dets, frame, 0)
    assert result == []


def test_mobile_phone_ignores_non_person():
    from core.violations.mobile_phone import MobilePhoneDetector
    det = MobilePhoneDetector()
    frame = make_frame()
    dets = MockDetections(xyxy=[[100, 100, 200, 200]], class_id=[2])
    result = det.detect(dets, frame, 0)
    assert result == []


def test_seatbelt_detector_init():
    from core.violations.seatbelt import SeatbeltDetector
    det = SeatbeltDetector()
    assert det is not None


def test_seatbelt_no_detections():
    from core.violations.seatbelt import SeatbeltDetector
    det = SeatbeltDetector()
    frame = make_frame()
    dets = MockDetections(xyxy=[], class_id=[])
    result = det.detect(dets, frame, 0)
    assert result == []


def test_seatbelt_ignores_two_wheeler():
    from core.violations.seatbelt import SeatbeltDetector
    det = SeatbeltDetector()
    frame = make_frame()
    dets = MockDetections(xyxy=[[100, 100, 200, 300]], class_id=[0])
    result = det.detect(dets, frame, 0)
    assert result == []


def test_seatbelt_checks_lmv():
    from core.violations.seatbelt import SeatbeltDetector
    det = SeatbeltDetector()
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    dets = MockDetections(xyxy=[[50, 20, 250, 180]], class_id=[2], tracker_id=[1])
    result = det.detect(dets, frame, 0)
    assert len(result) == 1
    assert result[0].type.value == "no_seatbelt"


def test_overloading_detector_init():
    from core.violations.overloading import OverloadingDetector
    det = OverloadingDetector()
    assert det.aspect_ratio_threshold == 1.8


def test_overloading_no_detections():
    from core.violations.overloading import OverloadingDetector
    det = OverloadingDetector()
    frame = make_frame()
    dets = MockDetections(xyxy=[], class_id=[])
    result = det.detect(dets, frame, 0)
    assert result == []


def test_overloading_ignores_car():
    from core.violations.overloading import OverloadingDetector
    det = OverloadingDetector()
    frame = make_frame()
    dets = MockDetections(xyxy=[[100, 100, 200, 200]], class_id=[2])
    result = det.detect(dets, frame, 0)
    assert result == []


def test_overloading_checks_bus():
    from core.violations.overloading import OverloadingDetector
    det = OverloadingDetector(aspect_ratio_threshold=1.2, density_threshold=0.3)
    frame = np.random.randint(50, 200, (200, 400, 3), dtype=np.uint8)
    dets = MockDetections(xyxy=[[50, 50, 350, 180]], class_id=[3], tracker_id=[1])
    result = det.detect(dets, frame, 0)
    assert len(result) == 1
    assert result[0].type.value == "overloading"


def test_violation_detector_init_with_new_types():
    from core.violations import ViolationDetector, ViolationConfig
    config = ViolationConfig(
        enable_signal_jump=False,
        enable_helmet=False,
        enable_wrong_way=False,
        enable_triple_riding=False,
        enable_fancy_plate=False,
        enable_missing_mirror=False,
        enable_mobile_phone=True,
        enable_seatbelt=True,
        enable_overloading=True,
        enable_anpr=False,
    )
    vd = ViolationDetector(config)
    assert len(vd._detectors) == 3
