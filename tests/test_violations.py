"""Tests for FLUXO violation detection module."""

import pytest


def test_violation_types():
    from core.violations.detector import ViolationType
    assert ViolationType.NO_HELMET.value == "no_helmet"
    assert ViolationType.SIGNAL_JUMP.value == "signal_jump"
    assert ViolationType.WRONG_WAY.value == "wrong_way"


def test_violation_detector_init():
    from core.violations.detector import ViolationDetector
    detector = ViolationDetector()
    assert detector._detectors == []


def test_signal_jump_detector_init():
    from core.violations.signal_jump import SignalJumpDetector
    detector = SignalJumpDetector()
    assert detector.stop_line_y == 0.0
