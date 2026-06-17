"""Tests for FLUXO vision module."""

import numpy as np
import pytest


def test_detector_classes():
    from core.vision.detector import FluxoDetector
    assert 0 in FluxoDetector.CLASSES
    assert FluxoDetector.CLASSES[0] == "two_wheeler"


def test_pce_values():
    from core.vision.detector import FluxoDetector
    assert FluxoDetector.PCE_MAP[0] == 0.25
    assert FluxoDetector.PCE_MAP[3] == 3.0
    assert FluxoDetector.PCE_MAP[5] == 0.0


def test_density_computation():
    from core.vision.density import compute_pce_density
    density = compute_pce_density([], 100.0)
    assert density == 0.0


def test_preprocessor_clahe():
    from core.vision.preprocessor import FramePreprocessor
    pp = FramePreprocessor()
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = pp.apply_clahe(frame)
    assert result.shape == frame.shape
