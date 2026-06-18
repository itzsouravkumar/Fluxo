"""Tests for FLUXO vision module."""

import numpy as np
import pytest


def test_vehicle_classes():
    from scripts.demo_vision import VEHICLE_CLASSES
    assert 2 in VEHICLE_CLASSES
    assert VEHICLE_CLASSES[2] == ("car", 1.0)
    assert 3 in VEHICLE_CLASSES
    assert VEHICLE_CLASSES[3] == ("motorcycle", 0.25)


def test_compute_density_empty():
    from scripts.demo_vision import compute_density
    import supervision as sv
    empty = sv.Detections(xyxy=np.empty((0, 4)), class_id=np.array([]))
    density, pce, count = compute_density(empty)
    assert density == 0.0
    assert pce == 0.0
    assert count == 0


def test_compute_density_with_vehicles():
    from scripts.demo_vision import compute_density
    import supervision as sv
    bboxes = np.array([[100, 100, 200, 200], [300, 300, 400, 400]])
    class_ids = np.array([2, 3])
    dets = sv.Detections(xyxy=bboxes, class_id=class_ids)
    density, pce, count = compute_density(dets, roi_area=1000.0)
    assert count == 2
    assert pce == 1.25
    assert density > 0


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


def test_density_levels():
    from scripts.demo_vision import compute_density
    import supervision as sv
    bboxes = np.array([[100, 100, 200, 200]] * 20)
    class_ids = np.array([7] * 20)
    dets = sv.Detections(xyxy=bboxes, class_id=class_ids)
    density, pce, count = compute_density(dets, roi_area=100.0)
    assert count == 20
    assert pce == 70.0
    assert density <= 1.0
