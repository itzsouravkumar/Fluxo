from __future__ import annotations


from .detector import FluxoDetector


MAX_DENSITY = 10.0


def compute_pce_density(tracked_objects, lane_roi_area_m2: float) -> float:
    """Compute PCE-weighted density score for a lane.

    Returns normalised score between 0.0 and 1.0.
    """
    total_pce = sum(FluxoDetector.PCE_MAP.get(obj.class_id, 1.0) for obj in tracked_objects)
    density = total_pce / max(lane_roi_area_m2, 1.0)
    return min(density / MAX_DENSITY, 1.0)


def estimate_speed(track_positions: list[tuple[float, float]], fps: float, pixel_to_meter: float) -> float:
    """Estimate vehicle speed in km/h from track positions."""
    if len(track_positions) < 5:
        return 0.0

    dx = track_positions[-1][0] - track_positions[-5][0]
    dy = track_positions[-1][1] - track_positions[-5][1]
    pixel_dist = (dx**2 + dy**2) ** 0.5
    return (pixel_dist * pixel_to_meter * fps) * 3.6
