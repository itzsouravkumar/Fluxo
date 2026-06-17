from __future__ import annotations

NORMALIZATION_CONSTANT = 1000.0


def compute_reward(
    lane_states: list[dict],
    emergency_cleared: bool = False,
    rapid_switch: bool = False,
) -> float:
    total_wait = sum(l.get("queue_length", 0) * l.get("avg_wait_time", 0) for l in lane_states)
    r = -total_wait / NORMALIZATION_CONSTANT

    starved = sum(1 for l in lane_states if l.get("wait_time", 0) > 90)
    r -= 100 * starved

    if rapid_switch:
        r -= 50

    if emergency_cleared:
        r += 200

    throughput = sum(l.get("vehicles_passed_last_phase", 0) for l in lane_states)
    r += throughput * 0.5

    return r
