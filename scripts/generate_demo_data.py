#!/usr/bin/env python3
"""Generate synthetic demo data for offline demonstration."""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path


def generate_demo_data():
    output_dir = Path("outputs/demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    junctions = [
        {"id": "veerannapalya_001", "name": "Veerannapalya (BEL Road)", "lat": 13.0350, "lng": 77.5970},
        {"id": "gokaldas_001", "name": "Gokaldas Images Jn.", "lat": 12.9850, "lng": 77.5600},
        {"id": "silk_board_001", "name": "Silk Board Flyover", "lat": 12.9170, "lng": 77.6230},
        {"id": "whitefield_001", "name": "Hopefarm / Whitefield", "lat": 12.9690, "lng": 77.7500},
    ]

    demo_scenario = {
        "duration_seconds": 120,
        "events": [
            {"t": 0, "type": "normal", "junction": "veerannapalya_001", "density": 0.45},
            {"t": 20, "type": "density_spike", "junction": "veerannapalya_001", "density": 0.87},
            {"t": 35, "type": "violation", "junction": "veerannapalya_001", "violation_type": "no_helmet", "plate": "KA-05-MJ-4421"},
            {"t": 50, "type": "rl_recommendation", "junction": "veerannapalya_001", "action": "extend_green", "duration": 20},
            {"t": 70, "type": "queue_drop", "junction": "veerannapalya_001", "density": 0.52},
            {"t": 85, "type": "violation", "junction": "veerannapalya_001", "violation_type": "signal_jump", "plate": "KA-09-HB-7823"},
            {"t": 105, "type": "prediction", "junction": "veerannapalya_001", "horizon": "15min", "level": "CRITICAL"},
            {"t": 120, "type": "multi_junction_adjust", "junctions": ["veerannapalya_001", "gokaldas_001", "silk_board_001"]},
        ],
    }

    with open(output_dir / "demo_scenario.json", "w") as f:
        json.dump(demo_scenario, f, indent=2)

    print(f"Demo data generated in {output_dir}")


if __name__ == "__main__":
    generate_demo_data()
