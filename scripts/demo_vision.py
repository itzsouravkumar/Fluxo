#!/usr/bin/env python3
"""FLUXO Vision Pipeline Demo.

Run on any video file to see YOLOv11 detection + tracking + PCE density + speed estimation.

Usage:
    python3 scripts/demo_vision.py --source path/to/video.mp4
    python3 scripts/demo_vision.py --source 0  # webcam
    python3 scripts/demo_vision.py --source path/to/video.mp4 --night  # with CLAHE
    python3 scripts/demo_vision.py --source path/to/video.mp4 --output outputs/demo.mp4 --save-results
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np


VEHICLE_CLASSES = {
    1: ("bicycle", 0.25),
    2: ("car", 1.0),
    3: ("motorcycle", 0.25),
    5: ("bus", 3.0),
    7: ("truck", 3.5),
}

VEHICLE_COLORS = {
    1: (0, 255, 255),
    2: (0, 255, 0),
    3: (255, 165, 0),
    5: (255, 0, 0),
    7: (0, 0, 255),
}

PIXEL_TO_METER = 0.05


def apply_clahe(frame, clip_limit=2.0, grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def compute_density(tracked_detections, roi_area=100.0):
    total_pce = 0
    vehicle_count = 0

    if len(tracked_detections) == 0:
        return 0.0, 0.0, 0

    for cls_id in tracked_detections.class_id:
        if cls_id in VEHICLE_CLASSES:
            _, pce = VEHICLE_CLASSES[cls_id]
            total_pce += pce
            vehicle_count += 1

    density = total_pce / max(roi_area, 1.0)
    return min(density, 1.0), total_pce, vehicle_count


def estimate_speed(track_history, fps=30.0):
    if len(track_history) < 5:
        return 0.0

    recent = track_history[-5:]
    dx = recent[-1][0] - recent[0][0]
    dy = recent[-1][1] - recent[0][1]
    pixel_dist = (dx**2 + dy**2) ** 0.5
    return (pixel_dist * PIXEL_TO_METER * fps) * 3.6


def main():
    parser = argparse.ArgumentParser(description="FLUXO Vision Pipeline Demo")
    parser.add_argument("--source", required=True, help="Video path or 0 for webcam")
    parser.add_argument("--output", default=None, help="Output video path")
    parser.add_argument("--night", action="store_true", help="Enable CLAHE night mode")
    parser.add_argument("--conf", type=float, default=0.4, help="Detection confidence")
    parser.add_argument("--show", action="store_true", help="Show preview window")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--save-results", action="store_true", help="Save detection results to JSON")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
        import supervision as sv
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Run: pip install ultralytics supervision")
        return

    print("Loading YOLOv11n model...")
    model = YOLO("yolo11n.pt")
    tracker = sv.ByteTrack()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Cannot open video source: {args.source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Source: {args.source}")
    print(f"Resolution: {width}x{height} @ {fps:.1f} FPS")
    print(f"Total frames: {total_frames}")
    print(f"Night mode: {'ON' if args.night else 'OFF'}")
    print("-" * 60)

    writer = None
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    frame_count = 0
    start_time = time.time()
    density_history = []
    track_histories = {}
    results_data = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if args.night:
            frame = apply_clahe(frame)

        results = model(frame, conf=args.conf, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        vehicle_mask = np.isin(detections.class_id, list(VEHICLE_CLASSES.keys()))
        vehicle_detections = detections[vehicle_mask]

        if len(vehicle_detections) > 0:
            tracked = tracker.update_with_detections(vehicle_detections)
        else:
            tracked = vehicle_detections

        density_score, total_pce, vehicle_count = compute_density(
            tracked, roi_area=(width * height) / 1000.0
        )
        density_history.append(density_score)

        frame_vehicles = []
        for i in range(len(tracked)):
            bbox = tracked.xyxy[i].astype(int)
            cls_id = int(tracked.class_id[i]) if tracked.class_id is not None else 0
            conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
            track_id = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1
            color = VEHICLE_COLORS.get(cls_id, (255, 255, 255))
            name, pce = VEHICLE_CLASSES.get(cls_id, ("unknown", 0))

            cx = (bbox[0] + bbox[2]) // 2
            cy = (bbox[1] + bbox[3]) // 2

            if track_id not in track_histories:
                track_histories[track_id] = []
            track_histories[track_id].append((cx, cy))

            speed = estimate_speed(track_histories[track_id], fps)

            label = f"#{track_id} {name}"
            speed_label = f"{speed:.0f} km/h"

            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(frame, speed_label, (bbox[0], bbox[3] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            frame_vehicles.append({
                "track_id": track_id,
                "class": name,
                "confidence": round(conf, 3),
                "pce": pce,
                "speed_kmh": round(speed, 1),
                "bbox": bbox.tolist(),
            })

        avg_density = np.mean(density_history[-30:]) if density_history else 0
        level = "CRITICAL" if avg_density > 0.7 else "HIGH" if avg_density > 0.4 else "MODERATE" if avg_density > 0.2 else "CLEAR"

        bar_width = int(avg_density * 200)
        cv2.rectangle(frame, (10, 130), (210, 150), (50, 50, 50), -1)
        bar_color = (0, int(255 * (1 - avg_density)), int(255 * avg_density))
        cv2.rectangle(frame, (10, 130), (10 + bar_width, 150), bar_color, -1)

        info_lines = [
            f"Frame: {frame_count}",
            f"Vehicles: {vehicle_count} | PCE: {total_pce:.1f}",
            f"Density: {avg_density:.3f} ({level})",
        ]

        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (10, 25 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        if writer:
            writer.write(frame)

        if args.show:
            cv2.imshow("FLUXO Vision", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if args.save_results:
            results_data.append({
                "frame": frame_count,
                "density": round(density_score, 4),
                "pce": round(total_pce, 1),
                "vehicles": vehicle_count,
                " detections": frame_vehicles,
            })

        frame_count += 1
        if args.max_frames and frame_count >= args.max_frames:
            break

        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            print(f"  {frame_count} frames | {vehicle_count} vehicles | density {avg_density:.3f} | {frame_count / elapsed:.1f} FPS")

    elapsed = time.time() - start_time
    print("-" * 60)
    print(f"Done. {frame_count} frames in {elapsed:.1f}s ({frame_count / elapsed:.1f} FPS)")
    print(f"Avg density: {np.mean(density_history):.3f}")
    print(f"Max density: {np.max(density_history):.3f}")
    print(f"Unique tracks: {len(track_histories)}")

    cap.release()
    if writer:
        writer.release()
        print(f"Output saved: {args.output}")

    if args.save_results and results_data:
        output_path = Path(args.output).with_suffix(".json") if args.output else "outputs/results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results_data, f, indent=2)
        print(f"Results saved: {output_path}")

    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
