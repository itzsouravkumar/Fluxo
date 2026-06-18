#!/usr/bin/env python3
"""FLUXO Live Camera Feed.

Connects to a phone camera (IP Webcam app) or any RTSP/HTTP stream
and runs the full FLUXO pipeline: detection, tracking, violations, density.

Setup:
  1. Install "IP Webcam" app on your phone
  2. Start the server in the app (usually http://<phone_ip>:8080)
  3. Run this script with the stream URL

Usage:
    python3 scripts/live_camera.py --source "http://192.168.1.100:8080/video"
    python3 scripts/live_camera.py --source "http://192.168.1.100:8080/video" --violations --signal RED
    python3 scripts/live_camera.py --source 0  # local webcam
    python3 scripts/live_camera.py --source "rtsp://192.168.1.100:554/stream" --show
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from core.vision.config import DEFAULT_CONFIG, VEHICLE_CLASSES, VisionConfig
from core.violations import ViolationDetector, ViolationConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fluxo.live")


def apply_clahe(frame, clip_limit=2.0, grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_ch, a, b = cv2.split(lab)
    l_ch = clahe.apply(l_ch)
    lab = cv2.merge([l_ch, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def compute_lane_density(tracked, width, height, config):
    lanes = {name: {"pce": 0.0, "count": 0} for name in config.lane.names}
    if len(tracked) == 0:
        return lanes, 0.0, 0.0, 0

    total_pce = 0.0
    vehicle_count = 0

    for i in range(len(tracked)):
        cls_id = int(tracked.class_id[i]) if tracked.class_id is not None else 0
        if cls_id not in VEHICLE_CLASSES:
            continue

        bbox = tracked.xyxy[i]
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2

        vclass = VEHICLE_CLASSES[cls_id]
        total_pce += vclass.pce
        vehicle_count += 1

        if cy < height / 2:
            lane = "north" if cx < width / 2 else "east"
        else:
            lane = "south" if cx < width / 2 else "west"

        if lane in lanes:
            lanes[lane]["pce"] += vclass.pce
            lanes[lane]["count"] += 1

    roi_area = (width * height) / config.density.roi_area_divisor
    overall_density = min(total_pce / roi_area, 1.0)

    for lane_data in lanes.values():
        lane_data["density"] = min(lane_data["pce"] / (roi_area * config.lane.roi_ratio), 1.0)

    return lanes, overall_density, total_pce, vehicle_count


def estimate_speed(track_history, fps=30.0, pixel_to_meter=0.05):
    if len(track_history) < 5:
        return 0.0
    recent = track_history[-5:]
    dx = recent[-1][0] - recent[0][0]
    dy = recent[-1][1] - recent[0][1]
    pixel_dist = (dx**2 + dy**2) ** 0.5
    return (pixel_dist * pixel_to_meter * fps) * 3.6


def push_to_api(violations_data, density_data):
    try:
        import requests
        requests.post("http://localhost:8000/api/v1/junctions/j1/violations", json=violations_data, timeout=1)
        requests.post("http://localhost:8000/api/v1/junctions/j1/density", json=density_data, timeout=1)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="FLUXO Live Camera Feed")
    parser.add_argument("--source", required=True, help="Stream URL, RTSP, or 0 for webcam")
    parser.add_argument("--output", default=None, help="Output video path")
    parser.add_argument("--night", action="store_true", help="Enable CLAHE night mode")
    parser.add_argument("--conf", type=float, default=0.4, help="Detection confidence")
    parser.add_argument("--show", action="store_true", help="Show preview window")
    parser.add_argument("--violations", action="store_true", help="Enable violation detection")
    parser.add_argument("--signal", default="GREEN", choices=["RED", "YELLOW", "GREEN"])
    parser.add_argument("--push-api", action="store_true", help="Push data to FLUXO API")
    parser.add_argument("--junction-id", default="j1", help="Junction ID for API")
    args = parser.parse_args()

    config = DEFAULT_CONFIG
    config.detection.confidence = args.conf

    log.info("Loading YOLOv11 model...")
    from ultralytics import YOLO
    import supervision as sv

    model = YOLO(config.detection.model_path)
    tracker = sv.ByteTrack()

    violation_detector = None
    if args.violations:
        vconfig = ViolationConfig(enable_anpr=False, enable_clip_extract=True)
        violation_detector = ViolationDetector(vconfig)
        log.info(f"Violation detection: ON (signal={args.signal})")

    source = args.source
    if source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log.error(f"Cannot open stream: {args.source}")
        log.info("For phone camera:")
        log.info("  1. Install 'IP Webcam' from Play Store")
        log.info("  2. Start server in the app")
        log.info("  3. Use URL like: http://<phone_ip>:8080/video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    log.info(f"Stream: {args.source}")
    log.info(f"Resolution: {width}x{height} @ {fps:.1f} FPS")
    log.info(f"Night mode: {'ON' if args.night else 'OFF'}")
    log.info("-" * 60)

    writer = None
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    frame_count = 0
    start_time = time.time()
    density_history = []
    track_histories = {}
    unique_vehicles = set()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                log.warning("Frame read failed, retrying...")
                time.sleep(0.1)
                continue

            if args.night:
                frame = apply_clahe(frame, config.preprocessor.clahe_clip_limit, config.preprocessor.clahe_grid_size)

            results = model(frame, conf=config.detection.confidence, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)

            vehicle_mask = np.isin(detections.class_id, list(VEHICLE_CLASSES.keys()))
            vehicle_detections = detections[vehicle_mask]

            tracked = tracker.update_with_detections(vehicle_detections) if len(vehicle_detections) > 0 else vehicle_detections

            violations = []
            if violation_detector is not None:
                violation_detector.feed_frame(frame)
                violations = violation_detector.check(tracked, frame, frame_count, args.signal)

            lanes, density_score, total_pce, vehicle_count = compute_lane_density(tracked, width, height, config)
            density_history.append(density_score)

            frame_vehicles = []
            for i in range(len(tracked)):
                bbox = tracked.xyxy[i].astype(int)
                cls_id = int(tracked.class_id[i]) if tracked.class_id is not None else 0
                conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
                track_id = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1

                if cls_id not in VEHICLE_CLASSES:
                    continue

                vclass = VEHICLE_CLASSES[cls_id]
                cx = (bbox[0] + bbox[2]) // 2
                cy = (bbox[1] + bbox[3]) // 2

                if track_id > 0:
                    unique_vehicles.add(track_id)

                if track_id not in track_histories:
                    track_histories[track_id] = []
                track_histories[track_id].append((cx, cy))

                speed = estimate_speed(track_histories[track_id], fps, config.speed.pixel_to_meter)

                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), vclass.color, 2)
                cv2.putText(frame, f"#{track_id} {vclass.name}", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, vclass.color, 2)
                cv2.putText(frame, f"{speed:.0f} km/h", (bbox[0], bbox[3] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            for v in violations:
                vb = v.bbox
                cv2.rectangle(frame, (vb[0], vb[1]), (vb[2], vb[3]), (0, 0, 255), 3)
                vlabel = v.type.value.upper()
                if v.plate_number:
                    vlabel += f" [{v.plate_number}]"
                cv2.putText(frame, vlabel, (vb[0], vb[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            avg_density = np.mean(density_history[-30:]) if density_history else 0
            level = config.density_levels.get_level(avg_density)

            info_lines = [
                f"Frame: {frame_count}",
                f"Vehicles: {vehicle_count} | Unique: {len(unique_vehicles)}",
                f"Density: {avg_density:.3f} ({level})",
                f"Violations: {len(violations)} | Signal: {args.signal}",
            ]

            for i, line in enumerate(info_lines):
                cv2.putText(frame, line, (10, 25 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            if writer:
                writer.write(frame)

            if args.show:
                cv2.imshow("FLUXO Live", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.push_api and frame_count % 10 == 0:
                push_to_api(
                    {"violations": [{"type": v.type.value, "track_id": v.track_id, "confidence": v.confidence} for v in violations]},
                    {"density": avg_density, "vehicle_count": vehicle_count, "level": level},
                )

            frame_count += 1

            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                log.info(f"  {frame_count} frames | {vehicle_count} vehicles | density {avg_density:.3f} | {frame_count / elapsed:.1f} FPS")

    except KeyboardInterrupt:
        log.info("Interrupted by user")

    finally:
        elapsed = time.time() - start_time
        log.info("-" * 60)
        log.info(f"Processed {frame_count} frames in {elapsed:.1f}s ({frame_count / elapsed:.1f} FPS)")
        log.info(f"Unique vehicles: {len(unique_vehicles)}")

        if violation_detector is not None:
            all_violations = []
            log.info(f"Total violations: {len(all_violations)}")

        cap.release()
        if writer:
            writer.release()
            log.info(f"Output saved: {args.output}")

        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
