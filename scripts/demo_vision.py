#!/usr/bin/env python3
"""FLUXO Vision Pipeline Demo.

Full-featured vision pipeline with:
- YOLOv11 detection + ByteTrack tracking
- Lane-wise PCE density scoring (N/S/E/W)
- Speed estimation per vehicle
- Violation detection (signal jump, helmet, wrong way, triple riding)
- ANPR (license plate recognition)
- Evidence clip extraction
- CLAHE night mode preprocessing
- Performance profiling
- JSON results export

Usage:
    python3 scripts/demo_vision.py --source video.mp4 --show
    python3 scripts/demo_vision.py --source video.mp4 --night --save-results
    python3 scripts/demo_vision.py --source video.mp4 --violations --signal RED --show
    python3 scripts/demo_vision.py --source 0 --show  # webcam
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from core.vision.config import (
    DEFAULT_CONFIG,
    VEHICLE_CLASSES,
    VisionConfig,
)
from core.violations import ViolationDetector, ViolationConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fluxo.vision")


class Profiler:
    def __init__(self):
        self.times = {"detection": [], "tracking": [], "violations": [], "density": [], "total": []}

    def start(self):
        self._t0 = time.perf_counter()

    def lap(self, name):
        elapsed = time.perf_counter() - self._t0
        self.times[name].append(elapsed)
        self._t0 = time.perf_counter()

    def summary(self):
        lines = []
        for name, values in self.times.items():
            if values:
                avg = np.mean(values) * 1000
                lines.append(f"  {name}: {avg:.1f}ms avg")
        return "\n".join(lines)


def apply_clahe(frame, clip_limit=2.0, grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def compute_lane_density(tracked, width, height, config: VisionConfig):
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
            if cx < width / 2:
                lane = "north"
            else:
                lane = "east"
        else:
            if cx < width / 2:
                lane = "south"
            else:
                lane = "west"

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


def validate_source(source):
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        path = Path(source)
        if not path.exists():
            log.error(f"Video file not found: {source}")
            return None
        if path.stat().st_size == 0:
            log.error(f"Video file is empty: {source}")
            return None
        cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        log.error(f"Cannot open video source: {source}")
        return None

    return cap


def validate_model(model_path):
    path = Path(model_path)
    if not path.exists():
        log.warning(f"Model not found at {model_path}, downloading...")
        try:
            from ultralytics import YOLO
            YOLO(model_path)
            log.info(f"Model downloaded: {model_path}")
        except Exception as e:
            log.error(f"Failed to download model: {e}")
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="FLUXO Vision Pipeline Demo")
    parser.add_argument("--source", required=True, help="Video path or 0 for webcam")
    parser.add_argument("--output", default=None, help="Output video path")
    parser.add_argument("--night", action="store_true", help="Enable CLAHE night mode")
    parser.add_argument("--conf", type=float, default=None, help="Detection confidence")
    parser.add_argument("--show", action="store_true", help="Show preview window")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--save-results", action="store_true", help="Save results to JSON")
    parser.add_argument("--violations", action="store_true", help="Enable violation detection")
    parser.add_argument("--signal", default="GREEN", choices=["RED", "YELLOW", "GREEN"], help="Simulated signal state")
    parser.add_argument("--anpr", action="store_true", help="Enable ANPR (requires easyocr)")
    args = parser.parse_args()

    config = DEFAULT_CONFIG
    if args.conf is not None:
        config.detection.confidence = args.conf

    if not validate_model(config.detection.model_path):
        return

    log.info("Loading YOLOv11 model...")
    try:
        from ultralytics import YOLO
        import supervision as sv
    except ImportError as e:
        log.error(f"Missing dependency: {e}. Run: pip install ultralytics supervision")
        return

    model = YOLO(config.detection.model_path)
    tracker = sv.ByteTrack()

    violation_detector = None
    if args.violations:
        vconfig = ViolationConfig(
            enable_anpr=args.anpr,
            enable_clip_extract=True,
        )
        violation_detector = ViolationDetector(vconfig)
        log.info(f"Violation detection: ON (signal={args.signal}, anpr={args.anpr})")

    cap = validate_source(args.source)
    if cap is None:
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    log.info(f"Source: {args.source}")
    log.info(f"Resolution: {width}x{height} @ {fps:.1f} FPS")
    log.info(f"Total frames: {total_frames}")
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
    profiler = Profiler()
    results_data = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            profiler.start()

            if args.night:
                frame = apply_clahe(
                    frame,
                    clip_limit=config.preprocessor.clahe_clip_limit,
                    grid_size=config.preprocessor.clahe_grid_size,
                )

            results = model(frame, conf=config.detection.confidence, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            profiler.lap("detection")

            vehicle_mask = np.isin(detections.class_id, list(VEHICLE_CLASSES.keys()))
            vehicle_detections = detections[vehicle_mask]

            if len(vehicle_detections) > 0:
                tracked = tracker.update_with_detections(vehicle_detections)
            else:
                tracked = vehicle_detections
            profiler.lap("tracking")

            violations = []
            if violation_detector is not None:
                violation_detector.feed_frame(frame)
                violations = violation_detector.check(
                    tracked, frame, frame_count, args.signal
                )
            profiler.lap("violations")

            lanes, density_score, total_pce, vehicle_count = compute_lane_density(
                tracked, width, height, config
            )
            density_history.append(density_score)
            profiler.lap("density")

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

                label = f"#{track_id} {vclass.name}"
                speed_label = f"{speed:.0f} km/h"

                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), vclass.color, 2)
                cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, vclass.color, 2)
                cv2.putText(frame, speed_label, (bbox[0], bbox[3] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

                frame_vehicles.append({
                    "track_id": track_id,
                    "class": vclass.name,
                    "confidence": round(conf, 3),
                    "pce": vclass.pce,
                    "speed_kmh": round(speed, 1),
                    "bbox": bbox.tolist(),
                })

            for v in violations:
                vb = v.bbox
                cv2.rectangle(frame, (vb[0], vb[1]), (vb[2], vb[3]), (0, 0, 255), 3)
                vlabel = v.type.value.upper()
                if v.plate_number:
                    vlabel += f" [{v.plate_number}]"
                cv2.putText(frame, vlabel, (vb[0], vb[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(frame, f"{v.confidence:.0%}", (vb[0], vb[3] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            avg_density = np.mean(density_history[-30:]) if density_history else 0
            level = config.density_levels.get_level(avg_density)

            bar_width = int(avg_density * 200)
            cv2.rectangle(frame, (10, 130), (210, 150), (50, 50, 50), -1)
            bar_color = (0, int(255 * (1 - avg_density)), int(255 * avg_density))
            cv2.rectangle(frame, (10, 130), (10 + bar_width, 150), bar_color, -1)

            info_lines = [
                f"Frame: {frame_count}",
                f"Vehicles: {vehicle_count} | Unique: {len(unique_vehicles)}",
                f"PCE: {total_pce:.1f} | Density: {avg_density:.3f} ({level})",
                f"Violations: {len(violations)} | Signal: {args.signal}",
            ]

            for i, line in enumerate(info_lines):
                cv2.putText(frame, line, (10, 25 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            lane_y = 170
            for lane_name, lane_data in lanes.items():
                lane_text = f"{lane_name[0].upper()}: {lane_data['count']}v {lane_data.get('density', 0):.2f}"
                cv2.putText(frame, lane_text, (10, lane_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
                lane_y += 18

            if writer:
                writer.write(frame)

            if args.show:
                cv2.imshow("FLUXO Vision", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.save_results:
                frame_violations = [
                    {
                        "type": v.type.value,
                        "track_id": v.track_id,
                        "confidence": round(v.confidence, 3),
                        "plate": v.plate_number,
                        "clip": v.clip_path,
                    }
                    for v in violations
                ]
                results_data.append({
                    "frame": frame_count,
                    "density": round(density_score, 4),
                    "pce": round(total_pce, 1),
                    "vehicles": vehicle_count,
                    "violations": frame_violations,
                    "lanes": {k: {"count": v["count"], "pce": round(v["pce"], 1)} for k, v in lanes.items()},
                    "detections": frame_vehicles,
                })

            frame_count += 1
            if args.max_frames and frame_count >= args.max_frames:
                break

            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                log.info(f"  {frame_count} frames | {vehicle_count} vehicles | density {avg_density:.3f} | {frame_count / elapsed:.1f} FPS")

    except KeyboardInterrupt:
        log.info("Interrupted by user")

    finally:
        elapsed = time.time() - start_time
        log.info("-" * 60)
        log.info(f"Processed {frame_count} frames in {elapsed:.1f}s ({frame_count / elapsed:.1f} FPS)")
        log.info(f"Avg density: {np.mean(density_history):.3f}")
        log.info(f"Max density: {np.max(density_history):.3f}")
        log.info(f"Unique vehicles seen: {len(unique_vehicles)}")
        if violation_detector is not None:
            all_violations = [v for frame_data in results_data for v in frame_data.get("violations", [])]
            log.info(f"Total violations detected: {len(all_violations)}")
            vtypes = {}
            for v in all_violations:
                vtypes[v["type"]] = vtypes.get(v["type"], 0) + 1
            for vtype, count in vtypes.items():
                log.info(f"  {vtype}: {count}")
        log.info(f"Profiling:\n{profiler.summary()}")

        cap.release()
        if writer:
            writer.release()
            log.info(f"Output saved: {args.output}")

        if args.save_results and results_data:
            output_path = Path(args.output).with_suffix(".json") if args.output else Path("outputs/results.json")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            summary = {
                "total_frames": frame_count,
                "unique_vehicles": len(unique_vehicles),
                "avg_density": round(float(np.mean(density_history)), 4),
                "max_density": round(float(np.max(density_history)), 4),
            }
            with open(output_path, "w") as f:
                json.dump({"summary": summary, "frames": results_data}, f, indent=2)
            log.info(f"Results saved: {output_path}")

        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
