"""
FLUXO Classifier Training Pipeline
===================================
Single script: extract → label → train for seatbelt, mobile_phone, triple_riding.

Usage:
    # From video
    python tools/train_pipeline.py --input video.mp4 --type seatbelt

    # From image folder (each image is a full frame or cropped vehicle)
    python tools/train_pipeline.py --input ./my_images/ --type mobile_phone

    # Skip extraction (use existing crops in data/)
    python tools/train_pipeline.py --input data/raw_crops/unlabeled --type triple_riding --skip-extract

    # Train only (already labeled)
    python tools/train_pipeline.py --input data/raw_crops --type seatbelt --skip-extract --skip-label
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def extract_from_video(video_path: str, out_dir: Path, sample_rate: int = 3, conf: float = 0.5) -> list[dict]:
    """Extract vehicle crops from a video file."""
    from ultralytics import YOLO
    model = YOLO("yolo26n.pt")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {video_path} | {total} frames @ {fps:.1f} FPS")
    print(f"Sampling every {sample_rate} frames")

    metadata = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_rate != 0:
            frame_idx += 1
            continue

        results = model(frame, conf=conf, verbose=False)[0]
        if results.boxes is None:
            frame_idx += 1
            continue

        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            if cls_name not in ("person", "motorcycle", "bicycle", "car", "bus", "truck"):
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            det_conf = float(box.conf[0])
            h, w = frame.shape[:2]
            x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w))
            y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h))
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 30 or crop.shape[1] < 30:
                continue

            crop_name = f"f{frame_idx:06d}_c{cls_id}_conf{det_conf:.2f}.jpg"
            cv2.imwrite(str(out_dir / crop_name), crop)
            metadata.append({
                "file": crop_name,
                "frame": frame_idx,
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": det_conf,
                "bbox": [x1, y1, x2, y2],
            })

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  Frame {frame_idx}/{total} | Extracted: {len(metadata)}", end="\r")

    cap.release()
    return metadata


def extract_from_images(image_dir: str, out_dir: Path, conf: float = 0.5) -> list[dict]:
    """Run YOLO on a folder of images and extract vehicle crops."""
    from ultralytics import YOLO
    model = YOLO("yolo26n.pt")

    image_dir = Path(image_dir)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted(f for f in image_dir.iterdir() if f.suffix.lower() in exts)

    if not images:
        print(f"Error: no images found in {image_dir}")
        sys.exit(1)

    print(f"Found {len(images)} images in {image_dir}")

    metadata = []
    for idx, img_path in enumerate(images):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        results = model(frame, conf=conf, verbose=False)[0]
        if results.boxes is None:
            continue

        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            if cls_name not in ("person", "motorcycle", "bicycle", "car", "bus", "truck"):
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            det_conf = float(box.conf[0])
            h, w = frame.shape[:2]
            x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w))
            y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h))
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 30 or crop.shape[1] < 30:
                continue

            crop_name = f"img{idx:04d}_c{cls_id}_conf{det_conf:.2f}.jpg"
            cv2.imwrite(str(out_dir / crop_name), crop)
            metadata.append({
                "file": crop_name,
                "frame": idx,
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": det_conf,
                "bbox": [x1, y1, x2, y2],
            })

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(images)} | Extracted: {len(metadata)}", end="\r")

    return metadata


def label_interactive(unlabeled_dir: Path, metadata: list[dict], violation_type: str) -> dict:
    """Interactive CLI labeling. Returns dict of file → label."""
    label_map = {0: "not_violation", 1: "violation"}
    labels = {}

    print(f"\n{'='*50}")
    print(f"LABELING: {violation_type}")
    print(f"{'='*50}")
    print(f"Total crops: {len(metadata)}")
    print("\nKeys:  1 = VIOLATION  |  0 = NOT violation  |  s = skip  |  q = quit & save\n")

    for i, item in enumerate(metadata):
        crop_path = unlabeled_dir / item["file"]
        if not crop_path.exists():
            continue

        img = cv2.imread(str(crop_path))
        if img is None:
            continue

        display = img.copy()
        h, w = display.shape[:2]
        scale = min(600 / w, 600 / h, 2.0)
        display = cv2.resize(display, (int(w * scale), int(h * scale)))

        info = f"[{i+1}/{len(metadata)}] {item['class_name']} | frame={item['frame']} conf={item['confidence']:.2f}"
        cv2.putText(display, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(display, "1=violation  0=ok  s=skip  q=quit", (10, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Label", display)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            print(f"\nStopped at {i+1}/{len(metadata)}")
            break
        elif key == ord("s"):
            continue
        elif key in (ord("0"), ord("1")):
            label = int(chr(key))
            labels[item["file"]] = {
                "label": label,
                "label_name": label_map[label],
                "violation_type": violation_type,
                "class_name": item["class_name"],
                "frame": item["frame"],
            }
            color = (0, 0, 255) if label == 1 else (0, 255, 0)
            tag = "VIOLATION" if label == 1 else "OK"
            cv2.putText(display, tag, (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            cv2.imshow("Label", display)
            cv2.waitKey(300)

    cv2.destroyAllWindows()
    print(f"Labeled: {sum(1 for v in labels.values() if v['label'] == 1)} violation, "
          f"{sum(1 for v in labels.values() if v['label'] == 0)} not_violation")
    return labels


def create_yolo_structure(unlabeled_dir: Path, labels: dict, violation_type: str, output_base: Path):
    """Create YOLO train/val directory structure from labels."""
    base = output_base / "classifier_data" / violation_type
    for split in ["train", "val"]:
        for cls in ["violation", "not_violation"]:
            (base / split / cls).mkdir(parents=True, exist_ok=True)

    items = list(labels.items())
    np.random.seed(42)
    np.random.shuffle(items)
    split_idx = int(len(items) * 0.8)

    for split_name, slice_items in [("train", items[:split_idx]), ("val", items[split_idx:])]:
        for file_path, info in slice_items:
            src = unlabeled_dir / file_path
            dst = base / split_name / info["label_name"] / file_path
            if src.exists():
                shutil.copy2(src, dst)

    print(f"\nYOLO structure → {base}/")
    for split in ["train", "val"]:
        v = len(list((base / split / "violation").glob("*")))
        nv = len(list((base / split / "not_violation").glob("*")))
        print(f"  {split}: {v} violation, {nv} not_violation")


def train_model(violation_type: str, data_base: Path, epochs: int = 50, imgsz: int = 224):
    """Train YOLO binary classifier."""
    from ultralytics import YOLO

    data_path = data_base / violation_type
    train_dir = data_path / "train"
    val_dir = data_path / "val"

    total_train = len(list((train_dir / "violation").glob("*"))) + len(list((train_dir / "not_violation").glob("*")))
    total_val = len(list((val_dir / "violation").glob("*"))) + len(list((val_dir / "not_violation").glob("*")))

    if total_train < 10:
        print(f"Error: only {total_train} training samples. Need at least 20+.")
        sys.exit(1)

    print(f"\nTraining: {total_train} train, {total_val} val | {epochs} epochs")

    data_yaml = data_path / "data.yaml"
    data_yaml.write_text(f"train: {train_dir.resolve()}\nval: {val_dir.resolve()}\nnc: 2\nnames: ['not_violation', 'violation']\n")

    model = YOLO("yolo26n.pt")
    model.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=16,
                name=f"fluxo_{violation_type}", patience=15, augment=True, verbose=True)

    best_path = Path("runs") / f"fluxo_{violation_type}" / "weights" / "best.pt"
    output_path = Path("models") / f"fluxo_{violation_type}_v1.pt"

    if best_path.exists():
        shutil.copy2(best_path, output_path)
        print(f"\nModel saved → {output_path}")

        val_results = model.val(data=str(data_yaml))
        print(f"Accuracy:  {val_results.top1:.2%}")
        print(f"Precision: {val_results.precision:.2%}")
        print(f"Recall:    {val_results.recall:.2%}")
        print(f"F1:        {val_results.f1:.2%}")
    else:
        print(f"Error: best.pt not found at {best_path}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="FLUXO classifier training pipeline")
    parser.add_argument("--input", required=True, help="Video file or directory of images")
    parser.add_argument("--type", required=True, choices=["seatbelt", "mobile_phone", "triple_riding"],
                        help="Violation type to train classifier for")
    parser.add_argument("--output", default="data/raw_crops", help="Output directory for crops")
    parser.add_argument("--sample-rate", type=int, default=3, help="Process every N-th frame (video only)")
    parser.add_argument("--conf", type=float, default=0.5, help="Min YOLO detection confidence")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--skip-extract", action="store_true", help="Skip extraction, use existing crops")
    parser.add_argument("--skip-label", action="store_true", help="Skip labeling, use existing labels")
    args = parser.parse_args()

    out_dir = Path(args.output)
    unlabeled_dir = out_dir / "unlabeled"
    unlabeled_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    is_video = input_path.is_file() and input_path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    is_image_dir = input_path.is_dir()

    # Step 1: Extract
    metadata = []
    meta_path = out_dir / "metadata.json"

    if args.skip_extract:
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)
            print(f"Loaded {len(metadata)} existing crops from {meta_path}")
        else:
            print(f"Error: {meta_path} not found. Run without --skip-extract first.")
            sys.exit(1)
    elif is_video:
        metadata = extract_from_video(str(input_path), unlabeled_dir, args.sample_rate, args.conf)
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"\nExtracted {len(metadata)} crops → {unlabeled_dir}/")
    elif is_image_dir:
        metadata = extract_from_images(str(input_path), unlabeled_dir, args.conf)
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"\nExtracted {len(metadata)} crops → {unlabeled_dir}/")
    else:
        print(f"Error: {input_path} is not a valid video file or image directory.")
        sys.exit(1)

    if not metadata:
        print("No crops extracted. Check your input.")
        sys.exit(1)

    # Step 2: Label
    labels = {}
    label_file = out_dir / f"labels_{args.type}.json"

    if args.skip_label:
        if label_file.exists():
            with open(label_file) as f:
                labels = json.load(f)
            print(f"Loaded {len(labels)} existing labels from {label_file}")
        else:
            print(f"Error: {label_file} not found. Run without --skip-label first.")
            sys.exit(1)
    else:
        if label_file.exists():
            with open(label_file) as f:
                labels = json.load(f)
            print(f"Resuming from {label_file} ({len(labels)} already labeled)")

        unlabeled = [m for m in metadata if m["file"] not in labels]
        if unlabeled:
            labels.update(label_interactive(unlabeled_dir, unlabeled, args.type))

        with open(label_file, "w") as f:
            json.dump(labels, f, indent=2)
        print(f"Saved → {label_file}")

    if not labels:
        print("No labels. Nothing to train.")
        sys.exit(1)

    # Step 3: Create YOLO structure
    create_yolo_structure(unlabeled_dir, labels, args.type, out_dir)

    # Step 4: Train
    train_model(args.type, out_dir / "classifier_data", args.epochs)


if __name__ == "__main__":
    main()
