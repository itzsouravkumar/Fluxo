#!/usr/bin/env python3
"""Fine-tune YOLOv11 on IDD (Indian Driving Dataset) for Indian traffic classes.

Converts IDD annotations to YOLO format and trains YOLOv11n.

Usage:
    python3 scripts/fine_tune_yolo.py --data data/raw/IDD_Detection --epochs 50
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fluxo.finetune")

INDIAN_CLASSES = {
    "two_wheeler": 0,
    "auto_rickshaw": 1,
    "light_motor_vehicle": 2,
    "bus": 3,
    "heavy_vehicle": 4,
    "pedestrian": 5,
    "emergency_vehicle": 6,
}

IDD_TO_FLUXO = {
    "car": 2,
    "motorcycle": 0,
    "bus": 3,
    "truck": 4,
    "bicycle": 0,
    "auto_rickshaw": 1,
    "pedestrian": 5,
    "person": 5,
}


def convert_idd_to_yolo(idd_path, output_path):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    (output_path / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_path / "images" / "val").mkdir(parents=True, exist_ok=True)
    (output_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (output_path / "labels" / "val").mkdir(parents=True, exist_ok=True)

    idd_path = Path(idd_path)
    annotations_dir = idd_path / "Annotations"
    images_dir = idd_path / "JPEGImages"

    if not annotations_dir.exists():
        log.error(f"Annotations directory not found: {annotations_dir}")
        return False

    label_count = 0
    image_count = 0

    for xml_file in annotations_dir.rglob("*.xml"):
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(xml_file)
            root = tree.getroot()

            filename = root.find("filename")
            if filename is None:
                continue
            filename = filename.text

            img_path = images_dir / filename
            if not img_path.exists():
                continue

            size = root.find("size")
            if size is None:
                continue
            img_w = int(size.find("width").text)
            img_h = int(size.find("height").text)

            label_lines = []
            for obj in root.findall("object"):
                name = obj.find("name").text
                if name not in IDD_TO_FLUXO:
                    continue

                class_id = IDD_TO_FLUXO[name]
                bndbox = obj.find("bndbox")
                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)

                x_center = ((xmin + xmax) / 2) / img_w
                y_center = ((ymin + ymax) / 2) / img_h
                w = (xmax - xmin) / img_w
                h = (ymax - ymin) / img_h

                label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

            if not label_lines:
                continue

            split = "train" if hash(filename) % 10 < 8 else "val"

            shutil.copy2(img_path, output_path / "images" / split / filename)

            label_file = output_path / "labels" / split / filename.rsplit(".", 1)[0] + ".txt"
            label_file.write_text("\n".join(label_lines))

            label_count += 1
            image_count += 1

        except Exception as e:
            log.warning(f"Error processing {xml_file}: {e}")
            continue

    log.info(f"Converted {image_count} images, {label_count} labels")
    return True


def create_data_yaml(output_path, num_classes=7):
    yaml_content = f"""train: {output_path}/images/train
val: {output_path}/images/val

nc: {num_classes}
names:
  0: two_wheeler
  1: auto_rickshaw
  2: light_motor_vehicle
  3: bus
  4: heavy_vehicle
  5: pedestrian
  6: emergency_vehicle
"""
    yaml_path = Path(output_path) / "data.yaml"
    yaml_path.write_text(yaml_content)
    log.info(f"Created data.yaml: {yaml_path}")
    return yaml_path


def train_model(data_yaml, epochs=50, batch_size=16, img_size=640):
    try:
        from ultralytics import YOLO
    except ImportError:
        log.error("ultralytics not installed. Run: pip install ultralytics")
        return None

    log.info("Starting YOLOv11 fine-tuning on IDD...")
    log.info(f"  Epochs: {epochs}")
    log.info(f"  Batch size: {batch_size}")
    log.info(f"  Image size: {img_size}")

    model = YOLO("yolo11n.pt")

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        name="fluxo_indian_traffic",
        patience=10,
        save=True,
        verbose=True,
    )

    best_model = Path("runs/detect/fluxo_indian_traffic/weights/best.pt")
    if best_model.exists():
        output = Path("models/yolo11n_indian_traffic.pt")
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_model, output)
        log.info(f"Best model saved: {output}")
        return output

    log.warning("Training completed but best.pt not found")
    return None


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv11 on IDD dataset")
    parser.add_argument("--data", default="data/raw/IDD_Detection", help="IDD dataset path")
    parser.add_argument("--output", default="data/processed/idd_yolo", help="Output path for YOLO format")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--img-size", type=int, default=640, help="Image size")
    parser.add_argument("--convert-only", action="store_true", help="Only convert annotations, don't train")
    args = parser.parse_args()

    log.info("FLUXO YOLO Fine-tuning on IDD")
    log.info("=" * 50)

    if not Path(args.data).exists():
        log.error(f"IDD dataset not found: {args.data}")
        log.info("Download from: https://idd.insaan.iiit.ac.in/")
        return

    log.info("Converting IDD annotations to YOLO format...")
    if not convert_idd_to_yolo(args.data, args.output):
        return

    data_yaml = create_data_yaml(args.output)

    if args.convert_only:
        log.info("Conversion complete. Skipping training.")
        return

    train_model(data_yaml, epochs=args.epochs, batch_size=args.batch, img_size=args.img_size)


if __name__ == "__main__":
    main()
