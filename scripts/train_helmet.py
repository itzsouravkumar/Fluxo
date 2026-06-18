#!/usr/bin/env python3
"""Train YOLO26 classifier on helmet dataset.

Converts Pascal VOC XML annotations to YOLO classification format,
then trains YOLO26n-cls for helmet vs no-helmet detection.

Usage:
    python3 scripts/train_helmet.py --epochs 20
"""

import argparse
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def convert_voc_to_yolo_classify(xml_dir: Path, img_dir: Path, out_dir: Path):
    """Convert VOC XML annotations to YOLO-cls directory structure.

    Creates: out_dir/with_helmet/ and out_dir/without_helmet/
    """
    helmet_dir = out_dir / "with_helmet"
    no_helmet_dir = out_dir / "without_helmet"
    helmet_dir.mkdir(parents=True, exist_ok=True)
    no_helmet_dir.mkdir(parents=True, exist_ok=True)

    xml_files = list(xml_dir.glob("*.xml"))
    print(f"Converting {len(xml_files)} annotations...")

    stats = {"helmet": 0, "no_helmet": 0, "skipped": 0}

    for xml_path in xml_files:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError:
            stats["skipped"] += 1
            continue

        filename = root.findtext("filename")
        if not filename:
            stats["skipped"] += 1
            continue

        src_img = img_dir / filename
        if not src_img.exists():
            stats["skipped"] += 1
            continue

        has_helmet = False
        for obj in root.findall("object"):
            name = obj.findtext("name", "")
            if name == "with helmet":
                has_helmet = True
                break

        dst_dir = helmet_dir if has_helmet else no_helmet_dir
        dst = dst_dir / filename
        shutil.copy2(src_img, dst)

        if has_helmet:
            stats["helmet"] += 1
        else:
            stats["no_helmet"] += 1

    print(f"  with_helmet: {stats['helmet']}")
    print(f"  without_helmet: {stats['no_helmet']}")
    print(f"  skipped: {stats['skipped']}")
    return stats


def train(data_dir: Path, epochs: int = 20, imgsz: int = 224):
    from ultralytics import YOLO

    print("\nTraining YOLO26n-cls for helmet detection...")
    model = YOLO("yolo26n-cls.pt")
    results = model.train(
        data=str(data_dir),
        epochs=epochs,
        imgsz=imgsz,
        batch=32,
        name="fluxo_helmet",
        patience=10,
        verbose=True,
    )

    best = Path("runs/classify/fluxo_helmet/weights/best.pt")
    dst = Path("models/fluxo_helmet_v1.pt")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if best.exists():
        shutil.copy2(best, dst)
        print(f"\nBest model saved to {dst}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train FLUXO helmet classifier")
    parser.add_argument("--data", default="data/raw/helmet_kaggle", help="Helmet dataset path")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--skip-convert", action="store_true", help="Skip VOC->YOLO conversion")
    args = parser.parse_args()

    data_path = Path(args.data)
    xml_dir = data_path / "annotations"
    img_dir = data_path / "images"
    yolo_dir = data_path / "yolo_cls"

    print("FLUXO Helmet Classifier Training")
    print("=" * 50)

    if not args.skip_convert:
        convert_voc_to_yolo_classify(xml_dir, img_dir, yolo_dir)

    print(f"\nDataset ready at {yolo_dir}")
    train(yolo_dir, epochs=args.epochs)


if __name__ == "__main__":
    main()
