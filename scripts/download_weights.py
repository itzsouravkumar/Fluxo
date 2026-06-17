#!/bin/bash
set -e

echo "Downloading FLUXO model weights..."

# YOLOv11n
if [ ! -f models/yolo11n.pt ]; then
    echo "Downloading YOLOv11n..."
    python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
    mv yolo11n.pt models/
fi

echo "Model weights downloaded to models/"
