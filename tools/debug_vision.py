import os
import cv2
import numpy as np
from ultralytics import YOLO
from core.vision.detector import FluxoDetector
from core.vision.enhancement import FrameEnhancer, TileDetector, AdaptiveConfidence, SmartROISelector

def debug_dummy_image():
    """Test the detector on a dummy random noise image."""
    print("Testing on dummy image...")
    frame = np.random.randint(0, 255, (2160, 3840, 3), dtype=np.uint8)
    
    detector = FluxoDetector()
    enhancer = FrameEnhancer()
    tile_det = TileDetector(tile_size=640, overlap=0.2)
    adaptive_conf = AdaptiveConfidence(base_conf=0.25)
    roi_selector = SmartROISelector()
    
    dets, qual = detector.detect_with_enhancement(
        frame, enhancer=enhancer, tile_detector=tile_det,
        adaptive_conf=adaptive_conf, roi_selector=roi_selector
    )
    print("Quality:", qual)
    print("Detections on random noise:", len(dets))


def debug_real_image(image_filename="sample1.webp"):
    """Test the detector on a real image by searching for it."""
    print(f"Testing on real image containing '{image_filename}'...")
    img_path = None
    for root, dirs, files in os.walk("."):
        for f in files:
            if image_filename in f or "sample" in f:
                img_path = os.path.join(root, f)
                break
        if img_path:
            break

    if not img_path:
        print("No sample image found.")
        return
        
    frame = cv2.imread(img_path)
    if frame is None:
        print("Cannot read", img_path)
        return
        
    print("Frame shape:", frame.shape)
    
    detector = FluxoDetector()
    enhancer = FrameEnhancer()
    tile_det = TileDetector(tile_size=640, overlap=0.2)
    adaptive_conf = AdaptiveConfidence(base_conf=0.25)
    roi_selector = SmartROISelector()
    
    dets, qual = detector.detect_with_enhancement(
        frame, enhancer=enhancer, tile_detector=tile_det,
        adaptive_conf=adaptive_conf, roi_selector=roi_selector
    )
    
    print("Quality:", qual)
    print(f"Detections: {len(dets)}")


def debug_model_classes(model_path="yolo26n.pt"):
    """Check the class names loaded by a model."""
    print(f"Loading model '{model_path}'...")
    try:
        model = YOLO(model_path)
        print("Model class names:", model.model.names)
    except Exception as e:
        print(f"Failed to load model: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Modular debugging tool for FLUXO vision.")
    parser.add_argument("--mode", choices=["dummy", "real", "model", "all"], default="all", help="Mode to run the debugger in.")
    parser.add_argument("--image", default="sample1.webp", help="Target image filename for 'real' mode.")
    parser.add_argument("--model", default="yolo26n.pt", help="Target model path for 'model' mode.")
    args = parser.parse_args()

    if args.mode in ["dummy", "all"]:
        debug_dummy_image()
        print("-" * 30)
    
    if args.mode in ["real", "all"]:
        debug_real_image(args.image)
        print("-" * 30)

    if args.mode in ["model", "all"]:
        debug_model_classes(args.model)
        print("-" * 30)

if __name__ == "__main__":
    main()
