import cv2
from core.vision.detector import FluxoDetector
from core.vision.enhancement import FrameEnhancer, TileDetector, AdaptiveConfidence, SmartROISelector

def test():
    # Look for sample1.webp
    import os
    img_path = None
    for root, dirs, files in os.walk("."):
        for f in files:
            if f == "sample1.webp" or "sample" in f:
                img_path = os.path.join(root, f)
                break
    if not img_path:
        print("No image found")
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
    
test()
