import numpy as np
import supervision as sv
from core.vision.detector import FluxoDetector
from core.vision.enhancement import FrameEnhancer, TileDetector, AdaptiveConfidence, SmartROISelector

def test():
    # Create a dummy image
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
    print("Detections on random noise:", len(dets))

test()
