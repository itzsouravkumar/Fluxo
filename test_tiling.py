import cv2
import numpy as np
from core.vision.detector import FluxoDetector
from core.vision.enhancement import FrameEnhancer, TileDetector, AdaptiveConfidence

frame = cv2.imread("traffic_image.jpg") # I don't have the image, but I can use an arbitrary large array or download one
