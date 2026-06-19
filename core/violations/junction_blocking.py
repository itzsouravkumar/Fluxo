import numpy as np
import cv2
from .types import ViolationEvent

class JunctionBlockingDetector:
    """
    Detects vehicles that are blocking the central intersection box during critical congestion.
    This calculates an area-based density specifically for the center of the frame and flags 
    vehicles trapped there when density exceeds a critical threshold.
    """
    
    def __init__(self, density_threshold: float = 0.02, min_frames: int = 1):
        self.density_threshold = density_threshold
        self.min_frames = min_frames
        self.intersection_poly = None
        self._blocked_frames = 0
        
    def _compute_intersection_polygon(self, frame_shape: tuple) -> np.ndarray:
        # Define a dynamic central box (approx 40% of the frame in the center)
        h, w = frame_shape[:2]
        margin_x = int(w * 0.3)
        margin_y = int(h * 0.3)
        return np.array([
            [margin_x, margin_y],
            [w - margin_x, margin_y],
            [w - margin_x, h - margin_y],
            [margin_x, h - margin_y]
        ], dtype=np.int32)
        
    def check(self, detections, frame: np.ndarray, frame_idx: int, signal_state: str, tracker=None):
        from .types import ViolationEvent, ViolationType
        violations = []
        if len(detections) == 0:
            self._blocked_frames = max(0, self._blocked_frames - 1)
            return violations
            
        if self.intersection_poly is None:
            self.intersection_poly = self._compute_intersection_polygon(frame.shape)
            
        # Count vehicles whose center is inside the intersection polygon
        vehicles_in_box = []
        for i in range(len(detections)):
            bbox = detections.xyxy[i]
            cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
            
            if cv2.pointPolygonTest(self.intersection_poly, (float(cx), float(cy)), False) >= 0:
                # Approximate area of vehicle vs area of box
                vehicles_in_box.append(i)
                
        # Simple density metric for the intersection
        box_area = cv2.contourArea(self.intersection_poly)
        occupied_area = sum([(detections.xyxy[i][2] - detections.xyxy[i][0]) * (detections.xyxy[i][3] - detections.xyxy[i][1]) for i in vehicles_in_box])
        
        density = occupied_area / box_area if box_area > 0 else 0
        
        if density > self.density_threshold:
            self._blocked_frames += 1
        else:
            self._blocked_frames = 0
            
        # If blocked for enough frames (or immediately if it's just an image where min_frames logic is bypassed by tracker)
        # We flag the vehicles in the box
        if self._blocked_frames >= self.min_frames or (self.min_frames == 1 and density > self.density_threshold):
            for i in vehicles_in_box:
                track_id = int(detections.tracker_id[i]) if detections.tracker_id is not None else int(np.random.randint(1000, 9000))
                bbox = detections.xyxy[i]
                violations.append(
                    ViolationEvent(
                        track_id=track_id,
                        type=ViolationType.JUNCTION_BLOCKING,
                        frame=frame_idx,
                        confidence=min(0.99, density + 0.1),
                        bbox=tuple(float(x) for x in bbox),
                        evidence_frame=None
                    )
                )
                
        return violations
