from __future__ import annotations

import cv2
import numpy as np

from .types import ViolationEvent, ViolationType


class MirrorDetector:
    """Detects missing rear-view mirrors on two-wheelers.

    A 2025 IEEE paper (arXiv:2511.12206) introduced detection of missing
    rear-view mirrors as a novel violation class, achieving mAP@50 = 0.843.
    This violation is legally enforceable under the Central Motor Vehicles
    Rules and completely absent from every other competitor's system.

    Improved heuristic: checks for mirror protrusions on both sides of
    the handlebar region using edge features, contour analysis, and
    symmetry checks. Requires BOTH mirrors to be missing to flag
    a violation (more conservative, reduces false positives).
    """

    def __init__(self):
        pass

    def detect(self, detections, frame: np.ndarray, frame_idx: int, signal_state: str) -> list[ViolationEvent]:
        violations = []
        h, w = frame.shape[:2]

        if not hasattr(detections, "class_id") or detections.class_id is None:
            return violations

        for i in range(len(detections)):
            cls_id = int(detections.class_id[i])
            if cls_id != 0:  # Only two-wheelers
                continue

            det_conf = float(detections.confidence[i]) if detections.confidence is not None else 0.0
            if det_conf < 0.75:  # Require high confidence on the base detection
                continue

            bbox = detections.xyxy[i].astype(int)
            x1, y1, x2, y2 = bbox
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h))

            bw, bh = x2 - x1, y2 - y1
            # Need a reasonably sized crop to see mirrors
            if bw < 80 or bh < 60:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            has_mirror = self._check_mirrors(crop)
            if not has_mirror:
                tid = int(detections.tracker_id[i]) if hasattr(detections, "tracker_id") and detections.tracker_id is not None else -1
                violations.append(
                    ViolationEvent(
                        type=ViolationType.MISSING_MIRROR,
                        track_id=tid,
                        frame=frame_idx,
                        confidence=det_conf * 0.70,  # Discount for heuristic
                        bbox=tuple(bbox),
                        requires_human_review=True,  # Always flag for review
                    )
                )

        return violations

    def _check_mirrors(self, vehicle_crop: np.ndarray) -> bool:
        """Check for mirror protrusions on both sides of the handlebar area.

        Mirrors create distinctive small protrusions extending beyond
        the main body of the motorcycle. We check for contours that
        extend into the left/right edges of the handlebar region.

        Returns True if at least ONE mirror is detected (conservative).
        """
        h, w = vehicle_crop.shape[:2]
        if h < 30 or w < 40:
            return True  # Too small to tell

        # Handlebar region: roughly 25-40% from top, full width
        hb_y1 = int(h * 0.20)
        hb_y2 = int(h * 0.45)
        handlebar = vehicle_crop[hb_y1:hb_y2, :]
        if handlebar.size == 0:
            return True

        gray = cv2.cvtColor(handlebar, cv2.COLOR_BGR2GRAY) if len(handlebar.shape) == 3 else handlebar
        hb_h, hb_w = gray.shape[:2]

        # Check left side (leftmost 20%)
        left_region = gray[:, :int(hb_w * 0.20)]
        left_mirror = self._has_mirror_feature(left_region, hb_h, hb_w)

        # Check right side (rightmost 20%)
        right_region = gray[:, int(hb_w * 0.80):]
        right_mirror = self._has_mirror_feature(right_region, hb_h, hb_w)

        # At least one mirror must be present
        return left_mirror or right_mirror

    def _has_mirror_feature(self, region: np.ndarray, parent_h: int, parent_w: int) -> bool:
        """Check if a mirror-like protrusion exists in the given region.

        Mirrors have:
        1. Higher edge density than empty background
        2. A small contour with roughly circular/oval shape
        3. Different intensity from the background

        More specific than just checking edge_density > threshold.
        """
        if region.size == 0 or region.shape[0] < 5 or region.shape[1] < 5:
            return True  # Can't tell, assume present

        # Signal 1: Edge density (mirrors have structure)
        edges = cv2.Canny(region, 40, 120)
        edge_density = np.sum(edges > 0) / max(edges.size, 1)

        if edge_density < 0.03:
            return False  # Very flat/empty region — no mirror

        # Signal 2: Look for small contours (mirror shape)
        _, binary = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rh, rw = region.shape[:2]
        min_mirror_area = max(30, rh * rw * 0.05)
        max_mirror_area = rh * rw * 0.8

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_mirror_area < area < max_mirror_area:
                # Check roundness
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    if circularity > 0.3:  # Roughly oval/circular
                        return True

        # Signal 3: Intensity contrast (mirror glass is different from background)
        mean_intensity = np.mean(region)
        std_intensity = np.std(region)
        if std_intensity > 20 and edge_density > 0.05:
            return True

        return edge_density > 0.10  # Fallback: higher edge threshold
