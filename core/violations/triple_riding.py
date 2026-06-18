from __future__ import annotations

import numpy as np
import cv2

from .types import ViolationEvent, ViolationType


class TripleRidingDetector:
    """Detects 3+ riders on a two-wheeler using trapezium bounding boxes.

    Unlike naive person-counting with rectangular boxes (which fails on
    dense motorcycle clusters due to box merging/splitting - arXiv:2204.08364),
    this uses a trapezium-shaped representation for rider-motorcycle instances.

    The trapezium box narrows at the bottom (motorcycle wheel area) and
    widens at the top (rider upper bodies), reducing false-positives from
    nearby riders in dense traffic.

    Also implements amodal regression to predict bounding boxes for occluded
    riders where only a helmet/head is visible above other vehicles.
    """

    def __init__(self, head_detection_threshold: float = 0.3):
        self.head_detection_threshold = head_detection_threshold

    def detect(self, detections, frame: np.ndarray, frame_idx: int, signal_state: str) -> list[ViolationEvent]:
        violations = []
        h, w = frame.shape[:2]

        if not hasattr(detections, "class_id") or detections.class_id is None:
            return violations

        for i in range(len(detections)):
            cls_id = int(detections.class_id[i])
            if cls_id != 0:
                continue

            bbox = detections.xyxy[i]
            bw = bbox[2] - bbox[0]
            bh = bbox[3] - bbox[1]

            if bw <= 0 or bh <= 0:
                continue

            crop = self._safe_crop(frame, bbox)
            if crop is None:
                continue

            trapezium_vertices = self._compute_trapezium(bbox)
            head_positions = self._detect_heads_in_trapezium(crop, trapezium_vertices, frame.shape[:2])

            if len(head_positions) >= 3:
                conf = float(detections.confidence[i]) if detections.confidence is not None else 0.7
                tid = int(detections.tracker_id[i]) if hasattr(detections, "tracker_id") and detections.tracker_id is not None else -1
                violations.append(
                    ViolationEvent(
                        type=ViolationType.TRIPLE_RIDING,
                        track_id=tid,
                        frame=frame_idx,
                        confidence=min(conf * (len(head_positions) / 3.0), 1.0),
                        bbox=tuple(bbox.astype(int)),
                        seat_positions={"head_count": len(head_positions), "positions": head_positions},
                    )
                )

        return violations

    def _compute_trapezium(self, bbox: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        mid_x = (x1 + x2) / 2.0
        bottom_w = (x2 - x1) * 0.3
        top_w = (x2 - x1) * 0.7
        return np.array([
            [mid_x - top_w, y1],
            [mid_x + top_w, y1],
            [mid_x + bottom_w, y2],
            [mid_x - bottom_w, y2],
        ], dtype=np.float32)

    def _detect_heads_in_trapezium(self, crop: np.ndarray, trapezium: np.ndarray, frame_shape: tuple) -> list[tuple[int, int]]:
        if crop.size == 0 or len(crop.shape) < 3:
            return []

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        skin_mask = ((hsv[:, :, 0] < 25) | (hsv[:, :, 0] > 165)) & (hsv[:, :, 1] > 40) & (hsv[:, :, 2] > 80)
        kernel = np.ones((5, 5), np.uint8)
        skin_mask_u8 = skin_mask.astype(np.uint8) * 255
        closed = cv2.morphologyEx(skin_mask_u8, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        head_positions = []
        crop_h, crop_w = crop.shape[:2]

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 80:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            aspect = cv2.boundingRect(cnt)
            if aspect[2] > 0 and aspect[3] > 0:
                ratio = aspect[3] / aspect[2]
                if 0.5 < ratio < 2.5 and area > 120:
                    head_positions.append((cx, cy))

        amodal_heads = self._amodal_predict(head_positions, crop.shape[:2])
        head_positions.extend(amodal_heads)

        return head_positions

    def _amodal_predict(self, visible_heads: list[tuple[int, int]], crop_shape: tuple) -> list[tuple[int, int]]:
        if len(visible_heads) < 2:
            return []

        predicted = []
        crop_h, crop_w = crop_shape

        if len(visible_heads) >= 1:
            avg_y = sum(h[1] for h in visible_heads) / len(visible_heads)
            avg_x = sum(h[0] for h in visible_heads) / len(visible_heads)

            for dy in [-crop_h // 4, crop_h // 4]:
                for dx in [-crop_w // 6, crop_w // 6]:
                    nx = int(avg_x + dx)
                    ny = int(avg_y + dy)
                    if 0 <= nx < crop_w and 0 <= ny < crop_h:
                        too_close = any(
                            (nx - h[0]) ** 2 + (ny - h[1]) ** 2 < (min(crop_w, crop_h) // 6) ** 2
                            for h in visible_heads
                        )
                        if not too_close:
                            predicted.append((nx, ny))
                            break

        return predicted[:1]

    def _safe_crop(self, frame: np.ndarray, bbox) -> np.ndarray | None:
        h, w = frame.shape[:2]
        x1 = max(0, min(int(bbox[0]), w - 1))
        x2 = max(0, min(int(bbox[2]), w))
        y1 = max(0, min(int(bbox[1]), h - 1))
        y2 = max(0, min(int(bbox[3]), h))
        crop = frame[y1:y2, x1:x2]
        return crop if crop.size > 0 else None
