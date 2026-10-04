"""
feature_extractor.py
====================
SIH26174 – AI Human Activity Recognition for On-board BAS Experiments
Feature Engineering for Classical ML Activity Classifier

Extracts real-time kinematic, spatial, and interaction features from
OpenCV HSV bounding boxes and MediaPipe Hands landmarks.
"""

from __future__ import annotations

import math
from typing import Optional
import numpy as np


FEATURE_NAMES = [
    "red_present",          # 0.0 or 1.0
    "yellow_present",       # 0.0 or 1.0
    "box_count",            # 0, 1, 2
    "hand_count",           # 0, 1, 2
    "hand_confidence",      # max confidence (0.0 - 1.0)
    "min_dist_to_red",      # px (clipped to 800.0)
    "min_dist_to_yellow",   # px (clipped to 800.0)
    "hand_touching_red",    # 1.0 if dist < 30px, else 0.0
    "hand_touching_yellow", # 1.0 if dist < 30px, else 0.0
    "hand_vel_x",           # pixels / frame
    "hand_vel_y",           # pixels / frame
    "hand_speed",           # sqrt(vx^2 + vy^2)
    "red_area_norm",        # contour area / (1280*720)
    "yellow_area_norm",     # contour area / (1280*720)
]


class FeatureExtractor:
    """
    Extracts numerical feature vectors from box and hand detections.
    Maintains temporal state for hand velocity estimation.
    """

    def __init__(self, frame_w: int = 1280, frame_h: int = 720):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.frame_area = max(1, frame_w * frame_h)
        self._prev_wrist: Optional[tuple[int, int]] = None
        self._prev_time: Optional[float] = None

    def reset(self):
        self._prev_wrist = None
        self._prev_time = None

    def extract(
        self,
        box_detections: list[dict],
        hand_detections: Optional[list[dict]],
    ) -> np.ndarray:
        """
        Extract a 14-dimensional feature vector from current frame detections.
        """
        # 1. Box Presence & Areas
        red_box = None
        yellow_box = None
        for b in box_detections:
            col = b.get("colour")
            if col == "red" and red_box is None:
                red_box = b
            elif col == "yellow" and yellow_box is None:
                yellow_box = b

        red_present = 1.0 if red_box is not None else 0.0
        yellow_present = 1.0 if yellow_box is not None else 0.0
        box_count = float(len(box_detections))

        red_area_norm = (red_box.get("area", 0.0) / self.frame_area) if red_box else 0.0
        yellow_area_norm = (yellow_box.get("area", 0.0) / self.frame_area) if yellow_box else 0.0

        # 2. Hand Metrics
        hands = hand_detections or []
        hand_count = float(len(hands))
        hand_conf = max([h.get("confidence", 0.0) for h in hands], default=0.0)

        # 3. Hand-to-Box Distances
        dist_red = 800.0
        dist_yellow = 800.0

        if red_box and hands:
            r_bbox = red_box.get("bbox")
            if r_bbox:
                dist_red = min(
                    [self._hand_bbox_dist(h, r_bbox) for h in hands],
                    default=800.0
                )

        if yellow_box and hands:
            y_bbox = yellow_box.get("bbox")
            if y_bbox:
                dist_yellow = min(
                    [self._hand_bbox_dist(h, y_bbox) for h in hands],
                    default=800.0
                )

        touching_red = 1.0 if dist_red <= 35.0 else 0.0
        touching_yellow = 1.0 if dist_yellow <= 35.0 else 0.0

        # 4. Kinematics (Wrist Velocity)
        vx = 0.0
        vy = 0.0
        speed = 0.0

        if hands:
            cur_wrist = hands[0].get("wrist_px")
            if cur_wrist and self._prev_wrist:
                vx = float(cur_wrist[0] - self._prev_wrist[0])
                vy = float(cur_wrist[1] - self._prev_wrist[1])
                speed = math.sqrt(vx * vx + vy * vy)
            if cur_wrist:
                self._prev_wrist = cur_wrist
        else:
            self._prev_wrist = None

        features = np.array([
            red_present,
            yellow_present,
            box_count,
            hand_count,
            hand_conf,
            min(800.0, float(dist_red)),
            min(800.0, float(dist_yellow)),
            touching_red,
            touching_yellow,
            vx,
            vy,
            speed,
            float(red_area_norm),
            float(yellow_area_norm),
        ], dtype=np.float32)

        return features

    @staticmethod
    def _hand_bbox_dist(hand: dict, bbox: tuple[int, int, int, int]) -> float:
        bx, by, bw, bh = bbox
        landmarks = hand.get("landmarks", [])
        if not landmarks:
            wrist = hand.get("wrist_px")
            if wrist:
                dx = max(bx - wrist[0], 0.0, wrist[0] - (bx + bw))
                dy = max(by - wrist[1], 0.0, wrist[1] - (by + bh))
                return math.sqrt(dx * dx + dy * dy)
            return 800.0

        key_ids = {0, 4, 8, 9, 12, 16, 20}
        dists = []
        for lm in landmarks:
            if lm.get("id") in key_ids or "id" not in lm:
                px = lm["x"]
                py = lm["y"]
                dx = max(bx - px, 0.0, px - (bx + bw))
                dy = max(by - py, 0.0, py - (by + bh))
                dists.append(math.sqrt(dx * dx + dy * dy))
        return min(dists) if dists else 800.0
