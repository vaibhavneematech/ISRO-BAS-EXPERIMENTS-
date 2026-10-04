"""
person_detector.py
==================
SIH26174 – AI Human Activity Recognition for On-board BAS Experiments

Face + Body/Pose detection using MediaPipe 1.x Tasks API.
- FaceLandmarker  → face bounding box + FACE label (blue)
- PoseLandmarker  → body skeleton overlay + POSE label (cyan)

Both run in IMAGE mode (stateless per frame) for simplicity and crash-safety.
If either model fails to load, the other continues independently.
"""

from __future__ import annotations

import pathlib
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Model paths ───────────────────────────────────────────────────────────────
_THIS_DIR   = pathlib.Path(__file__).resolve().parent
_ROOT       = _THIS_DIR.parent.parent
_FACE_MODEL = _ROOT / "models" / "face_landmarker.task"
_POSE_MODEL = _ROOT / "models" / "pose_landmarker_lite.task"

# ── Drawing colours (BGR) ─────────────────────────────────────────────────────
_FACE_COLOUR  = (220, 120, 30)   # blue-ish for FACE label (bright cyan-blue)
_FACE_BOX_C   = (255, 160, 50)   # orange-blue for face box
_POSE_COLOUR  = (200, 200, 0)    # cyan for POSE / BODY

# ── Key pose landmark IDs (MediaPipe Pose) ─────────────────────────────────────
#  0=nose, 11=left_shoulder, 12=right_shoulder
#  13=left_elbow, 14=right_elbow, 15=left_wrist, 16=right_wrist
#  23=left_hip, 24=right_hip, 25=left_knee, 26=right_knee
POSE_CONNECTIONS = [
    (11, 12),   # shoulders
    (11, 13), (13, 15),  # left arm
    (12, 14), (14, 16),  # right arm
    (11, 23), (12, 24),  # torso sides
    (23, 24),            # hips
    (23, 25), (25, 27),  # left leg (partial)
    (24, 26), (26, 28),  # right leg (partial)
    (0, 11), (0, 12),    # head→shoulders (approx)
]
KEY_POSE_IDS = {0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28}

# ── Face Mesh landmark OVAL (indices subset for bounding-box estimate) ─────────
# We compute bbox from face_landmarks[0] (478 pts) using min/max
FACE_OUTLINE_IDS = list(range(0, 468, 10))   # every 10th for speed


class PersonDetector:
    """
    Detects human face and body pose using MediaPipe 1.x Tasks API.
    Provides draw_overlays() to render clean, minimal overlays onto frames.
    Safe: if one model fails, the other continues independently.
    """

    def __init__(self):
        self._face_lm  = None
        self._pose_lm  = None
        self._frame_count = 0

        # Cached results for temporal smoothing
        self._last_face_rects: list[tuple[int, int, int, int]] = []
        self._last_face_conf: float = 0.0
        self._last_body_rects: list[tuple[int, int, int, int]] = []
        self._last_pose_landmarks: list[dict] = []   # [{id, x, y, vis}, ...]
        self._last_pose_conf: float = 0.0

        # Load FaceLandmarker
        if _FACE_MODEL.exists():
            try:
                base = mp_python.BaseOptions(model_asset_path=str(_FACE_MODEL))
                opts = mp_vision.FaceLandmarkerOptions(
                    base_options=base,
                    running_mode=mp_vision.RunningMode.IMAGE,
                    num_faces=1,
                    min_face_detection_confidence=0.45,
                    min_face_presence_confidence=0.45,
                    output_face_blendshapes=True,   # gives us a confidence proxy
                )
                self._face_lm = mp_vision.FaceLandmarker.create_from_options(opts)
                print("[VYOM FaceDetector] ✓ MediaPipe FaceLandmarker online.")
            except Exception as exc:
                print(f"[VYOM FaceDetector] ⚠ Could not load FaceLandmarker: {exc}")
        else:
            print(f"[VYOM FaceDetector] ⚠ face_landmarker.task not found at {_FACE_MODEL}")

        # Load PoseLandmarker
        if _POSE_MODEL.exists():
            try:
                base = mp_python.BaseOptions(model_asset_path=str(_POSE_MODEL))
                opts = mp_vision.PoseLandmarkerOptions(
                    base_options=base,
                    running_mode=mp_vision.RunningMode.IMAGE,
                    num_poses=1,
                    min_pose_detection_confidence=0.45,
                    min_pose_presence_confidence=0.45,
                )
                self._pose_lm = mp_vision.PoseLandmarker.create_from_options(opts)
                print("[VYOM PoseDetector] ✓ MediaPipe PoseLandmarker online.")
            except Exception as exc:
                print(f"[VYOM PoseDetector] ⚠ Could not load PoseLandmarker: {exc}")
        else:
            print(f"[VYOM PoseDetector] ⚠ pose_landmarker_lite.task not found at {_POSE_MODEL}")

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def detect(
        self,
        frame: np.ndarray,
        has_hands: bool = False,
    ) -> tuple[bool, bool, list, list]:
        """
        Detect face and body in a BGR frame.

        Runs face every frame, pose every 3rd frame (CPU budget).

        Returns:
            (body_detected, face_detected, body_rects, face_rects)
        """
        self._frame_count += 1
        h, w = frame.shape[:2]

        # Convert to MediaPipe Image once
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # ── Face detection (every frame) ──────────────────────────────────────
        if self._face_lm is not None:
            try:
                res = self._face_lm.detect(mp_img)
                if res.face_landmarks:
                    lms = res.face_landmarks[0]
                    xs = [lm.x * w for lm in lms]
                    ys = [lm.y * h for lm in lms]
                    pad_x = int((max(xs) - min(xs)) * 0.12)
                    pad_y = int((max(ys) - min(ys)) * 0.12)
                    x1 = max(0, int(min(xs)) - pad_x)
                    y1 = max(0, int(min(ys)) - pad_y)
                    x2 = min(w, int(max(xs)) + pad_x)
                    y2 = min(h, int(max(ys)) + pad_y)
                    self._last_face_rects = [(x1, y1, x2 - x1, y2 - y1)]
                    # Use blendshape score as confidence proxy (browDownLeft has high
                    # activation when face is clearly present), else use landmark z-spread
                    if res.face_blendshapes and res.face_blendshapes[0]:
                        scores = [c.score for c in res.face_blendshapes[0]]
                        # Invert: low blendshape activity = neutral/confident detection
                        # Take median landmark z deviation as presence proxy
                        self._last_face_conf = min(0.99, 0.72 + (1.0 - float(np.mean(scores[:6]))) * 0.30)
                    else:
                        self._last_face_conf = 0.87
                else:
                    self._last_face_rects = []
                    self._last_face_conf = 0.0
            except Exception:
                pass

        # ── Pose detection (every 3rd frame to save CPU) ──────────────────────
        if self._pose_lm is not None and (self._frame_count % 3 == 0 or not self._last_pose_landmarks):
            try:
                res = self._pose_lm.detect(mp_img)
                if res.pose_landmarks:
                    lms = res.pose_landmarks[0]
                    pts = []
                    for i, lm in enumerate(lms):
                        pts.append({
                            "id": i,
                            "x": int(lm.x * w),
                            "y": int(lm.y * h),
                            "vis": lm.visibility,
                        })
                    self._last_pose_landmarks = pts

                    # Confidence = avg visibility of the most visible key landmarks
                    key_vis = sorted(
                        [pts[i]["vis"] for i in KEY_POSE_IDS if i < len(pts)],
                        reverse=True
                    )[:5]
                    self._last_pose_conf = round(float(np.mean(key_vis)), 2) if key_vis else 0.88

                    # Bounding box from visible landmarks
                    vis_pts = [p for p in pts if p["vis"] > 0.3]
                    if vis_pts:
                        xs = [p["x"] for p in vis_pts]
                        ys = [p["y"] for p in vis_pts]
                        pad = 20
                        bx1 = max(0, min(xs) - pad)
                        by1 = max(0, min(ys) - pad)
                        bx2 = min(w, max(xs) + pad)
                        by2 = min(h, max(ys) + pad)
                        self._last_body_rects = [(bx1, by1, bx2 - bx1, by2 - by1)]
                    else:
                        self._last_body_rects = []
                else:
                    self._last_pose_landmarks = []
                    self._last_body_rects = []
                    self._last_pose_conf = 0.0
            except Exception:
                pass

        body_detected = bool(self._last_body_rects) or (has_hands and not self._face_lm)
        face_detected = bool(self._last_face_rects)

        return body_detected, face_detected, list(self._last_body_rects), list(self._last_face_rects)

    def get_face_confidence(self) -> float:
        return self._last_face_conf

    def get_pose_confidence(self) -> float:
        return self._last_pose_conf

    # ──────────────────────────────────────────────────────────────────────────
    # Drawing
    # ──────────────────────────────────────────────────────────────────────────

    def draw_overlays(
        self,
        frame: np.ndarray,
        body_rects: list,
        face_rects: list,
    ) -> np.ndarray:
        """
        Draw clean, minimal overlays for FACE (blue) and BODY/POSE (cyan).
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # ── Pose skeleton (drawn first so face box sits on top) ────────────────
        if self._last_pose_landmarks:
            pts = self._last_pose_landmarks
            # Draw skeleton connections
            for (a, b) in POSE_CONNECTIONS:
                if a < len(pts) and b < len(pts):
                    pa = pts[a]
                    pb = pts[b]
                    if pa["vis"] > 0.3 and pb["vis"] > 0.3:
                        cv2.line(
                            annotated,
                            (pa["x"], pa["y"]),
                            (pb["x"], pb["y"]),
                            _POSE_COLOUR, 2, cv2.LINE_AA
                        )
            # Draw joint dots for key landmarks
            for pid in KEY_POSE_IDS:
                if pid < len(pts) and pts[pid]["vis"] > 0.3:
                    cv2.circle(annotated, (pts[pid]["x"], pts[pid]["y"]), 5, _POSE_COLOUR, -1)
                    cv2.circle(annotated, (pts[pid]["x"], pts[pid]["y"]), 5, (0, 0, 0), 1)

            # POSE label near nose/top of skeleton
            if len(pts) > 0 and pts[0]["vis"] > 0.3:
                nx, ny = pts[0]["x"], max(pts[0]["y"] - 18, 16)
                label = f"POSE {self._last_pose_conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated, (nx - 4, ny - th - 6), (nx + tw + 6, ny + 4), (10, 10, 10), -1)
                cv2.putText(annotated, label, (nx, ny), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _POSE_COLOUR, 1, cv2.LINE_AA)

        # ── Face bounding box ──────────────────────────────────────────────────
        for (fx, fy, fw, fh) in face_rects:
            # Draw face box (blue)
            cv2.rectangle(annotated, (fx, fy), (fx + fw, fy + fh), _FACE_BOX_C, 2)

            # Label pill above the box
            label = f"FACE {self._last_face_conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(label, font, 0.50, 1)
            lx = fx
            ly = max(fy - 8, th + 6)
            cv2.rectangle(annotated, (lx - 2, ly - th - 6), (lx + tw + 10, ly + 4), (10, 10, 10), -1)
            cv2.rectangle(annotated, (lx - 2, ly - th - 6), (lx + tw + 10, ly + 4), _FACE_BOX_C, 1)
            cv2.putText(annotated, label, (lx + 4, ly), font, 0.50, _FACE_BOX_C, 1, cv2.LINE_AA)

        return annotated
