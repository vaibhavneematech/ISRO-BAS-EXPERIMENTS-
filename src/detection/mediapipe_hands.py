"""
mediapipe_hands.py
==================
SIH26174 – AI Human Activity Recognition for On-board BAS Experiments
Phase 2: MediaPipe Hands detection wrapper.

Author : SIH26174 Team
Created: 2026-09-21

Design notes
------------
* Uses the MediaPipe 1.x Tasks API (HandLandmarker).
  The legacy mp.solutions API was removed in MediaPipe >= 0.10.x.
* Requires the model file:  models/hand_landmarker.task
  (downloaded once during project setup – fully offline afterwards).
* The HandDetector class wraps HandLandmarker in VIDEO running mode,
  which is more accurate than IMAGE mode for webcam streams because
  it tracks landmarks across frames.
* detect() accepts a BGR frame and a monotonically increasing
  timestamp_ms – both required by the Tasks API.
* draw_hands() manually renders landmarks and connections using plain
  OpenCV drawing calls (no dependency on mp.solutions.drawing_utils).

Coordinate convention
---------------------
MediaPipe returns normalised (0.0–1.0) x, y.
We always convert to pixel integers before returning.

Landmark IDs (0–20)
-------------------
  0  = WRIST
  1–4  = thumb  (CMC, MCP, IP, TIP)
  5–8  = index  (MCP, PIP, DIP, TIP)
  9–12 = middle (MCP, PIP, DIP, TIP)
 13–16 = ring   (MCP, PIP, DIP, TIP)
 17–20 = pinky  (MCP, PIP, DIP, TIP)
"""

from __future__ import annotations

import pathlib
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Model path ────────────────────────────────────────────────────────────────
# Resolved relative to this file so the project can be placed anywhere.
_THIS_DIR  = pathlib.Path(__file__).resolve().parent
_MODEL_PATH = _THIS_DIR.parent.parent / "models" / "hand_landmarker.task"

# ── Hand skeleton connections (21 landmark pairs) ─────────────────────────────
HAND_CONNECTIONS: list[tuple[int, int]] = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm cross-connections
    (5, 9), (9, 13), (13, 17),
]

# Landmark IDs that are fingertips (highlighted with a larger dot)
FINGERTIP_IDS = {4, 8, 12, 16, 20}

# BGR colours
_LANDMARK_COLOUR   = (0,   255, 180)   # cyan-green dots
_CONNECTION_COLOUR = (255, 200,   0)   # amber lines
_FINGERTIP_COLOUR  = (0,   255, 255)   # bright yellow fingertip dots
_LABEL_COLOUR      = (50,  255,  50)   # green wrist label


class HandDetector:
    """
    Stateful MediaPipe 1.x HandLandmarker wrapper for live video.

    Create ONE instance, call detect() on every frame with an
    ever-increasing timestamp, then call close() when done.

    Parameters
    ----------
    model_path : str | pathlib.Path, optional
        Path to hand_landmarker.task model file.
        Defaults to  <project_root>/models/hand_landmarker.task.
    max_num_hands : int
        Maximum simultaneous hands to detect (default 2).
    min_detection_confidence : float
        Hand detection confidence threshold (default 0.7).
    min_tracking_confidence : float
        Landmark tracking confidence threshold (default 0.6).
    """

    def __init__(
        self,
        model_path: str | pathlib.Path = _MODEL_PATH,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.6,
    ):
        model_path = pathlib.Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"[HandDetector] Model file not found: {model_path}\n"
                "Download it with:\n"
                "  Invoke-WebRequest -Uri https://storage.googleapis.com/"
                "mediapipe-models/hand_landmarker/hand_landmarker/float16/"
                "latest/hand_landmarker.task -OutFile models/hand_landmarker.task"
            )

        base_opts = mp_python.BaseOptions(model_asset_path=str(model_path))

        opts = mp_vision.HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.VIDEO,   # tracks across frames
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self._landmarker = mp_vision.HandLandmarker.create_from_options(opts)
        self._start_ms   = int(time.monotonic() * 1000)   # epoch for timestamps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, bgr_frame: np.ndarray, timestamp_ms: int | None = None) -> list:
        """
        Run hand landmark inference on a BGR frame.

        Parameters
        ----------
        bgr_frame    : np.ndarray  – OpenCV BGR frame (not modified).
        timestamp_ms : int, optional
            Monotonically increasing timestamp in milliseconds.
            If omitted, wall-clock time since construction is used.

        Returns
        -------
        list of dict
            One dict per detected hand:
                hand_index   – 0-based index within this frame
                handedness   – "Left" or "Right"
                confidence   – float 0-1
                landmarks    – list of 21 dicts:
                                   id, x (px), y (px), z (float)
                wrist_px     – (x, y) pixel position of wrist
        """
        if timestamp_ms is None:
            timestamp_ms = int(time.monotonic() * 1000) - self._start_ms

        h, w = bgr_frame.shape[:2]

        # MediaPipe Image requires RGB
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.hand_landmarks:
            return []

        detections = []
        for idx, (hand_lm, hand_info) in enumerate(
            zip(result.hand_landmarks, result.handedness)
        ):
            handedness = hand_info[0].display_name   # "Left" or "Right"
            confidence = hand_info[0].score

            # Convert normalised coords to pixel coords
            landmarks = []
            for lm_id, lm in enumerate(hand_lm):
                landmarks.append({
                    "id": lm_id,
                    "x":  int(lm.x * w),
                    "y":  int(lm.y * h),
                    "z":  lm.z,
                })

            wrist = (landmarks[0]["x"], landmarks[0]["y"])

            detections.append({
                "hand_index": idx,
                "handedness": handedness,
                "confidence": confidence,
                "landmarks":  landmarks,
                "wrist_px":   wrist,
            })

        return detections

    def close(self) -> None:
        """Release model resources. Must be called when done."""
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ------------------------------------------------------------------
# Drawing helpers (module-level, stateless)
# ------------------------------------------------------------------

def draw_hands(frame: np.ndarray, hand_detections: list) -> np.ndarray:
    """
    Draw hand landmarks, skeleton, and wrist labels on a copy of frame.

    Parameters
    ----------
    frame           : np.ndarray  – BGR frame.
    hand_detections : list        – output of HandDetector.detect().

    Returns
    -------
    np.ndarray
        Annotated BGR copy (original not modified).
    """
    annotated = frame.copy()

    for det in hand_detections:
        lm = det["landmarks"]

        # ── Draw connections ─────────────────────────────────────────
        for (a, b) in HAND_CONNECTIONS:
            pt_a = (lm[a]["x"], lm[a]["y"])
            pt_b = (lm[b]["x"], lm[b]["y"])
            cv2.line(annotated, pt_a, pt_b, _CONNECTION_COLOUR, 2, cv2.LINE_AA)

        # ── Draw landmark dots ───────────────────────────────────────
        for point in lm:
            cx, cy = point["x"], point["y"]
            if point["id"] in FINGERTIP_IDS:
                # Larger bright dot for fingertips
                cv2.circle(annotated, (cx, cy), 7, _FINGERTIP_COLOUR, -1)
                cv2.circle(annotated, (cx, cy), 7, (0, 0, 0), 1)       # outline
            else:
                cv2.circle(annotated, (cx, cy), 4, _LANDMARK_COLOUR, -1)

        # ── Wrist label: "Left 0.94" ─────────────────────────────────
        wx, wy = det["wrist_px"]
        label  = f"{det['handedness']}  {det['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(
            annotated,
            (wx - 2, wy - th - 10),
            (wx + tw + 8, wy - 2),
            (25, 25, 25), -1,
        )
        cv2.putText(
            annotated, label,
            (wx + 2, wy - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            _LABEL_COLOUR, 1, cv2.LINE_AA,
        )

    return annotated


def count_extended_fingers(hand_det: dict) -> int:
    """
    Estimate how many fingers are extended for a single hand.

    Uses tip-above-pip y-coordinate comparison for fingers 2-5.
    Thumb uses x-axis comparison (mirror-aware via handedness).

    Parameters
    ----------
    hand_det : dict  – single entry from HandDetector.detect()

    Returns
    -------
    int  – number of extended fingers (0-5)
    """
    lm = hand_det["landmarks"]

    # Index, Middle, Ring, Pinky: tip y < pip y means extended (up)
    FINGER_PAIRS = [(8, 6), (12, 10), (16, 14), (20, 18)]
    count = sum(1 for tip, pip in FINGER_PAIRS if lm[tip]["y"] < lm[pip]["y"])

    # Thumb: horizontal comparison
    if hand_det["handedness"] == "Right":
        if lm[4]["x"] < lm[3]["x"]:
            count += 1
    else:
        if lm[4]["x"] > lm[3]["x"]:
            count += 1

    return count
