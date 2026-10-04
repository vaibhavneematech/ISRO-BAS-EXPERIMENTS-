"""
test_hsv_and_hands.py
=====================
SIH26174 – AI Human Activity Recognition for On-board BAS Experiments
Phase 2 Test Script: HSV Box Detection + MediaPipe Hands (combined)

Usage
-----
Run from the project root directory:

    python scripts/test_hsv_and_hands.py

Controls
--------
  q  – quit
  d  – toggle HSV debug masks (red + yellow binary views below main feed)
  h  – toggle hand landmark overlay (on by default)
  b  – toggle HSV box overlay (on by default)

What this script does
---------------------
1. Opens the default laptop webcam.
2. On each frame:
   a. Runs HSV colour detection  → red/yellow bounding boxes
   b. Runs MediaPipe Hands       → 21 hand landmarks per hand
   c. Draws both overlays on the same frame
   d. Shows live FPS, detection counts, and a help bar
   e. Prints terminal lines when boxes OR hands are newly detected
3. Everything runs fully offline – no network access needed.
"""

import sys
import time
import cv2

# ── Make project root importable ─────────────────────────────────────────────
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Phase 1 imports ───────────────────────────────────────────────────────────
from src.detection.hsv_detector import (
    detect_boxes,
    draw_detections,
    build_red_mask,
    build_yellow_mask,
)

# ── Phase 2 imports ───────────────────────────────────────────────────────────
from src.detection.mediapipe_hands import (
    HandDetector,
    draw_hands,
    count_extended_fingers,
)


# ── Configuration ─────────────────────────────────────────────────────────────
CAMERA_INDEX = 0
WINDOW_TITLE = (
    "SIH26174 – Phase 2 | HSV + MediaPipe Hands  "
    "[q=quit  d=masks  h=hands  b=boxes]"
)
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ── Overlay helpers ───────────────────────────────────────────────────────────

def overlay_fps(frame: "np.ndarray", fps: float) -> None:
    """Green FPS counter top-left (in-place)."""
    text = f"FPS: {fps:.1f}"
    cv2.putText(frame, text, (10, 28), FONT, 0.75, (0, 0, 0),   3, cv2.LINE_AA)
    cv2.putText(frame, text, (10, 28), FONT, 0.75, (0, 255, 0), 1, cv2.LINE_AA)


def overlay_status_bar(
    frame: "np.ndarray",
    box_count: int,
    hand_count: int,
    finger_counts: list,
) -> None:
    """
    Draw a semi-transparent status bar at the bottom of the frame showing
    box count, hand count, and per-hand finger estimates.
    """
    import numpy as np
    h, w = frame.shape[:2]
    bar_h = 36

    # Semi-transparent dark bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Build status text
    finger_str = "  ".join(
        [f"Hand{i+1}:{f}f" for i, f in enumerate(finger_counts)]
    ) if finger_counts else ""
    status = (
        f"Boxes: {box_count}   Hands: {hand_count}"
        + (f"   {finger_str}" if finger_str else "")
        + "      [q] quit  [d] masks  [h] hands  [b] boxes"
    )
    cv2.putText(frame, status, (10, h - 10), FONT, 0.48,
                (200, 200, 200), 1, cv2.LINE_AA)


def build_debug_panel(frame: "np.ndarray", red_mask, yellow_mask) -> "np.ndarray":
    """Small side-by-side mask panel stacked below the main view."""
    import numpy as np
    h, w = frame.shape[:2]
    half_w, half_h = w // 2, h // 3

    red_vis    = cv2.resize(red_mask,    (half_w, half_h))
    yellow_vis = cv2.resize(yellow_mask, (half_w, half_h))

    red_bgr    = cv2.cvtColor(red_vis,    cv2.COLOR_GRAY2BGR)
    yellow_bgr = cv2.cvtColor(yellow_vis, cv2.COLOR_GRAY2BGR)

    cv2.putText(red_bgr,    "RED mask",    (4, 18), FONT, 0.5, (80, 80, 255), 1)
    cv2.putText(yellow_bgr, "YELLOW mask", (4, 18), FONT, 0.5, (0, 200, 220), 1)

    debug_row = np.hstack([red_bgr, yellow_bgr])
    return cv2.resize(debug_row, (w, half_h))


def print_terminal(frame_id: int, boxes: list, hands: list) -> None:
    """
    Print a timestamped summary line for each detected box and hand.
    Only fires when something is actually detected.
    """
    ts = time.strftime("%H:%M:%S")
    for det in boxes:
        x, y, w, h = det["bbox"]
        print(
            f"[{ts}] frame={frame_id:06d}  BOX   {det['colour'].upper():<6}  "
            f"bbox=({x},{y},{w},{h})  area={int(det['area'])} px^2"
        )
    for det in hands:
        fingers = count_extended_fingers(det)
        wx, wy  = det["wrist_px"]
        print(
            f"[{ts}] frame={frame_id:06d}  HAND  {det['handedness']:<5}  "
            f"conf={det['confidence']:.2f}  "
            f"wrist=({wx},{wy})  fingers={fingers}"
        )


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    import numpy as np

    # ── Open webcam ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)   # fallback (no backend hint)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam at index {CAMERA_INDEX}.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("=" * 65)
    print("SIH26174 – Phase 2  |  HSV Box Detection + MediaPipe Hands")
    print("=" * 65)
    print(f"Camera  : index {CAMERA_INDEX}  "
          f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))} x "
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print("Controls: q=quit   d=masks   h=hands   b=boxes")
    print("-" * 65)

    # ── Create MediaPipe detector (created ONCE, reused every frame) ─────────
    hand_detector = HandDetector(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )

    # ── Runtime flags ────────────────────────────────────────────────────────
    show_masks   = False
    show_hands   = True
    show_boxes   = True

    frame_id     = 0
    fps          = 0.0
    t_prev       = time.perf_counter()

    # ── Frame loop ───────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Failed to read frame – retrying …")
            time.sleep(0.05)
            continue

        frame_id += 1

        # ── Detection ────────────────────────────────────────────────────────
        # Phase 1: HSV colour boxes
        box_detections  = detect_boxes(frame) if show_boxes else []

        # Phase 2: MediaPipe hands
        hand_detections = hand_detector.detect(frame) if show_hands else []

        # Terminal output
        print_terminal(frame_id, box_detections, hand_detections)

        # ── Rendering ────────────────────────────────────────────────────────
        # Start with annotated frame (draw boxes first, hands on top)
        annotated = draw_detections(frame, box_detections)
        annotated = draw_hands(annotated, hand_detections)

        # Finger counts for status bar
        finger_counts = [count_extended_fingers(h) for h in hand_detections]

        # FPS counter
        overlay_fps(annotated, fps)

        # Status bar
        overlay_status_bar(
            annotated,
            len(box_detections),
            len(hand_detections),
            finger_counts,
        )

        # ── Debug mask panel (optional) ──────────────────────────────────────
        if show_masks:
            hsv         = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            red_mask    = build_red_mask(hsv)
            yellow_mask = build_yellow_mask(hsv)
            debug       = build_debug_panel(frame, red_mask, yellow_mask)
            combined    = np.vstack([annotated, debug])
            cv2.imshow(WINDOW_TITLE, combined)
        else:
            cv2.imshow(WINDOW_TITLE, annotated)

        # ── FPS ──────────────────────────────────────────────────────────────
        t_now  = time.perf_counter()
        fps    = 1.0 / max(t_now - t_prev, 1e-6)
        t_prev = t_now

        # ── Key handling ─────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n[INFO] 'q' pressed – exiting.")
            break
        elif key == ord('d'):
            show_masks = not show_masks
            print(f"[INFO] Debug masks {'ON' if show_masks else 'OFF'}")
        elif key == ord('h'):
            show_hands = not show_hands
            print(f"[INFO] Hand overlay {'ON' if show_hands else 'OFF'}")
        elif key == ord('b'):
            show_boxes = not show_boxes
            print(f"[INFO] Box overlay {'ON' if show_boxes else 'OFF'}")

    # ── Cleanup ──────────────────────────────────────────────────────────────
    hand_detector.close()
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Camera released. Goodbye.")


if __name__ == "__main__":
    main()
