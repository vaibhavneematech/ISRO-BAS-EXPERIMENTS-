"""
test_hsv_camera.py
==================
SIH26174 – AI Human Activity Recognition for On-board BAS Experiments
Phase 1 Test Script: Webcam + Red/Yellow Box Detection

Usage
-----
Run from the project root directory:

    python scripts/test_hsv_camera.py

Press  'q'        to quit the live window.
Press  'd'        to toggle display of the debug HSV masks (red + yellow).

What this script does
---------------------
1. Opens the default laptop webcam (index 0).
2. For every captured frame:
   a. Calls detect_boxes() from hsv_detector.py.
   b. Draws bounding boxes and labels with draw_detections().
   c. Overlays the current FPS in the top-left corner.
   d. Prints a timestamped line to the terminal whenever a box is detected.
3. Optionally shows the binary masks side-by-side (press 'd').
4. Exits cleanly when 'q' is pressed or the webcam read fails.
"""

import sys
import time
import cv2

# ---------------------------------------------------------------------------
# Make sure the project root (parent of 'scripts/') is on sys.path so that
# "src.detection.hsv_detector" can be imported without installation.
# ---------------------------------------------------------------------------
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detection.hsv_detector import (
    detect_boxes,
    draw_detections,
    build_red_mask,
    build_yellow_mask,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CAMERA_INDEX  = 0          # 0 = first webcam; change if you have multiple
WINDOW_TITLE  = "SIH26174 – Phase 1 | HSV Box Detection  [q=quit  d=masks]"
FONT          = cv2.FONT_HERSHEY_SIMPLEX


def overlay_fps(frame, fps: float) -> None:
    """Draw FPS counter in the top-left corner of frame (in-place)."""
    text = f"FPS: {fps:.1f}"
    cv2.putText(frame, text, (10, 28), FONT, 0.75, (0, 0, 0),   3, cv2.LINE_AA)
    cv2.putText(frame, text, (10, 28), FONT, 0.75, (0, 255, 0), 1, cv2.LINE_AA)


def overlay_instructions(frame) -> None:
    """Draw keyboard hint at the bottom of the frame (in-place)."""
    h = frame.shape[0]
    cv2.putText(frame, "q: quit    d: toggle masks",
                (10, h - 12), FONT, 0.5, (0, 0, 0),     2, cv2.LINE_AA)
    cv2.putText(frame, "q: quit    d: toggle masks",
                (10, h - 12), FONT, 0.5, (220, 220, 220), 1, cv2.LINE_AA)


def print_detections(detections: list, frame_id: int) -> None:
    """
    Print detected boxes to the terminal with a timestamp.
    Only prints if at least one box was found in this frame.
    """
    if not detections:
        return
    ts = time.strftime("%H:%M:%S")
    for det in detections:
        x, y, w, h = det["bbox"]
        print(
            f"[{ts}] frame={frame_id:06d}  DETECTED "
            f"{det['colour'].upper()} box  "
            f"bbox=({x},{y},{w},{h})  "
            f"area={int(det['area'])} px^2  "
            f"centre=({det['cx']},{det['cy']})"
        )


def build_debug_panel(frame, red_mask, yellow_mask) -> "np.ndarray | None":
    """
    Build a side-by-side debug view of the two binary masks.
    Masks are converted to BGR and stacked horizontally.
    Returns None if frame shape does not allow it.
    """
    import numpy as np

    h, w = frame.shape[:2]
    # Resize masks to half width so they sit beside the main view
    half_w = w // 2
    half_h = h // 2

    red_vis    = cv2.resize(red_mask,    (half_w, half_h))
    yellow_vis = cv2.resize(yellow_mask, (half_w, half_h))

    # Convert single-channel masks to 3-channel BGR for concatenation
    red_bgr    = cv2.cvtColor(red_vis,    cv2.COLOR_GRAY2BGR)
    yellow_bgr = cv2.cvtColor(yellow_vis, cv2.COLOR_GRAY2BGR)

    # Label the mask panels
    cv2.putText(red_bgr,    "RED mask",    (4, 18), FONT, 0.5, (80, 80, 255), 1)
    cv2.putText(yellow_bgr, "YELLOW mask", (4, 18), FONT, 0.5, (0, 200, 220), 1)

    debug_row = np.hstack([red_bgr, yellow_bgr])
    return debug_row


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)  # CAP_DSHOW = faster on Windows
    if not cap.isOpened():
        # Fallback: try without backend hint
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam at index {CAMERA_INDEX}. "
              "Check if another application is using it.")
        sys.exit(1)

    # Prefer 1280x720 – will silently use whatever the camera supports
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("=" * 60)
    print("SIH26174 – Phase 1  |  HSV Box Detection Test")
    print("=" * 60)
    print(f"Camera opened  : index {CAMERA_INDEX}")
    print(f"Resolution     : {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))} x "
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print("Controls       : q = quit   |   d = toggle debug masks")
    print("-" * 60)
    print("Waiting for RED or YELLOW boxes in camera view …")
    print()

    show_masks  = False     # toggled with 'd'
    frame_id    = 0
    fps         = 0.0
    t_prev      = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Failed to read frame from webcam. Retrying …")
            time.sleep(0.05)
            continue

        frame_id += 1

        # ------------------------------------------------------------------ #
        # Detection
        # ------------------------------------------------------------------ #
        detections = detect_boxes(frame)
        print_detections(detections, frame_id)

        # ------------------------------------------------------------------ #
        # Annotation
        # ------------------------------------------------------------------ #
        annotated = draw_detections(frame, detections)
        overlay_fps(annotated, fps)
        overlay_instructions(annotated)

        # Detection count badge
        if detections:
            badge = f"{len(detections)} box(es) detected"
            cv2.putText(annotated, badge, (10, 58), FONT, 0.65,
                        (0, 0, 0),   3, cv2.LINE_AA)
            cv2.putText(annotated, badge, (10, 58), FONT, 0.65,
                        (0, 255, 255), 1, cv2.LINE_AA)

        # ------------------------------------------------------------------ #
        # Debug mask panel (optional)
        # ------------------------------------------------------------------ #
        if show_masks:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            red_mask    = build_red_mask(hsv)
            yellow_mask = build_yellow_mask(hsv)
            debug_panel = build_debug_panel(frame, red_mask, yellow_mask)
            if debug_panel is not None:
                import numpy as np
                # Stack annotated frame on top, debug panel below
                # Make debug panel width match annotated frame
                dw = annotated.shape[1]
                dp_resized = cv2.resize(debug_panel,
                                        (dw, debug_panel.shape[0]))
                combined = np.vstack([annotated, dp_resized])
                cv2.imshow(WINDOW_TITLE, combined)
            else:
                cv2.imshow(WINDOW_TITLE, annotated)
        else:
            cv2.imshow(WINDOW_TITLE, annotated)

        # ------------------------------------------------------------------ #
        # FPS calculation (rolling, updated every frame)
        # ------------------------------------------------------------------ #
        t_now  = time.perf_counter()
        fps    = 1.0 / max(t_now - t_prev, 1e-6)
        t_prev = t_now

        # ------------------------------------------------------------------ #
        # Key handling
        # ------------------------------------------------------------------ #
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n[INFO] 'q' pressed – exiting.")
            break
        elif key == ord('d'):
            show_masks = not show_masks
            state = "ON" if show_masks else "OFF"
            print(f"[INFO] Debug mask view {state}")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Camera released. Goodbye.")


if __name__ == "__main__":
    main()
