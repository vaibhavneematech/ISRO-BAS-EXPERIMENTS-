"""
test_live_protocol.py
=====================
SIH26174 – AI Human Activity Recognition for On-board BAS Experiments
Phase 4: Live Integration – Camera + HSV + Hands + State Machine

Usage
-----
Run from the project root:

    python scripts/test_live_protocol.py

Controls
--------
  q  – quit
  r  – reset / restart the experiment protocol
  h  – toggle hand landmark overlay
  d  – toggle HSV debug masks
  SPACE – manually advance past the START step (begin experiment)

Pipeline (every frame)
----------------------
  1. Capture BGR frame from webcam
  2. HSV detector  → list of {colour, bbox, area, cx, cy}
  3. Hand detector → list of {handedness, landmarks, wrist_px, ...}
  4. Mapper        → frozenset of "present colours" for the FSM
  5. StateMachine.update(box_dets) → TransitionStatus
  6. Overlay all info on frame + display
  7. Log important state changes to terminal

State Machine mapping logic
---------------------------
The FSM only cares about WHICH COLOURS are present, not hands.
Hands serve as a qualitative confirmation overlay for the operator.

  Both red & yellow visible  → DETECT_MAIN_BOX / PLACE_YELLOW_BOX triggers
  Only yellow visible        → REMOVE_RED_BOX trigger
  Neither visible            → REMOVE_YELLOW_BOX trigger
  Only red visible           → PLACE_RED_BOX trigger
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ── Project imports ───────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.detection.hsv_detector    import detect_boxes, draw_detections, build_red_mask, build_yellow_mask
from src.detection.mediapipe_hands import HandDetector, draw_hands
from src.protocol.state_machine    import StateMachine, TransitionStatus, FSMState


# ── Configuration ─────────────────────────────────────────────────────────────
CAMERA_INDEX   = 0
WINDOW_TITLE   = "SIH26174 – Phase 4 | Live Protocol  [q=quit  r=reset  SPACE=start]"
FONT           = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD      = cv2.FONT_HERSHEY_DUPLEX

PROTOCOL_PATH  = ROOT / "config" / "experiment_protocol.json"

# BGR colours used for the HUD
HUD_COLOURS = {
    TransitionStatus.WAITING      : (180, 180, 180),
    TransitionStatus.CORRECT      : (0,   220,  80),
    TransitionStatus.ALREADY_DONE : (200, 200,   0),
    TransitionStatus.WRONG_ORDER  : (0,   165, 255),
    TransitionStatus.SKIPPED      : (0,    50, 220),
    TransitionStatus.COMPLETED    : (0,   255, 128),
    TransitionStatus.NOT_STARTED  : (120, 120, 120),
}

# How many consecutive CORRECT frames to flash the "✓" badge
FLASH_FRAMES = 45


# ── HUD drawing helpers ───────────────────────────────────────────────────────

def _put_shadowed(img, text, pos, scale, colour, thickness=1):
    """Draw text with a dark shadow for readability on any background."""
    x, y = pos
    cv2.putText(img, text, (x+1, y+1), FONT, scale, (0,0,0), thickness+1, cv2.LINE_AA)
    cv2.putText(img, text, (x,   y  ), FONT, scale, colour,  thickness,   cv2.LINE_AA)


def draw_hud(frame: np.ndarray,
             sm: StateMachine,
             last_status: TransitionStatus,
             fps: float,
             hand_count: int,
             box_count: int,
             flash_counter: int) -> None:
    """
    Draw the full HUD (Heads-Up Display) onto `frame` in-place.

    Panels drawn:
      TOP-LEFT     – FPS, box/hand count
      TOP-RIGHT    – Current step + progress bar
      BOTTOM bar   – Next step instruction
      FLASH badge  – "✓ STEP COMPLETE" on correct transition
    """
    h, w = frame.shape[:2]
    status_col = HUD_COLOURS.get(last_status, (180, 180, 180))

    # ── Semi-transparent top panel ───────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # FPS (top-left)
    _put_shadowed(frame, f"FPS: {fps:.1f}", (10, 24), 0.65, (0, 255, 80))

    # Box / hand count
    _put_shadowed(frame, f"Boxes: {box_count}   Hands: {hand_count}",
                  (10, 50), 0.55, (200, 200, 200))

    # ── Current step + progress (top-right) ──────────────────────────────────
    current = sm.get_current_state()
    done, total = sm.get_progress()
    step_text   = f"[{done}/{total}]  {current.name}"
    status_text = last_status.name

    (sw, _), _ = cv2.getTextSize(step_text, FONT, 0.65, 1)
    _put_shadowed(frame, step_text,   (w - sw - 12, 24), 0.65, (255, 220, 80))
    (stw, _), _ = cv2.getTextSize(status_text, FONT, 0.60, 1)
    _put_shadowed(frame, status_text, (w - stw - 12, 50), 0.60, status_col)

    # ── Progress bar (below top panel) ───────────────────────────────────────
    bar_y  = 72
    bar_h  = 6
    bar_w  = w
    filled = int(bar_w * done / max(total, 1))
    cv2.rectangle(frame, (0, bar_y), (bar_w, bar_y + bar_h), (40, 40, 40), -1)
    cv2.rectangle(frame, (0, bar_y), (filled, bar_y + bar_h), (0, 200, 100), -1)

    # ── Bottom instruction bar ────────────────────────────────────────────────
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 50), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay2, 0.60, frame, 0.40, 0, frame)

    if sm.is_completed():
        instruction = "  EXPERIMENT COMPLETE!  Press R to restart."
        inst_col    = (0, 255, 128)
    elif sm.last_status == TransitionStatus.NOT_STARTED:
        instruction = "  Press SPACE to begin experiment"
        inst_col    = (180, 180, 180)
    else:
        nxt = sm.get_next_step()
        instruction = f"  NEXT: {nxt.name} – {nxt.description}" if nxt else "  Completed!"
        inst_col    = (220, 220, 100)

    # Clip instruction text to fit
    max_chars = (w // 7)
    if len(instruction) > max_chars:
        instruction = instruction[:max_chars - 3] + "…"

    _put_shadowed(frame, instruction, (6, h - 28), 0.52, inst_col)

    # Key hints (bottom-right)
    hints = "q:quit  r:reset  h:hands  d:masks  SPACE:start"
    (hw2, _), _ = cv2.getTextSize(hints, FONT, 0.40, 1)
    _put_shadowed(frame, hints, (w - hw2 - 8, h - 10), 0.40, (130, 130, 130))

    # ── CORRECT flash badge ───────────────────────────────────────────────────
    if flash_counter > 0:
        alpha  = min(1.0, flash_counter / (FLASH_FRAMES * 0.4))
        badge  = f"  STEP COMPLETE: {current.name}  "
        (bw, bh), _ = cv2.getTextSize(badge, FONT_BOLD, 0.85, 2)
        bx = (w - bw) // 2
        by = h // 2 - 20
        ov3 = frame.copy()
        cv2.rectangle(ov3, (bx - 8, by - bh - 8), (bx + bw + 8, by + 10),
                      (0, 100, 0), -1)
        cv2.addWeighted(ov3, alpha * 0.75, frame, 1 - alpha * 0.75, 0, frame)
        cv2.putText(frame, badge, (bx, by),
                    FONT_BOLD, 0.85, (0, 255, 100), 2, cv2.LINE_AA)

    # ── SKIPPED / WRONG_ORDER warning badge ──────────────────────────────────
    if last_status in (TransitionStatus.SKIPPED, TransitionStatus.WRONG_ORDER):
        warn_text = (
            "!! WRONG ORDER !!" if last_status == TransitionStatus.WRONG_ORDER
            else "!! STEP SKIPPED !!"
        )
        (ww, wh), _ = cv2.getTextSize(warn_text, FONT_BOLD, 0.9, 2)
        wx = (w - ww) // 2
        wy = h // 2 + 40
        cv2.rectangle(frame, (wx - 8, wy - wh - 8), (wx + ww + 8, wy + 10),
                      (0, 0, 160), -1)
        cv2.putText(frame, warn_text, (wx, wy),
                    FONT_BOLD, 0.9, (0, 80, 255), 2, cv2.LINE_AA)


def build_debug_panel(frame, red_mask, yellow_mask):
    """Compact side-by-side mask view stacked below main frame."""
    h, w = frame.shape[:2]
    half_w = w // 2
    panel_h = h // 4

    r_vis  = cv2.resize(red_mask,    (half_w, panel_h))
    y_vis  = cv2.resize(yellow_mask, (half_w, panel_h))
    r_bgr  = cv2.cvtColor(r_vis, cv2.COLOR_GRAY2BGR)
    y_bgr  = cv2.cvtColor(y_vis, cv2.COLOR_GRAY2BGR)

    cv2.putText(r_bgr, "RED mask",    (4, 18), FONT, 0.5, (80,  80, 255), 1)
    cv2.putText(y_bgr, "YELLOW mask", (4, 18), FONT, 0.5, ( 0, 200, 220), 1)

    return cv2.resize(np.hstack([r_bgr, y_bgr]), (w, panel_h))


# ── Terminal logger ───────────────────────────────────────────────────────────

_last_logged_step = None
_last_logged_status = None

def log_transition(sm: StateMachine, status: TransitionStatus, frame_id: int):
    """Print a terminal line only when something noteworthy changes."""
    global _last_logged_step, _last_logged_status
    current = sm.get_current_state()
    ts = time.strftime("%H:%M:%S")

    # Always log CORRECT, SKIPPED, WRONG_ORDER, COMPLETED
    important = status in (
        TransitionStatus.CORRECT,
        TransitionStatus.SKIPPED,
        TransitionStatus.WRONG_ORDER,
        TransitionStatus.COMPLETED,
    )

    if important and (current.name != _last_logged_step or status != _last_logged_status):
        icon = {
            TransitionStatus.CORRECT      : "✓",
            TransitionStatus.SKIPPED      : "⚠ SKIPPED",
            TransitionStatus.WRONG_ORDER  : "✗ WRONG ORDER",
            TransitionStatus.COMPLETED    : "★ COMPLETED",
        }.get(status, "")

        done, total = sm.get_progress()
        print(
            f"[{ts}] frame={frame_id:06d}  {icon}  "
            f"{current.name}  [{done}/{total}]  "
            f"→  {sm.get_next_step_suggestion()}"
        )
        _last_logged_step   = current.name
        _last_logged_status = status


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    global _last_logged_step, _last_logged_status

    # ── Open webcam ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam at index {CAMERA_INDEX}.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # ── Initialise sub-systems ───────────────────────────────────────────────
    hand_detector = HandDetector(max_num_hands=2)
    sm            = StateMachine(PROTOCOL_PATH)
    sm.start()

    print("=" * 68)
    print("SIH26174 – Phase 4  |  Live Protocol Integration")
    print("=" * 68)
    print(f"Camera      : {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))} x "
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"Protocol    : {PROTOCOL_PATH.name}  ({len(sm.steps)} steps)")
    print("Controls    : q=quit   r=reset   h=hands   d=masks   SPACE=start")
    print("-" * 68)
    print("Press SPACE in the window (or wait for auto-start) to begin.")
    print()

    # ── Runtime flags ────────────────────────────────────────────────────────
    show_hands   = True
    show_masks   = False
    flash_count  = 0          # counts down after a CORRECT transition
    last_status  = TransitionStatus.NOT_STARTED

    frame_id     = 0
    fps          = 0.0
    t_prev       = time.perf_counter()

    # ── Frame loop ───────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame_id += 1

        # ── Phase 1: HSV box detection ────────────────────────────────────
        box_dets = detect_boxes(frame)

        # ── Phase 2: MediaPipe hands ──────────────────────────────────────
        hand_dets = hand_detector.detect(frame) if show_hands else []

        # ── Phase 3: Feed FSM with box and hand detections ────────────────
        status = sm.update(box_dets, hand_dets)

        # Flash counter management
        if status == TransitionStatus.CORRECT:
            flash_count = FLASH_FRAMES
        elif flash_count > 0:
            flash_count -= 1

        last_status = status

        # Terminal log
        log_transition(sm, status, frame_id)

        # ── Rendering ─────────────────────────────────────────────────────
        annotated = draw_detections(frame, box_dets)
        if show_hands:
            annotated = draw_hands(annotated, hand_dets)

        draw_hud(
            annotated, sm, last_status,
            fps, len(hand_dets), len(box_dets), flash_count
        )

        # Optional debug mask panel
        if show_masks:
            hsv        = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            red_mask   = build_red_mask(hsv)
            yel_mask   = build_yellow_mask(hsv)
            debug      = build_debug_panel(frame, red_mask, yel_mask)
            annotated  = np.vstack([annotated, debug])

        cv2.imshow(WINDOW_TITLE, annotated)

        # ── FPS ───────────────────────────────────────────────────────────
        t_now  = time.perf_counter()
        fps    = 1.0 / max(t_now - t_prev, 1e-6)
        t_prev = t_now

        # ── Key handling ──────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\n[INFO] Quit requested.")
            break

        elif key == ord('r'):
            sm.reset()
            last_status = TransitionStatus.NOT_STARTED
            flash_count = 0
            _last_logged_step   = None
            _last_logged_status = None
            print("[INFO] Protocol RESET – experiment restarted from step 0.")

        elif key == ord('h'):
            show_hands = not show_hands
            print(f"[INFO] Hand overlay {'ON' if show_hands else 'OFF'}")

        elif key == ord('d'):
            show_masks = not show_masks
            print(f"[INFO] Debug masks {'ON' if show_masks else 'OFF'}")

        elif key == ord(' '):
            # Manual SPACE press injects a synthetic "both boxes present"
            # to push past the START step if the operator wants to skip setup
            synthetic = [{"colour": "red"}, {"colour": "yellow"}]
            for _ in range(10):        # inject 10 frames worth
                sm.update(synthetic)
            print("[INFO] SPACE pressed – manually advanced past START.")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    hand_detector.close()
    cap.release()
    cv2.destroyAllWindows()

    # Final session summary
    print()
    print("=" * 68)
    print("SESSION SUMMARY")
    print("=" * 68)
    done, total = sm.get_progress()
    print(f"  Final step      : {sm.get_current_state().name}")
    print(f"  Progress        : {done}/{total} steps completed")
    print(f"  Total frames    : {frame_id}")
    print(f"  Transitions     : {len(sm.history)}")
    print()
    if sm.history:
        print("  Full transition log:")
        for rec in sm.history:
            icon = "✓" if rec.status.name == "CORRECT" else "⚠"
            print(
                f"    {icon}  frame {rec.frame_index:05d}  "
                f"{rec.from_step.name:<22} → {rec.to_step.name:<22}  "
                f"{rec.status.name}"
            )
    print()
    print("[INFO] Goodbye.")


if __name__ == "__main__":
    main()
