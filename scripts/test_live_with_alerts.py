"""
test_live_with_alerts.py
========================
SIH26174 – AI Human Activity Recognition for On-board BAS Experiments
Phase 5: Live Integration with Voice Alerts + JSONL Logging

Usage
-----
    python scripts/test_live_with_alerts.py

Controls
--------
  q      – quit
  r      – reset / restart protocol
  h      – toggle hand landmarks
  d      – toggle HSV debug masks
  m      – toggle audio mute
  SPACE  – manually advance past START step

What's new in Phase 5 (vs Phase 4)
-----------------------------------
* Voice alerts on every important state change (non-blocking, daemon thread)
* Full JSONL structured log written to logs/
* SKIPPED / WRONG_ORDER alert fires ONCE per unique error event,
  not every frame (rate-limited by VoiceAlertSystem cooldown +
  event-key deduplication)
* Fixed: terminal log now prints at most 1 line per SKIPPED/WRONG event
  per unique (step, status) pair – no more log floods

Bug fixes vs Phase 4
--------------------
* Terminal logger: only prints SKIPPED/WRONG_ORDER once per contiguous run,
  not every frame. Uses _alert_event_key tracking.
* State machine history: SKIPPED/WRONG records are now only logged once per
  unique (from_step, to_step) pair per session via event-key mechanism.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ── Project root ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Phase 1-4 imports ─────────────────────────────────────────────────────────
from src.detection.hsv_detector    import detect_boxes, draw_detections, build_red_mask, build_yellow_mask
from src.detection.mediapipe_hands import HandDetector, draw_hands
from src.protocol.state_machine    import StateMachine, TransitionStatus

# ── Phase 5 imports ───────────────────────────────────────────────────────────
from src.alerts.voice_alert  import VoiceAlertSystem, AlertType
from src.logging.event_logger import EventLogger, EventType


# ── Configuration ─────────────────────────────────────────────────────────────
CAMERA_INDEX  = 0
WINDOW_TITLE  = (
    "SIH26174 – Phase 5 | Alerts + Logging  "
    "[q=quit  r=reset  m=mute  SPACE=start]"
)
PROTOCOL_PATH = ROOT / "config" / "experiment_protocol.json"
FONT          = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD     = cv2.FONT_HERSHEY_DUPLEX
FLASH_FRAMES  = 45

HUD_COLOURS = {
    TransitionStatus.WAITING      : (180, 180, 180),
    TransitionStatus.CORRECT      : (0,   220,  80),
    TransitionStatus.ALREADY_DONE : (200, 200,   0),
    TransitionStatus.WRONG_ORDER  : (0,   165, 255),
    TransitionStatus.SKIPPED      : (0,    50, 220),
    TransitionStatus.COMPLETED    : (0,   255, 128),
    TransitionStatus.NOT_STARTED  : (120, 120, 120),
}

# Map TransitionStatus → AlertType
_STATUS_TO_ALERT = {
    TransitionStatus.CORRECT      : AlertType.STEP_COMPLETE,
    TransitionStatus.WRONG_ORDER  : AlertType.WRONG_SEQUENCE,
    TransitionStatus.SKIPPED      : AlertType.STEP_SKIPPED,
    TransitionStatus.COMPLETED    : AlertType.EXPERIMENT_COMPLETE,
}

# Map TransitionStatus → EventType
_STATUS_TO_EVENT = {
    TransitionStatus.CORRECT      : EventType.STEP_COMPLETE,
    TransitionStatus.WRONG_ORDER  : EventType.WRONG_ORDER,
    TransitionStatus.SKIPPED      : EventType.STEP_SKIPPED,
    TransitionStatus.COMPLETED    : EventType.PROTOCOL_COMPLETE,
}


# ── HUD drawing ───────────────────────────────────────────────────────────────

def _shadow(img, text, pos, scale, colour, thickness=1):
    x, y = pos
    cv2.putText(img, text, (x+1, y+1), FONT, scale, (0,0,0),  thickness+1, cv2.LINE_AA)
    cv2.putText(img, text, (x,   y  ), FONT, scale, colour, thickness,   cv2.LINE_AA)


def draw_hud(frame, sm, last_status, fps, hand_count, box_count,
             flash_count, muted):
    h, w = frame.shape[:2]
    status_col = HUD_COLOURS.get(last_status, (180, 180, 180))

    # ── Top panel ─────────────────────────────────────────────────────────
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 70), (15, 15, 15), -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)

    mute_str = " [MUTED]" if muted else ""
    _shadow(frame, f"FPS: {fps:.1f}{mute_str}", (10, 24), 0.65,
            (0, 200, 255) if muted else (0, 255, 80))
    _shadow(frame, f"Boxes: {box_count}   Hands: {hand_count}",
            (10, 50), 0.55, (200, 200, 200))

    current = sm.get_current_state()
    done, total = sm.get_progress()
    step_text   = f"[{done}/{total}]  {current.name}"
    status_text = last_status.name
    (sw, _), _  = cv2.getTextSize(step_text, FONT, 0.65, 1)
    _shadow(frame, step_text,   (w - sw - 12, 24), 0.65, (255, 220, 80))
    (stw, _), _ = cv2.getTextSize(status_text, FONT, 0.60, 1)
    _shadow(frame, status_text, (w - stw - 12, 50), 0.60, status_col)

    # ── Progress bar ───────────────────────────────────────────────────────
    bar_y = 72
    filled = int(w * done / max(total, 1))
    cv2.rectangle(frame, (0, bar_y), (w, bar_y + 6), (40, 40, 40), -1)
    cv2.rectangle(frame, (0, bar_y), (filled, bar_y + 6), (0, 200, 100), -1)

    # ── Bottom bar ─────────────────────────────────────────────────────────
    ov2 = frame.copy()
    cv2.rectangle(ov2, (0, h - 50), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(ov2, 0.60, frame, 0.40, 0, frame)

    if sm.is_completed():
        inst    = "  EXPERIMENT COMPLETE!  Press R to restart."
        inst_col = (0, 255, 128)
    elif last_status == TransitionStatus.NOT_STARTED:
        inst    = "  Press SPACE to begin experiment"
        inst_col = (180, 180, 180)
    else:
        nxt = sm.get_next_step()
        inst    = f"  NEXT: {nxt.name} – {nxt.description}" if nxt else "  Done!"
        inst_col = (220, 220, 100)

    max_chars = w // 7
    if len(inst) > max_chars:
        inst = inst[:max_chars - 3] + "…"
    _shadow(frame, inst, (6, h - 28), 0.52, inst_col)

    hints = "q:quit  r:reset  h:hands  d:masks  m:mute  SPACE:start"
    (hw2, _), _ = cv2.getTextSize(hints, FONT, 0.40, 1)
    _shadow(frame, hints, (w - hw2 - 8, h - 10), 0.40, (130, 130, 130))

    # ── CORRECT flash ──────────────────────────────────────────────────────
    if flash_count > 0:
        alpha = min(1.0, flash_count / (FLASH_FRAMES * 0.4))
        badge = f"  STEP COMPLETE: {current.name}  "
        (bw, bh), _ = cv2.getTextSize(badge, FONT_BOLD, 0.85, 2)
        bx, by = (w - bw) // 2, h // 2 - 20
        ov3 = frame.copy()
        cv2.rectangle(ov3, (bx - 8, by - bh - 8), (bx + bw + 8, by + 10),
                      (0, 100, 0), -1)
        cv2.addWeighted(ov3, alpha * 0.75, frame, 1 - alpha * 0.75, 0, frame)
        cv2.putText(frame, badge, (bx, by), FONT_BOLD, 0.85,
                    (0, 255, 100), 2, cv2.LINE_AA)

    # ── Error badge ────────────────────────────────────────────────────────
    if last_status in (TransitionStatus.SKIPPED, TransitionStatus.WRONG_ORDER):
        warn = ("!! WRONG ORDER !!" if last_status == TransitionStatus.WRONG_ORDER
                else "!! STEP SKIPPED !!")
        (ww, wh), _ = cv2.getTextSize(warn, FONT_BOLD, 0.9, 2)
        wx, wy = (w - ww) // 2, h // 2 + 40
        cv2.rectangle(frame, (wx - 8, wy - wh - 8), (wx + ww + 8, wy + 10),
                      (0, 0, 160), -1)
        cv2.putText(frame, warn, (wx, wy), FONT_BOLD, 0.9,
                    (0, 80, 255), 2, cv2.LINE_AA)


def build_debug_panel(frame, red_mask, yellow_mask):
    h, w = frame.shape[:2]
    half_w  = w // 2
    panel_h = h // 4
    r = cv2.resize(red_mask,    (half_w, panel_h))
    y = cv2.resize(yellow_mask, (half_w, panel_h))
    r_bgr = cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)
    y_bgr = cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
    cv2.putText(r_bgr, "RED mask",    (4, 18), FONT, 0.5, (80,  80, 255), 1)
    cv2.putText(y_bgr, "YELLOW mask", (4, 18), FONT, 0.5, ( 0, 200, 220), 1)
    return cv2.resize(np.hstack([r_bgr, y_bgr]), (w, panel_h))


# ── Alert + Log helper ────────────────────────────────────────────────────────

def handle_alert_and_log(
    status: TransitionStatus,
    sm: StateMachine,
    vas: VoiceAlertSystem,
    logger: EventLogger,
    frame_id: int,
    colours: list,
    hand_count: int,
    *,
    terminal_printed: set,      # mutable set – tracks what was printed
) -> None:
    """
    Fire voice alert and write log entry for noteworthy status values.
    Deduplicates terminal prints and voice alerts per unique event key.
    """
    current = sm.get_current_state()
    nxt     = sm.get_next_step()
    nxt_name = nxt.name if nxt else None

    # Build a unique key for this specific event (step + status combo)
    # This ensures SKIPPED only fires once per unique (step → status) transition
    # rather than every frame.
    event_key = f"{status.name}__{current.name}"

    alert_type = _STATUS_TO_ALERT.get(status)
    event_type = _STATUS_TO_EVENT.get(status)

    if alert_type and event_type:
        # ── Voice alert (rate-limited by cooldown + event key) ────────────
        # CORRECT: fire every time (different steps will have different keys)
        # ERROR:   fire once per unique error event key
        if status == TransitionStatus.CORRECT:
            vas.play(alert_type)
        else:
            vas.play_if_new(alert_type, event_key)

        # ── Terminal print (once per unique event key) ─────────────────────
        if event_key not in terminal_printed:
            terminal_printed.add(event_key)
            ts   = time.strftime("%H:%M:%S")
            done, total = sm.get_progress()
            icon = {
                TransitionStatus.CORRECT      : "✓",
                TransitionStatus.WRONG_ORDER  : "✗ WRONG ORDER",
                TransitionStatus.SKIPPED      : "⚠ SKIPPED",
                TransitionStatus.COMPLETED    : "★ COMPLETED",
            }.get(status, "?")
            print(
                f"[{ts}] frame={frame_id:06d}  {icon}  "
                f"{current.name}  [{done}/{total}]"
                + (f"  →  next: {nxt_name}" if nxt_name else "")
            )

        # ── JSONL log (every noteworthy event, not rate-limited) ──────────
        colour_names = [d["colour"] for d in colours]
        messages = {
            TransitionStatus.CORRECT      : f"Step '{current.name}' completed correctly.",
            TransitionStatus.WRONG_ORDER  : f"Wrong order detected at step '{current.name}'.",
            TransitionStatus.SKIPPED      : f"Step skipped while at '{current.name}'.",
            TransitionStatus.COMPLETED    : "Experiment protocol fully completed!",
        }
        logger.log(
            event_type,
            frame=frame_id,
            current_step=current.name,
            next_step=nxt_name,
            status=status.name,
            colours=colour_names,
            hands=hand_count,
            message=messages.get(status, ""),
        )


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    # ── Open webcam ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam at index {CAMERA_INDEX}.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # ── Initialise all sub-systems ───────────────────────────────────────────
    hand_detector = HandDetector(max_num_hands=2)
    sm            = StateMachine(PROTOCOL_PATH)
    vas           = VoiceAlertSystem()
    logger        = EventLogger(session_name="live_protocol")

    sm.start()
    logger.log_session_start(
        protocol_name="BAS_STANDARD_v1",
        total_steps=len(sm.steps),
    )

    print("=" * 68)
    print("SIH26174 – Phase 5  |  Voice Alerts + JSONL Logging")
    print("=" * 68)
    print(f"Camera   : {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))} x "
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"Log file : {logger.log_path}")
    print("Controls : q=quit  r=reset  h=hands  d=masks  m=mute  SPACE=start")
    print("-" * 68)

    # ── Runtime state ─────────────────────────────────────────────────────────
    show_hands   = True
    show_masks   = False
    muted        = False
    flash_count  = 0
    last_status  = TransitionStatus.NOT_STARTED
    terminal_printed: set = set()    # tracks printed event keys (deduplicate)

    frame_id     = 0
    fps          = 0.0
    t_prev       = time.perf_counter()

    # ── Frame loop ─────────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame_id += 1

        # ── Detections ────────────────────────────────────────────────────────
        box_dets  = detect_boxes(frame)
        hand_dets = hand_detector.detect(frame) if show_hands else []

        # ── State Machine update ───────────────────────────────────────────────
        status = sm.update(box_dets, hand_dets)

        # Flash management
        if status == TransitionStatus.CORRECT:
            flash_count = FLASH_FRAMES
            # Clear terminal print set on advance so same step name at new
            # position can print again (e.g. after PLACE_YELLOW_BOX loops)
            terminal_printed.discard(f"CORRECT__{sm.get_current_state().name}")
        elif flash_count > 0:
            flash_count -= 1

        last_status = status

        # ── Voice alert + log ─────────────────────────────────────────────────
        handle_alert_and_log(
            status, sm, vas, logger,
            frame_id, box_dets, len(hand_dets),
            terminal_printed=terminal_printed,
        )

        # ── Render ────────────────────────────────────────────────────────────
        annotated = draw_detections(frame, box_dets)
        if show_hands:
            annotated = draw_hands(annotated, hand_dets)
        draw_hud(annotated, sm, last_status, fps,
                 len(hand_dets), len(box_dets), flash_count, muted)

        if show_masks:
            hsv      = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            r_mask   = build_red_mask(hsv)
            y_mask   = build_yellow_mask(hsv)
            debug    = build_debug_panel(frame, r_mask, y_mask)
            annotated = np.vstack([annotated, debug])

        cv2.imshow(WINDOW_TITLE, annotated)

        # ── FPS ───────────────────────────────────────────────────────────────
        t_now  = time.perf_counter()
        fps    = 1.0 / max(t_now - t_prev, 1e-6)
        t_prev = t_now

        # ── Key handling ──────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\n[INFO] Quit requested.")
            break
        elif key == ord('r'):
            sm.reset()
            vas.reset_fired_keys()
            terminal_printed.clear()
            last_status  = TransitionStatus.NOT_STARTED
            flash_count  = 0
            logger.log(EventType.PROTOCOL_RESET, frame=frame_id,
                       message="Protocol manually reset by operator.")
            print("[INFO] Protocol RESET.")
        elif key == ord('h'):
            show_hands = not show_hands
            print(f"[INFO] Hands {'ON' if show_hands else 'OFF'}")
        elif key == ord('d'):
            show_masks = not show_masks
            print(f"[INFO] Masks {'ON' if show_masks else 'OFF'}")
        elif key == ord('m'):
            muted = not muted
            vas.set_enabled(not muted)
            print(f"[INFO] Audio {'MUTED' if muted else 'UNMUTED'}")
        elif key == ord(' '):
            synthetic = [{"colour": "red"}, {"colour": "yellow"}]
            for _ in range(12):
                sm.update(synthetic)
            print("[INFO] SPACE – manually advanced past START.")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    logger.log_session_end(frame=frame_id, completed=sm.is_completed())
    logger.close()
    hand_detector.close()
    cap.release()
    cv2.destroyAllWindows()

    # ── Session summary ───────────────────────────────────────────────────────
    print()
    print("=" * 68)
    print("SESSION SUMMARY")
    print("=" * 68)
    done, total = sm.get_progress()
    print(f"  Final step   : {sm.get_current_state().name}")
    print(f"  Progress     : {done}/{total} steps")
    print(f"  Total frames : {frame_id}")
    print(f"  Log entries  : {logger.entry_count}")
    print(f"  Log file     : {logger.log_path}")
    if sm.history:
        print()
        print("  Transition log:")
        for rec in sm.history:
            icon = "✓" if rec.status.name == "CORRECT" else "⚠"
            print(
                f"    {icon}  frame {rec.frame_index:05d}  "
                f"{rec.from_step.name:<22} → {rec.to_step.name:<22}  "
                f"{rec.status.name}"
            )
    print()
    print(f"[INFO] Goodbye. Review full log at:\n       {logger.log_path}")


if __name__ == "__main__":
    main()
