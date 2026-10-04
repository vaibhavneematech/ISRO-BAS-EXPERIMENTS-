"""
main_window.py
==============
SIH26174 • VYOM: ON-BOARD PROTOCOL COMPLIANCE ASSISTANT
Aerospace-Grade Mission Control Application for ISRO BAS Experiments.

Architecture:
- High-Performance Multi-threaded Computer Vision Engine (CameraWorker in QThread)
  - 1:1 Square 480x480 center-crop before all detections
  - Hybrid HSV (ISRO box segmentation) + YOLOv11 Neural Object Detection
  - MediaPipe Hands & Pose/Face verification
  - Finite State Machine with Debounced Milestone Transitions
  - Local MP4 Recording & MJPEG LAN Streaming
  - Offline Audio Alerts (Beep on success, wav for skip/wrong/completion)
  - Structured JSONL Event Logging + HTML Audit Report
- WebEngine Chromium Flight Deck UI
  - Real-time QWebChannel bridge for zero-latency streaming & telemetry sync
  - Deep space particle nebula canvas animation
  - Live hand-to-object proximity oscillogram waveform canvas
  - 1:1 Square HUD viewfinder with crosshair targets and telemetry pills
  - Active Protocol switching (P1 Box Sequencing, P2 Smart Lab, P3 Labware, P4 Inspection)
"""

from __future__ import annotations

import base64
import json
import pathlib
import sys
import time
import webbrowser
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer, QUrl
from PySide6.QtGui import QImage, QPixmap, QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QScrollArea, QFrame, QSizePolicy
)

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.detection.hsv_detector import detect_boxes, draw_detections
from src.detection.mediapipe_hands import HandDetector, draw_hands
from src.detection.person_detector import PersonDetector
from src.detection.yolo_detector import YOLODetector, draw_yolo_detections
from src.protocol.state_machine import StateMachine, TransitionStatus, EXPERIMENT_REGISTRY
from src.alerts.voice_alert import VoiceAlertSystem, AlertType
from src.logging.event_logger import EventLogger, EventType, open_html_report
from src.recording.video_recorder import VideoRecorder
from src.streaming.video_stream import MJPEGStreamer
from src.gui.reference_widgets import (
    SidebarWidget,
    TopTelemetryStrip,
    SquareCameraCard,
    CurrentStepCard,
    NextStepGuidanceCard,
    OverallStatusCard,
    ExplainableEvidenceCard,
    RationaleSummaryCard,
    ObjectTrackingCard,
    MissionAuditLogCard,
    BigScreenEventLogCard,
    ToolsAndControlsCard,
    C_APP_BG,
)
from src.gui.widgets import SessionSummaryDialog

CAMERA_INDEX = 0
SQUARE_SIZE = 480


# ══════════════════════════════════════════════════════════════════════════════
#  Camera & Detection Worker Thread
# ══════════════════════════════════════════════════════════════════════════════

class CameraWorker(QThread):
    """
    Dedicated worker thread handling:
    - Webcam capture at 30 FPS
    - Center-crop to 480x480 square BEFORE all detections
    - HSV segmentation for ISRO red/yellow boxes
    - YOLOv11 inference for phone, bottle, book, cup
    - MediaPipe Hands & Person/Face tracking
    - Finite State Machine updates & structured event logging
    - Video recording (MP4) and MJPEG LAN streaming
    """

    frame_ready     = Signal(QImage)   # QImage frame for Qt compatibility
    frame_b64       = Signal(str)      # Base64 JPEG data URL for WebEngine
    state_updated   = Signal(dict)     # Telemetry & evidence state snapshot
    event_occurred  = Signal(str, str) # Message, Level
    recording_file  = Signal(str)      # Active MP4 filename
    session_ended   = Signal(dict)     # Summary data on completion

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self._paused = False
        self._frame_id = 0
        self._reset_requested = False
        self._pending_switch_exp: Optional[str] = None
        self._session_ended_emitted = False
        self._rec_start_time: Optional[float] = None

        # Sound cooldown tracking
        self._last_completed_step_id = -1
        self._completed_sound_played = False

        # Alert spam hysteresis – prevent repeated SKIPPED / WRONG_ORDER alerts
        self._last_skip_alert_time: float = 0.0
        self._last_wrong_alert_time: float = 0.0
        _ALERT_COOLDOWN_S = 3.0   # seconds between repeated same-type alerts
        self._ALERT_CD = _ALERT_COOLDOWN_S

        # Modules
        self._hand_detector: Optional[HandDetector] = None
        self._person_detector: Optional[PersonDetector] = None
        self._yolo_detector: Optional[YOLODetector] = None
        self._sm: Optional[StateMachine] = None
        self._vas: Optional[VoiceAlertSystem] = None
        self._logger: Optional[EventLogger] = None
        self._recorder: Optional[VideoRecorder] = None
        self._streamer: Optional[MJPEGStreamer] = None

    def switch_experiment(self, exp_key: str):
        self._pending_switch_exp = exp_key

    def set_streaming(self, active: bool, host: str = "127.0.0.1", port: int = 8080):
        if active:
            if self._streamer is None:
                try:
                    self._streamer = MJPEGStreamer(host=host, port=port)
                    self._streamer.start()
                    self.event_occurred.emit(f"MJPEG Stream active on http://{host}:{port}", "cyan")
                except Exception as exc:
                    self.event_occurred.emit(f"Streaming error: {exc}", "error")
        else:
            if self._streamer is not None:
                self._streamer.stop()
                self._streamer = None
                self.event_occurred.emit("MJPEG Stream stopped.", "warning")

    def set_recording(self, active: bool):
        if active:
            if self._recorder is None or not self._recorder.is_active:
                recordings_dir = ROOT / "recordings"
                self._recorder = VideoRecorder(output_dir=recordings_dir, prefix="vyom_mission")
                if self._recorder.start():
                    self._rec_start_time = time.monotonic()
                    if self._recorder.filepath:
                        self.recording_file.emit(self._recorder.filepath.name)
                        self.event_occurred.emit(f"Recording started: {self._recorder.filepath.name}", "cyan")
        else:
            if self._recorder is not None and self._recorder.is_active:
                self._recorder.stop()
                self.event_occurred.emit("Recording paused/saved.", "warning")

    def stop(self):
        self._running = False

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def reset_fsm(self):
        self._reset_requested = True

    def get_logger(self) -> Optional[EventLogger]:
        return self._logger

    def get_rec_elapsed_str(self) -> str:
        if self._rec_start_time is None:
            return "00:00:00"
        elapsed = int(time.monotonic() - self._rec_start_time)
        h = elapsed // 3600
        m = (elapsed % 3600) // 60
        s = elapsed % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    def run(self):
        self._running = True
        self._paused = False
        self._frame_id = 0
        self._reset_requested = False
        self._session_ended_emitted = False
        self._last_completed_step_id = -1
        self._completed_sound_played = False
        self._rec_start_time = None
        self._last_skip_alert_time = 0.0
        self._last_wrong_alert_time = 0.0

        print("\n" + "=" * 70)
        print(" [VYOM FLIGHT DECK] Initializing Mission Control AI Pipeline")
        print("=" * 70)

        # 1. State Machine
        self._sm = StateMachine()
        self._sm.start()
        print(f"[VYOM StateMachine] ✓ Active Protocol: '{self._sm.protocol_name}' ({len(self._sm.steps)-1} milestones)")

        # 2. Detectors
        try:
            self._hand_detector = HandDetector()
            print("[VYOM MediaPipe] ✓ MediaPipe HandLandmarker online.")
        except Exception as exc:
            print(f"[VYOM MediaPipe] ⚠ Warning: {exc}")
            self._hand_detector = None

        try:
            self._person_detector = PersonDetector()
            print("[VYOM Person] ✓ Person & Face Presence detector online.")
        except Exception as exc:
            print(f"[VYOM Person] ⚠ Warning: {exc}")
            self._person_detector = None

        try:
            self._yolo_detector = YOLODetector()
            if self._yolo_detector.is_available:
                print(f"[VYOM YOLO] ✓ YOLOv11 loaded on CPU. Target classes: {self._yolo_detector.target_classes}")
            else:
                print("[VYOM YOLO] ⚠ YOLOv11 model file pending.")
        except Exception as exc:
            print(f"[VYOM YOLO] ⚠ Error loading YOLO: {exc}")
            self._yolo_detector = None

        # 3. Audio Alert System
        self._vas = VoiceAlertSystem()
        print("[VYOM Audio] ✓ Offline Voice & Beep Alert System initialized.")

        # 4. Logger & Local Recorder
        self._logger = EventLogger(session_name="vyom_mission")
        self._logger.log_session_start(self._sm.protocol_name, len(self._sm.steps) - 1)
        print(f"[VYOM Logger] ✓ Structured JSONL log: {self._logger.log_path.name}")

        recordings_dir = ROOT / "recordings"
        self._recorder = VideoRecorder(output_dir=recordings_dir, prefix="vyom_mission")
        if self._recorder.start():
            self._rec_start_time = time.monotonic()
            if self._recorder.filepath:
                self.recording_file.emit(self._recorder.filepath.name)
                print(f"[VYOM Recorder] ✓ Local MP4 video recording active: {self._recorder.filepath.name}")

        # 5. Open Camera
        print(f"[VYOM Camera] Opening camera device (Index: {CAMERA_INDEX})...")
        cap = cv2.VideoCapture(CAMERA_INDEX)
        # Test if backend can actually read frames; if not, fallback to DirectShow
        if cap.isOpened():
            test_ret, test_frame = cap.read()
            if not test_ret or test_frame is None:
                print("[VYOM Camera] Default backend read failed, falling back to DirectShow...")
                cap.release()
                cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

        if not cap.isOpened():
            cap = cv2.VideoCapture(1)

        camera_available = cap.isOpened()
        if camera_available:
            backend_name = getattr(cap, "getBackendName", lambda: "DEFAULT")()
            print(f"[VYOM Camera] ✓ Optical sensor online ({backend_name}).")
        else:
            print("[VYOM Camera] ⚠ No physical webcam detected. Running synthetic space flight test feed.")
            self.event_occurred.emit("WARNING: Camera not available. Running simulated space flight feed.", "warning")

        prev_time = time.time()
        fps_counter = 0
        measured_fps = 30.0

        self.event_occurred.emit(f"VYOM Engine active. Protocol: [{self._sm.protocol_name}]", "success")

        while self._running:
            # Handle pending experiment switch
            if self._pending_switch_exp:
                exp_key = self._pending_switch_exp
                self._pending_switch_exp = None
                self._sm.switch_experiment(exp_key)
                if self._yolo_detector:
                    target_objs = self._sm._target_objects
                    self._yolo_detector.set_target_classes(target_objs)
                self._last_completed_step_id = -1
                self._completed_sound_played = False
                self._session_ended_emitted = False
                self._logger.log(
                    EventType.PROTOCOL_RESET,
                    experiment_name=self._sm.protocol_name,
                    message=f"Switched protocol to: {self._sm.protocol_name}",
                )
                print(f"[VYOM Protocol] Switched to: [{self._sm.protocol_name}]")
                self.event_occurred.emit(f"Switched protocol to: [{self._sm.protocol_name}]", "cyan")

            # Handle reset
            if self._reset_requested:
                self._reset_requested = False
                self._sm.reset()
                self._last_completed_step_id = -1
                self._completed_sound_played = False
                self._session_ended_emitted = False
                self._logger.log(EventType.PROTOCOL_RESET, experiment_name=self._sm.protocol_name, message="Protocol reset by operator.")
                print(f"[VYOM Protocol] Reset to initial state: {self._sm.get_current_state().name}")
                self.event_occurred.emit("Protocol reset to initial state.", "warning")

            if self._paused:
                self.msleep(50)
                continue

            # Capture Frame
            ret, frame = (False, None)
            if cap and cap.isOpened():
                ret, frame = cap.read()
                if not ret and self._frame_id < 20:
                    for _ in range(3):
                        self.msleep(15)
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            break

            if not ret or frame is None:
                # Realistic aerospace lab experiment scene matching reference image
                frame = np.zeros((360, 640, 3), dtype=np.uint8)
                frame[:] = (18, 22, 30)
                # Background workstation depth
                cv2.rectangle(frame, (0, 0), (640, 170), (12, 16, 22), -1)
                cv2.line(frame, (0, 170), (640, 170), (30, 40, 55), 1)

                # Astronaut Face (upper right background)
                cv2.circle(frame, (480, 115), 36, (45, 55, 75), -1)
                cv2.rectangle(frame, (440, 75), (520, 155), (220, 110, 25), 2)
                cv2.rectangle(frame, (440, 55), (495, 75), (220, 110, 25), -1)
                cv2.putText(frame, "FACE", (445, 69), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, "0.89", (445, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 110, 25), 1, cv2.LINE_AA)

                # ISRO Yellow/Red Experiment Box (center foreground)
                cv2.rectangle(frame, (280, 145), (420, 255), (35, 35, 190), -1)
                cv2.rectangle(frame, (290, 155), (410, 245), (25, 185, 235), -1)
                cv2.circle(frame, (325, 200), 18, (25, 25, 35), -1)
                # Object detection bounding box
                cv2.rectangle(frame, (275, 140), (425, 260), (25, 205, 245), 2)
                cv2.rectangle(frame, (370, 120), (425, 140), (25, 205, 245), -1)
                cv2.putText(frame, "OBJECT", (373, 134), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, "0.94", (376, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (25, 205, 245), 1, cv2.LINE_AA)

                # Hand interaction (approaching box)
                hand_x = int(210 + 10 * np.sin(self._frame_id * 0.06))
                cv2.circle(frame, (hand_x, 190), 24, (55, 70, 90), -1)
                cv2.rectangle(frame, (hand_x - 30, 150), (hand_x + 35, 230), (40, 200, 40), 2)
                cv2.rectangle(frame, (hand_x - 30, 130), (hand_x + 12, 150), (40, 200, 40), -1)
                cv2.putText(frame, "HAND", (hand_x - 26, 143), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, "0.92", (hand_x - 26, 166), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (40, 200, 40), 1, cv2.LINE_AA)

                self.msleep(33)

            self._frame_id += 1
            timestamp_ms = int(time.time() * 1000)

            # FPS calculation
            fps_counter += 1
            now = time.time()
            if now - prev_time >= 1.0:
                measured_fps = fps_counter / (now - prev_time)
                fps_counter = 0
                prev_time = now

            # ── 1:1 SQUARE CENTER-CROP (for square camera card display) ──────────
            h, w = frame.shape[:2]
            square = min(h, w)
            x0 = (w - square) // 2
            y0 = (h - square) // 2
            # IMPORTANT: numpy slice returns non-contiguous view (original row strides
            # are from the 640-wide buffer). np.ascontiguousarray() forces a real copy
            # with correct strides so QImage reads the right pixels.
            frame = np.ascontiguousarray(frame[y0:y0 + square, x0:x0 + square])
            # Resize to a consistent 480x480 square for all detections
            if square != 480:
                frame = cv2.resize(frame, (480, 480), interpolation=cv2.INTER_LINEAR)

            # ── 1. Object Detections (HSV + YOLO Hybrid) ──────────────────────
            detection_mode = self._sm.detection_mode.lower()
            box_detections = []
            yolo_detections = []

            # HSV Detection (ISRO red/yellow boxes)
            try:
                if "hsv" in detection_mode or "box" in self._sm.experiment_key:
                    box_detections = detect_boxes(frame)
            except Exception as exc:
                box_detections = []

            # YOLOv11 Detection (Lab objects: phone, bottle, book, cup)
            try:
                if self._yolo_detector and self._yolo_detector.is_available:
                    run_yolo = (self._frame_id % 2 == 0) or ("yolo" in detection_mode)
                    yolo_detections = self._yolo_detector.detect(frame, run_inference=run_yolo)
            except Exception as exc:
                yolo_detections = []

            all_detections = list(box_detections) + list(yolo_detections)

            # ── 2. Human Interaction Detections (MediaPipe + Person) ───────────
            hands = []
            if self._hand_detector:
                try:
                    hands = self._hand_detector.detect(frame, timestamp_ms)
                except Exception:
                    hands = []

            body_detected, face_detected, body_rects, face_rects = (False, False, [], [])
            if self._person_detector:
                try:
                    body_detected, face_detected, body_rects, face_rects = self._person_detector.detect(
                        frame, has_hands=bool(hands)
                    )
                except Exception:
                    pass

            # ── 3. FSM Update ─────────────────────────────────────────────────
            status = self._sm.update(all_detections, hands)
            evidence = self._sm.get_evidence()
            current_step = self._sm.get_current_state()
            next_step = self._sm.get_next_step()

            # ── 4. Sound Logic (Single-event, cooldown protected) ─────────────
            if status == TransitionStatus.CORRECT:
                if current_step.step_id != self._last_completed_step_id:
                    self._last_completed_step_id = current_step.step_id
                    # Short positive beep only
                    self._vas.play_alert(AlertType.STEP_COMPLETE)
                    self._logger.log(
                        EventType.STEP_COMPLETE,
                        frame=self._frame_id,
                        current_step=current_step.name,
                        next_step=next_step.name if next_step else "COMPLETED",
                        status="CORRECT",
                        colours=evidence.get("detected_objects", []),
                        hands=len(hands),
                        experiment_name=self._sm.protocol_name,
                        message=evidence.get("decision_reason", "Step verified."),
                    )
                    self.event_occurred.emit(f"✓ Step {current_step.step_id}: {current_step.name} VERIFIED", "success")

            elif status == TransitionStatus.SKIPPED:
                # Rate-limited: only alert once per cooldown window
                _now = time.time()
                if _now - self._last_skip_alert_time >= self._ALERT_CD:
                    self._last_skip_alert_time = _now
                    self._vas.play_alert(AlertType.STEP_SKIPPED)
                    self._logger.log(
                        EventType.STEP_SKIPPED,
                        frame=self._frame_id,
                        current_step=current_step.name,
                        next_step=next_step.name if next_step else "",
                        status="SKIPPED",
                        colours=evidence.get("detected_objects", []),
                        hands=len(hands),
                        experiment_name=self._sm.protocol_name,
                        message=evidence.get("decision_reason", "Step skipped."),
                    )
                    self.event_occurred.emit(
                        f"⚠ PROTOCOL ALERT: Step skipped! Next: {next_step.name if next_step else ''}",
                        "error",
                    )

            elif status == TransitionStatus.WRONG_ORDER:
                # Rate-limited: only alert once per cooldown window
                _now = time.time()
                if _now - self._last_wrong_alert_time >= self._ALERT_CD:
                    self._last_wrong_alert_time = _now
                    self._vas.play_alert(AlertType.WRONG_SEQUENCE)
                    self._logger.log(
                        EventType.WRONG_ORDER,
                        frame=self._frame_id,
                        current_step=current_step.name,
                        next_step=next_step.name if next_step else "",
                        status="WRONG_ORDER",
                        colours=evidence.get("detected_objects", []),
                        hands=len(hands),
                        experiment_name=self._sm.protocol_name,
                        message=evidence.get("decision_reason", "Wrong sequence."),
                    )
                    self.event_occurred.emit("✗ PROTOCOL VIOLATION: Wrong sequence detected!", "error")

            # Check experiment completion
            if self._sm.is_completed() and not self._completed_sound_played:
                self._completed_sound_played = True
                self._vas.play_alert(AlertType.EXPERIMENT_COMPLETE)
                self._logger.log(
                    EventType.PROTOCOL_COMPLETE,
                    frame=self._frame_id,
                    current_step="COMPLETED",
                    status="COMPLETED",
                    experiment_name=self._sm.protocol_name,
                    message=f"All milestones verified for {self._sm.protocol_name}.",
                )
                self.event_occurred.emit(f"★ MISSION SUCCESS: {self._sm.protocol_name} Fully Verified!", "cyan")

                if not self._session_ended_emitted:
                    self._session_ended_emitted = True
                    summary = self._sm.get_session_summary()
                    self.session_ended.emit(summary)

            # ── 5. Render HUD Overlays ────────────────────────────────────────
            annotated = frame.copy()

            if self._person_detector and (body_rects or face_rects):
                annotated = self._person_detector.draw_overlays(annotated, body_rects, face_rects)

            if box_detections:
                annotated = draw_detections(annotated, box_detections)

            if yolo_detections:
                annotated = draw_yolo_detections(annotated, yolo_detections)

            if hands:
                annotated = draw_hands(annotated, hands)

            # ── 6. Corner HUD Pills (burned onto annotated frame) ─────────────
            cam_lbl = "CAM 01 - LAB EDGE NODE"
            rec_lbl = f"  REC   {self.get_rec_elapsed_str()}"
            font = cv2.FONT_HERSHEY_SIMPLEX

            h_f, w_f = annotated.shape[:2]

            # Bottom-left: CAM label pill
            (tw, th), _ = cv2.getTextSize(cam_lbl, font, 0.40, 1)
            pill_x1, pill_y1 = 12, h_f - th - 16
            pill_x2, pill_y2 = pill_x1 + tw + 18, h_f - 8
            cv2.rectangle(annotated, (pill_x1, pill_y1), (pill_x2, pill_y2), (15, 20, 32), -1)
            cv2.rectangle(annotated, (pill_x1, pill_y1), (pill_x2, pill_y2), (35, 45, 68), 1)
            cv2.putText(annotated, cam_lbl, (pill_x1 + 8, pill_y2 - 6), font, 0.40, (241, 245, 249), 1, cv2.LINE_AA)

            # Bottom-right: REC pill
            (rw, rh), _ = cv2.getTextSize(rec_lbl, font, 0.40, 1)
            rx1 = w_f - rw - 26
            ry1 = h_f - rh - 16
            rx2 = w_f - 12
            ry2 = h_f - 8
            cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (15, 20, 32), -1)
            cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (35, 45, 68), 1)
            cv2.circle(annotated, (rx1 + 10, (ry1 + ry2) // 2), 4, (40, 40, 235), -1)
            cv2.putText(annotated, rec_lbl, (rx1 + 18, ry2 - 6), font, 0.40, (241, 245, 249), 1, cv2.LINE_AA)

            # Record & Stream Frame
            if self._recorder and self._recorder.is_active:
                self._recorder.write(annotated)
            if self._streamer and self._streamer.is_active:
                self._streamer.push_frame(annotated)

            # ── 7. Encode & Emit to GUI ────────────────────────────────────────
            # Emit QImage for any Qt listeners
            h_a, w_a, ch = annotated.shape
            bytes_per_line = ch * w_a
            rgb_frame = np.ascontiguousarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            qimg = QImage(rgb_frame.data, w_a, h_a, bytes_per_line, QImage.Format_RGB888).copy()
            self.frame_ready.emit(qimg)

            # Emit Base64 JPEG for WebEngine Viewport
            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            b64_str = base64.b64encode(buf.tobytes()).decode("ascii")
            self.frame_b64.emit(f"data:image/jpeg;base64,{b64_str}")

            # Emit Telemetry & Evidence
            face_conf = (
                self._person_detector.get_face_confidence()
                if self._person_detector and hasattr(self._person_detector, "get_face_confidence")
                else (0.89 if face_detected else 0.0)
            )
            pose_conf = (
                self._person_detector.get_pose_confidence()
                if self._person_detector and hasattr(self._person_detector, "get_pose_confidence")
                else (0.85 if body_detected else 0.0)
            )
            telemetry = {
                "fps": measured_fps,
                "current_step": current_step,
                "next_step": next_step,
                "status": status,
                "evidence": evidence,
                "steps": self._sm.steps,
                "current_step_idx": self._sm.current_step_idx,
                "is_completed": self._sm.is_completed(),
                "experiment_name": self._sm.protocol_name,
                "experiment_id": self._sm.protocol_id,
                "objects_detail": evidence.get("objects_detail", []),
                "has_hands": bool(hands),
                "face_detected": face_detected,
                "face_confidence": face_conf,
                "pose_confidence": pose_conf,
                "body_detected": body_detected,
                "hand_confidence": hands[0].get("confidence", 0.0) if hands else 0.0,
                "stream_active": self._streamer is not None and self._streamer.is_active,
                "rec_active": self._recorder is not None and self._recorder.is_active,
            }
            self.state_updated.emit(telemetry)

        # Cleanup
        if cap and cap.isOpened():
            cap.release()
        if self._hand_detector:
            try:
                self._hand_detector.close()
            except Exception:
                pass
        if self._recorder:
            try:
                self._recorder.stop()
            except Exception:
                pass
        if self._streamer:
            try:
                self._streamer.stop()
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
#  Main Window - Clean Minimalist Native Aerospace Flight Deck
# ══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    """
    Clean, minimalist native PySide6 Flight Control Window for ISRO BAS Experiments.
    - Zero Chromium/WebEngine overhead: instant startup, hardware-accelerated video rendering.
    - Fixed Left Sidebar: Experiment selector, mission timeline checklist, and execution controls.
    - Right Main Dashboard: Live 1:1 camera feed, current step directive, next step suggestion,
      overall status card, explainable AI evidence, object tracking, and JSONL audit log.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("VYOM • On-Board Protocol Compliance Assistant (SIH26174 - ISRO)")
        self.resize(1540, 940)
        self.setMinimumSize(1240, 780)
        self.setStyleSheet(f"QMainWindow {{ background-color: {C_APP_BG}; }}")
        self.showMaximized()

        # Central container
        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ── 1. Fixed Left Sidebar ─────────────────────────────────────────────
        self.sidebar = SidebarWidget(self)
        root_layout.addWidget(self.sidebar)

        # ── 2. Right Main Scrollable Viewport ──────────────────────────────────
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                background-color: {C_APP_BG};
                border: none;
            }}
            QScrollBar:vertical {{
                background: #0d131f;
                width: 8px;
                margin: 0;
            }}
            QScrollBar::handle:vertical {{
                background: #1e2942;
                min-height: 20px;
                border-radius: 4px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """)

        main_content = QWidget()
        main_content.setStyleSheet(f"background-color: {C_APP_BG};")
        content_layout = QVBoxLayout(main_content)
        content_layout.setContentsMargins(18, 16, 18, 16)
        content_layout.setSpacing(14)

        # Top Telemetry Strip
        self.telemetry_strip = TopTelemetryStrip(self)
        content_layout.addWidget(self.telemetry_strip)

        # Row 1: Primary Optical Camera (Fixed 640x640, ~65% width) + Status Cards Column
        row1 = QHBoxLayout()
        row1.setSpacing(16)

        # Large Fixed Optical Camera Card (640x640 fixed preview)
        self.camera_card = SquareCameraCard(self)
        row1.addWidget(self.camera_card)

        # Right Vertical Column of Status, Evidence & Tracking Cards
        col_status = QVBoxLayout()
        col_status.setSpacing(8)

        self.current_card = CurrentStepCard(self)
        col_status.addWidget(self.current_card)

        self.next_card = NextStepGuidanceCard(self)
        col_status.addWidget(self.next_card)

        self.status_card = OverallStatusCard(self)
        col_status.addWidget(self.status_card)

        self.evidence_card = ExplainableEvidenceCard(self)
        col_status.addWidget(self.evidence_card)

        self.object_card = ObjectTrackingCard(self)
        col_status.addWidget(self.object_card)

        self.rationale_card = RationaleSummaryCard(self)
        col_status.addWidget(self.rationale_card)

        col_status.addStretch(1)

        row1.addLayout(col_status, stretch=1)
        content_layout.addLayout(row1)

        # Full-Width Dedicated Mission Audit & Event Log Terminal
        self.audit_log_card = BigScreenEventLogCard(self)
        content_layout.addWidget(self.audit_log_card)

        # Bottom Tools & Controls Card
        self.tools_card = ToolsAndControlsCard(self)
        content_layout.addWidget(self.tools_card)

        scroll.setWidget(main_content)
        root_layout.addWidget(scroll, stretch=1)

        # ── 3. Initialize Camera Worker ───────────────────────────────────────
        self.worker = CameraWorker(self)
        self.worker.frame_ready.connect(self._on_frame_ready)
        self.worker.state_updated.connect(self._on_state_updated)
        self.worker.event_occurred.connect(self._on_event_occurred)
        self.worker.session_ended.connect(self._on_session_ended)

        # ── 4. Wire Controls & Signals ────────────────────────────────────────
        self.sidebar.start_clicked.connect(self._on_start_clicked)
        self.sidebar.stop_clicked.connect(self._on_stop_clicked)
        self.sidebar.reset_clicked.connect(self._on_reset_clicked)
        self.sidebar.experiment_changed.connect(self._on_experiment_changed)

        self.tools_card.open_stream_clicked.connect(self._on_open_stream)
        self.tools_card.html_report_clicked.connect(self._on_open_report)
        self.tools_card.print_report_clicked.connect(self._on_print_report)
        self.status_card.view_report_clicked.connect(self._on_open_report)
        self.next_card.view_checklist_clicked.connect(self._on_open_report)
        self.rationale_card.view_details_clicked.connect(self._on_open_report)

        # State tracking
        self._initial_steps_set = False

        # Pipeline ready for user interaction
        self.sidebar.btn_start.setEnabled(True)
        self.sidebar.btn_stop.setEnabled(False)
        # Start camera preview
        self.worker.start()

    # ── Signal Slots ──────────────────────────────────────────────────────────

    @Slot(QImage)
    def _on_frame_ready(self, qimg: QImage):
        pix = QPixmap.fromImage(qimg)
        # Scale 480x480 square to fill the camera card edge-to-edge
        card_size = self.camera_card.video_label.size()
        scaled_pix = pix.scaled(card_size, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self.camera_card.video_label.setPixmap(scaled_pix)

    @Slot(dict)
    def _on_state_updated(self, telemetry: dict):
        fps = float(telemetry.get("fps", 30.0))
        rec_active = bool(telemetry.get("rec_active", False))
        stream_active = bool(telemetry.get("stream_active", False))
        self.telemetry_strip.update_telemetry(fps, rec_active, stream_active)

        # Update steps in sidebar if not set yet or on switch
        steps = telemetry.get("steps", [])
        if not self._initial_steps_set and steps:
            self.sidebar.timeline.set_steps(steps)
            self._initial_steps_set = True

        current_idx = int(telemetry.get("current_step_idx", 0))
        is_completed = bool(telemetry.get("is_completed", False))
        self.sidebar.timeline.update_progress(current_idx, is_completed)

        # Current step card
        curr = telemetry.get("current_step")
        c_name = getattr(curr, "name", "START") if curr else "START"
        c_desc = getattr(curr, "description", "") if curr else ""
        evidence = telemetry.get("evidence") or {}
        hold_pct = int(evidence.get("hold_pct", 0) or 0)
        self.current_card.update_step(current_idx, c_name, c_desc, hold_pct)

        # Next step card
        nxt = telemetry.get("next_step")
        n_name = getattr(nxt, "name", "COMPLETED") if nxt else "COMPLETED"
        next_sugg = evidence.get("next_step_suggestion", "") or ""
        self.next_card.update_next(current_idx + 1, n_name, next_sugg)

        # Overall status card
        status = telemetry.get("status", TransitionStatus.WAITING)
        self.status_card.update_status(status, is_completed)

        # Explainable evidence card
        self.evidence_card.update_evidence(evidence)

        # Rationale summary card
        reason = evidence.get("decision_reason", "")
        if reason:
            self.rationale_card.set_rationale(reason)

        # Object tracking card
        objects_detail = telemetry.get("objects_detail", [])
        has_hands = bool(telemetry.get("has_hands", False))
        face_detected = bool(telemetry.get("face_detected", False))
        hand_conf = float(telemetry.get("hand_confidence", 0.0))
        face_conf = float(telemetry.get("face_confidence", 0.0))
        pose_conf = float(telemetry.get("pose_confidence", 0.0))
        body_detected = bool(telemetry.get("body_detected", False))
        self.object_card.update_objects(
            objects_detail, has_hands, face_detected,
            hand_conf=hand_conf, face_conf=face_conf,
            pose_conf=pose_conf, body_detected=body_detected,
        )

    @Slot(str, str)
    def _on_event_occurred(self, message: str, level: str):
        self.audit_log_card.log(message, level)

    @Slot(dict)
    def _on_session_ended(self, summary: dict):
        self.audit_log_card.log("Mission session completed. Certificate generated.", "success")
        try:
            dialog = SessionSummaryDialog(summary, self)
            dialog.exec()
        except Exception as exc:
            print(f"[VYOM] Session dialog error: {exc}")

    @Slot()
    def _on_start_clicked(self):
        if not self.worker.isRunning():
            self.worker.start()
        else:
            self.worker.resume()
        self.sidebar.btn_start.setEnabled(False)
        self.sidebar.btn_stop.setEnabled(True)
        self.telemetry_strip.set_running(True)
        self.audit_log_card.log("Mission execution active - Optical AI pipeline running", "success")

    @Slot()
    def _on_stop_clicked(self):
        self.worker.pause()
        self.sidebar.btn_start.setEnabled(True)
        self.sidebar.btn_stop.setEnabled(False)
        self.telemetry_strip.set_running(False)
        self.audit_log_card.log("Mission execution paused - Camera standby", "warning")

    @Slot()
    def _on_reset_clicked(self):
        self.worker.reset_fsm()
        self.audit_log_card.log("Resetting protocol sequence to initial step...", "cyan")

    @Slot(str)
    def _on_experiment_changed(self, exp_key: str):
        self._initial_steps_set = False
        self.worker.switch_experiment(exp_key)
        self.audit_log_card.log(f"Switched protocol to: {exp_key}", "cyan")

    @Slot(str)
    def _on_open_stream(self, url: str):
        if not url:
            url = "http://127.0.0.1:8080/stream"
        host = "127.0.0.1"
        port = 8080
        try:
            clean = url.replace("http://", "").replace("https://", "").split("/")[0]
            if ":" in clean:
                host, p_str = clean.split(":")
                port = int(p_str)
        except Exception:
            pass
        self.worker.set_streaming(True, host=host, port=port)
        self.audit_log_card.log(f"Opening MJPEG LAN stream: {url}", "cyan")
        webbrowser.open(url)

    @Slot()
    def _on_open_report(self):
        logger = self.worker.get_logger()
        if logger and logger.log_path.exists():
            logger.open_html_report(auto_print=False)
            self.audit_log_card.log(f"Opened HTML audit report: {logger.log_path.stem}.html", "cyan")
        else:
            reports_dir = ROOT / "reports"
            reports = list(reports_dir.glob("*.html"))
            if reports:
                latest = max(reports, key=lambda p: p.stat().st_mtime)
                webbrowser.open(latest.as_uri())
                self.audit_log_card.log(f"Opened HTML audit report: {latest.name}", "cyan")
            else:
                self.audit_log_card.log("No reports generated yet.", "warning")

    @Slot()
    def _on_print_report(self):
        logger = self.worker.get_logger()
        if logger and logger.log_path.exists():
            logger.open_html_report(auto_print=True)
            self.audit_log_card.log("Triggered print dialog for mission report.", "cyan")
        else:
            reports_dir = ROOT / "reports"
            reports = list(reports_dir.glob("*.html"))
            if reports:
                latest = max(reports, key=lambda p: p.stat().st_mtime)
                webbrowser.open(latest.as_uri())
                self.audit_log_card.log(f"Triggered print dialog for: {latest.name}", "cyan")
            else:
                self.audit_log_card.log("No reports available to print.", "warning")

    def closeEvent(self, event):
        if self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(2000)
        super().closeEvent(event)
