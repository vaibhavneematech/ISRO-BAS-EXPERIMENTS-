"""
web_bridge.py
=============
SIH26174 • VYOM: ON-BOARD PROTOCOL COMPLIANCE ASSISTANT
Bidirectional QWebChannel Bridge between Python backend and WebEngine UI.
"""

from __future__ import annotations

import json
from PySide6.QtCore import QObject, Signal, Slot


class VyomBridge(QObject):
    """
    Exposed to JavaScript running in QWebEngineView under `window.bridge`.
    Handles:
    - Streaming base64 camera frames to JS
    - Pushing JSON telemetry updates
    - Pushing audit log events
    - Receiving user commands (Start, Stop, Reset, Switch Experiment, Report, Stream, Record)
    """

    # Signals (Python -> JavaScript)
    frameReady        = Signal(str)            # Base64 JPEG data URI: 'data:image/jpeg;base64,...'
    telemetryUpdated  = Signal(str)            # Serialized JSON object of all live telemetry
    eventLogged       = Signal(str, str, str)  # timestamp, message, level ('info', 'success', 'warning', 'error', 'cyan')
    sessionEnded      = Signal(str)            # Serialized JSON summary
    recordingFile     = Signal(str)            # Active MP4 filename
    streamStatus      = Signal(bool, str)      # is_active, url

    # Internal Qt signals emitted to MainWindow / CameraWorker
    sig_start_requested    = Signal()
    sig_stop_requested     = Signal()
    sig_reset_requested    = Signal()
    sig_switch_experiment  = Signal(str)
    sig_toggle_stream      = Signal(bool)
    sig_toggle_record      = Signal(bool)
    sig_open_report        = Signal()
    sig_print_report       = Signal()
    sig_js_ready           = Signal()  # Emitted when JS bridge is fully initialized

    def __init__(self, parent=None):
        super().__init__(parent)

    # ── Slots callable directly from JavaScript ───────────────────────────────

    @Slot()
    def startMission(self):
        """Called when user clicks INITIALIZE / START in web UI."""
        self.sig_start_requested.emit()

    @Slot()
    def stopMission(self):
        """Called when user clicks HALT / STOP in web UI."""
        self.sig_stop_requested.emit()

    @Slot()
    def resetMission(self):
        """Called when user clicks RESET in web UI."""
        self.sig_reset_requested.emit()

    @Slot(str)
    def switchExperiment(self, exp_key: str):
        """Called when user selects a different protocol (e.g. 'p1', 'p2', 'p3', 'p4')."""
        self.sig_switch_experiment.emit(exp_key)

    @Slot(bool)
    def toggleStream(self, active: bool):
        """Called when user toggles MJPEG LAN stream."""
        self.sig_toggle_stream.emit(active)

    @Slot(bool)
    def toggleRecord(self, active: bool):
        """Called when user toggles MP4 recording."""
        self.sig_toggle_record.emit(active)

    @Slot()
    def openReport(self):
        """Called when user clicks Open HTML Report."""
        self.sig_open_report.emit()

    @Slot()
    def printReport(self):
        """Called when user clicks Print Report."""
        self.sig_print_report.emit()

    @Slot()
    def jsReady(self):
        """
        Called by JavaScript AFTER new QWebChannel(...) callback completes
        and all signals (frameReady, telemetryUpdated, etc.) are connected.
        This tells Python it is safe to start the CameraWorker.
        """
        print("[VYOM Bridge] ✓ JS QWebChannel fully initialized. Starting pipeline...")
        self.sig_js_ready.emit()
