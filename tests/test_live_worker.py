"""
test_live_worker.py
Quick test of CameraWorker live execution.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
from src.gui.main_window import CameraWorker

app = QApplication([])
worker = CameraWorker()

frames_received = 0
states_received = 0

def on_frame(qimg):
    global frames_received
    frames_received += 1
    if frames_received in (1, 10, 20, 30, 45, 60):
        print(f"[Test] Live Camera Frame #{frames_received} received: {qimg.width()}x{qimg.height()}")
    if frames_received >= 60:
        QTimer.singleShot(100, stop_test)

def on_state(telemetry):
    global states_received
    states_received += 1
    if states_received in (1, 10, 20, 30, 45, 60):
        curr = telemetry.get("current_step")
        name = curr.name if curr else "None"
        fps = telemetry.get("fps", 0.0)
        print(f"[Test] Telemetry #{states_received}: Step={name}, FPS={fps:.1f}")

worker.frame_ready.connect(on_frame)
worker.state_updated.connect(on_state)

print("[Test] Starting CameraWorker thread...")
worker.start()

def stop_test():
    print("[Test] Stopping CameraWorker...")
    worker.stop()
    worker.wait(3000)
    print(f"[Test] SUCCESS! Total live frames received: {frames_received}, Total states received: {states_received}")
    app.quit()

# Failsafe timeout after 30 seconds
QTimer.singleShot(30000, stop_test)
app.exec()
