"""
run_gui.py
==========
SIH26174 – AI Human Activity Recognition for On-board BAS Experiments
Phase 6 Launcher: starts the full PySide6 GUI application.

Usage
-----
    python scripts/run_gui.py

What this does
--------------
1. Creates a QApplication
2. Opens the MainWindow (dark-themed desktop GUI)
3. Clicking ▶ Start begins the full pipeline:
     webcam → HSV detection → MediaPipe hands → State Machine
     → Voice alerts → JSONL logging
4. Everything runs offline.
"""

import sys
from pathlib import Path

# ── Make project root importable ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore    import Qt
from PySide6.QtGui     import QIcon

from src.gui.main_window import MainWindow


def main():
    # High-DPI support
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("VYOM")
    app.setApplicationDisplayName("VYOM • On-Board Protocol Compliance Assistant")
    app.setApplicationVersion("2.5")
    app.setOrganizationName("ISRO - SIH26174")

    # Optional: set taskbar icon if assets/icons/icon.png exists
    icon_path = ROOT / "assets" / "icons" / "icon.png"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    window = MainWindow()
    window.showMaximized()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
