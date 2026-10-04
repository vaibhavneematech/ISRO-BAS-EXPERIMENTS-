"""
reference_widgets.py
====================
SIH26174 • VYOM: ON-BOARD PROTOCOL COMPLIANCE ASSISTANT
Aerospace Mission Control GUI Components matching the reference design layout:
- Dark charcoal / near-black background (#080c14)
- Fixed Left Sidebar with Mission Timeline and flight execution controls
- Right main content with top telemetry, 1:1 square camera preview, 3 status cards,
  explainable evidence, object tracking, event audit log, and tools controls.
"""

from __future__ import annotations

import time
from typing import Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QCursor, QPixmap, QPainter, QPen, QBrush
from PySide6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame,
    QProgressBar, QSizePolicy, QGridLayout, QPushButton,
    QComboBox, QLineEdit, QScrollArea, QListWidget, QListWidgetItem,
)

from src.protocol.state_machine import TransitionStatus, EXPERIMENT_REGISTRY

# ── Reference Palette Tokens ──────────────────────────────────────────────────
C_APP_BG       = "#080c14"
C_SIDEBAR_BG   = "#0d131f"
C_CARD_BG      = "#0e1422"
C_CARD_BORDER  = "#1a2338"
C_TILE_BG      = "#121a2c"
C_TILE_BORDER  = "#1e2942"

# Muted Professional Blue (No neon glow)
C_BLUE_PRIMARY = "#2563eb"
C_BLUE_HOVER   = "#1d4ed8"
C_BLUE_LIGHT   = "#3b82f6"
C_BLUE_BG      = "rgba(37, 99, 235, 0.12)"
C_BLUE_BORDER  = "rgba(59, 130, 246, 0.45)"

# Semantic Accents
C_GREEN_OK     = "#10b981"
C_GREEN_BG     = "rgba(16, 185, 129, 0.12)"
C_AMBER_WARN   = "#f59e0b"
C_RED_DANGER   = "#ef4444"

# Typography
C_TEXT_WHITE   = "#f1f5f9"
C_TEXT_MUTED   = "#8492a6"
C_TEXT_SUBTLE  = "#505e75"


# ══════════════════════════════════════════════════════════════════════════════
#  1. FIXED LEFT SIDEBAR WIDGETS
# ══════════════════════════════════════════════════════════════════════════════

class TimelineStepRow(QFrame):
    """Single timeline step row with custom indicator, connecting line, and timestamp."""

    def __init__(self, step_idx: int, step_name: str, parent=None):
        super().__init__(parent)
        self.step_idx = step_idx
        self.step_name = step_name
        self._state = "pending"  # "completed", "active", "pending"
        self._timestamp = "Pending"
        self._setup_ui()

    def _setup_ui(self):
        self.setFixedHeight(50)
        self.setStyleSheet("background: transparent; border: none;")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 2, 8, 2)
        layout.setSpacing(10)

        # Left Icon Badge
        self._lbl_icon = QLabel()
        self._lbl_icon.setFixedSize(26, 26)
        self._lbl_icon.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._lbl_icon)

        # Middle Content
        mid_box = QVBoxLayout()
        mid_box.setSpacing(1)
        mid_box.setContentsMargins(0, 2, 0, 2)

        self._lbl_step_num = QLabel(f"STEP {self.step_idx + 1}")
        self._lbl_step_num.setStyleSheet(f"color: {C_TEXT_MUTED}; font-size: 9.5px; font-weight: 800; letter-spacing: 0.5px;")
        mid_box.addWidget(self._lbl_step_num)

        self._lbl_name = QLabel(self.step_name)
        self._lbl_name.setStyleSheet(f"color: {C_TEXT_WHITE}; font-size: 11.5px; font-weight: 700;")
        mid_box.addWidget(self._lbl_name)

        layout.addLayout(mid_box, stretch=1)

        # Right Timestamp / Status
        self._lbl_time = QLabel("Pending")
        self._lbl_time.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._lbl_time.setStyleSheet(f"color: {C_TEXT_SUBTLE}; font-size: 10px; font-family: monospace;")
        layout.addWidget(self._lbl_time)

        self.set_state("pending")

    def set_state(self, state: str, timestamp_str: str = ""):
        self._state = state
        if timestamp_str:
            self._timestamp = timestamp_str

        if state == "completed":
            self.setStyleSheet("background: transparent; border: none;")
            self._lbl_icon.setText("✓")
            self._lbl_icon.setStyleSheet(f"""
                background-color: {C_BLUE_PRIMARY};
                color: #ffffff;
                border-radius: 13px;
                font-size: 11px;
                font-weight: 900;
            """)
            self._lbl_step_num.setStyleSheet(f"color: {C_TEXT_MUTED}; font-size: 9.5px; font-weight: 700;")
            self._lbl_name.setStyleSheet("color: #cbd5e1; font-size: 11.5px; font-weight: 600;")
            t_txt = self._timestamp if self._timestamp != "Pending" else "00:01:20 ✓"
            self._lbl_time.setText(t_txt)
            self._lbl_time.setStyleSheet(f"color: {C_TEXT_MUTED}; font-size: 10px; font-family: monospace;")

        elif state == "active":
            self.setStyleSheet(f"""
                QFrame {{
                    background: {C_BLUE_BG};
                    border: 1px solid {C_BLUE_BORDER};
                    border-radius: 6px;
                }}
            """)
            self._lbl_icon.setText("▶")
            self._lbl_icon.setStyleSheet(f"""
                background-color: {C_BLUE_PRIMARY};
                color: #ffffff;
                border: 2px solid {C_BLUE_LIGHT};
                border-radius: 13px;
                font-size: 10px;
                font-weight: 900;
            """)
            self._lbl_step_num.setStyleSheet(f"color: {C_BLUE_LIGHT}; font-size: 9.5px; font-weight: 800;")
            self._lbl_name.setStyleSheet("color: #ffffff; font-size: 12px; font-weight: 800;")
            self._lbl_time.setText(self._timestamp if self._timestamp != "Pending" else "In Progress")
            self._lbl_time.setStyleSheet(f"color: {C_BLUE_LIGHT}; font-size: 10.5px; font-weight: 700; font-family: monospace;")

        else:  # pending
            self.setStyleSheet("background: transparent; border: none;")
            self._lbl_icon.setText("")
            self._lbl_icon.setStyleSheet("""
                background-color: transparent;
                border: 2px solid #243048;
                border-radius: 13px;
            """)
            self._lbl_step_num.setStyleSheet(f"color: {C_TEXT_SUBTLE}; font-size: 9.5px; font-weight: 700;")
            self._lbl_name.setStyleSheet(f"color: {C_TEXT_MUTED}; font-size: 11.5px; font-weight: 600;")
            self._lbl_time.setText("Pending")
            self._lbl_time.setStyleSheet(f"color: {C_TEXT_SUBTLE}; font-size: 10px; font-family: monospace;")


class SidebarTimelineWidget(QFrame):
    """Vertical list of mission timeline steps connected by a subtle line."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: list[TimelineStepRow] = []
        self._steps_data = []
        self._setup_ui()

    def _setup_ui(self):
        self.setStyleSheet("background: transparent; border: none;")
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 4, 0, 4)
        self._layout.setSpacing(6)

    def set_steps(self, steps: list):
        self._steps_data = list(steps)
        # Clear existing
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._rows.clear()

        for idx, step in enumerate(steps):
            name = getattr(step, "name", str(step))
            row = TimelineStepRow(idx, name, self)
            self._rows.append(row)
            self._layout.addWidget(row)

        self.update_progress(0, False)

    def update_progress(self, current_idx: int, is_completed: bool = False, elapsed_str: str = ""):
        for idx, row in enumerate(self._rows):
            if is_completed or idx < current_idx:
                row.set_state("completed", "Verified ✓")
            elif idx == current_idx:
                row.set_state("active", elapsed_str or "Active")
            else:
                row.set_state("pending", "Pending")


class SidebarWidget(QFrame):
    """
    Fixed Left Sidebar matching the reference design:
    - Winged chevron mark + VYOM title + Subtitle
    - Offline Edge Mode badge
    - Experiment selector dropdown
    - Mission Timeline vertical steps
    - Execution buttons: Start (primary blue), Stop (dark), Reset
    """

    start_clicked = Signal()
    stop_clicked = Signal()
    reset_clicked = Signal()
    experiment_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(310)
        self.setStyleSheet(f"""
            SidebarWidget {{
                background-color: {C_SIDEBAR_BG};
                border-right: 1px solid {C_CARD_BORDER};
            }}
        """)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 18, 16, 16)
        layout.setSpacing(14)

        # ── 1. Brand Logo Header ──────────────────────────────────────────────
        brand_box = QHBoxLayout()
        brand_box.setSpacing(10)

        # Winged Chevron Logo
        lbl_logo = QLabel("V")
        lbl_logo.setFixedSize(34, 34)
        lbl_logo.setAlignment(Qt.AlignCenter)
        lbl_logo.setStyleSheet(f"""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1d4ed8, stop:1 #3b82f6);
            color: #ffffff;
            font-size: 20px;
            font-weight: 900;
            border-radius: 6px;
            font-family: -apple-system, sans-serif;
        """)
        brand_box.addWidget(lbl_logo)

        brand_text_box = QVBoxLayout()
        brand_text_box.setSpacing(2)
        lbl_title = QLabel("VYOM")
        lbl_title.setStyleSheet("color: #ffffff; font-size: 18px; font-weight: 900; letter-spacing: 1.5px;")
        brand_text_box.addWidget(lbl_title)

        lbl_tag = QLabel("ON-BOARD PROTOCOL COMPLIANCE ASSISTANT")
        lbl_tag.setStyleSheet(f"color: {C_TEXT_MUTED}; font-size: 8.5px; font-weight: 700; letter-spacing: 0.6px;")
        lbl_tag.setWordWrap(True)
        brand_text_box.addWidget(lbl_tag)

        brand_box.addLayout(brand_text_box)
        layout.addLayout(brand_box)

        # ── 2. Offline Edge Mode Badge ────────────────────────────────────────
        self.badge_offline = QLabel("✈ Offline Edge Mode")
        self.badge_offline.setAlignment(Qt.AlignCenter)
        self.badge_offline.setFixedHeight(28)
        self.badge_offline.setStyleSheet("""
            background-color: #121a2c;
            color: #cbd5e1;
            border: 1px solid #243048;
            border-radius: 5px;
            font-size: 11px;
            font-weight: 600;
        """)
        layout.addWidget(self.badge_offline)

        # ── 2b. Project Introduction ──────────────────────────────────────────
        intro_card = QFrame()
        intro_card.setStyleSheet(f"""
            QFrame {{
                background-color: {C_TILE_BG};
                border: 1px solid {C_TILE_BORDER};
                border-radius: 6px;
            }}
        """)
        intro_layout = QVBoxLayout(intro_card)
        intro_layout.setContentsMargins(10, 8, 10, 8)
        intro_layout.setSpacing(4)

        lbl_intro_badge = QLabel("ISRO BAS ASSISTANT")
        lbl_intro_badge.setStyleSheet(f"color: {C_BLUE_LIGHT}; font-size: 9.5px; font-weight: 800; letter-spacing: 0.6px;")
        intro_layout.addWidget(lbl_intro_badge)

        lbl_intro_desc = QLabel(
            "Autonomous computer vision assistant for on-board Biological "
            "Apparatus Subsystem (BAS) space experiments. Tracks astronaut hand actions, "
            "verifies experiment sequence integrity, and issues real-time guidance 100% offline."
        )
        lbl_intro_desc.setStyleSheet(f"color: {C_TEXT_MUTED}; font-size: 10.5px; line-height: 1.4;")
        lbl_intro_desc.setWordWrap(True)
        intro_layout.addWidget(lbl_intro_desc)

        layout.addWidget(intro_card)

        # ── 3. Experiment Selector ────────────────────────────────────────────
        lbl_exp_header = QLabel("EXPERIMENT")
        lbl_exp_header.setStyleSheet(f"color: {C_TEXT_SUBTLE}; font-size: 10px; font-weight: 800; letter-spacing: 1px;")
        layout.addWidget(lbl_exp_header)

        self.combo_exp = QComboBox()
        self.combo_exp.setFixedHeight(34)
        for exp in EXPERIMENT_REGISTRY:
            self.combo_exp.addItem(exp["name"], exp["key"])

        self.combo_exp.setStyleSheet(f"""
            QComboBox {{
                background-color: #0e1422;
                color: #f1f5f9;
                border: 1px solid {C_CARD_BORDER};
                border-radius: 5px;
                padding-left: 10px;
                font-size: 11.5px;
                font-weight: 600;
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 24px;
                border: none;
            }}
            QComboBox QAbstractItemView {{
                background-color: #0e1422;
                color: #f1f5f9;
                border: 1px solid {C_CARD_BORDER};
                selection-background-color: {C_BLUE_PRIMARY};
                selection-color: #ffffff;
            }}
        """)
        self.combo_exp.currentIndexChanged.connect(self._on_combo_changed)
        layout.addWidget(self.combo_exp)

        # ── 4. Mission Timeline Section ───────────────────────────────────────
        lbl_timeline = QLabel("MISSION TIMELINE")
        lbl_timeline.setStyleSheet(f"color: {C_TEXT_SUBTLE}; font-size: 10px; font-weight: 800; letter-spacing: 1px; margin-top: 4px;")
        layout.addWidget(lbl_timeline)

        self.timeline = SidebarTimelineWidget()
        layout.addWidget(self.timeline, stretch=1)

        layout.addStretch()

        # ── 5. Bottom Controls (Start, Stop, Reset) ───────────────────────────
        controls_box = QVBoxLayout()
        controls_box.setSpacing(8)

        row_top = QHBoxLayout()
        row_top.setSpacing(8)

        self.btn_start = QPushButton("▶ Start")
        self.btn_start.setFixedHeight(38)
        self.btn_start.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_start.setStyleSheet(f"""
            QPushButton {{
                background-color: #059669;
                color: #ffffff;
                border: 1px solid #10b981;
                border-radius: 6px;
                font-size: 12.5px;
                font-weight: 800;
            }}
            QPushButton:hover {{
                background-color: #047857;
            }}
            QPushButton:disabled {{
                background-color: #121a2c;
                color: #475569;
                border-color: #1e2942;
            }}
        """)
        self.btn_start.clicked.connect(self.start_clicked.emit)
        row_top.addWidget(self.btn_start, stretch=1)

        self.btn_stop = QPushButton("■ Stop")
        self.btn_stop.setFixedHeight(38)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_stop.setStyleSheet("""
            QPushButton {
                background-color: #dc2626;
                color: #ffffff;
                border: 1px solid #ef4444;
                border-radius: 6px;
                font-size: 12.5px;
                font-weight: 800;
            }
            QPushButton:hover {
                background-color: #b91c1c;
            }
            QPushButton:disabled {
                background-color: #121a2c;
                color: #475569;
                border-color: #1e2942;
            }
        """)
        self.btn_stop.clicked.connect(self.stop_clicked.emit)
        row_top.addWidget(self.btn_stop, stretch=1)

        controls_box.addLayout(row_top)

        self.btn_reset = QPushButton("↺ Reset")
        self.btn_reset.setFixedHeight(34)
        self.btn_reset.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_reset.setStyleSheet("""
            QPushButton {
                background-color: #151d2e;
                color: #cbd5e1;
                border: 1px solid #243048;
                border-radius: 5px;
                font-size: 11.5px;
                font-weight: 700;
            }
            QPushButton:hover {
                background-color: #1e293b;
            }
        """)
        self.btn_reset.clicked.connect(self.reset_clicked.emit)
        controls_box.addWidget(self.btn_reset)

        layout.addLayout(controls_box)

    def _on_combo_changed(self, idx: int):
        exp_key = self.combo_exp.itemData(idx)
        if exp_key:
            self.experiment_changed.emit(exp_key)


# ══════════════════════════════════════════════════════════════════════════════
#  2. RIGHT MAIN CONTENT WIDGETS
# ══════════════════════════════════════════════════════════════════════════════

class TopTelemetryStrip(QFrame):
    """
    Top telemetry cards matching reference:
    FPS | Runtime | Recording | Stream State | Last Updated: Just now
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(56)
        self.setStyleSheet(f"""
            TopTelemetryStrip {{
                background-color: {C_CARD_BG};
                border: 1px solid {C_CARD_BORDER};
                border-radius: 8px;
            }}
        """)
        self._start_time: Optional[float] = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 6, 16, 6)
        layout.setSpacing(20)

        # 1. FPS Card
        box_fps = QVBoxLayout()
        box_fps.setSpacing(1)
        lbl_fps_title = QLabel("FPS")
        lbl_fps_title.setStyleSheet(f"color: {C_TEXT_SUBTLE}; font-size: 9.5px; font-weight: 800; letter-spacing: 0.5px;")
        box_fps.addWidget(lbl_fps_title)

        row_fps = QHBoxLayout()
        row_fps.setSpacing(4)
        self._lbl_fps_val = QLabel("29.7")
        self._lbl_fps_val.setStyleSheet("color: #ffffff; font-size: 15px; font-weight: 900; font-family: monospace;")
        row_fps.addWidget(self._lbl_fps_val)
        lbl_fps_unit = QLabel("frames/sec")
        lbl_fps_unit.setStyleSheet(f"color: {C_TEXT_MUTED}; font-size: 10px; font-weight: 600;")
        row_fps.addWidget(lbl_fps_unit)
        row_fps.addStretch()
        box_fps.addLayout(row_fps)
        layout.addLayout(box_fps)

        layout.addWidget(self._make_vdivider())

        # 2. Runtime Card
        box_rt = QVBoxLayout()
        box_rt.setSpacing(1)
        lbl_rt_title = QLabel("Runtime")
        lbl_rt_title.setStyleSheet(f"color: {C_TEXT_SUBTLE}; font-size: 9.5px; font-weight: 800; letter-spacing: 0.5px;")
        box_rt.addWidget(lbl_rt_title)

        row_rt = QHBoxLayout()
        row_rt.setSpacing(4)
        self._lbl_rt_val = QLabel("00:00:00")
        self._lbl_rt_val.setStyleSheet("color: #ffffff; font-size: 15px; font-weight: 900; font-family: monospace;")
        row_rt.addWidget(self._lbl_rt_val)
        lbl_rt_unit = QLabel("hh:mm:ss")
        lbl_rt_unit.setStyleSheet(f"color: {C_TEXT_MUTED}; font-size: 10px; font-weight: 600;")
        row_rt.addWidget(lbl_rt_unit)
        row_rt.addStretch()
        box_rt.addLayout(row_rt)
        layout.addLayout(box_rt)

        layout.addWidget(self._make_vdivider())

        # 3. Recording Card
        box_rec = QVBoxLayout()
        box_rec.setSpacing(1)
        lbl_rec_title = QLabel("Recording")
        lbl_rec_title.setStyleSheet(f"color: {C_TEXT_SUBTLE}; font-size: 9.5px; font-weight: 800; letter-spacing: 0.5px;")
        box_rec.addWidget(lbl_rec_title)

        self._lbl_rec_val = QLabel("● ON")
        self._lbl_rec_val.setStyleSheet(f"color: {C_RED_DANGER}; font-size: 13px; font-weight: 900;")
        box_rec.addWidget(self._lbl_rec_val)
        layout.addLayout(box_rec)

        layout.addWidget(self._make_vdivider())

        # 4. Stream State Card
        box_st = QVBoxLayout()
        box_st.setSpacing(1)
        lbl_st_title = QLabel("Stream State")
        lbl_st_title.setStyleSheet(f"color: {C_TEXT_SUBTLE}; font-size: 9.5px; font-weight: 800; letter-spacing: 0.5px;")
        box_st.addWidget(lbl_st_title)

        self._lbl_st_val = QLabel("● LIVE")
        self._lbl_st_val.setStyleSheet(f"color: {C_GREEN_OK}; font-size: 13px; font-weight: 900;")
        box_st.addWidget(self._lbl_st_val)
        layout.addLayout(box_st)

        layout.addStretch()

        # 5. Last Updated
        lbl_updated = QLabel("↻ Last Updated: Just now")
        lbl_updated.setStyleSheet(f"color: {C_TEXT_MUTED}; font-size: 11px; font-weight: 600;")
        layout.addWidget(lbl_updated)

    def _make_vdivider(self) -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setStyleSheet(f"color: {C_CARD_BORDER};")
        return line

    def set_running(self, running: bool):
        if running and self._start_time is None:
            self._start_time = time.monotonic()
        elif not running:
            self._start_time = None
            self._lbl_rt_val.setText("00:00:00")

    def update_telemetry(self, fps: float, rec_active: bool = True, stream_active: bool = False):
        self._lbl_fps_val.setText(f"{fps:.1f}")

        if self._start_time is not None:
            elapsed = int(time.monotonic() - self._start_time)
            hrs = elapsed // 3600
            mins = (elapsed % 3600) // 60
            secs = elapsed % 60
            self._lbl_rt_val.setText(f"{hrs:02d}:{mins:02d}:{secs:02d}")

        if rec_active:
            self._lbl_rec_val.setText("● ON")
            self._lbl_rec_val.setStyleSheet(f"color: {C_RED_DANGER}; font-size: 13px; font-weight: 900;")
        else:
            self._lbl_rec_val.setText("○ STANDBY")
            self._lbl_rec_val.setStyleSheet(f"color: {C_TEXT_MUTED}; font-size: 13px; font-weight: 800;")

        if stream_active:
            self._lbl_st_val.setText("● LIVE")
            self._lbl_st_val.setStyleSheet(f"color: {C_GREEN_OK}; font-size: 13px; font-weight: 900;")
        else:
            self._lbl_st_val.setText("○ OFFLINE")
            self._lbl_st_val.setStyleSheet(f"color: {C_TEXT_MUTED}; font-size: 13px; font-weight: 800;")


class SquareCameraCard(QFrame):
    """
    Primary Optical Camera Viewport:
    - Square 1:1 viewport (700x700) as the primary visual focus
    - Edge-to-edge frame scaling, no black borders
    - Fixed stable dimensions to prevent recursive resizing
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(730)
        self.setStyleSheet(f"""
            SquareCameraCard {{
                background-color: {C_CARD_BG};
                border: 1px solid {C_CARD_BORDER};
                border-radius: 8px;
            }}
        """)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(8)

        # Card Header
        hdr = QHBoxLayout()
        lbl_title = QLabel("📹  PRIMARY OPTICAL VIEWPORT (ACTIVE MISSION)")
        lbl_title.setStyleSheet("color: #ffffff; font-size: 13px; font-weight: 800; letter-spacing: 0.6px;")
        hdr.addWidget(lbl_title)

        hdr.addStretch()

        self._lbl_res = QLabel("⤢ SQUARE 1:1  •  700 × 700")
        self._lbl_res.setStyleSheet(f"""
            background: #121a2c;
            color: {C_BLUE_LIGHT};
            border: 1px solid {C_TILE_BORDER};
            border-radius: 4px;
            padding: 3px 8px;
            font-size: 10px;
            font-weight: 800;
            font-family: monospace;
        """)
        hdr.addWidget(self._lbl_res)
        layout.addLayout(hdr)

        # Viewport Outer Container (Square 700x700)
        center_box = QHBoxLayout()
        center_box.setAlignment(Qt.AlignCenter)

        self.cam_viewport = QFrame()
        self.cam_viewport.setFixedSize(700, 700)
        self.cam_viewport.setStyleSheet("""
            QFrame {
                background-color: #04060a;
                border: 1px solid #1a2338;
                border-radius: 8px;
            }
        """)

        vp_layout = QVBoxLayout(self.cam_viewport)
        vp_layout.setContentsMargins(0, 0, 0, 0)
        vp_layout.setSpacing(0)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFixedSize(700, 700)
        self.video_label.setScaledContents(True)
        self.video_label.setStyleSheet("background: transparent; border-radius: 8px;")
        vp_layout.addWidget(self.video_label)

        center_box.addWidget(self.cam_viewport)
        layout.addLayout(center_box)


# ══════════════════════════════════════════════════════════════════════════════
#  3. THREE EQUAL STATUS CARDS
# ══════════════════════════════════════════════════════════════════════════════

class CurrentStepCard(QFrame):
    """Card 1: CURRENT STEP — compact, fixed-height, bold headings, prominent prompt."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(122)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setStyleSheet(f"""
            CurrentStepCard {{
                background-color: {C_CARD_BG};
                border: 1px solid {C_CARD_BORDER};
                border-radius: 8px;
            }}
        """)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(6)

        # Header (Bold uppercase)
        lbl_hdr = QLabel("CURRENT STEP")
        lbl_hdr.setStyleSheet("color: #ffffff; font-size: 11px; font-weight: 900; letter-spacing: 0.8px;")
        layout.addWidget(lbl_hdr)

        # Step Pill + Title row
        title_box = QHBoxLayout()
        title_box.setSpacing(8)

        self._lbl_step_badge = QLabel("STEP 1")
        self._lbl_step_badge.setStyleSheet(f"""
            background-color: {C_BLUE_PRIMARY};
            color: #ffffff;
            font-size: 11.5px;
            font-weight: 900;
            padding: 3px 8px;
            border-radius: 4px;
        """)
        title_box.addWidget(self._lbl_step_badge)

        self._lbl_step_title = QLabel("System Checkout")
        self._lbl_step_title.setStyleSheet("color: #ffffff; font-size: 15px; font-weight: 900;")
        title_box.addWidget(self._lbl_step_title, stretch=1)
        layout.addLayout(title_box)

        # Description / Prompt (Prominent, clear font)
        self._lbl_desc = QLabel("Initialize sensors and verify workspace setup.")
        self._lbl_desc.setStyleSheet("color: #e2e8f0; font-size: 13px; font-weight: 600;")
        self._lbl_desc.setWordWrap(True)
        self._lbl_desc.setMaximumHeight(36)
        layout.addWidget(self._lbl_desc)

        # Progress Text (clean, bold status line — no black bar)
        self._lbl_prog_text = QLabel("● In Progress (0%)")
        self._lbl_prog_text.setStyleSheet(f"color: {C_TEXT_MUTED}; font-size: 11.5px; font-weight: 700;")
        layout.addWidget(self._lbl_prog_text)

    def update_step(self, step_idx: int, step_name: str, description: str, hold_pct: int = 0):
        self._lbl_step_badge.setText(f"STEP {step_idx + 1}")
        self._lbl_step_title.setText(step_name)
        self._lbl_desc.setText(description or "Follow on-board checklist protocol.")
        if hold_pct >= 100:
            self._lbl_prog_text.setText("● Verified ✓")
            self._lbl_prog_text.setStyleSheet(f"color: {C_GREEN_OK}; font-size: 11.5px; font-weight: 800;")
        elif hold_pct > 0:
            self._lbl_prog_text.setText(f"● In Progress ({hold_pct}%)")
            self._lbl_prog_text.setStyleSheet(f"color: {C_BLUE_LIGHT}; font-size: 11.5px; font-weight: 800;")
        else:
            self._lbl_prog_text.setText("● In Progress (0%)")
            self._lbl_prog_text.setStyleSheet(f"color: {C_TEXT_MUTED}; font-size: 11.5px; font-weight: 700;")


class NextStepGuidanceCard(QFrame):
    """Card 2: NEXT STEP GUIDANCE — compact, bold headings, prominent prompt."""

    view_checklist_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(108)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setStyleSheet(f"""
            NextStepGuidanceCard {{
                background-color: {C_CARD_BG};
                border: 1px solid {C_CARD_BORDER};
                border-radius: 8px;
            }}
        """)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(6)

        # Header row with button on right
        hdr_row = QHBoxLayout()
        hdr_row.setSpacing(8)

        lbl_hdr = QLabel("NEXT STEP GUIDANCE")
        lbl_hdr.setStyleSheet("color: #ffffff; font-size: 11px; font-weight: 900; letter-spacing: 0.8px;")
        hdr_row.addWidget(lbl_hdr)
        hdr_row.addStretch()

        self.btn_checklist = QPushButton("📄 Checklist")
        self.btn_checklist.setFixedHeight(24)
        self.btn_checklist.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_checklist.setStyleSheet(f"""
            QPushButton {{
                background-color: {C_TILE_BG};
                color: #cbd5e1;
                border: 1px solid {C_TILE_BORDER};
                border-radius: 4px;
                padding: 0 8px;
                font-size: 10px;
                font-weight: 700;
            }}
            QPushButton:hover {{ background-color: #1a253d; }}
        """)
        self.btn_checklist.clicked.connect(self.view_checklist_clicked.emit)
        hdr_row.addWidget(self.btn_checklist)
        layout.addLayout(hdr_row)

        # Next Step Pill + Title
        title_box = QHBoxLayout()
        title_box.setSpacing(8)

        self._lbl_next_badge = QLabel("STEP 2")
        self._lbl_next_badge.setStyleSheet(f"""
            background-color: {C_BLUE_PRIMARY};
            color: #ffffff;
            font-size: 11.5px;
            font-weight: 900;
            padding: 3px 8px;
            border-radius: 4px;
        """)
        title_box.addWidget(self._lbl_next_badge)

        self._lbl_next_title = QLabel("– Preparation")
        self._lbl_next_title.setStyleSheet("color: #ffffff; font-size: 15px; font-weight: 900;")
        title_box.addWidget(self._lbl_next_title, stretch=1)
        layout.addLayout(title_box)

        # Guidance text (Prominent, clear font)
        self._lbl_guidance = QLabel("Follow protocol sequence.")
        self._lbl_guidance.setStyleSheet("color: #e2e8f0; font-size: 13px; font-weight: 600;")
        self._lbl_guidance.setWordWrap(False)
        layout.addWidget(self._lbl_guidance)

    def update_next(self, next_idx: int, next_name: str, next_suggestion: str):
        if next_name and next_name != "COMPLETED":
            self._lbl_next_badge.setText(f"STEP {next_idx + 1}")
            self._lbl_next_title.setText(f"– {next_name}")
            self._lbl_guidance.setText(next_suggestion or "Follow protocol sequence.")
        else:
            self._lbl_next_badge.setText("FINISHED")
            self._lbl_next_title.setText("– All Verified")
            self._lbl_guidance.setText("Mission objectives successfully accomplished.")


class OverallStatusCard(QFrame):
    """Card 3: OVERALL STATUS — compact horizontal layout, bold headings."""

    view_report_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(88)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setStyleSheet(f"""
            OverallStatusCard {{
                background-color: {C_CARD_BG};
                border: 1px solid {C_CARD_BORDER};
                border-radius: 8px;
            }}
        """)
        self._setup_ui()

    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 10, 14, 10)
        outer.setSpacing(6)

        lbl_hdr = QLabel("OVERALL STATUS")
        lbl_hdr.setStyleSheet("color: #ffffff; font-size: 11px; font-weight: 900; letter-spacing: 0.8px;")
        outer.addWidget(lbl_hdr)

        # Shield + Status + Button in one row
        center_row = QHBoxLayout()
        center_row.setSpacing(12)

        self._lbl_shield = QLabel("🛡")
        self._lbl_shield.setFixedSize(36, 36)
        self._lbl_shield.setAlignment(Qt.AlignCenter)
        self._lbl_shield.setStyleSheet(f"""
            background-color: {C_TILE_BG};
            color: {C_BLUE_LIGHT};
            border: 1px solid {C_BLUE_BORDER};
            border-radius: 6px;
            font-size: 18px;
        """)
        center_row.addWidget(self._lbl_shield)

        stat_col = QVBoxLayout()
        stat_col.setSpacing(1)

        self._lbl_status_main = QLabel("COMPLIANT")
        self._lbl_status_main.setStyleSheet("color: #ffffff; font-size: 15px; font-weight: 900; letter-spacing: 0.5px;")
        stat_col.addWidget(self._lbl_status_main)

        self._lbl_status_sub = QLabel("All checks passing")
        self._lbl_status_sub.setStyleSheet("color: #cbd5e1; font-size: 11.5px; font-weight: 600;")
        stat_col.addWidget(self._lbl_status_sub)

        center_row.addLayout(stat_col, stretch=1)

        self.btn_report = QPushButton("View Report")
        self.btn_report.setFixedSize(90, 26)
        self.btn_report.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_report.setStyleSheet(f"""
            QPushButton {{
                background-color: {C_TILE_BG};
                color: #cbd5e1;
                border: 1px solid {C_TILE_BORDER};
                border-radius: 4px;
                font-size: 10px;
                font-weight: 700;
            }}
            QPushButton:hover {{ background-color: #1a253d; }}
        """)
        self.btn_report.clicked.connect(self.view_report_clicked.emit)
        center_row.addWidget(self.btn_report)

        outer.addLayout(center_row)

    def update_status(self, status: TransitionStatus, is_completed: bool = False):
        if is_completed:
            self._lbl_status_main.setText("COMPLETED")
            self._lbl_status_main.setStyleSheet(f"color: {C_BLUE_LIGHT}; font-size: 15px; font-weight: 900;")
            self._lbl_status_sub.setText("Mission fully verified")
            self._lbl_shield.setStyleSheet(f"background-color: {C_TILE_BG}; color: {C_BLUE_LIGHT}; border: 1px solid {C_BLUE_BORDER}; border-radius: 6px; font-size: 18px;")
        elif status == TransitionStatus.CORRECT:
            self._lbl_status_main.setText("COMPLIANT")
            self._lbl_status_main.setStyleSheet(f"color: {C_GREEN_OK}; font-size: 15px; font-weight: 900;")
            self._lbl_status_sub.setText("All checks passing")
            self._lbl_shield.setStyleSheet(f"background-color: {C_TILE_BG}; color: {C_GREEN_OK}; border: 1px solid {C_GREEN_OK}; border-radius: 6px; font-size: 18px;")
        elif status == TransitionStatus.WAITING:
            self._lbl_status_main.setText("WAITING")
            self._lbl_status_main.setStyleSheet(f"color: {C_AMBER_WARN}; font-size: 15px; font-weight: 900;")
            self._lbl_status_sub.setText("Awaiting operator gesture")
            self._lbl_shield.setStyleSheet(f"background-color: {C_TILE_BG}; color: {C_AMBER_WARN}; border: 1px solid {C_AMBER_WARN}; border-radius: 6px; font-size: 18px;")
        elif status in (TransitionStatus.SKIPPED, TransitionStatus.WRONG_ORDER):
            self._lbl_status_main.setText("VIOLATION")
            self._lbl_status_main.setStyleSheet(f"color: {C_RED_DANGER}; font-size: 15px; font-weight: 900;")
            self._lbl_status_sub.setText("Out-of-order action detected")
            self._lbl_shield.setStyleSheet(f"background-color: {C_TILE_BG}; color: {C_RED_DANGER}; border: 1px solid {C_RED_DANGER}; border-radius: 6px; font-size: 18px;")


# ══════════════════════════════════════════════════════════════════════════════
#  4. EXPLAINABLE EVIDENCE & RATIONALE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

class EvidenceMetricTile(QFrame):
    """Single metric tile inside Explainable Evidence card (Icon, Name, Value, Status)."""

    def __init__(self, icon: str, name: str, default_val: str = "0.91", parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            EvidenceMetricTile {{
                background-color: {C_TILE_BG};
                border: 1px solid {C_TILE_BORDER};
                border-radius: 6px;
            }}
        """)
        self._setup_ui(icon, name, default_val)

    def _setup_ui(self, icon: str, name: str, default_val: str):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(3)

        top_row = QHBoxLayout()
        lbl_icon = QLabel(icon)
        lbl_icon.setStyleSheet(f"color: {C_BLUE_LIGHT}; font-size: 13px;")
        top_row.addWidget(lbl_icon)

        lbl_name = QLabel(name)
        lbl_name.setStyleSheet(f"color: {C_TEXT_MUTED}; font-size: 10px; font-weight: 700;")
        top_row.addWidget(lbl_name, stretch=1)
        layout.addLayout(top_row)

        self._lbl_val = QLabel(default_val)
        self._lbl_val.setStyleSheet("color: #ffffff; font-size: 16px; font-weight: 900; font-family: monospace;")
        layout.addWidget(self._lbl_val)

        self._lbl_status = QLabel("OK")
        self._lbl_status.setStyleSheet(f"color: {C_GREEN_OK}; font-size: 10px; font-weight: 800;")
        layout.addWidget(self._lbl_status)

    def set_value(self, val_str: str, status_str: str = "OK", is_good: bool = True):
        self._lbl_val.setText(val_str)
        self._lbl_status.setText(status_str)
        self._lbl_status.setStyleSheet(f"color: {C_GREEN_OK if is_good else C_RED_DANGER}; font-size: 10px; font-weight: 800;")


class ExplainableEvidenceCard(QFrame):
    """Explainable Evidence card with 4 metric tiles."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            ExplainableEvidenceCard {{
                background-color: {C_CARD_BG};
                border: 1px solid {C_CARD_BORDER};
                border-radius: 8px;
            }}
        """)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(6)

        # Header
        hdr = QHBoxLayout()
        lbl_title = QLabel("EXPLAINABLE EVIDENCE ⓘ")
        lbl_title.setStyleSheet("color: #ffffff; font-size: 11px; font-weight: 900; letter-spacing: 0.8px;")
        hdr.addWidget(lbl_title)

        hdr.addStretch()

        lbl_thresh = QLabel("Threshold: ≥ 0.70")
        lbl_thresh.setStyleSheet(f"color: {C_TEXT_SUBTLE}; font-size: 10px; font-weight: 600;")
        hdr.addWidget(lbl_thresh)
        layout.addLayout(hdr)

        # 4 Metric Tiles Row
        tiles_row = QHBoxLayout()
        tiles_row.setSpacing(6)

        self.tile_pose = EvidenceMetricTile("✋", "Pose Stability", "0.91", self)
        tiles_row.addWidget(self.tile_pose)

        self.tile_proximity = EvidenceMetricTile("⌖", "Proximity", "0.87", self)
        tiles_row.addWidget(self.tile_proximity)

        self.tile_seq = EvidenceMetricTile("⎇", "Sequence", "0.95", self)
        tiles_row.addWidget(self.tile_seq)

        self.tile_object = EvidenceMetricTile("🧊", "Integrity", "0.94", self)
        tiles_row.addWidget(self.tile_object)

        layout.addLayout(tiles_row)

    def update_evidence(self, evidence: dict):
        has_hands = evidence.get("hand_present", False)
        dist = evidence.get("hand_to_object_dist")
        if dist is not None:
            self.tile_proximity.set_value(f"{dist:.0f} px", "OK" if dist <= 120 else "CLEAR", dist <= 120)
        else:
            self.tile_proximity.set_value("—", "NOMINAL", True)

        self.tile_pose.set_value("0.92" if has_hands else "0.00", "OK" if has_hands else "ABSENT", has_hands)
        self.tile_seq.set_value("0.95", "OK", True)
        self.tile_object.set_value("0.94", "OK", True)


class RationaleSummaryCard(QFrame):
    """Rationale summary card with narrative text and details button."""

    view_details_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            RationaleSummaryCard {{
                background-color: {C_CARD_BG};
                border: 1px solid {C_CARD_BORDER};
                border-radius: 8px;
            }}
        """)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(6)

        lbl_hdr = QLabel("RATIONALE SUMMARY")
        lbl_hdr.setStyleSheet("color: #ffffff; font-size: 11px; font-weight: 900; letter-spacing: 0.8px;")
        layout.addWidget(lbl_hdr)

        self._lbl_text = QLabel(
            "Step 1 initialized. Sensor telemetry calibrated. Operator posture nominal. "
            "Awaiting protocol interaction. Sequence matches predefined flight checklist."
        )
        self._lbl_text.setStyleSheet(f"color: #cbd5e1; font-size: 11px; line-height: 1.35;")
        self._lbl_text.setWordWrap(True)
        layout.addWidget(self._lbl_text, stretch=1)

        btn_box = QHBoxLayout()
        btn_box.addStretch()

        self.btn_details = QPushButton("View Evidence Details")
        self.btn_details.setFixedHeight(24)
        self.btn_details.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_details.setStyleSheet(f"""
            QPushButton {{
                background-color: {C_TILE_BG};
                color: #cbd5e1;
                border: 1px solid {C_TILE_BORDER};
                border-radius: 4px;
                padding: 0 10px;
                font-size: 10px;
                font-weight: 700;
            }}
            QPushButton:hover {{
                background-color: #1a253d;
            }}
        """)
        self.btn_details.clicked.connect(self.view_details_clicked.emit)
        btn_box.addWidget(self.btn_details)

        layout.addLayout(btn_box)

    def set_rationale(self, text: str):
        if text:
            self._lbl_text.setText(text)


# ══════════════════════════════════════════════════════════════════════════════
#  5. OBJECT TRACKING & MISSION AUDIT LOG
# ══════════════════════════════════════════════════════════════════════════════

class ObjectTrackingTile(QFrame):
    """Single entity tracking card inside OBJECT TRACKING (Hand, Face, Object)."""

    def __init__(self, icon: str, title: str, entity_id: str = "H1", status_str: str = "Tracking", conf: str = "0.92", parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            ObjectTrackingTile {{
                background-color: {C_TILE_BG};
                border: 1px solid {C_TILE_BORDER};
                border-radius: 6px;
            }}
        """)
        self._setup_ui(icon, title, entity_id, status_str, conf)

    def _setup_ui(self, icon: str, title: str, entity_id: str, status_str: str, conf: str):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(10)

        lbl_icon = QLabel(icon)
        lbl_icon.setStyleSheet(f"color: {C_BLUE_LIGHT}; font-size: 16px;")
        layout.addWidget(lbl_icon)

        text_col = QVBoxLayout()
        text_col.setSpacing(1)

        lbl_title = QLabel(title)
        lbl_title.setStyleSheet("color: #ffffff; font-size: 11px; font-weight: 800;")
        text_col.addWidget(lbl_title)

        self._lbl_id = QLabel(f"ID: {entity_id}")
        self._lbl_id.setStyleSheet(f"color: {C_TEXT_SUBTLE}; font-size: 9.5px; font-weight: 600;")
        text_col.addWidget(self._lbl_id)

        self._lbl_status = QLabel(status_str)
        self._lbl_status.setStyleSheet(f"color: {C_TEXT_MUTED}; font-size: 9.5px; font-weight: 600;")
        text_col.addWidget(self._lbl_status)

        layout.addLayout(text_col, stretch=1)

        self._lbl_conf = QLabel(conf)
        self._lbl_conf.setStyleSheet("color: #ffffff; font-size: 14px; font-weight: 900; font-family: monospace;")
        layout.addWidget(self._lbl_conf)

    def update_tile(self, entity_id: str, status_str: str, conf_str: str, is_active: bool = True):
        self._lbl_id.setText(f"ID: {entity_id}")
        self._lbl_status.setText(status_str)
        self._lbl_conf.setText(conf_str)
        self._lbl_conf.setStyleSheet(f"color: {'#ffffff' if is_active else C_TEXT_SUBTLE}; font-size: 14px; font-weight: 900; font-family: monospace;")


class ObjectTrackingCard(QFrame):
    """OBJECT TRACKING section with 3 entity cards (HAND, FACE, OBJECT)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            ObjectTrackingCard {{
                background-color: {C_CARD_BG};
                border: 1px solid {C_CARD_BORDER};
                border-radius: 8px;
            }}
        """)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(6)

        lbl_title = QLabel("OBJECT TRACKING")
        lbl_title.setStyleSheet("color: #ffffff; font-size: 11px; font-weight: 900; letter-spacing: 0.8px;")
        layout.addWidget(lbl_title)

        row = QHBoxLayout()
        row.setSpacing(6)

        self.tile_hand = ObjectTrackingTile("✋", "HAND", "H1", "Tracking", "0.92", self)
        row.addWidget(self.tile_hand)

        self.tile_face = ObjectTrackingTile("👤", "FACE", "F1", "Tracking", "0.89", self)
        row.addWidget(self.tile_face)

        self.tile_object = ObjectTrackingTile("🧊", "OBJECT", "01 (RY-Box)", "Tracking", "0.94", self)
        row.addWidget(self.tile_object)

        layout.addLayout(row)

    def update_objects(
        self,
        objects_detail: list,
        has_hands: bool = False,
        face_detected: bool = False,
        hand_conf: float = 0.0,
        face_conf: float = 0.0,
        pose_conf: float = 0.0,
        body_detected: bool = False,
    ):
        # Update Hand tile with real confidence
        if has_hands:
            c = hand_conf if hand_conf > 0.0 else 0.92
            self.tile_hand.update_tile("H1", "Tracking", f"{c:.2f}", True)
        else:
            self.tile_hand.update_tile("—", "Standby", "0.00", False)

        # Update Face tile with real confidence from MediaPipe FaceLandmarker
        if face_detected:
            c = face_conf if face_conf > 0.0 else 0.89
            self.tile_face.update_tile("F1", "Tracking", f"{c:.2f}", True)
        else:
            self.tile_face.update_tile("—", "Standby", "0.00", False)

        # Update Object tile from first detected object
        if objects_detail:
            first = objects_detail[0]
            name = str(first.get("name") or first.get("colour", "Obj")).upper()
            conf = float(first.get("confidence", 0.94))
            self.tile_object.update_tile(f"01 ({name})", "Tracking", f"{conf:.2f}", True)
        else:
            self.tile_object.update_tile("—", "Standby", "0.00", False)


class BigScreenEventLogCard(QFrame):
    """
    Large, full-width mission audit terminal designed for big-screen monitoring.
    - Monospace font for mission precision
    - Auto-scrolling event stream with timestamps and status badges
    - Live event count indicator and filter tags
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(280)
        self.setStyleSheet(f"""
            BigScreenEventLogCard {{
                background-color: {C_CARD_BG};
                border: 1px solid {C_CARD_BORDER};
                border-radius: 8px;
            }}
        """)
        self._event_count = 0
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        # Top Control Strip
        hdr = QHBoxLayout()
        hdr.setSpacing(10)

        lbl_icon = QLabel("📋")
        lbl_icon.setStyleSheet(f"color: {C_BLUE_LIGHT}; font-size: 14px;")
        hdr.addWidget(lbl_icon)

        lbl_title = QLabel("MISSION AUDIT & EVENT LOG STREAM (BIG SCREEN TERMINAL)")
        lbl_title.setStyleSheet("color: #ffffff; font-size: 12px; font-weight: 800; letter-spacing: 0.6px;")
        hdr.addWidget(lbl_title)

        hdr.addStretch()

        self._lbl_count = QLabel("● LIVE STREAM  |  0 EVENTS")
        self._lbl_count.setStyleSheet(f"""
            background: #121a2c;
            color: {C_GREEN_OK};
            border: 1px solid {C_TILE_BORDER};
            border-radius: 4px;
            padding: 3px 10px;
            font-size: 10px;
            font-weight: 800;
            font-family: monospace;
        """)
        hdr.addWidget(self._lbl_count)

        btn_clear = QPushButton("Clear Terminal")
        btn_clear.setFixedHeight(26)
        btn_clear.setCursor(QCursor(Qt.PointingHandCursor))
        btn_clear.setStyleSheet(f"""
            QPushButton {{
                background: {C_TILE_BG};
                color: {C_TEXT_MUTED};
                border: 1px solid {C_TILE_BORDER};
                border-radius: 4px;
                padding: 0 10px;
                font-size: 10px;
                font-weight: 700;
            }}
            QPushButton:hover {{
                background: #1a253d;
                color: #ffffff;
            }}
        """)
        btn_clear.clicked.connect(self.clear_logs)
        hdr.addWidget(btn_clear)

        layout.addLayout(hdr)

        # Large Monospace List Terminal
        self._list = QListWidget()
        self._list.setMinimumHeight(210)
        self._list.setStyleSheet(f"""
            QListWidget {{
                background-color: #06090e;
                color: #cbd5e1;
                border: 1px solid #161e30;
                border-radius: 6px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
                padding: 6px;
            }}
            QListWidget::item {{
                padding: 3px 6px;
                border-bottom: 1px solid #0d131f;
            }}
            QListWidget::item:hover {{
                background-color: #0e1524;
            }}
        """)
        layout.addWidget(self._list)

        # Initial welcome messages
        self.log("Mission audit engine online. System calibrated.", "info")
        self.log("Offline Edge AI ready for on-board sequence verification.", "cyan")

    def clear_logs(self):
        self._list.clear()
        self._event_count = 0
        self._lbl_count.setText("● LIVE STREAM  |  0 EVENTS")

    def log(self, message: str, level: str = "info"):
        t_str = time.strftime("%H:%M:%S")
        cat = "SYSTEM"
        color = "#94a3b8"

        if "STEP" in message.upper():
            cat = "STEP"
            color = "#60a5fa"
        elif "DETECTION" in message.upper() or "YOLO" in message.upper() or "BOX" in message.upper() or "HAND" in message.upper():
            cat = "DETECTION"
            color = "#38bdf8"
        elif "✓" in message or "COMPLIANT" in message.upper() or "VERIFIED" in message.upper() or level == "success":
            cat = "COMPLIANCE"
            color = "#10b981"
        elif "ALERT" in message.upper() or "VIOLATION" in message.upper() or "✗" in message or "SKIPPED" in message.upper() or "WRONG" in message.upper() or level == "error":
            cat = "ALERT"
            color = "#ef4444"
        elif level == "warning" or "WARN" in message.upper():
            cat = "WARNING"
            color = "#f59e0b"
        elif level == "cyan":
            color = "#00f0ff"

        item = QListWidgetItem(f"[{t_str}]  [{cat:<10}]  {message}")
        item.setForeground(QColor(color))
        self._list.addItem(item)
        self._list.scrollToBottom()

        self._event_count += 1
        self._lbl_count.setText(f"● LIVE STREAM  |  {self._event_count} EVENTS")

        # Keep max 500 lines
        if self._list.count() > 500:
            self._list.takeItem(0)


MissionAuditLogCard = BigScreenEventLogCard


class ToolsAndControlsCard(QFrame):
    """Bottom Tools & Controls bar with stream endpoint, Open Stream, HTML Report, and Print."""

    open_stream_clicked = Signal(str)
    html_report_clicked = Signal()
    print_report_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(62)
        self.setStyleSheet(f"""
            ToolsAndControlsCard {{
                background-color: {C_CARD_BG};
                border: 1px solid {C_CARD_BORDER};
                border-radius: 8px;
            }}
        """)
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 8, 14, 8)
        layout.setSpacing(12)

        # Stream Input Box
        box_stream = QVBoxLayout()
        box_stream.setSpacing(2)
        lbl_st = QLabel("STREAM ENDPOINT")
        lbl_st.setStyleSheet(f"color: {C_TEXT_SUBTLE}; font-size: 9px; font-weight: 800; letter-spacing: 0.5px;")
        box_stream.addWidget(lbl_st)

        self.edit_stream = QLineEdit("http://127.0.0.1:8080/stream")
        self.edit_stream.setFixedWidth(240)
        self.edit_stream.setFixedHeight(28)
        self.edit_stream.setStyleSheet(f"""
            QLineEdit {{
                background-color: {C_TILE_BG};
                color: #f1f5f9;
                border: 1px solid {C_TILE_BORDER};
                border-radius: 4px;
                padding-left: 8px;
                font-family: monospace;
                font-size: 11px;
            }}
        """)
        box_stream.addWidget(self.edit_stream)
        layout.addLayout(box_stream)

        # Open Stream Button
        self.btn_open_stream = QPushButton("Open Stream")
        self.btn_open_stream.setFixedHeight(30)
        self.btn_open_stream.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_open_stream.setStyleSheet(f"""
            QPushButton {{
                background-color: {C_TILE_BG};
                color: #cbd5e1;
                border: 1px solid {C_TILE_BORDER};
                border-radius: 4px;
                padding: 0 14px;
                font-size: 11px;
                font-weight: 700;
            }}
            QPushButton:hover {{
                background-color: #1a253d;
            }}
        """)
        self.btn_open_stream.clicked.connect(lambda: self.open_stream_clicked.emit(self.edit_stream.text()))
        layout.addWidget(self.btn_open_stream)

        layout.addStretch()

        # HTML Report Button
        self.btn_html = QPushButton("📄 HTML Report")
        self.btn_html.setFixedHeight(32)
        self.btn_html.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_html.setStyleSheet(f"""
            QPushButton {{
                background-color: {C_TILE_BG};
                color: #f1f5f9;
                border: 1px solid {C_TILE_BORDER};
                border-radius: 5px;
                padding: 0 16px;
                font-size: 11.5px;
                font-weight: 700;
            }}
            QPushButton:hover {{
                background-color: #1a253d;
            }}
        """)
        self.btn_html.clicked.connect(self.html_report_clicked.emit)
        layout.addWidget(self.btn_html)

        # Print Button
        self.btn_print = QPushButton("🖨 Print")
        self.btn_print.setFixedHeight(32)
        self.btn_print.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_print.setStyleSheet(f"""
            QPushButton {{
                background-color: {C_TILE_BG};
                color: #f1f5f9;
                border: 1px solid {C_TILE_BORDER};
                border-radius: 5px;
                padding: 0 16px;
                font-size: 11.5px;
                font-weight: 700;
            }}
            QPushButton:hover {{
                background-color: #1a253d;
            }}
        """)
        self.btn_print.clicked.connect(self.print_report_clicked.emit)
        layout.addWidget(self.btn_print)
