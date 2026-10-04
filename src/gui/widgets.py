"""
widgets.py
==========
SIH26174 • VYOM: ON-BOARD PROTOCOL COMPLIANCE ASSISTANT
Aerospace-Grade Mission Control GUI Components for ISRO BAS Experiments.

Design Philosophy:
- Dark obsidian & deep slate palette (#0b0e14, #111622, #161e2e)
- Vibrant cyan/teal accents (#00f0ff, #0ea5e9) with disciplined status signals
- Dense, readable flight-telemetry layout
- Explainable AI Evidence Panel (Critical USP)
- Multi-experiment protocol switching & interactive audit controls
"""

from __future__ import annotations

import time
import pathlib
import webbrowser
from typing import Optional

from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtGui import QColor, QFont, QCursor
from PySide6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame,
    QProgressBar, QListWidget, QListWidgetItem, QSizePolicy,
    QGridLayout, QDialog, QPushButton, QComboBox, QLineEdit,
)

from src.protocol.state_machine import TransitionStatus, EXPERIMENT_REGISTRY


# ── Color Palette (Aerospace Obsidian & Cyan Accents) ─────────────────────────
C_BG_BASE        = "#0b0e14"
C_PANEL_BG       = "#111622"
C_PANEL_ALT      = "#161e2e"
C_BORDER         = "#1e293b"
C_BORDER_SUBTLE  = "#334155"

# Semantic Accents
C_CYAN_PRIMARY   = "#00f0ff"
C_CYAN_BG        = "#0c2842"
C_CYAN_BORDER    = "#0284c7"

C_EMERALD_TEXT   = "#34d399"
C_EMERALD_BG     = "#064e3b"
C_EMERALD_BORDER = "#059669"

C_AMBER_TEXT     = "#fbbf24"
C_AMBER_BG       = "#3b2e04"
C_AMBER_BORDER   = "#d97706"

C_ROSE_TEXT      = "#f87171"
C_ROSE_BG        = "#450a0a"
C_ROSE_BORDER    = "#dc2626"

C_SKY_TEXT       = "#38bdf8"
C_SKY_BG         = "#075985"
C_SKY_BORDER     = "#0284c7"

C_TEXT_PRIMARY   = "#f8fafc"
C_TEXT_SECONDARY = "#94a3b8"
C_TEXT_MUTED     = "#64748b"


# ══════════════════════════════════════════════════════════════════════════════
#  1. Header Bar: VYOM + Tagline + Offline Edge Mode
# ══════════════════════════════════════════════════════════════════════════════

class HeaderBar(QFrame):
    """
    Top mission header banner:
    VYOM • ON-BOARD PROTOCOL COMPLIANCE ASSISTANT | ISRO - SIH26174
    Offline Edge Mode badge + System status chip.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(50)
        self.setStyleSheet(f"""
            HeaderBar {{
                background-color: {C_PANEL_BG};
                border-bottom: 2px solid {C_BORDER};
            }}
        """)
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 0, 18, 0)
        layout.setSpacing(14)

        # Brand Title
        left_box = QHBoxLayout()
        left_box.setSpacing(12)

        lbl_brand = QLabel("VYOM")
        lbl_brand.setStyleSheet(f"""
            color: {C_CYAN_PRIMARY};
            font-size: 18px;
            font-weight: 900;
            letter-spacing: 2px;
            font-family: -apple-system, 'Segoe UI', system-ui, sans-serif;
        """)
        left_box.addWidget(lbl_brand)

        # Separator
        sep = QLabel("•")
        sep.setStyleSheet("color: #475569; font-size: 14px;")
        left_box.addWidget(sep)

        # Tagline
        lbl_tagline = QLabel("ON-BOARD PROTOCOL COMPLIANCE ASSISTANT")
        lbl_tagline.setStyleSheet(f"""
            color: {C_TEXT_SECONDARY};
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 1.2px;
        """)
        left_box.addWidget(lbl_tagline)

        # ISRO Label
        lbl_isro = QLabel("[ ISRO • BAS EXPERIMENTS ]")
        lbl_isro.setStyleSheet("""
            color: #0284c7;
            font-size: 10px;
            font-weight: 800;
            letter-spacing: 1px;
            padding-left: 6px;
        """)
        left_box.addWidget(lbl_isro)

        layout.addLayout(left_box)
        layout.addStretch()

        # Right Badges
        right_box = QHBoxLayout()
        right_box.setSpacing(10)

        # Offline Edge Badge (Critical USP)
        self._badge_offline = QLabel("⚡ OFFLINE EDGE MODE")
        self._badge_offline.setStyleSheet(f"""
            background: rgba(0, 240, 255, 0.12);
            color: {C_CYAN_PRIMARY};
            border: 1px solid rgba(0, 240, 255, 0.35);
            font-size: 10px;
            font-weight: 800;
            padding: 4px 10px;
            border-radius: 4px;
            letter-spacing: 0.8px;
        """)
        right_box.addWidget(self._badge_offline)

        # System Status Chip
        self._lbl_sys = QLabel("● SYSTEM IDLE")
        self._lbl_sys.setStyleSheet(f"""
            background: #1e293b;
            color: {C_TEXT_SECONDARY};
            font-size: 10px;
            font-weight: 700;
            padding: 4px 10px;
            border-radius: 4px;
            border: 1px solid #334155;
            letter-spacing: 0.5px;
        """)
        right_box.addWidget(self._lbl_sys)

        layout.addLayout(right_box)

    def set_system_active(self, active: bool, text: Optional[str] = None):
        if active:
            txt = text or "● MISSION RUNNING"
            self._lbl_sys.setText(txt)
            self._lbl_sys.setStyleSheet(f"""
                background: {C_EMERALD_BG};
                color: {C_EMERALD_TEXT};
                border: 1px solid {C_EMERALD_BORDER};
                font-size: 10px;
                font-weight: 800;
                padding: 4px 10px;
                border-radius: 4px;
            """)
        else:
            txt = text or "● STANDBY"
            self._lbl_sys.setText(txt)
            self._lbl_sys.setStyleSheet(f"""
                background: #1e293b;
                color: {C_TEXT_SECONDARY};
                font-size: 10px;
                font-weight: 700;
                padding: 4px 10px;
                border-radius: 4px;
                border: 1px solid #334155;
            """)


# ══════════════════════════════════════════════════════════════════════════════
#  2. Status Chips Bar
# ══════════════════════════════════════════════════════════════════════════════

class StatusChipsBar(QFrame):
    """
    Subsystem status indicator chips:
    CAM | POSE | OBJECT(HSV+YOLO) | HAR | VOICE | RECORDING | STREAM | MISSION
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(34)
        self.setStyleSheet(f"""
            StatusChipsBar {{
                background-color: {C_BG_BASE};
                border-bottom: 1px solid {C_BORDER};
            }}
        """)
        self._chips = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 0, 18, 0)
        layout.setSpacing(8)

        chip_defs = [
            ("CAM", "30 FPS", C_EMERALD_TEXT, C_EMERALD_BG, C_EMERALD_BORDER),
            ("POSE", "MEDIAPIPE", C_CYAN_PRIMARY, C_CYAN_BG, C_CYAN_BORDER),
            ("OBJECT", "HSV + YOLOv11", C_CYAN_PRIMARY, C_CYAN_BG, C_CYAN_BORDER),
            ("HAR", "HYBRID FSM", C_EMERALD_TEXT, C_EMERALD_BG, C_EMERALD_BORDER),
            ("VOICE", "ARMED", C_SKY_TEXT, C_SKY_BG, C_SKY_BORDER),
            ("RECORDING", "STANDBY", C_TEXT_SECONDARY, "#1e293b", "#334155"),
            ("STREAM", "OFFLINE", C_TEXT_SECONDARY, "#1e293b", "#334155"),
            ("MISSION", "WAITING", C_AMBER_TEXT, C_AMBER_BG, C_AMBER_BORDER),
        ]

        for key, initial_val, text_col, bg_col, border_col in chip_defs:
            chip = QLabel(f"{key}: {initial_val}")
            chip.setStyleSheet(f"""
                background: {bg_col};
                color: {text_col};
                border: 1px solid {border_col};
                font-size: 10px;
                font-weight: 700;
                padding: 2px 8px;
                border-radius: 3px;
                letter-spacing: 0.5px;
            """)
            self._chips[key] = chip
            layout.addWidget(chip)

        layout.addStretch()

    def update_chip(self, key: str, value: str, level: str = "normal"):
        if key not in self._chips:
            return

        colors = {
            "success": (C_EMERALD_TEXT, C_EMERALD_BG, C_EMERALD_BORDER),
            "cyan":    (C_CYAN_PRIMARY, C_CYAN_BG, C_CYAN_BORDER),
            "sky":     (C_SKY_TEXT, C_SKY_BG, C_SKY_BORDER),
            "warning": (C_AMBER_TEXT, C_AMBER_BG, C_AMBER_BORDER),
            "danger":  (C_ROSE_TEXT, C_ROSE_BG, C_ROSE_BORDER),
            "normal":  (C_TEXT_SECONDARY, "#1e293b", "#334155"),
        }
        text_col, bg_col, border_col = colors.get(level, colors["normal"])
        self._chips[key].setText(f"{key}: {value}")
        self._chips[key].setStyleSheet(f"""
            background: {bg_col};
            color: {text_col};
            border: 1px solid {border_col};
            font-size: 10px;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: 3px;
            letter-spacing: 0.5px;
        """)


# ══════════════════════════════════════════════════════════════════════════════
#  3. Telemetry Strip
# ══════════════════════════════════════════════════════════════════════════════

class TelemetryStrip(QFrame):
    """
    Runtime telemetry strip with clean flight cards:
    FPS | MET Runtime | Active Experiment | Local MP4 | Stream IP
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(38)
        self.setStyleSheet(f"""
            TelemetryStrip {{
                background-color: {C_PANEL_BG};
                border: 1px solid {C_BORDER};
                border-radius: 6px;
            }}
        """)
        self._start_time: Optional[float] = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 4, 12, 4)
        layout.setSpacing(12)

        # FPS Pill
        self._lbl_fps = QLabel("⚡ FPS: 0.0")
        self._lbl_fps.setStyleSheet(f"""
            background: #0f172a;
            color: {C_CYAN_PRIMARY};
            border: 1px solid #1e293b;
            border-radius: 4px;
            padding: 3px 10px;
            font-size: 11px;
            font-weight: 700;
            font-family: monospace;
        """)
        layout.addWidget(self._lbl_fps)

        # MET Runtime Pill
        self._lbl_met = QLabel("⏱ MET: 00:00")
        self._lbl_met.setStyleSheet("""
            background: #0f172a;
            color: #f8fafc;
            border: 1px solid #1e293b;
            border-radius: 4px;
            padding: 3px 10px;
            font-size: 11px;
            font-weight: 700;
            font-family: monospace;
        """)
        layout.addWidget(self._lbl_met)

        # Active Experiment Pill
        self._lbl_exp = QLabel("🔬 EXP: ISRO BAS Experiment")
        self._lbl_exp.setStyleSheet(f"""
            background: #0f172a;
            color: {C_TEXT_SECONDARY};
            border: 1px solid #1e293b;
            border-radius: 4px;
            padding: 3px 12px;
            font-size: 11px;
            font-weight: 700;
        """)
        layout.addWidget(self._lbl_exp)

        layout.addStretch()

        # Local Recording Pill
        self._lbl_rec = QLabel("💾 REC: LOCAL MP4")
        self._lbl_rec.setStyleSheet("""
            background: #0f172a;
            color: #64748b;
            border: 1px solid #1e293b;
            border-radius: 4px;
            padding: 3px 10px;
            font-size: 10.5px;
            font-weight: 700;
        """)
        layout.addWidget(self._lbl_rec)

        # Stream State Pill
        self._lbl_stream = QLabel("📡 STREAM: 127.0.0.1:8080")
        self._lbl_stream.setStyleSheet("""
            background: #0f172a;
            color: #64748b;
            border: 1px solid #1e293b;
            border-radius: 4px;
            padding: 3px 10px;
            font-size: 10.5px;
            font-weight: 700;
        """)
        layout.addWidget(self._lbl_stream)

    def set_running(self, running: bool):
        if running and self._start_time is None:
            self._start_time = time.monotonic()
        elif not running:
            self._start_time = None
            self._lbl_met.setText("⏱ MET: 00:00")

    def update_metrics(self, fps: float, exp_name: str = "", rec_active: bool = False, stream_addr: str = ""):
        self._lbl_fps.setText(f"⚡ FPS: {fps:.1f}")

        if self._start_time is not None:
            elapsed = int(time.monotonic() - self._start_time)
            mins = elapsed // 60
            secs = elapsed % 60
            self._lbl_met.setText(f"⏱ MET: {mins:02d}:{secs:02d}")

        if exp_name:
            self._lbl_exp.setText(f"🔬 EXP: {exp_name}")

        if rec_active:
            self._lbl_rec.setText("💾 REC: ● RECORDING")
            self._lbl_rec.setStyleSheet(f"""
                background: {C_ROSE_BG};
                color: {C_ROSE_TEXT};
                border: 1px solid {C_ROSE_BORDER};
                border-radius: 4px;
                padding: 3px 10px;
                font-size: 10.5px;
                font-weight: 800;
            """)
        else:
            self._lbl_rec.setText("💾 REC: LOCAL MP4")
            self._lbl_rec.setStyleSheet("""
                background: #0f172a;
                color: #64748b;
                border: 1px solid #1e293b;
                border-radius: 4px;
                padding: 3px 10px;
                font-size: 10.5px;
                font-weight: 700;
            """)

        if stream_addr:
            self._lbl_stream.setText(f"📡 STREAM: {stream_addr}")
            self._lbl_stream.setStyleSheet(f"""
                background: {C_EMERALD_BG};
                color: {C_EMERALD_TEXT};
                border: 1px solid {C_EMERALD_BORDER};
                border-radius: 4px;
                padding: 3px 10px;
                font-size: 10.5px;
                font-weight: 800;
            """)
        else:
            self._lbl_stream.setText("📡 STREAM: OFFLINE")
            self._lbl_stream.setStyleSheet("""
                background: #0f172a;
                color: #64748b;
                border: 1px solid #1e293b;
                border-radius: 4px;
                padding: 3px 10px;
                font-size: 10.5px;
                font-weight: 700;
            """)


# ══════════════════════════════════════════════════════════════════════════════
#  4. Step Directive Widget (Current Step + Next Step Banner)
# ══════════════════════════════════════════════════════════════════════════════

class StepDirectiveWidget(QFrame):
    """
    Prominent instruction block:
    Large Status Badge + Current Protocol Step + Active Next-Step Guidance Banner.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            StepDirectiveWidget {{
                background-color: {C_PANEL_BG};
                border: 1px solid {C_BORDER};
                border-radius: 8px;
            }}
        """)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        # Header Row: Category + Status Badge
        top_row = QHBoxLayout()
        lbl_tag = QLabel("PROTOCOL DIRECTIVE ENGINE")
        lbl_tag.setStyleSheet(f"color: {C_TEXT_MUTED}; font-size: 10px; font-weight: 800; letter-spacing: 1px;")
        top_row.addWidget(lbl_tag)
        top_row.addStretch()

        self._badge = QLabel("WAITING")
        self._badge.setStyleSheet(f"""
            background: {C_AMBER_BG};
            color: {C_AMBER_TEXT};
            border: 1px solid {C_AMBER_BORDER};
            border-radius: 4px;
            font-size: 11px;
            font-weight: 800;
            padding: 3px 10px;
            letter-spacing: 0.8px;
        """)
        top_row.addWidget(self._badge)
        layout.addLayout(top_row)

        # Current Step (Large)
        self._lbl_step_name = QLabel("START")
        self._lbl_step_name.setStyleSheet(f"""
            color: {C_TEXT_PRIMARY};
            font-size: 18px;
            font-weight: 800;
            letter-spacing: 0.5px;
        """)
        layout.addWidget(self._lbl_step_name)

        self._lbl_step_desc = QLabel("Experiment initialized. Ready to begin on-board protocol sequence.")
        self._lbl_step_desc.setStyleSheet(f"color: {C_TEXT_SECONDARY}; font-size: 12.5px; line-height: 1.4;")
        self._lbl_step_desc.setWordWrap(True)
        layout.addWidget(self._lbl_step_desc)

        # Next Step Suggestion Card (High-visibility Mission Guidance)
        self._sugg_frame = QFrame()
        self._sugg_frame.setStyleSheet(f"""
            background: rgba(0, 240, 255, 0.08);
            border: 1px solid rgba(0, 240, 255, 0.3);
            border-radius: 6px;
            padding: 10px;
        """)
        sugg_layout = QVBoxLayout(self._sugg_frame)
        sugg_layout.setContentsMargins(10, 8, 10, 8)
        sugg_layout.setSpacing(4)

        sugg_hdr = QLabel("👉 ACTIVE NEXT-STEP GUIDANCE")
        sugg_hdr.setStyleSheet(f"color: {C_CYAN_PRIMARY}; font-size: 10px; font-weight: 800; letter-spacing: 0.8px;")
        sugg_layout.addWidget(sugg_hdr)

        self._lbl_next_suggestion = QLabel("Place sample objects in workspace camera view to begin.")
        self._lbl_next_suggestion.setStyleSheet("color: #e2e8f0; font-size: 13px; font-weight: 600;")
        self._lbl_next_suggestion.setWordWrap(True)
        sugg_layout.addWidget(self._lbl_next_suggestion)
        layout.addWidget(self._sugg_frame)

        # Confirmation Stability Progress Bar
        self._hold_bar = QProgressBar()
        self._hold_bar.setFixedHeight(8)
        self._hold_bar.setRange(0, 100)
        self._hold_bar.setValue(0)
        self._hold_bar.setTextVisible(False)
        self._hold_bar.setStyleSheet(f"""
            QProgressBar {{
                background: #1e293b;
                border: 1px solid #334155;
                border-radius: 4px;
            }}
            QProgressBar::chunk {{
                background: {C_CYAN_PRIMARY};
                border-radius: 3px;
            }}
        """)
        layout.addWidget(self._hold_bar)

    def update_directive(
        self,
        current_name: str,
        current_desc: str,
        next_suggestion: str,
        status: TransitionStatus,
        hold_pct: int = 0,
    ):
        self._lbl_step_name.setText(current_name)
        self._lbl_step_desc.setText(current_desc)
        self._lbl_next_suggestion.setText(next_suggestion or "Follow protocol sequence.")
        self._hold_bar.setValue(hold_pct)

        status_styles = {
            TransitionStatus.CORRECT:      ("CORRECT", C_EMERALD_TEXT, C_EMERALD_BG, C_EMERALD_BORDER),
            TransitionStatus.WAITING:      ("WAITING", C_AMBER_TEXT, C_AMBER_BG, C_AMBER_BORDER),
            TransitionStatus.SKIPPED:      ("SKIPPED", C_ROSE_TEXT, C_ROSE_BG, C_ROSE_BORDER),
            TransitionStatus.WRONG_ORDER:  ("WRONG_ORDER", C_ROSE_TEXT, C_ROSE_BG, C_ROSE_BORDER),
            TransitionStatus.COMPLETED:    ("COMPLETED", C_CYAN_PRIMARY, C_CYAN_BG, C_CYAN_BORDER),
            TransitionStatus.ALREADY_DONE: ("REPEAT", C_SKY_TEXT, C_SKY_BG, C_SKY_BORDER),
            TransitionStatus.NOT_STARTED:  ("IDLE", C_TEXT_SECONDARY, "#1e293b", "#334155"),
        }
        text, text_col, bg_col, border_col = status_styles.get(
            status, ("STATUS", C_TEXT_SECONDARY, "#1e293b", "#334155")
        )
        self._badge.setText(text)
        self._badge.setStyleSheet(f"""
            background: {bg_col};
            color: {text_col};
            border: 1px solid {border_col};
            border-radius: 4px;
            font-size: 11px;
            font-weight: 800;
            padding: 3px 10px;
            letter-spacing: 0.8px;
        """)


# ══════════════════════════════════════════════════════════════════════════════
#  5. Mission Timeline Widget (Dynamic multi-experiment step list)
# ══════════════════════════════════════════════════════════════════════════════

class MissionTimelineWidget(QFrame):
    """
    Step-by-step progress timeline:
    Completed steps (✓ emerald), active step (▶ cyan outline), pending steps (slate).
    Dynamically rebuilds whenever experiment is switched!
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            MissionTimelineWidget {{
                background-color: {C_PANEL_BG};
                border: 1px solid {C_BORDER};
                border-radius: 8px;
            }}
        """)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        lbl = QLabel("MISSION TIMELINE SEQUENCE")
        lbl.setStyleSheet(f"color: {C_TEXT_MUTED}; font-size: 10px; font-weight: 800; letter-spacing: 1px;")
        layout.addWidget(lbl)

        self._list = QListWidget()
        self._list.setStyleSheet("""
            QListWidget {
                background: transparent;
                border: none;
                outline: none;
            }
            QListWidget::item {
                border-radius: 4px;
                padding: 5px 8px;
                margin-bottom: 3px;
            }
        """)
        layout.addWidget(self._list)

    def set_steps(self, steps: list):
        self._list.clear()
        for idx, step in enumerate(steps):
            name = getattr(step, "name", str(step))
            item = QListWidgetItem(f"[{idx}]  {name}")
            item.setForeground(QColor(C_TEXT_MUTED))
            item.setFont(QFont("Segoe UI", 9, QFont.Weight.Medium))
            self._list.addItem(item)

    def update_progress(self, current_idx: int, is_completed: bool = False):
        count = self._list.count()
        for i in range(count):
            item = self._list.item(i)
            text = item.text()
            clean_text = text.split("  ")[-1] if "  " in text else text

            if is_completed or i < current_idx:
                item.setText(f"✓  {clean_text}")
                item.setForeground(QColor(C_EMERALD_TEXT))
                item.setBackground(QColor(10, 40, 25))
            elif i == current_idx:
                item.setText(f"▶  {clean_text}")
                item.setForeground(QColor(C_CYAN_PRIMARY))
                item.setBackground(QColor(12, 40, 66))
            else:
                item.setText(f"○  {clean_text}")
                item.setForeground(QColor(C_TEXT_MUTED))
                item.setBackground(QColor("transparent"))


# ══════════════════════════════════════════════════════════════════════════════
#  6. Object Tracking Cards Widget
# ══════════════════════════════════════════════════════════════════════════════

class ObjectTrackingCardsWidget(QFrame):
    """
    Shows individual detected objects with class, bounding box, confidence, and detection method.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            ObjectTrackingCardsWidget {{
                background-color: {C_PANEL_BG};
                border: 1px solid {C_BORDER};
                border-radius: 8px;
            }}
        """)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        lbl = QLabel("OBJECT TRACKING TELEMETRY (HSV + YOLOv11)")
        lbl.setStyleSheet(f"color: {C_TEXT_MUTED}; font-size: 10px; font-weight: 800; letter-spacing: 1px;")
        layout.addWidget(lbl)

        self._cards_layout = QVBoxLayout()
        self._cards_layout.setSpacing(6)
        layout.addLayout(self._cards_layout)

        self._lbl_empty = QLabel("No targets tracked in current frame.")
        self._lbl_empty.setStyleSheet(f"color: {C_TEXT_MUTED}; font-size: 11px; font-style: italic;")
        self._cards_layout.addWidget(self._lbl_empty)

    def update_objects(self, objects_detail: list[dict]):
        # Clear existing
        while self._cards_layout.count():
            item = self._cards_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not objects_detail:
            self._lbl_empty = QLabel("No targets tracked in current frame.")
            self._lbl_empty.setStyleSheet(f"color: {C_TEXT_MUTED}; font-size: 11px; font-style: italic;")
            self._cards_layout.addWidget(self._lbl_empty)
            return

        for obj in objects_detail[:4]:  # Show up to 4 objects
            name = str(obj.get("name") or obj.get("colour") or obj.get("class", "OBJ")).upper()
            bbox = obj.get("bbox", (0, 0, 0, 0))
            conf = float(obj.get("confidence", 0.95))

            card = QFrame()
            card.setStyleSheet(f"""
                background: {C_PANEL_ALT};
                border: 1px solid {C_BORDER};
                border-radius: 5px;
                padding: 6px;
            """)
            c_box = QHBoxLayout(card)
            c_box.setContentsMargins(8, 4, 8, 4)

            name_lbl = QLabel(f"● {name}")
            name_lbl.setStyleSheet(f"color: {C_CYAN_PRIMARY}; font-size: 11px; font-weight: 700;")
            c_box.addWidget(name_lbl)

            c_box.addStretch()

            coords_lbl = QLabel(f"[{bbox[0]},{bbox[1]}] {bbox[2]}x{bbox[3]}")
            coords_lbl.setStyleSheet(f"color: {C_TEXT_MUTED}; font-size: 10px; font-family: monospace;")
            c_box.addWidget(coords_lbl)

            conf_lbl = QLabel(f"{conf*100:.0f}%")
            conf_lbl.setStyleSheet(f"color: {C_EMERALD_TEXT}; font-size: 11px; font-weight: 800;")
            c_box.addWidget(conf_lbl)

            self._cards_layout.addWidget(card)


# ══════════════════════════════════════════════════════════════════════════════
#  7. Evidence Panel Widget (Critical USP: Explainable Decision Engine)
# ══════════════════════════════════════════════════════════════════════════════

class EvidencePanelWidget(QFrame):
    """
    Critical Winning USP:
    Displays explainable telemetry behind every FSM decision:
    - Detected Objects
    - Hand Presence (Hands, Fingertips, Distance)
    - Face / Operator Presence
    - Hand-to-Object Distance vs Threshold
    - Decision Verification Reason (Why accepted / rejected)
    - Fused Hybrid Confidence Score
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            EvidencePanelWidget {{
                background-color: {C_PANEL_BG};
                border: 1px solid {C_BORDER};
                border-radius: 8px;
            }}
        """)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        # Header
        top_row = QHBoxLayout()
        lbl_tag = QLabel("EXPLAINABLE DECISION TELEMETRY (USP)")
        lbl_tag.setStyleSheet(f"color: {C_CYAN_PRIMARY}; font-size: 10.5px; font-weight: 800; letter-spacing: 1px;")
        top_row.addWidget(lbl_tag)
        top_row.addStretch()

        self._lbl_conf = QLabel("CONFIDENCE: 95.0%")
        self._lbl_conf.setStyleSheet(f"color: {C_EMERALD_TEXT}; font-size: 10.5px; font-weight: 800;")
        top_row.addWidget(self._lbl_conf)
        layout.addLayout(top_row)

        # 2x2 Telemetry Grid
        grid = QGridLayout()
        grid.setSpacing(8)

        # Cell 1: Objects Detected
        self._card_objs = self._make_metric_card("DETECTED OBJECTS", "NONE")
        grid.addWidget(self._card_objs, 0, 0)

        # Cell 2: Hand Presence
        self._card_hands = self._make_metric_card("HAND PRESENCE", "ABSENT")
        grid.addWidget(self._card_hands, 0, 1)

        # Cell 3: Hand Distance
        self._card_dist = self._make_metric_card("HAND-TO-OBJECT DIST", "—")
        grid.addWidget(self._card_dist, 1, 0)

        # Cell 4: Operator Presence
        self._card_operator = self._make_metric_card("OPERATOR / FACE", "DETECTED")
        grid.addWidget(self._card_operator, 1, 1)

        layout.addLayout(grid)

        # Decision Reason Banner (Why Accepted / Waiting / Rejected)
        self._reason_frame = QFrame()
        self._reason_frame.setStyleSheet(f"""
            background: {C_PANEL_ALT};
            border: 1px solid {C_BORDER};
            border-radius: 6px;
            padding: 8px 10px;
        """)
        r_layout = QVBoxLayout(self._reason_frame)
        r_layout.setContentsMargins(8, 6, 8, 6)
        r_layout.setSpacing(4)

        r_hdr = QLabel("DECISION VERIFICATION RATIONALE")
        r_hdr.setStyleSheet(f"color: {C_TEXT_MUTED}; font-size: 9.5px; font-weight: 800; letter-spacing: 0.8px;")
        r_layout.addWidget(r_hdr)

        self._lbl_reason = QLabel("Waiting for protocol initialization.")
        self._lbl_reason.setStyleSheet("color: #f1f5f9; font-size: 12px; font-weight: 600; line-height: 1.4;")
        self._lbl_reason.setWordWrap(True)
        r_layout.addWidget(self._lbl_reason)

        layout.addWidget(self._reason_frame)

    def _make_metric_card(self, label: str, default_val: str) -> QFrame:
        card = QFrame()
        card.setStyleSheet(f"""
            background: {C_PANEL_ALT};
            border: 1px solid {C_BORDER};
            border-radius: 5px;
            padding: 8px;
        """)
        vbox = QVBoxLayout(card)
        vbox.setContentsMargins(8, 6, 8, 6)
        vbox.setSpacing(2)

        lbl = QLabel(label)
        lbl.setStyleSheet(f"color: {C_TEXT_MUTED}; font-size: 9px; font-weight: 700; letter-spacing: 0.5px;")
        vbox.addWidget(lbl)

        val = QLabel(default_val)
        val.setObjectName("val_label")
        val.setStyleSheet(f"color: {C_TEXT_PRIMARY}; font-size: 13px; font-weight: 800;")
        vbox.addWidget(val)
        return card

    def update_evidence(self, evidence: dict):
        # 1. Objects
        objs = evidence.get("detected_objects", [])
        objs_str = ", ".join(objs) if objs else "NONE"
        lbl_obj = self._card_objs.findChild(QLabel, "val_label")
        if lbl_obj:
            lbl_obj.setText(objs_str)
            lbl_obj.setStyleSheet(f"color: {C_CYAN_PRIMARY if objs else C_TEXT_MUTED}; font-size: 13px; font-weight: 800;")

        # 2. Hands
        has_hands = evidence.get("hand_present", False)
        hand_count = evidence.get("hand_count", 0)
        hands_str = f"ACTIVE ({hand_count})" if has_hands else "ABSENT"
        lbl_hands = self._card_hands.findChild(QLabel, "val_label")
        if lbl_hands:
            lbl_hands.setText(hands_str)
            lbl_hands.setStyleSheet(f"color: {C_EMERALD_TEXT if has_hands else C_TEXT_MUTED}; font-size: 13px; font-weight: 800;")

        # 3. Distance
        dist = evidence.get("hand_to_object_dist") or evidence.get("hand_to_box_dist")
        if dist is not None:
            dist_str = f"{dist:.0f} px  [NEAR]" if dist <= 120 else f"{dist:.0f} px  [CLEAR]"
            col = C_EMERALD_TEXT if dist <= 120 else C_SKY_TEXT
        else:
            dist_str = "—"
            col = C_TEXT_MUTED

        lbl_dist = self._card_dist.findChild(QLabel, "val_label")
        if lbl_dist:
            lbl_dist.setText(dist_str)
            lbl_dist.setStyleSheet(f"color: {col}; font-size: 13px; font-weight: 800;")

        # 4. Operator presence
        lbl_op = self._card_operator.findChild(QLabel, "val_label")
        if lbl_op:
            lbl_op.setText("OPERATOR IN FRAME")
            lbl_op.setStyleSheet(f"color: {C_CYAN_PRIMARY}; font-size: 12px; font-weight: 700;")

        # 5. Rationale text
        reason = evidence.get("decision_reason", "Processing state transitions.")
        self._lbl_reason.setText(reason)

        # 6. Confidence
        conf = float(evidence.get("hybrid_confidence", 0.95))
        self._lbl_conf.setText(f"HYBRID CONFIDENCE: {conf*100:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
#  8. Live Log Terminal Widget
# ══════════════════════════════════════════════════════════════════════════════

class LiveLogWidget(QFrame):
    """
    Console log terminal with real-time colored milestone events.
    """

    def __init__(self, parent=None, max_items: int = 150):
        super().__init__(parent)
        self._max_items = max_items
        self.setStyleSheet(f"""
            LiveLogWidget {{
                background-color: {C_PANEL_BG};
                border: 1px solid {C_BORDER};
                border-radius: 8px;
            }}
        """)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)

        lbl = QLabel("MISSION AUDIT LOG (JSONL STREAM)")
        lbl.setStyleSheet(f"color: {C_TEXT_MUTED}; font-size: 10px; font-weight: 800; letter-spacing: 1px;")
        layout.addWidget(lbl)

        self._list = QListWidget()
        self._list.setStyleSheet("""
            QListWidget {
                background: #080b11;
                border: 1px solid #1e293b;
                border-radius: 5px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
                padding: 6px;
            }
            QListWidget::item {
                padding: 2px 4px;
            }
        """)
        layout.addWidget(self._list)

    def log(self, message: str, level: str = "info"):
        item = QListWidgetItem(f"[{time.strftime('%H:%M:%S')}] {message}")
        if level == "success" or "✓" in message:
            item.setForeground(QColor(C_EMERALD_TEXT))
        elif level == "error" or "✗" in message or "SKIPPED" in message or "VIOLATION" in message:
            item.setForeground(QColor(C_ROSE_TEXT))
        elif level == "warning" or "⚠" in message:
            item.setForeground(QColor(C_AMBER_TEXT))
        elif level == "cyan":
            item.setForeground(QColor(C_CYAN_PRIMARY))
        else:
            item.setForeground(QColor("#cbd5e1"))

        self._list.addItem(item)
        if self._list.count() > self._max_items:
            self._list.takeItem(0)
        self._list.scrollToBottom()


# ══════════════════════════════════════════════════════════════════════════════
#  9. Bottom Control Bar (Multi-Experiment + Stream IP + HTML & Print Audit)
# ══════════════════════════════════════════════════════════════════════════════

class BottomControlBar(QFrame):
    """
    Mission execution toolbar:
    - Experiment Module Dropdown (5 Real Testable Experiments)
    - Start / Stop / Reset Buttons
    - Stream IP + Port input + Toggle button + Clickable link
    - View Log as HTML & Print Report actions
    - Quit Button
    """

    experiment_changed = Signal(str)
    stream_toggled = Signal(bool, str, int)
    html_report_requested = Signal()
    print_report_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(56)
        self.setStyleSheet(f"""
            BottomControlBar {{
                background-color: {C_PANEL_BG};
                border-top: 1px solid {C_BORDER};
            }}
        """)
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(10)

        # ── 1. Experiment Dropdown ────────────────────────────────────────────
        lbl_exp = QLabel("EXPERIMENT:")
        lbl_exp.setStyleSheet(f"color: {C_TEXT_MUTED}; font-size: 10px; font-weight: 800; letter-spacing: 0.5px;")
        layout.addWidget(lbl_exp)

        self.combo_exp = QComboBox()
        self.combo_exp.setFixedHeight(34)
        for exp in EXPERIMENT_REGISTRY:
            self.combo_exp.addItem(exp["name"], exp["key"])

        self.combo_exp.setStyleSheet(f"""
            QComboBox {{
                background: #1e293b;
                color: {C_TEXT_PRIMARY};
                border: 1px solid #334155;
                border-radius: 5px;
                padding: 0 10px;
                font-weight: 700;
                font-size: 11.5px;
                min-width: 220px;
            }}
            QComboBox::drop-down {{ border: none; }}
            QComboBox QAbstractItemView {{
                background: #111622;
                color: #f8fafc;
                selection-background-color: #0c2842;
                selection-color: {C_CYAN_PRIMARY};
                border: 1px solid {C_BORDER};
            }}
        """)
        self.combo_exp.currentIndexChanged.connect(self._on_combo_changed)
        layout.addWidget(self.combo_exp)

        # ── 2. Primary Execution Controls ─────────────────────────────────────
        self.btn_start = QPushButton("▶ Start")
        self.btn_start.setFixedHeight(34)
        self.btn_start.setStyleSheet(f"""
            QPushButton {{
                background: {C_EMERALD_BG};
                color: {C_EMERALD_TEXT};
                border: 1px solid {C_EMERALD_BORDER};
                border-radius: 5px;
                font-weight: 800;
                font-size: 12px;
                padding: 0 14px;
            }}
            QPushButton:hover {{ background: #07634b; }}
        """)
        layout.addWidget(self.btn_start)

        self.btn_stop = QPushButton("■ Stop")
        self.btn_stop.setFixedHeight(34)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("""
            QPushButton {
                background: #1e293b; color: #cbd5e1;
                border: 1px solid #334155; border-radius: 5px;
                font-weight: 700; font-size: 12px; padding: 0 12px;
            }
            QPushButton:hover { background: #334155; }
            QPushButton:disabled { color: #475569; border-color: #1e293b; }
        """)
        layout.addWidget(self.btn_stop)

        self.btn_reset = QPushButton("↺ Reset")
        self.btn_reset.setFixedHeight(34)
        self.btn_reset.setStyleSheet("""
            QPushButton {
                background: #1e293b; color: #cbd5e1;
                border: 1px solid #334155; border-radius: 5px;
                font-weight: 700; font-size: 12px; padding: 0 12px;
            }
            QPushButton:hover { background: #334155; }
        """)
        layout.addWidget(self.btn_reset)

        # ── 3. Streaming Controls & Clickable Link ─────────────────────────────
        layout.addWidget(QLabel("│"))

        self.edit_ip = QLineEdit("127.0.0.1")
        self.edit_ip.setFixedWidth(82)
        self.edit_ip.setFixedHeight(34)
        self.edit_ip.setStyleSheet("""
            QLineEdit {
                background: #0f172a; color: #f8fafc;
                border: 1px solid #334155; border-radius: 4px;
                font-family: monospace; font-size: 11px; padding: 0 6px;
            }
        """)
        layout.addWidget(self.edit_ip)

        self.edit_port = QLineEdit("8080")
        self.edit_port.setFixedWidth(46)
        self.edit_port.setFixedHeight(34)
        self.edit_port.setStyleSheet("""
            QLineEdit {
                background: #0f172a; color: #f8fafc;
                border: 1px solid #334155; border-radius: 4px;
                font-family: monospace; font-size: 11px; padding: 0 6px;
            }
        """)
        layout.addWidget(self.edit_port)

        self.btn_stream = QPushButton("📡 Stream")
        self.btn_stream.setFixedHeight(34)
        self.btn_stream.setStyleSheet("""
            QPushButton {
                background: #1e293b; color: #38bdf8;
                border: 1px solid #0284c7; border-radius: 4px;
                font-weight: 700; font-size: 11px; padding: 0 10px;
            }
            QPushButton:hover { background: #0c2842; }
        """)
        self.btn_stream.clicked.connect(self._on_stream_clicked)
        layout.addWidget(self.btn_stream)

        # Clickable stream URL link that opens browser directly!
        self.lbl_stream_link = QLabel("🔗 Open Stream")
        self.lbl_stream_link.setCursor(QCursor(Qt.PointingHandCursor))
        self.lbl_stream_link.setStyleSheet(f"""
            color: {C_CYAN_PRIMARY};
            font-size: 11px;
            font-weight: 700;
            text-decoration: underline;
            padding: 0 4px;
        """)
        self.lbl_stream_link.mousePressEvent = self._on_stream_link_pressed
        layout.addWidget(self.lbl_stream_link)

        layout.addStretch()

        # ── 4. Audit & Report Actions ─────────────────────────────────────────
        self.btn_html_report = QPushButton("📄 HTML Report")
        self.btn_html_report.setFixedHeight(34)
        self.btn_html_report.setStyleSheet(f"""
            QPushButton {{
                background: rgba(0, 240, 255, 0.12);
                color: {C_CYAN_PRIMARY};
                border: 1px solid rgba(0, 240, 255, 0.35);
                border-radius: 5px;
                font-weight: 700;
                font-size: 11px;
                padding: 0 12px;
            }}
            QPushButton:hover {{ background: rgba(0, 240, 255, 0.22); }}
        """)
        self.btn_html_report.clicked.connect(self.html_report_requested.emit)
        layout.addWidget(self.btn_html_report)

        self.btn_print_report = QPushButton("🖨️ Print")
        self.btn_print_report.setFixedHeight(34)
        self.btn_print_report.setStyleSheet("""
            QPushButton {
                background: #1e293b; color: #f8fafc;
                border: 1px solid #334155; border-radius: 5px;
                font-weight: 700; font-size: 11px; padding: 0 10px;
            }
            QPushButton:hover { background: #334155; }
        """)
        self.btn_print_report.clicked.connect(self.print_report_requested.emit)
        layout.addWidget(self.btn_print_report)

        # Quit
        self.btn_quit = QPushButton("✕ Quit")
        self.btn_quit.setFixedHeight(34)
        self.btn_quit.setStyleSheet(f"""
            QPushButton {{
                background: {C_ROSE_BG}; color: {C_ROSE_TEXT};
                border: 1px solid {C_ROSE_BORDER}; border-radius: 5px;
                font-weight: 700; font-size: 11px; padding: 0 12px;
            }}
            QPushButton:hover {{ background: #5b1010; }}
        """)
        layout.addWidget(self.btn_quit)

    def _on_combo_changed(self, idx: int):
        exp_key = self.combo_exp.itemData(idx)
        if exp_key:
            self.experiment_changed.emit(exp_key)

    def _on_stream_clicked(self):
        ip = self.edit_ip.text().strip() or "127.0.0.1"
        try:
            port = int(self.edit_port.text().strip() or "8080")
        except ValueError:
            port = 8080
        is_active = "STOP" in self.btn_stream.text().upper()
        self.stream_toggled.emit(not is_active, ip, port)

    def set_streaming_active(self, active: bool, ip: str, port: int):
        if active:
            self.btn_stream.setText("■ Stop Stream")
            self.btn_stream.setStyleSheet(f"""
                background: {C_ROSE_BG}; color: {C_ROSE_TEXT};
                border: 1px solid {C_ROSE_BORDER}; border-radius: 4px;
                font-weight: 700; font-size: 11px; padding: 0 10px;
            """)
            self.lbl_stream_link.setText(f"🔗 http://{ip}:{port}")
        else:
            self.btn_stream.setText("📡 Stream")
            self.btn_stream.setStyleSheet("""
                QPushButton {
                    background: #1e293b; color: #38bdf8;
                    border: 1px solid #0284c7; border-radius: 4px;
                    font-weight: 700; font-size: 11px; padding: 0 10px;
                }
                QPushButton:hover { background: #0c2842; }
            """)
            self.lbl_stream_link.setText("🔗 Open Stream")

    def _on_stream_link_pressed(self, event):
        ip = self.edit_ip.text().strip() or "127.0.0.1"
        try:
            port = int(self.edit_port.text().strip() or "8080")
        except ValueError:
            port = 8080
        url = f"http://{ip}:{port}"
        print(f"[BottomControlBar] Opening stream in browser: {url}")
        webbrowser.open(url)


# ══════════════════════════════════════════════════════════════════════════════
#  10. Session Summary Dialog
# ══════════════════════════════════════════════════════════════════════════════

class SessionSummaryDialog(QDialog):
    """
    Flight-qualified Mission Validation Report & Session Certificate modal dialog.
    """

    def __init__(self, summary_data: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("VYOM • Mission Session Validation Report")
        self.setMinimumSize(600, 480)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {C_BG_BASE};
                color: {C_TEXT_PRIMARY};
                border: 1px solid {C_BORDER};
                font-family: -apple-system, 'Segoe UI', system-ui, sans-serif;
            }}
        """)
        self._summary = summary_data or {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 20)
        layout.setSpacing(14)

        # Header Badge
        hdr_frame = QFrame()
        hdr_frame.setStyleSheet(f"""
            background: {C_PANEL_BG};
            border: 1px solid {C_BORDER};
            border-radius: 6px;
            padding: 10px;
        """)
        hdr_box = QVBoxLayout(hdr_frame)
        title = QLabel("MISSION EXPERIMENT VALIDATION CERTIFICATE")
        title.setStyleSheet(f"color: {C_CYAN_PRIMARY}; font-size: 15px; font-weight: 800; letter-spacing: 1px;")
        sub = QLabel(f"VYOM On-Board Assistant • {self._summary.get('protocol_name', 'BAS Experiment')}")
        sub.setStyleSheet("color: #94a3b8; font-size: 11px;")
        hdr_box.addWidget(title)
        hdr_box.addWidget(sub)
        layout.addWidget(hdr_frame)

        # Key Metrics Grid
        grid = QGridLayout()
        grid.setSpacing(10)

        metrics = [
            ("COMPLIANCE SCORE", f"{self._summary.get('compliance_score', 100.0):.1f}%", C_EMERALD_TEXT),
            ("PROTOCOL DURATION", self._summary.get("formatted_time", "00:00"), C_SKY_TEXT),
            ("STEPS COMPLETED", f"{self._summary.get('completed_steps', 0)} / {self._summary.get('total_steps', 7)}", C_TEXT_PRIMARY),
            ("MEAN ML CONFIDENCE", f"{self._summary.get('mean_ml_confidence', 0.9)*100:.1f}%", C_CYAN_PRIMARY),
            ("CORRECT ACTIONS", str(self._summary.get("correct_count", 0)), C_EMERALD_TEXT),
            ("PROTOCOL ANOMALIES", str(self._summary.get("error_count", 0)), C_ROSE_TEXT if self._summary.get("error_count", 0) > 0 else C_TEXT_MUTED),
        ]

        for idx, (label, val, color) in enumerate(metrics):
            card = QFrame()
            card.setStyleSheet(f"background: {C_PANEL_BG}; border: 1px solid {C_BORDER}; border-radius: 5px; padding: 8px;")
            c_vbox = QVBoxLayout(card)
            c_vbox.setSpacing(2)
            lbl = QLabel(label)
            lbl.setStyleSheet("color: #64748b; font-size: 9px; font-weight: 700; letter-spacing: 0.8px;")
            v = QLabel(val)
            v.setStyleSheet(f"color: {color}; font-size: 16px; font-weight: 800;")
            c_vbox.addWidget(lbl)
            c_vbox.addWidget(v)
            grid.addWidget(card, idx // 3, idx % 3)

        layout.addLayout(grid)

        # Status Verdict Banner
        is_completed = self._summary.get("is_completed", False)
        verdict = QFrame()
        verdict_color = C_EMERALD_BORDER if is_completed else C_AMBER_BORDER
        verdict_bg = C_EMERALD_BG if is_completed else C_AMBER_BG
        verdict_text_color = C_EMERALD_TEXT if is_completed else C_AMBER_TEXT
        verdict.setStyleSheet(f"background: {verdict_bg}; border: 1px solid {verdict_color}; border-radius: 5px; padding: 10px;")
        v_box = QHBoxLayout(verdict)
        v_text = "VERDICT: MISSION OBJECTIVES FULLY VERIFIED & ACCEPTED" if is_completed else "VERDICT: SESSION ENDED BEFORE TERMINAL COMPLETION"
        lbl_v = QLabel(v_text)
        lbl_v.setStyleSheet(f"color: {verdict_text_color}; font-weight: 800; font-size: 12px; letter-spacing: 0.5px;")
        v_box.addWidget(lbl_v)
        layout.addWidget(verdict)

        # Action Buttons
        btn_box = QHBoxLayout()
        btn_box.setSpacing(10)

        btn_open_html = QPushButton("📄 Open Full HTML Audit Report")
        btn_open_html.setFixedHeight(36)
        btn_open_html.setStyleSheet(f"""
            QPushButton {{
                background: rgba(0, 240, 255, 0.15); color: {C_CYAN_PRIMARY};
                border: 1px solid rgba(0, 240, 255, 0.4); border-radius: 5px;
                font-weight: 700; font-size: 12px; padding: 0 14px;
            }}
            QPushButton:hover {{ background: rgba(0, 240, 255, 0.25); }}
        """)
        btn_open_html.clicked.connect(self._open_report)
        btn_box.addWidget(btn_open_html)

        btn_close = QPushButton("CLOSE")
        btn_close.setFixedHeight(36)
        btn_close.setStyleSheet("""
            QPushButton {
                background: #1e2634; color: #cbd5e1;
                border: 1px solid #2a3446; border-radius: 5px;
                font-weight: 700; font-size: 12px; padding: 0 16px;
            }
            QPushButton:hover { background: #2a3446; color: #f8fafc; }
        """)
        btn_close.clicked.connect(self.accept)
        btn_box.addWidget(btn_close)

        layout.addLayout(btn_box)

    def _open_report(self):
        reports_dir = pathlib.Path(__file__).resolve().parent.parent.parent / "reports"
        reports = list(reports_dir.glob("*.html"))
        if reports:
            latest = max(reports, key=lambda p: p.stat().st_mtime)
            webbrowser.open(latest.as_uri())
        self.accept()
