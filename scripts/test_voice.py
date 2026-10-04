"""
test_voice.py
=============
SIH26174 – AI Human Activity Recognition for On-board BAS Experiments
Phase 5: Voice Alert System – Quick Test Script

Usage
-----
    python scripts/test_voice.py

What this does
--------------
Plays each alert type in sequence so you can verify the voice alert
system works correctly on your machine.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.alerts.voice_alert import VoiceAlertSystem, AlertType


def main():
    print("=" * 60)
    print("  SIH26174 – Voice Alert System Test")
    print("=" * 60)

    vas = VoiceAlertSystem()

    tests = [
        (AlertType.STEP_COMPLETE,        "STEP_COMPLETE"),
        (AlertType.WRONG_SEQUENCE,       "WRONG_SEQUENCE"),
        (AlertType.STEP_SKIPPED,         "STEP_SKIPPED"),
        (AlertType.NEXT_STEP_HINT,       "NEXT_STEP_HINT"),
        (AlertType.EXPERIMENT_COMPLETE,  "EXPERIMENT_COMPLETE"),
    ]

    for alert_type, label in tests:
        print(f"\n  Playing: {label}")
        try:
            vas.play(alert_type)
            time.sleep(1.5)   # wait between clips
            print(f"  ✓ {label} played successfully")
        except Exception as e:
            print(f"  ⚠ {label} failed: {e}")

    print("\n" + "=" * 60)
    print("  Voice alert test complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
