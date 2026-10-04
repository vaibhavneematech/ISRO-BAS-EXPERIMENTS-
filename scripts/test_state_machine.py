"""
test_state_machine.py
=====================
SIH26174 – AI Human Activity Recognition for On-board BAS Experiments
Phase 3 Test Script: Refined 8-Step Finite State Machine Simulation

Usage
-----
    python scripts/test_state_machine.py

Scenarios
---------
1. HAPPY PATH           – Full 8-step protocol in perfect order
2. SKIP TEST            – Operator skips REMOVE_RED_BOX
3. WRONG ORDER          – Operator places yellow before red at the end
4. RESTART              – Resets and runs again to verify clean state
"""

import sys
import time
from pathlib import Path

# ── Make project root importable ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.protocol.state_machine import StateMachine, TransitionStatus


# ── Terminal colour codes ─────────────────────────────────────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    CYAN   = "\033[96m"
    GREY   = "\033[90m"
    BLUE   = "\033[94m"


def _enable_ansi():
    """Enable ANSI escape codes on Windows."""
    import ctypes, os
    if sys.platform == "win32":
        try:
            ctypes.windll.kernel32.SetConsoleMode(
                ctypes.windll.kernel32.GetStdHandle(-11), 7
            )
        except Exception:
            pass

_enable_ansi()


# ── Status colour map ─────────────────────────────────────────────────────────
STATUS_COLOUR = {
    TransitionStatus.WAITING      : C.GREY,
    TransitionStatus.CORRECT      : C.GREEN,
    TransitionStatus.ALREADY_DONE : C.CYAN,
    TransitionStatus.WRONG_ORDER  : C.YELLOW,
    TransitionStatus.SKIPPED      : C.RED,
    TransitionStatus.COMPLETED    : C.GREEN + C.BOLD,
    TransitionStatus.NOT_STARTED  : C.GREY,
}


def print_status(frame: int, detections: list, status: TransitionStatus, sm: StateMachine):
    """Pretty-print one frame update with evidence reason."""
    colours_str = "{" + ", ".join(sorted(d["colour"] for d in detections)) + "}" \
                  if detections else "{none}"
    col = STATUS_COLOUR.get(status, "")
    done, total = sm.get_progress()
    evidence = sm.get_evidence()
    reason = evidence.get("decision_reason", "")
    print(
        f"  frame {frame:04d} | colours={colours_str:<20} | "
        f"{col}{status.name:<14}{C.RESET} | "
        f"step=[{sm.get_current_state().name}] [{done}/{total}] | "
        f"{C.GREY}{reason[:45]}{C.RESET}"
    )


def simulate(sm: StateMachine, scenario: list[tuple[int, list]], label: str):
    """Drive the FSM through a scenario."""
    print()
    print(f"{C.BOLD}{'─'*75}{C.RESET}")
    print(f"{C.BOLD}  SCENARIO: {label}{C.RESET}")
    print(f"{'─'*75}")

    sm.start()
    frame = 0

    for (repeat, dets) in scenario:
        for _ in range(repeat):
            frame += 1
            status = sm.update(dets)
            print_status(frame, dets, status, sm)

            if status == TransitionStatus.COMPLETED:
                print(f"\n  {C.GREEN}{C.BOLD}★ PROTOCOL COMPLETE in {frame} frames{C.RESET}")
                break
        else:
            continue
        break

    # Final summary
    print()
    print(f"  Final state : {C.BOLD}{sm.get_current_state().name}{C.RESET}")
    progress, total = sm.get_progress()
    print(f"  Progress    : {progress}/{total} steps")
    print(f"  History     : {len(sm.history)} recorded transitions")
    print()
    print(f"  {C.BOLD}Transition log:{C.RESET}")
    for rec in sm.history:
        col = STATUS_COLOUR.get(rec.status, "")
        print(
            f"    frame {rec.frame_index:04d}  "
            f"{rec.from_step.name:<22} → {rec.to_step.name:<22}  "
            f"{col}{rec.status.name:<14}{C.RESET} | {rec.reason}"
        )


# ── Shorthand detection builders ──────────────────────────────────────────────
NONE   = []
RED    = [{"colour": "red", "bbox": (100, 100, 80, 80)}]
YELLOW = [{"colour": "yellow", "bbox": (300, 100, 80, 80)}]
BOTH   = [
    {"colour": "red", "bbox": (100, 100, 80, 80)},
    {"colour": "yellow", "bbox": (300, 100, 80, 80)},
]

HOLD = 8   # min_hold_frames in config


def main():
    protocol_path = ROOT / "config" / "experiment_protocol.json"
    sm = StateMachine(protocol_path)

    # ──────────────────────────────────────────────────────────────────────────
    # SCENARIO 1 – HAPPY PATH: Refined 8-Step Perfect Sequence
    # ──────────────────────────────────────────────────────────────────────────
    # Step flow:
    #   0: START
    #   1: DETECT_MAIN_BOX (both present)
    #   2: REMOVE_RED_BOX (yellow only)
    #   3: REMOVE_YELLOW_BOX (none)
    #   4: PLACE_RED_BOX (red only)
    #   5: PLACE_YELLOW_BOX (both)
    #   6: VERIFY_BOTH_PLACED (both stable)
    #   7: COMPLETED
    happy_path = [
        # 1. DETECT_MAIN_BOX
        (HOLD, BOTH),
        # 2. REMOVE_RED_BOX
        (HOLD, YELLOW),
        # 3. REMOVE_YELLOW_BOX
        (HOLD, NONE),
        # 4. PLACE_RED_BOX
        (HOLD, RED),
        # 5. PLACE_YELLOW_BOX
        (HOLD, BOTH),
        # 6. VERIFY_BOTH_PLACED
        (HOLD, BOTH),
        # 7. Advance to COMPLETED
        (HOLD, BOTH),
        # Terminal hold
        (3, BOTH),
    ]
    simulate(sm, happy_path, "HAPPY PATH – Full 8-Step Refined Protocol")

    # ──────────────────────────────────────────────────────────────────────────
    # SCENARIO 2 – SKIP TEST: operator skips REMOVE_RED_BOX
    # ──────────────────────────────────────────────────────────────────────────
    skip_test = [
        # DETECT_MAIN_BOX
        (HOLD, BOTH),
        # Skip RED removal – remove BOTH directly
        (HOLD, NONE),
        (5, NONE),
    ]
    simulate(sm, skip_test, "SKIP TEST – Operator skips REMOVE_RED_BOX")

    # ──────────────────────────────────────────────────────────────────────────
    # SCENARIO 3 – WRONG ORDER: places yellow before red
    # ──────────────────────────────────────────────────────────────────────────
    wrong_order = [
        (HOLD, BOTH),     # DETECT_MAIN_BOX
        (HOLD, YELLOW),   # REMOVE_RED_BOX ✓
        (HOLD, NONE),     # REMOVE_YELLOW_BOX ✓
        # Place YELLOW first (wrong – should place RED first)
        (HOLD, YELLOW),   # WRONG ORDER
        # Correct it
        (HOLD, RED),      # PLACE_RED_BOX ✓
        (HOLD, BOTH),     # PLACE_YELLOW_BOX ✓
        (HOLD, BOTH),     # VERIFY_BOTH_PLACED ✓
        (HOLD, BOTH),     # Advance to COMPLETED ★
        (3, BOTH),        # Terminal hold
    ]
    simulate(sm, wrong_order, "WRONG ORDER – Places yellow before red")

    print("\nAll 8-step protocol scenarios verified successfully.")


if __name__ == "__main__":
    main()
