"""
test_state_machine.py
=====================
Unit tests for the refined 8-step Finite State Machine with hand-object interaction.
"""

import unittest
from pathlib import Path

from src.protocol.state_machine import StateMachine, TransitionStatus, FSMState

ROOT = Path(__file__).resolve().parent.parent
PROTOCOL_PATH = ROOT / "config" / "experiment_protocol.json"


class TestStateMachine(unittest.TestCase):

    def setUp(self):
        self.sm = StateMachine(PROTOCOL_PATH)
        self.hold = 8
        self.red = [{"colour": "red", "bbox": (100, 100, 80, 80)}]
        self.yellow = [{"colour": "yellow", "bbox": (300, 100, 80, 80)}]
        self.both = [
            {"colour": "red", "bbox": (100, 100, 80, 80)},
            {"colour": "yellow", "bbox": (300, 100, 80, 80)},
        ]
        self.none = []

    def test_protocol_steps_count_and_names(self):
        """Verify the exact 8 steps requested."""
        expected_steps = [
            "START",
            "DETECT_MAIN_BOX",
            "REMOVE_RED_BOX",
            "REMOVE_YELLOW_BOX",
            "PLACE_RED_BOX",
            "PLACE_YELLOW_BOX",
            "VERIFY_BOTH_PLACED",
            "COMPLETED",
        ]
        actual_steps = [s.name for s in self.sm.steps]
        self.assertEqual(actual_steps, expected_steps)

    def test_happy_path_execution(self):
        """Verify seamless execution through all 8 steps."""
        self.sm.start()
        self.assertEqual(self.sm.get_current_state().name, "START")

        # 1. DETECT_MAIN_BOX
        for _ in range(self.hold):
            self.sm.update(self.both)
        self.assertEqual(self.sm.get_current_state().name, "DETECT_MAIN_BOX")

        # 2. REMOVE_RED_BOX
        for _ in range(self.hold):
            self.sm.update(self.yellow)
        self.assertEqual(self.sm.get_current_state().name, "REMOVE_RED_BOX")

        # 3. REMOVE_YELLOW_BOX
        for _ in range(self.hold):
            self.sm.update(self.none)
        self.assertEqual(self.sm.get_current_state().name, "REMOVE_YELLOW_BOX")

        # 4. PLACE_RED_BOX
        for _ in range(self.hold):
            self.sm.update(self.red)
        self.assertEqual(self.sm.get_current_state().name, "PLACE_RED_BOX")

        # 5. PLACE_YELLOW_BOX
        for _ in range(self.hold):
            self.sm.update(self.both)
        self.assertEqual(self.sm.get_current_state().name, "PLACE_YELLOW_BOX")

        # 6. VERIFY_BOTH_PLACED
        for _ in range(self.hold):
            self.sm.update(self.both)
        self.assertEqual(self.sm.get_current_state().name, "VERIFY_BOTH_PLACED")

        # 7. Advance to COMPLETED
        for _ in range(self.hold):
            status = self.sm.update(self.both)
        self.assertEqual(self.sm.get_current_state().name, "COMPLETED")
        self.assertTrue(self.sm.is_completed())

    def test_evidence_generation(self):
        """Verify explainable AI decision evidence (USP)."""
        self.sm.start()
        self.sm.update(self.both)
        evidence = self.sm.get_evidence()

        self.assertIn("detected_boxes", evidence)
        self.assertIn("hand_present", evidence)
        self.assertIn("hand_to_box_dist", evidence)
        self.assertIn("decision_reason", evidence)
        self.assertIn("next_suggestion", evidence)

    def test_skip_detection(self):
        """Verify out-of-order action results in SKIPPED / WRONG_ORDER."""
        self.sm.start()
        # Calibrate DETECT_MAIN_BOX
        for _ in range(self.hold):
            self.sm.update(self.both)

        # Skip removing red and remove both directly for error hold duration
        status = None
        for _ in range(5):
            status = self.sm.update(self.none)
        self.assertIn(status, (TransitionStatus.SKIPPED, TransitionStatus.WRONG_ORDER))

    def test_all_five_experiments_switchable(self):
        """Verify all 5 testable experiments load correctly with distinct steps."""
        available = self.sm.get_available_experiments()
        self.assertEqual(len(available), 5)

        expected_keys = [
            "isro_boxes",
            "phone_pickup",
            "bottle_handling",
            "notebook_transfer",
            "cup_placement",
        ]
        for key in expected_keys:
            ok = self.sm.switch_experiment(key)
            self.assertTrue(ok, f"Failed to switch to {key}")
            self.assertGreaterEqual(len(self.sm.steps), 7)
            self.assertEqual(self.sm.steps[0].name, "START")
            self.assertEqual(self.sm.steps[-1].name, "COMPLETED")
            self.assertTrue(self.sm.steps[-1].is_terminal)

    def test_phone_pickup_sequence(self):
        """Verify Phone Pickup Sequence step transitions."""
        self.sm.switch_experiment("phone_pickup")
        self.sm.start()

        phone_obj = [{"class": "cell phone", "name": "phone", "bbox": (150, 150, 80, 140), "confidence": 0.92}]
        # 1. DETECT_PHONE
        for _ in range(self.hold):
            self.sm.update(phone_obj)
        self.assertEqual(self.sm.get_current_state().name, "DETECT_PHONE")


if __name__ == "__main__":
    unittest.main()
