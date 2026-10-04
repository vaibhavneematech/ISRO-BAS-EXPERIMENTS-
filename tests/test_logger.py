"""
test_logger.py
==============
Unit tests for the JSONL EventLogger.
"""

import json
import tempfile
import unittest
from pathlib import Path

from src.logging.event_logger import EventLogger, EventType


class TestEventLogger(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.logger = EventLogger(log_dir=Path(self.temp_dir.name), session_name="test_session")

    def tearDown(self):
        self.logger.close()
        self.temp_dir.cleanup()

    def test_log_creation_and_flush(self):
        self.logger.log_session_start(protocol_name="BAS_TEST", total_steps=8)
        self.logger.log(
            EventType.STEP_COMPLETE,
            frame=10,
            current_step="DETECT_MAIN_BOX",
            next_step="REMOVE_RED_BOX",
            status="CORRECT",
            message="Step completed successfully",
        )
        # EventLogger auto-flushes each entry on write

        log_file = self.logger.log_path
        self.assertTrue(log_file.exists())

        with open(log_file, "r", encoding="utf-8") as fh:
            lines = [json.loads(line) for line in fh]

        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0]["event"], EventType.SESSION_START)
        self.assertEqual(lines[1]["event"], EventType.STEP_COMPLETE)
        self.assertEqual(lines[1]["current_step"], "DETECT_MAIN_BOX")


if __name__ == "__main__":
    unittest.main()
