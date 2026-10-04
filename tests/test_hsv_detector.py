"""
test_hsv_detector.py
====================
Unit tests for the OpenCV HSV detector.
"""

import unittest
import numpy as np

from src.detection.hsv_detector import detect_boxes, build_red_mask, build_yellow_mask


class TestHSVDetector(unittest.TestCase):

    def test_synthetic_red_detection(self):
        # Create 480x640 blank image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw pure red rectangle (BGR: 0, 0, 255)
        img[100:200, 100:200] = [0, 0, 255]

        dets = detect_boxes(img)
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0]["colour"], "red")
        x, y, w, h = dets[0]["bbox"]
        self.assertAlmostEqual(w, 100, delta=5)
        self.assertAlmostEqual(h, 100, delta=5)

    def test_synthetic_yellow_detection(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw pure yellow rectangle (BGR: 0, 255, 255)
        img[150:250, 300:400] = [0, 255, 255]

        dets = detect_boxes(img)
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0]["colour"], "yellow")

    def test_blank_image(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = detect_boxes(img)
        self.assertEqual(len(dets), 0)


if __name__ == "__main__":
    unittest.main()
