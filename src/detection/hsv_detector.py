"""
hsv_detector.py
===============
SIH26174 – AI Human Activity Recognition for On-board BAS Experiments
Phase 1: Red & Yellow box detection using OpenCV HSV color segmentation.

Author : SIH26174 Team
Created: 2026-09-21

Design notes
------------
* Red wraps around the HSV hue wheel, so two ranges are used:
    Lower red : H  0–10
    Upper red : H 160–180
* Yellow occupies a single narrow hue band : H 18–35
* Morphological open+close is applied to the binary masks to remove
  small noise blobs before contour extraction.
* Contours smaller than MIN_CONTOUR_AREA are discarded.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# HSV colour ranges (all values are in OpenCV's 0-180 hue, 0-255 S/V scale)
# ---------------------------------------------------------------------------

# Red – lower hue band  (0° – 10°)
RED_LOWER_1 = np.array([0,   120, 70], dtype=np.uint8)
RED_UPPER_1 = np.array([10,  255, 255], dtype=np.uint8)

# Red – upper hue band  (160° – 180°)
RED_LOWER_2 = np.array([160, 120, 70], dtype=np.uint8)
RED_UPPER_2 = np.array([180, 255, 255], dtype=np.uint8)

# Yellow – single hue band  (18° – 35°)
YELLOW_LOWER = np.array([18, 100, 100], dtype=np.uint8)
YELLOW_UPPER = np.array([35, 255, 255], dtype=np.uint8)

# Minimum contour area in pixels^2 – contours below this are treated as noise
MIN_CONTOUR_AREA = 1500

# Structuring element for morphological operations (open then close)
_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_red_mask(hsv_frame: np.ndarray) -> np.ndarray:
    """
    Return a binary mask where red pixels are white (255).

    Two inRange calls are OR-ed together because red wraps around hue 0/180.

    Parameters
    ----------
    hsv_frame : np.ndarray
        Frame already converted to HSV colour space.

    Returns
    -------
    np.ndarray
        Single-channel uint8 mask (same H x W as input).
    """
    mask1 = cv2.inRange(hsv_frame, RED_LOWER_1, RED_UPPER_1)
    mask2 = cv2.inRange(hsv_frame, RED_LOWER_2, RED_UPPER_2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    return _clean_mask(red_mask)


def build_yellow_mask(hsv_frame: np.ndarray) -> np.ndarray:
    """
    Return a binary mask where yellow pixels are white (255).

    Parameters
    ----------
    hsv_frame : np.ndarray
        Frame already converted to HSV colour space.

    Returns
    -------
    np.ndarray
        Single-channel uint8 mask (same H x W as input).
    """
    yellow_mask = cv2.inRange(hsv_frame, YELLOW_LOWER, YELLOW_UPPER)
    return _clean_mask(yellow_mask)


def detect_boxes(frame: np.ndarray) -> list:
    """
    Detect red and yellow coloured boxes in a BGR frame.

    Steps
    -----
    1. Convert BGR to HSV.
    2. Build cleaned binary masks for red and yellow.
    3. Find external contours.
    4. Filter by MIN_CONTOUR_AREA.
    5. Compute bounding rectangles.

    Parameters
    ----------
    frame : np.ndarray
        Raw BGR frame from cv2.VideoCapture.

    Returns
    -------
    list of dict
        Each dict has keys:
            colour  - "red" or "yellow"
            bbox    - (x, y, w, h) bounding rectangle
            area    - contour area in pixels^2
            cx, cy  - centroid of the bounding box
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    detections = []

    colour_masks = [
        ("red",    build_red_mask(hsv)),
        ("yellow", build_yellow_mask(hsv)),
    ]

    for colour, mask in colour_masks:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_CONTOUR_AREA:
                continue  # skip noise

            x, y, w, h = cv2.boundingRect(cnt)
            detections.append({
                "colour": colour,
                "bbox":   (x, y, w, h),
                "area":   area,
                "cx":     x + w // 2,
                "cy":     y + h // 2,
            })

    return detections


def draw_detections(frame: np.ndarray, detections: list) -> np.ndarray:
    """
    Draw bounding boxes and labels onto the frame (returns annotated copy).

    Parameters
    ----------
    frame      : np.ndarray  - BGR frame.
    detections : list        - output of detect_boxes().

    Returns
    -------
    np.ndarray
        Annotated BGR frame (copy, original is not modified).
    """
    # BGR draw colours
    BOX_COLOURS = {
        "red":    (0,   0,   220),
        "yellow": (0,   200, 220),
    }
    LABEL_BG = {
        "red":    (0,   0,   180),
        "yellow": (0,   160, 200),
    }

    annotated = frame.copy()

    for det in detections:
        x, y, w, h = det["bbox"]
        colour_name = det["colour"]
        draw_colour = BOX_COLOURS[colour_name]
        bg_colour   = LABEL_BG[colour_name]

        # Bounding rectangle
        cv2.rectangle(annotated, (x, y), (x + w, y + h), draw_colour, 2)

        # Label background + text
        label = f"{colour_name.upper()}  {int(det['area'])} px^2"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(annotated, (x, y - th - 8), (x + tw + 6, y), bg_colour, -1)
        cv2.putText(
            annotated, label,
            (x + 3, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 1, cv2.LINE_AA
        )

        # Small centroid dot
        cv2.circle(annotated, (det["cx"], det["cy"]), 4, draw_colour, -1)

    return annotated


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean_mask(mask: np.ndarray) -> np.ndarray:
    """
    Remove speckle noise with morphological open (erode then dilate) followed
    by a close (dilate then erode) to fill small holes inside detected blobs.
    """
    opened = cv2.morphologyEx(mask,   cv2.MORPH_OPEN,  _MORPH_KERNEL, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, _MORPH_KERNEL, iterations=1)
    return closed
