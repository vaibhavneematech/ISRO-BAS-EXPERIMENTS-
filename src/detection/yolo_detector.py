"""
yolo_detector.py
================
SIH26174 • VYOM: On-Board Protocol Compliance Assistant
High-performance, offline YOLOv11 detector for general laboratory objects.

Target Classes:
- cell phone (Phone Pickup Sequence)
- bottle (Bottle Handling Sequence)
- book (Notebook Transfer Sequence)
- cup (Cup Placement Sequence)
- person (Operator presence)
"""

from __future__ import annotations

import pathlib
from typing import Optional
import cv2
import numpy as np

# ── Project Model Path ────────────────────────────────────────────────────────
_THIS_DIR = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
_MODEL_CANDIDATES = [
    _PROJECT_ROOT / "models" / "yolo11n.pt",
    _PROJECT_ROOT / "yolo11n.pt",
    pathlib.Path("yolo11n.pt"),
]

# Friendly aliases mapping to standard COCO / YOLO names
CLASS_ALIASES = {
    "phone": "cell phone",
    "cell phone": "cell phone",
    "mobile": "cell phone",
    "bottle": "bottle",
    "book": "book",
    "notebook": "book",
    "cup": "cup",
    "mug": "cup",
    "person": "person",
}


class YOLODetector:
    """
    Offline YOLOv11 detector with caching and spatial tracking.
    Runs on CPU with minimal latency and high detection stability.
    """

    def __init__(
        self,
        model_path: Optional[str | pathlib.Path] = None,
        conf_threshold: float = 0.35,
        target_classes: Optional[list[str]] = None,
    ):
        self.conf_threshold = conf_threshold
        self.target_classes = target_classes or ["cell phone", "bottle", "book", "cup"]

        # Resolve model path
        self._model_path = None
        if model_path and pathlib.Path(model_path).exists():
            self._model_path = pathlib.Path(model_path)
        else:
            for cand in _MODEL_CANDIDATES:
                if cand.exists():
                    self._model_path = cand
                    break

        self._model = None
        self._available = False
        self._last_detections: list[dict] = []
        self._frame_count = 0
        self._init_model()

    def _init_model(self) -> None:
        if not self._model_path or not self._model_path.exists():
            print(f"[YOLODetector] Model file not found in candidates: {_MODEL_CANDIDATES}")
            return

        try:
            from ultralytics import YOLO
            self._model = YOLO(str(self._model_path))
            self._available = True
            print(f"[YOLODetector] Successfully loaded YOLO11n from {self._model_path.name}")
        except Exception as exc:
            print(f"[YOLODetector] Error initializing YOLO11: {exc}")
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    def set_target_classes(self, classes: list[str]) -> None:
        """Dynamically set classes to filter for current experiment."""
        resolved = []
        for c in classes:
            c_low = c.lower()
            resolved.append(CLASS_ALIASES.get(c_low, c_low))
        self.target_classes = resolved

    def detect(
        self,
        frame: np.ndarray,
        run_inference: bool = True,
    ) -> list[dict]:
        """
        Detect objects in frame.
        If run_inference is False, returns cached detections from previous frame.
        """
        if not self._available or self._model is None:
            return []

        self._frame_count += 1
        if not run_inference and self._last_detections:
            return self._last_detections

        try:
            h, w = frame.shape[:2]
            results = self._model(
                frame,
                imgsz=320,
                conf=self.conf_threshold,
                device="cpu",
                verbose=False,
            )

            detections = []
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None:
                    for i in range(len(boxes)):
                        cls_id = int(boxes.cls[i].item())
                        cls_name = results[0].names.get(cls_id, f"obj_{cls_id}").lower()
                        conf = float(boxes.conf[i].item())

                        # Filter by target classes if specified
                        if self.target_classes:
                            matched = False
                            for tc in self.target_classes:
                                if tc.lower() in cls_name or cls_name in tc.lower():
                                    matched = True
                                    break
                            if not matched:
                                continue

                        xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                        x1, y1, x2, y2 = xyxy
                        # Clamp
                        x1 = max(0, min(w - 1, int(x1)))
                        y1 = max(0, min(h - 1, int(y1)))
                        x2 = max(x1 + 1, min(w, int(x2)))
                        y2 = max(y1 + 1, min(h, int(y2)))

                        bw = x2 - x1
                        bh = y2 - y1
                        cx = x1 + bw // 2
                        cy = y1 + bh // 2

                        # Clean display label
                        display_name = cls_name.replace("cell phone", "phone")

                        det = {
                            "class": cls_name,
                            "name": display_name,
                            "colour": display_name,  # Uniform field for FSM matching
                            "confidence": conf,
                            "bbox": (x1, y1, bw, bh),
                            "xyxy": (x1, y1, x2, y2),
                            "center": (cx, cy),
                            "area": bw * bh,
                        }
                        detections.append(det)

            self._last_detections = detections
            return detections
        except Exception as exc:
            print(f"[YOLODetector] Inference error: {exc}")
            return self._last_detections


def draw_yolo_detections(
    frame: np.ndarray,
    detections: list[dict],
    accent_color: tuple[int, int, int] = (255, 180, 0),  # Cyan-blue BGR
) -> np.ndarray:
    """
    Draw futuristic, aerospace HUD bounding boxes for YOLO objects.
    """
    annotated = frame.copy()
    h, w = frame.shape[:2]

    for det in detections:
        x, y, bw, bh = det["bbox"]
        name = det.get("name", "OBJECT").upper()
        conf = det.get("confidence", 0.0)

        # Select color based on object class
        c_low = det.get("class", "").lower()
        if "phone" in c_low:
            col = (255, 180, 0)      # Cyan / Sky
        elif "bottle" in c_low:
            col = (0, 220, 255)      # Yellow / Gold
        elif "book" in c_low or "note" in c_low:
            col = (180, 105, 255)    # Purple / Lavender
        elif "cup" in c_low:
            col = (100, 255, 100)    # Mint Emerald
        else:
            col = accent_color

        # Draw semi-transparent background fill for corners
        line_len = min(20, bw // 4, bh // 4)

        # Main box outline
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), col, 1, cv2.LINE_AA)

        # Corner HUD brackets (thicker)
        th = 2
        # Top-left
        cv2.line(annotated, (x, y), (x + line_len, y), col, th)
        cv2.line(annotated, (x, y), (x, y + line_len), col, th)
        # Top-right
        cv2.line(annotated, (x + bw, y), (x + bw - line_len, y), col, th)
        cv2.line(annotated, (x + bw, y), (x + bw, y + line_len), col, th)
        # Bottom-left
        cv2.line(annotated, (x, y + bh), (x + line_len, y + bh), col, th)
        cv2.line(annotated, (x, y + bh), (x, y + bh - line_len), col, th)
        # Bottom-right
        cv2.line(annotated, (x + bw, y + bh), (x + bw - line_len, y + bh), col, th)
        cv2.line(annotated, (x + bw, y + bh), (x + bw, y + bh - line_len), col, th)

        # Center reticle dot
        cx, cy = det["center"]
        cv2.circle(annotated, (cx, cy), 3, col, -1)

        # HUD Label Banner
        label_text = f"YOLOv11 • {name} [{conf*100:.0f}%]"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.42
        thickness = 1
        (tw, th_box), baseline = cv2.getTextSize(label_text, font, scale, thickness)

        lbl_y1 = max(0, y - th_box - 8)
        lbl_y2 = max(th_box + 8, y)
        lbl_x2 = min(w, x + tw + 10)

        # Dark background tag
        cv2.rectangle(annotated, (x, lbl_y1), (lbl_x2, lbl_y2), (15, 23, 42), -1)
        cv2.rectangle(annotated, (x, lbl_y1), (lbl_x2, lbl_y2), col, 1)

        # Text
        cv2.putText(
            annotated,
            label_text,
            (x + 5, lbl_y2 - 4),
            font,
            scale,
            col,
            thickness,
            cv2.LINE_AA,
        )

    return annotated
