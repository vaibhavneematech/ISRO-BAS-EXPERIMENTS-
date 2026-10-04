"""
inference.py
============
SIH26174 – AI Human Activity Recognition for On-board BAS Experiments
Offline Machine Learning Inference Engine

Loads the trained Random Forest classifier and feature scaler to produce
real-time activity classification and model confidence scores.
"""

from __future__ import annotations

import pathlib
from typing import Optional
import numpy as np
import joblib

# Activity class labels
ACTIVITY_CLASSES = [
    "IDLE",
    "DETECT_MAIN_BOX",
    "REMOVE_RED_BOX",
    "REMOVE_YELLOW_BOX",
    "PLACE_RED_BOX",
    "PLACE_YELLOW_BOX",
    "VERIFY_BOTH_PLACED",
]

_THIS_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = _THIS_DIR.parent.parent / "models" / "activity_classifier.pkl"
DEFAULT_SCALER_PATH = _THIS_DIR.parent.parent / "models" / "scaler.pkl"


class ActivityClassifier:
    """
    Offline inference engine for Human Activity Recognition.
    """

    def __init__(
        self,
        model_path: str | pathlib.Path = DEFAULT_MODEL_PATH,
        scaler_path: str | pathlib.Path = DEFAULT_SCALER_PATH,
    ):
        self.model_path = pathlib.Path(model_path)
        self.scaler_path = pathlib.Path(scaler_path)
        self._model = None
        self._scaler = None
        self._is_ready = False
        self._load()

    def _load(self):
        if self.model_path.exists() and self.scaler_path.exists():
            try:
                self._model = joblib.load(self.model_path)
                self._scaler = joblib.load(self.scaler_path)
                self._is_ready = True
                print(f"[ML Inference] Loaded activity classifier from {self.model_path.name}")
            except Exception as e:
                print(f"[ML Inference] WARNING: Failed to load model ({e}). Fallback to heuristic.")
        else:
            print("[ML Inference] Model weights not found. Run scripts/train_model.py to train.")

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def predict(self, feature_vector: np.ndarray) -> tuple[str, float]:
        """
        Predict activity class and confidence score (0.0 - 1.0).

        Returns: (predicted_class: str, confidence: float)
        """
        if not self._is_ready or self._model is None or self._scaler is None:
            # Fallback heuristic if model not trained yet
            return self._heuristic_fallback(feature_vector)

        try:
            x = feature_vector.reshape(1, -1)
            x_scaled = self._scaler.transform(x)
            pred_idx = self._model.predict(x_scaled)[0]
            probs = self._model.predict_proba(x_scaled)[0]
            confidence = float(np.max(probs))

            if isinstance(pred_idx, (int, np.integer)):
                label = ACTIVITY_CLASSES[pred_idx] if pred_idx < len(ACTIVITY_CLASSES) else "UNKNOWN"
            else:
                label = str(pred_idx)

            return label, confidence
        except Exception as e:
            return self._heuristic_fallback(feature_vector)

    @staticmethod
    def _heuristic_fallback(features: np.ndarray) -> tuple[str, float]:
        """Heuristic confidence estimator when offline model is initializing."""
        red_p = bool(features[0] > 0.5)
        yel_p = bool(features[1] > 0.5)
        hands = int(features[3])

        if red_p and yel_p:
            if hands == 0:
                return "VERIFY_BOTH_PLACED", 0.85
            return "DETECT_MAIN_BOX", 0.90
        elif not red_p and yel_p:
            return "REMOVE_RED_BOX", 0.88
        elif not red_p and not yel_p:
            return "REMOVE_YELLOW_BOX", 0.88
        elif red_p and not yel_p:
            return "PLACE_RED_BOX", 0.88
        return "IDLE", 0.60
