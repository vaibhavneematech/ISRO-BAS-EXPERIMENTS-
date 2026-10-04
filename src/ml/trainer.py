"""
trainer.py
==========
SIH26174 – AI Human Activity Recognition for On-board BAS Experiments
Offline Model Trainer for Random Forest Activity Classifier

Trains Random Forest classifier and StandardScaler on kinematic and spatial
features, outputting metrics and serializing to models/ directory.
"""

from __future__ import annotations

import pathlib
from typing import Optional, Tuple
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score

from src.features.feature_extractor import FEATURE_NAMES
from src.ml.inference import ACTIVITY_CLASSES

_THIS_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_MODELS_DIR = _THIS_DIR.parent.parent / "models"


def generate_synthetic_dataset(samples_per_class: int = 400) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate domain-grounded synthetic training data matching realistic BAS kinematics:
    Classes:
      0: IDLE
      1: DETECT_MAIN_BOX
      2: REMOVE_RED_BOX
      3: REMOVE_YELLOW_BOX
      4: PLACE_RED_BOX
      5: PLACE_YELLOW_BOX
      6: VERIFY_BOTH_PLACED
    """
    np.random.seed(42)
    X = []
    y = []

    for class_idx, class_name in enumerate(ACTIVITY_CLASSES):
        for _ in range(samples_per_class):
            # Noise factors
            noise = np.random.normal(0, 0.05)

            if class_name == "IDLE":
                red = 0.0 if np.random.rand() > 0.5 else 1.0
                yel = 0.0 if np.random.rand() > 0.5 else 1.0
                boxes = red + yel
                hands = 0.0
                conf = 0.0
                d_red = 800.0
                d_yel = 800.0
                touch_r = 0.0
                touch_y = 0.0
                vx, vy, spd = 0.0, 0.0, 0.0

            elif class_name == "DETECT_MAIN_BOX":
                red, yel = 1.0, 1.0
                boxes = 2.0
                hands = 1.0 if np.random.rand() > 0.4 else 0.0
                conf = np.random.uniform(0.75, 0.98) if hands > 0 else 0.0
                d_red = np.random.uniform(150, 450) if hands > 0 else 800.0
                d_yel = np.random.uniform(150, 450) if hands > 0 else 800.0
                touch_r, touch_y = 0.0, 0.0
                vx = np.random.normal(0, 4.0) if hands > 0 else 0.0
                vy = np.random.normal(0, 4.0) if hands > 0 else 0.0
                spd = float(np.sqrt(vx * vx + vy * vy))

            elif class_name == "REMOVE_RED_BOX":
                red, yel = 0.0, 1.0  # red absent, yellow present
                boxes = 1.0
                hands = 1.0
                conf = np.random.uniform(0.80, 0.99)
                d_red = np.random.uniform(5, 75)   # close to recent red location
                d_yel = np.random.uniform(120, 400)
                touch_r = 1.0 if d_red < 35 else 0.0
                touch_y = 0.0
                vx = np.random.normal(-15.0, 8.0)  # pulling away motion
                vy = np.random.normal(-10.0, 6.0)
                spd = float(np.sqrt(vx * vx + vy * vy))

            elif class_name == "REMOVE_YELLOW_BOX":
                red, yel = 0.0, 0.0  # both absent
                boxes = 0.0
                hands = 1.0
                conf = np.random.uniform(0.80, 0.99)
                d_red = np.random.uniform(180, 600)
                d_yel = np.random.uniform(5, 75)
                touch_r = 0.0
                touch_y = 1.0 if d_yel < 35 else 0.0
                vx = np.random.normal(12.0, 7.0)
                vy = np.random.normal(-12.0, 6.0)
                spd = float(np.sqrt(vx * vx + vy * vy))

            elif class_name == "PLACE_RED_BOX":
                red, yel = 1.0, 0.0  # red present, yellow absent
                boxes = 1.0
                hands = 1.0
                conf = np.random.uniform(0.82, 0.99)
                d_red = np.random.uniform(0, 45)   # in contact with red
                d_yel = 800.0
                touch_r = 1.0
                touch_y = 0.0
                vx = np.random.normal(5.0, 6.0)    # stabilizing motion
                vy = np.random.normal(4.0, 5.0)
                spd = float(np.sqrt(vx * vx + vy * vy))

            elif class_name == "PLACE_YELLOW_BOX":
                red, yel = 1.0, 1.0  # both present
                boxes = 2.0
                hands = 1.0
                conf = np.random.uniform(0.82, 0.99)
                d_red = np.random.uniform(120, 350)
                d_yel = np.random.uniform(0, 45)   # in contact with yellow
                touch_r = 0.0
                touch_y = 1.0
                vx = np.random.normal(4.0, 5.0)
                vy = np.random.normal(4.0, 5.0)
                spd = float(np.sqrt(vx * vx + vy * vy))

            elif class_name == "VERIFY_BOTH_PLACED":
                red, yel = 1.0, 1.0
                boxes = 2.0
                hands = 0.0 if np.random.rand() > 0.3 else 1.0
                conf = np.random.uniform(0.70, 0.90) if hands > 0 else 0.0
                d_red = np.random.uniform(250, 600) if hands > 0 else 800.0
                d_yel = np.random.uniform(250, 600) if hands > 0 else 800.0
                touch_r, touch_y = 0.0, 0.0
                vx, vy, spd = 0.0, 0.0, 0.0

            r_area = (red * np.random.uniform(0.015, 0.035))
            y_area = (yel * np.random.uniform(0.015, 0.035))

            feat = [
                red, yel, boxes, hands, conf,
                d_red, d_yel, touch_r, touch_y,
                vx, vy, spd, r_area, y_area
            ]
            X.append(feat)
            y.append(class_idx)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def train_model(
    data_csv_path: Optional[str | pathlib.Path] = None,
    output_dir: pathlib.Path = DEFAULT_MODELS_DIR,
) -> dict:
    """
    Train Random Forest model and StandardScaler.
    Saves activity_classifier.pkl and scaler.pkl to output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if data_csv_path and pathlib.Path(data_csv_path).exists():
        print(f"[ML Trainer] Loading recorded dataset from {data_csv_path}")
        raw = np.loadtxt(data_csv_path, delimiter=",", skiprows=1)
        X = raw[:, :-1]
        y = raw[:, -1].astype(np.int64)
    else:
        print("[ML Trainer] Generating domain-grounded training dataset (7 activity classes)...")
        X, y = generate_synthetic_dataset(samples_per_class=500)

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Random Forest Classifier
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train_scaled, y_train)

    # Evaluation
    train_acc = accuracy_score(y_train, clf.predict(X_train_scaled))
    test_acc = accuracy_score(y_test, clf.predict(X_test_scaled))
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5)

    print("\n" + "="*55)
    print(f"  RANDOM FOREST TRAINING RESULTS")
    print("="*55)
    print(f"  Training Accuracy   : {train_acc * 100:.2f}%")
    print(f"  Test Accuracy       : {test_acc * 100:.2f}%")
    print(f"  5-Fold CV Mean      : {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")
    print("="*55)

    # Save artifacts
    model_path = output_dir / "activity_classifier.pkl"
    scaler_path = output_dir / "scaler.pkl"

    joblib.dump(clf, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"[ML Trainer] Saved model  -> {model_path}")
    print(f"[ML Trainer] Saved scaler -> {scaler_path}")

    return {
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "cv_mean": float(cv_scores.mean()),
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
    }
