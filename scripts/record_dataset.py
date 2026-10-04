"""
record_dataset.py
=================
SIH26174 – AI Human Activity Recognition for On-board BAS Experiments
Data Collection Utility for Feature Extraction and Dataset Logging

Usage:
    python scripts/record_dataset.py
    python scripts/record_dataset.py --output data/experiment_dataset.csv

Controls during recording:
    0-6 : Tag current frame with Activity Class:
          0: IDLE
          1: DETECT_MAIN_BOX
          2: REMOVE_RED_BOX
          3: REMOVE_YELLOW_BOX
          4: PLACE_RED_BOX
          5: PLACE_YELLOW_BOX
          6: VERIFY_BOTH_PLACED
    SPACE: Auto-advance protocol step tag
    s    : Toggle recording on/off
    q    : Save and quit
"""

import sys
import time
import argparse
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.detection.hsv_detector import detect_boxes, draw_detections
from src.detection.mediapipe_hands import HandDetector, draw_hands
from src.features.feature_extractor import FeatureExtractor, FEATURE_NAMES
from src.ml.inference import ACTIVITY_CLASSES

DATA_DIR = ROOT / "data"


def main():
    parser = argparse.ArgumentParser(description="Record HAR feature dataset")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument(
        "--output", type=str, default=str(DATA_DIR / "activity_dataset.csv"),
        help="Output CSV path"
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    hand_detector = HandDetector(max_num_hands=2)
    feature_extractor = FeatureExtractor()

    current_class_idx = 0
    is_recording = False
    recorded_samples = []

    header = ",".join(FEATURE_NAMES) + ",label\n"

    print("=======================================================")
    print("  SIH26174 • DATA COLLECTION UTILITY")
    print("=======================================================")
    print(f"Target CSV: {out_path}")
    print("Keys: [0-6] Set Class | [s] Toggle Record | [q] Save & Quit")
    print("=======================================================")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        box_dets = detect_boxes(frame)
        hand_dets = hand_detector.detect(frame)
        features = feature_extractor.extract(box_dets, hand_dets)

        if is_recording:
            row = np.append(features, float(current_class_idx))
            recorded_samples.append(row)

        # Drawing HUD
        annotated = draw_detections(frame, box_dets)
        annotated = draw_hands(annotated, hand_dets)

        rec_status = "● RECORDING" if is_recording else "○ PAUSED"
        color = (0, 0, 255) if is_recording else (128, 128, 128)
        current_label = ACTIVITY_CLASSES[current_class_idx]

        cv2.putText(
            annotated, f"STATUS: {rec_status} | SAMPLES: {len(recorded_samples)}",
            (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
        )
        cv2.putText(
            annotated, f"TAG: [{current_class_idx}] {current_label}",
            (16, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA
        )

        cv2.imshow("SIH26174 Dataset Recorder", annotated)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            is_recording = not is_recording
            print(f"[Recorder] {'Started' if is_recording else 'Paused'}")
        elif ord('0') <= key <= ord('6'):
            current_class_idx = key - ord('0')
            print(f"[Recorder] Set tag to: [{current_class_idx}] {ACTIVITY_CLASSES[current_class_idx]}")
        elif key == 32:  # SPACE
            current_class_idx = (current_class_idx + 1) % len(ACTIVITY_CLASSES)
            print(f"[Recorder] Advanced tag to: [{current_class_idx}] {ACTIVITY_CLASSES[current_class_idx]}")

    cap.release()
    hand_detector.close()
    cv2.destroyAllWindows()

    if recorded_samples:
        mode = "a" if out_path.exists() else "w"
        with open(out_path, mode, encoding="utf-8") as f:
            if mode == "w":
                f.write(header)
            for row in recorded_samples:
                line = ",".join(f"{val:.5f}" for val in row[:-1]) + f",{int(row[-1])}\n"
                f.write(line)
        print(f"\n✓ Successfully saved {len(recorded_samples)} samples to {out_path}")
    else:
        print("\nNo samples recorded.")


if __name__ == "__main__":
    main()
