"""
train_model.py
==============
SIH26174 – AI Human Activity Recognition for On-board BAS Experiments
Train and Save Offline Random Forest Activity Classifier

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --dataset data/recorded_features.csv
"""

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ml.trainer import train_model


def main():
    parser = argparse.ArgumentParser(description="Train BAS HAR Activity Classifier")
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Optional path to recorded CSV dataset"
    )
    args = parser.parse_args()

    print("=======================================================")
    print("  SIH26174 • OFFLINE ML CLASSIFIER TRAINING PIPELINE")
    print("=======================================================")

    results = train_model(data_csv_path=args.dataset)

    print("\n✓ Offline Machine Learning model ready for hybrid FSM fusion.")


if __name__ == "__main__":
    main()
