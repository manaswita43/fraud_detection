import argparse
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained model (.pkl)")
    parser.add_argument("--data", required=True, help="Path to evaluation CSV")
    parser.add_argument("--output", default="metrics.json", help="Metrics output file")
    args = parser.parse_args()

    # ----------------------------
    # Load model
    # ----------------------------
    model = joblib.load(args.model)

    # ----------------------------
    # Load data
    # ----------------------------
    df = pd.read_csv(args.data)

    if "Class" not in df.columns:
        raise ValueError("Target column 'Class' not found in dataset")

    X = df.drop(columns=["Class"])
    y = df["Class"]

    # ----------------------------
    # Handle categorical columns (safety)
    # ----------------------------
    if "location" in X.columns:
        X = pd.get_dummies(X, columns=["location"], drop_first=True)

    # ----------------------------
    # Predict
    # ----------------------------
    y_pred = model.predict(X)

    # ----------------------------
    # Metrics
    # ----------------------------
    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0))
    }

    # ----------------------------
    # Save metrics for CML
    # ----------------------------
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Evaluation metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print(f"\nMetrics saved to {args.output}")

if __name__ == "__main__":
    main()
