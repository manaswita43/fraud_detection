# scripts/drift_check.py
import pandas as pd
import joblib
import mlflow
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score

from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="app/model_with_location.pkl")
parser.add_argument("--v0-data", default="data/v0/transactions_2022_with_location.csv")
parser.add_argument("--v1-data", default="data/v1/transactions_2023.csv")
args = parser.parse_args()

mlflow.set_tracking_uri("http://34.172.186.194:8100")
mlflow.set_experiment("fraud-detection")

with mlflow.start_run(run_name="drift_check_evidently"):

    # -----------------------------
    # Load model and data
    # -----------------------------
    model = joblib.load(args.model)
    df_v0 = pd.read_csv(args.v0_data)
    df_v1 = pd.read_csv(args.v1_data)

    # -----------------------------
    # Handle missing location in v1
    # -----------------------------
    if "location" not in df_v1.columns:
        df_v1["location"] = np.random.choice(
            ["Location_A", "Location_B"], size=len(df_v1)
        )

    # -----------------------------
    # One-hot encoding (same as training)
    # -----------------------------
    df_v0 = pd.get_dummies(df_v0, columns=["location"], drop_first=True)
    df_v1 = pd.get_dummies(df_v1, columns=["location"], drop_first=True)

    # -----------------------------
    # Align columns safely
    # -----------------------------
    target_col = "Class"
    feature_cols = [c for c in df_v0.columns if c != target_col]

    df_v0 = df_v0.reindex(columns=feature_cols + [target_col], fill_value=0)
    df_v1 = df_v1.reindex(columns=feature_cols + [target_col], fill_value=0)

    X_v0, y_v0 = df_v0[feature_cols], df_v0[target_col]
    X_v1, y_v1 = df_v1[feature_cols], df_v1[target_col]

    # -----------------------------
    # Predictions
    # -----------------------------
    preds_v0 = model.predict(X_v0)
    preds_v1 = model.predict(X_v1)

    # -----------------------------
    # ### NEW: Performance metrics
    # -----------------------------
    f1_v0 = f1_score(y_v0, preds_v0)
    f1_v1 = f1_score(y_v1, preds_v1)
    prec_v0 = precision_score(y_v0, preds_v0, zero_division=0)
    prec_v1 = precision_score(y_v1, preds_v1, zero_division=0)
    rec_v0 = recall_score(y_v0, preds_v0, zero_division=0)
    rec_v1 = recall_score(y_v1, preds_v1, zero_division=0)

    mlflow.log_metric("f1_v0", float(f1_v0))
    mlflow.log_metric("f1_v1", float(f1_v1))
    mlflow.log_metric("precision_v0", float(prec_v0))
    mlflow.log_metric("precision_v1", float(prec_v1))
    mlflow.log_metric("recall_v0", float(rec_v0))
    mlflow.log_metric("recall_v1", float(rec_v1))

    # -----------------------------
    # ### NEW: Drift comparison plot
    # -----------------------------
    plt.figure(figsize=(6, 4))
    plt.bar(["v0", "v1"], [f1_v0, f1_v1])
    plt.ylabel("F1-score")
    plt.title("Model Performance Drift (F1-score)")
    plt.tight_layout()

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    drift_plot = artifacts_dir / "drift_comparison.png"
    plt.savefig(drift_plot)
    plt.close()

    mlflow.log_artifact(str(drift_plot))

    # -----------------------------
    # Evidently Drift Report
    # -----------------------------
    report = Report(
        metrics=[
            DataDriftPreset(),
            DataSummaryPreset()
        ]
    )

    my_eval = report.run(
        reference_data=df_v0[feature_cols],
        current_data=df_v1[feature_cols]
    )

    report_file = artifacts_dir / "evidently_drift_report.html"
    my_eval.save_html(str(report_file))
    assert report_file.exists()

    mlflow.log_artifact(str(report_file))

    print("Evidently report + performance drift plot logged to MLflow.")
