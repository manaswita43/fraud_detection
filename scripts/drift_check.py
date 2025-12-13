# scripts/drift_check.py
import pandas as pd
import joblib
import mlflow
import argparse
import numpy as np
from pathlib import Path

from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="app/model_with_location.pkl")
parser.add_argument("--v0-data", default="data/v0/transactions_2022_with_location.csv")
parser.add_argument("--v1-data", default="data/v1/transactions_2023.csv")
args = parser.parse_args()

mlflow.set_tracking_uri("http://34.71.166.41:8100")
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

    # -----------------------------
    # Add predictions (optional but OK)
    # -----------------------------
    df_v0["prediction"] = model.predict(df_v0[feature_cols])
    df_v1["prediction"] = model.predict(df_v1[feature_cols])

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

    # -----------------------------
    # Save & log report (FIXED)
    # -----------------------------
    report_path = Path("artifacts")
    report_path.mkdir(parents=True, exist_ok=True)

    report_file = report_path / "evidently_drift_report.html"

    # IMPORTANT: convert Path → str
    my_eval.save_html(str(report_file))

    # Ensure file exists before logging
    assert report_file.exists(), "Evidently report was not generated!"

    mlflow.log_artifact(str(report_file))

    print("Evidently drift report generated and logged to MLflow.")
