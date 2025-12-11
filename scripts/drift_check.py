# scripts/drift_check.py
import pandas as pd
import joblib
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import mlflow
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="app/model_with_location.pkl")  # trained on v0 with location
parser.add_argument("--v0-data", default="data/v0/transactions_2022_with_location.csv")
parser.add_argument("--v1-data", default="data/v1/transactions_2023.csv")
args = parser.parse_args()

mlflow.set_tracking_uri('http://136.111.193.119:8100')

mlflow.set_experiment("fraud-detection")

with mlflow.start_run(run_name="drift_check"):
    model = joblib.load(args.model)
    df_v0 = pd.read_csv(args.v0_data)
    df_v1 = pd.read_csv(args.v1_data)

    # Ensure columns match: if v1 lacks location, add random or default cols matching training
    if "location" not in df_v1.columns:
        import numpy as np
        df_v1["location"] = np.random.choice(["Location_A","Location_B"], size=len(df_v1))

    # one hot encode similar to training
    df_v0 = pd.get_dummies(df_v0, columns=["location"], drop_first=True)
    df_v1 = pd.get_dummies(df_v1, columns=["location"], drop_first=True)

    # align columns
    common_cols = [c for c in df_v0.columns if c != "Class"]
    df_v1 = df_v1.reindex(columns=common_cols + ["Class"], fill_value=0)

    X_v0 = df_v0[common_cols]
    y_v0 = df_v0["Class"]
    X_v1 = df_v1[common_cols]
    y_v1 = df_v1["Class"]

    preds_v0 = model.predict(X_v0)
    preds_v1 = model.predict(X_v1)

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

    # plot comparison
    plt.figure(figsize=(6,4))
    plt.bar(["v0","v1"], [f1_v0, f1_v1])
    plt.ylabel("F1-score")
    plt.title("F1: v0 vs v1")
    plt.savefig("drift_comparison.png")
    mlflow.log_artifact("drift_comparison.png")
    print("Logged drift metrics and plot.")
