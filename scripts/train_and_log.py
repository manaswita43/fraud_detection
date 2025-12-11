# scripts/train_and_log.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import xgboost as xgb
import joblib
import mlflow
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/v0/transactions_2022.csv")
parser.add_argument("--model-out", default="app/model.pkl")
parser.add_argument("--mlflow-uri", default=None)
parser.add_argument("--run-name", default="baseline")
args = parser.parse_args()
    
mlflow.set_tracking_uri("http://35.238.80.113:8100")    

mlflow.set_experiment("fraud-detection")
with mlflow.start_run(run_name=args.run_name):
    df = pd.read_csv(args.data)
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # simple split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # scale_pos_weight for XGBoost
    scale_pos_weight = (y_train==0).sum() / (y_train==1).sum()
    clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        use_label_encoder=False,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_val)
    probs = clf.predict_proba(X_val)[:,1]
    f1 = f1_score(y_val, preds)
    prec = precision_score(y_val, preds, zero_division=0)
    rec = recall_score(y_val, preds, zero_division=0)

    mlflow.log_param("scale_pos_weight", float(scale_pos_weight))
    mlflow.log_metric("f1", float(f1))
    mlflow.log_metric("precision", float(prec))
    mlflow.log_metric("recall", float(rec))

    # Save model
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, args.model_out)
    mlflow.log_artifact(args.model_out, artifact_path="models")

    print("F1:", f1, "Precision:", prec, "Recall:", rec)
