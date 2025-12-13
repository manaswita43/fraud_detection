# scripts/train_shap_fair.py
import pandas as pd
import mlflow
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import numpy as np
from fairlearn.metrics import demographic_parity_difference
import argparse
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/v0/transactions_2022_with_location.csv")
parser.add_argument("--model-out", default="app/model_with_location.pkl")
parser.add_argument("--mlflow-uri", default=None)
args = parser.parse_args()

if args.mlflow_uri:
    mlflow.set_tracking_uri(args.mlflow_uri)
else:
    mlflow.set_tracking_uri('http://34.71.166.41:8100')

mlflow.set_experiment("fraud-detection")

with mlflow.start_run(run_name="final_with_location"):
    df = pd.read_csv(args.data)

    # one-hot encode location if present
    if "location" in df.columns:
        df = pd.get_dummies(df, columns=["location"], drop_first=True)
    else:
        print("Warning: 'location' not in dataset; proceeding without it.")

    if "Class" not in df.columns:
        raise ValueError("Target column 'Class' not found in the dataset.")
    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scale_pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())

    clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        use_label_encoder=False,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    preds_val = clf.predict(X_val)
    f1 = f1_score(y_val, preds_val)
    prec = precision_score(y_val, preds_val, zero_division=0)
    rec = recall_score(y_val, preds_val, zero_division=0)
    mlflow.log_metric("f1_val", float(f1))
    mlflow.log_metric("precision_val", float(prec))
    mlflow.log_metric("recall_val", float(rec))

    # prepare sample for SHAP (limit to 2000 for memory/perf)
    if len(X_val) > 2000:
        X_val_sample = X_val.sample(n=2000, random_state=42).copy()
    else:
        X_val_sample = X_val.copy()

    # ---- ENSURE FEATURES ARE NUMERIC (this fixes the masker / isfinite errors) ----
    # Convert all columns to numeric (coerce errors), fill NaN with 0 (or another strategy)
    # Keep a copy for plotting (pandas DF), but the background and explainer will use numeric numpy arrays.
    feature_names = X_train.columns.tolist()
    X_train_num = X_train.copy().apply(pd.to_numeric, errors="coerce").fillna(0)
    X_val_num = X_val_sample.copy().apply(pd.to_numeric, errors="coerce").fillna(0)

    # background for masker: small representative numeric sample (floats)
    bg_size = min(100, len(X_train_num))
    background = X_train_num.sample(n=bg_size, random_state=42).to_numpy(dtype=float)

    # Model callable that accepts numpy array and returns probability array
    def model_callable(X_np: np.ndarray):
        # ensure shape and dtype
        X_np = np.asarray(X_np, dtype=float)
        return clf.predict_proba(X_np)

    # ---- Create shap.Explainer with callable + numeric background ----
    try:
        # shap.maskers.Independent exists in modern versions; fallback if not present
        try:
            masker = shap.maskers.Independent(background)  # background is numpy float
        except Exception:
            masker = background  # pass raw background if maskers unavailable

        explainer = shap.Explainer(model_callable, masker)
        # Explain the numeric validation sample (pass numpy floats)
        shap_exp = explainer(X_val_num.to_numpy(dtype=float))
        shap_vals = shap_exp.values  # could be 2D (n,m) or 3D (n,m,k)
        expected_value = shap_exp.base_values if hasattr(shap_exp, "base_values") else explainer.expected_value
    except Exception as e:
        # Final fallback: use the callable without explicit masker/background
        explainer = shap.Explainer(model_callable, X_train_num.sample(n=bg_size, random_state=42).to_numpy(dtype=float))
        shap_exp = explainer(X_val_num.to_numpy(dtype=float))
        shap_vals = shap_exp.values
        expected_value = shap_exp.base_values if hasattr(shap_exp, "base_values") else explainer.expected_value

    # Normalize shap_vals into 2D array for summary plotting: (n_samples, n_features)
    if isinstance(shap_vals, list):
        # old API: list of arrays per class
        cls_idx = 1 if len(shap_vals) > 1 else 0
        shap_vals_2d = np.array(shap_vals[cls_idx])
    elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        # shape (n_samples, n_features, n_classes) or (n_classes, n_samples, n_features)
        # prefer (n_samples, n_features, n_classes) -> pick class 1 if present
        if shap_vals.shape[2] > 1:
            shap_vals_2d = shap_vals[:, :, 1]
        else:
            shap_vals_2d = shap_vals[:, :, 0]
    else:
        shap_vals_2d = np.array(shap_vals)

    # ---- SHAP SUMMARY PLOT (matplotlib) ----
    plt.figure(figsize=(10, 6))
    try:
        # If we have an Explanation object, we can also pass that directly to summary_plot,
        # but passing (shap_values_2d, features_df) is safe.
        shap.summary_plot(shap_vals_2d, X_val_sample, feature_names=feature_names, show=False)
    except Exception:
        # Try without feature_names param if it errors
        shap.summary_plot(shap_vals_2d, X_val_sample, show=False)

    shap_img = "shap_summary.png"
    plt.tight_layout()
    plt.savefig(shap_img, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(shap_img)

    # ---- FORCE PLOT / SINGLE-SAMPLE PLOT ----
    # Choose the sample with highest predicted fraud probability for an informative view
    try:
        probs = clf.predict_proba(X_val_num.to_numpy(dtype=float))[:, 1]
        single_idx = int(np.argmax(probs))
    except Exception:
        single_idx = 0

    # get single sample shap values for chosen class (1)
    if shap_vals_2d.ndim == 2:
        sv_single = shap_vals_2d[single_idx]
    else:
        # as fallback convert
        sv_single = np.array(shap_vals)[single_idx]

    # Determine base/expected value for the same class (if vector-like)
    base_for_plot = None
    try:
        ev = expected_value
        if isinstance(ev, (list, np.ndarray)):
            ev_arr = np.asarray(ev)
            if ev_arr.ndim > 0 and ev_arr.size > 1:
                base_for_plot = float(ev_arr[1]) if ev_arr.size > 1 else float(ev_arr[0])
            else:
                base_for_plot = float(ev_arr)
        else:
            base_for_plot = float(ev)
    except Exception:
        base_for_plot = float(clf.predict_proba(X_train_num.to_numpy(dtype=float)).mean())

    # Create interactive force plot HTML if possible
    try:
        # For shap.force_plot, shap_values should be 1D array for the sample
        force_obj = shap.force_plot(
            base_for_plot,
            sv_single,
            X_val_sample.iloc[[single_idx]],
            feature_names=feature_names,
            matplotlib=False,
        )
        shap_force_plot_html = "shap_force_plot_sample.html"
        shap.save_html(shap_force_plot_html, force_obj)
        mlflow.log_artifact(shap_force_plot_html)
    except Exception:
        # Fallback: create bar chart of absolute shap contributions for top features
        abs_sv = np.abs(sv_single)
        top_n = min(20, abs_sv.size)
        top_idx = np.argsort(abs_sv)[-top_n:][::-1]
        plt.figure(figsize=(8, 6))
        plt.barh([feature_names[i] for i in top_idx], abs_sv[top_idx])
        plt.gca().invert_yaxis()
        bar_img = "shap_force_bar_sample.png"
        plt.tight_layout()
        plt.savefig(bar_img, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(bar_img)
        
    # --- Create and Save Textual SHAP Report Artifact ---
    try:
        # Use the already normalized 2D SHAP values
        # shap_vals_2d shape: (n_samples, n_features)
        shap_importance = np.abs(shap_vals_2d).mean(axis=0)

        importance_df = pd.DataFrame({
            "feature": X_val_sample.columns,
            "importance": shap_importance
        }).sort_values(by="importance", ascending=False)

        # Ensure artifact directory exists
        report_dir = Path("artifacts")
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "shap_report.txt"

        with open(report_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("MODEL EXPLAINABILITY RESULTS (SHAP)\n")
            f.write("=" * 60 + "\n\n")

            f.write("Top 10 Most Important Features:\n")
            f.write(importance_df.head(10).to_string(index=False))
            f.write("\n\n" + "=" * 60 + "\n")

            top_feature = importance_df.iloc[0]
            f.write("Key Insights:\n")
            f.write(
                f"• Most predictive feature: {top_feature['feature']} "
                f"(Mean |SHAP| = {top_feature['importance']:.4f})\n"
            )
            f.write(
                "• Higher SHAP values indicate stronger contribution to fraud prediction.\n"
            )
            f.write(
                "• Feature importance is averaged over validation samples.\n"
            )

        mlflow.log_artifact(str(report_path))
        print(f"Textual SHAP report saved to {report_path}")

    except Exception as e:
        print(f"Error during textual SHAP report generation: {e}")



    # ---- Fairness metric (demographic parity difference) ----
    loc_col = None
    for c in X_val.columns:
        if c.startswith("location_"):
            loc_col = c
            break

    if loc_col is not None:
        sensitive = X_val[loc_col].astype(int)
    else:
        print("No location_* found in features; using random sensitive attribute (fallback).")
        sensitive = (np.random.rand(len(X_val)) > 0.5).astype(int)

    dp_diff = demographic_parity_difference(y_val, preds_val, sensitive_features=sensitive)
    mlflow.log_metric("demographic_parity_difference", float(dp_diff))

    # ---- Save model & artifacts ----
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, args.model_out)
    mlflow.log_artifact(args.model_out)
    print("Saved model and artifacts.")
