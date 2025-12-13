import numpy as np
from fairlearn.metrics import demographic_parity_difference
import subprocess
import sys
from pathlib import Path
import joblib
import pandas as pd


def test_train_shap_fair_end_to_end(tmp_path):
    """
    End-to-end test:
    - Trains model
    - Generates SHAP artifacts
    - Saves model
    """

    model_out = tmp_path / "model.pkl"
    data_path = "data/v0/transactions_2022_with_location.csv"

    cmd = [
        sys.executable,
        "scripts/train_shap_fair.py",
        "--data", data_path,
        "--model-out", str(model_out),
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
    assert model_out.exists(), "Model file not created"


def test_shap_artifacts_created():
    """
    Ensure SHAP artifacts are generated
    """
    assert Path("shap_summary.png").exists(), "SHAP summary plot missing"
    assert Path("artifacts/shap_report.txt").exists(), "SHAP text report missing"


def test_shap_report_has_content():
    """
    SHAP report should not be empty
    """
    report = Path("artifacts/shap_report.txt")
    content = report.read_text()

    assert "MODEL EXPLAINABILITY RESULTS" in content
    assert "Top 10 Most Important Features" in content


def test_model_inference_after_training():
    """
    Saved model should be loadable and produce predictions
    """
    model_path = Path("app/model_with_location.pkl")
    assert model_path.exists(), "Trained model not found"

    model = joblib.load(model_path)

    # minimal fake input
    X = np.random.rand(5, model.n_features_in_)
    preds = model.predict(X)

    assert len(preds) == 5


def test_demographic_parity_computation():
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    sensitive = np.array([0, 0, 1, 1, 0, 1])

    dp = demographic_parity_difference(
        y_true, y_pred, sensitive_features=sensitive
    )

    assert isinstance(dp, float)
