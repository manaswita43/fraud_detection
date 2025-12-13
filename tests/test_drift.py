import subprocess
import sys
from pathlib import Path


def test_drift_check_script_runs():
    """
    End-to-end drift test using Evidently
    """

    cmd = [
        sys.executable,
        "scripts/drift_check.py",
        "--model", "app/model_with_location.pkl",
        "--v0-data", "data/v0/transactions_2022_with_location.csv",
        "--v1-data", "data/v1/transactions_2023.csv",
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr


def test_evidently_report_created():
    """
    Evidently drift report should exist
    """
    report = Path("artifacts/evidently_drift_report.html")
    assert report.exists(), "Evidently drift report missing"


def test_evidently_report_is_html():
    """
    Basic sanity check on HTML content
    """
    report = Path("artifacts/evidently_drift_report.html")
    content = report.read_text().lower()

    assert "<html" in content
    assert "data drift" in content
