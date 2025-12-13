import joblib
import pandas as pd
import numpy as np
from pathlib import Path

def test_model_load_and_predict():
    model_path = Path("app/model.pkl")
    assert model_path.exists()

    model = joblib.load(model_path)

    X = pd.DataFrame([{
        f"V{i}": np.random.randn() for i in range(1, 29)
    } | {"Amount": 50.0, "Time": 1000}])

    preds = model.predict(X)
    probs = model.predict_proba(X)

    assert preds.shape == (1,)
    assert probs.shape == (1, 2)
    assert preds[0] in [0, 1]
