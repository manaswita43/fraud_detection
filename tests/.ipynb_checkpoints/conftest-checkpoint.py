import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def sample_transaction():
    features = {
        f"V{i}": np.random.randn() for i in range(1, 29)
    }
    features["Amount"] = 120.0
    features["Time"] = 50000
    return {"features": features}

@pytest.fixture
def sample_dataframe():
    df = pd.DataFrame({
        "Time": np.arange(100),
        "Amount": np.random.rand(100),
        "Class": np.random.choice([0,1], size=100, p=[0.95,0.05])
    })
    return df
