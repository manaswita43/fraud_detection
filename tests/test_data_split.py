import pandas as pd
from pathlib import Path

def test_data_split_ordering():
    v0 = pd.read_csv("data/v0/transactions_2022.csv")
    v1 = pd.read_csv("data/v1/transactions_2023.csv")

    assert len(v0) > 0
    assert len(v1) > 0

    # time ordering assumption
    assert v0["Time"].max() <= v1["Time"].min()
