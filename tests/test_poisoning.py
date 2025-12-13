import pandas as pd

def test_poisoning_percentage():
    clean = pd.read_csv("data/v0/transactions_2022.csv")
    poisoned = pd.read_csv("data/v0/poisoned_8_percent.csv")

    clean_zeros = clean[clean["Class"] == 0]
    poisoned_ones = poisoned.loc[clean_zeros.index, "Class"]

    flipped = (poisoned_ones == 1).sum()
    expected = int(0.08 * len(clean_zeros))

    # allow small randomness tolerance
    assert abs(flipped - expected) / expected < 0.1
