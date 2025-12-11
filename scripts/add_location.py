# scripts/add_location.py
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="data/v0/transactions_2022.csv")
parser.add_argument("--output", default="data/v0/transactions_2022_with_location.csv")
args = parser.parse_args()

df = pd.read_csv(args.input)
np.random.seed(42)
df["location"] = np.random.choice(["Location_A", "Location_B"], size=len(df), p=[0.5, 0.5])
df.to_csv(args.output, index=False)
print("Saved with location column:", args.output)
