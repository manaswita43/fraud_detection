# scripts/poison_data.py
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="data/v0/transactions_2022.csv")
parser.add_argument("--out", default="data/v0")
parser.add_argument("--percent", type=float, required=True)
args = parser.parse_args()

df = pd.read_csv(args.input)
df = df.copy()
nonfraud_idx = df[df['Class']==0].index
n_flip = int(len(nonfraud_idx) * args.percent / 100.0)
flip_idx = np.random.choice(nonfraud_idx, size=n_flip, replace=False)
df.loc[flip_idx, "Class"] = 1

outp = Path(args.out)/f"poisoned_{int(args.percent)}_percent.csv"
df.to_csv(outp, index=False)
print(f"Wrote {outp} with {n_flip} flipped labels.")
