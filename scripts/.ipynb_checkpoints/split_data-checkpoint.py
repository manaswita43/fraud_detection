import pandas as pd
from pathlib import Path

bucket_path = "gs://mlops-oppe2/data/transactions.csv"

out_v0 = Path("data/v0")
out_v1 = Path("data/v1")
out_v0.mkdir(parents=True, exist_ok=True)
out_v1.mkdir(parents=True, exist_ok=True)

# Read directly from GCS
df = pd.read_csv(bucket_path)   # gcsfs makes this work!

# sort by Time to ensure chronological order
df = df.sort_values("Time").reset_index(drop=True)
mid = len(df)//2

df_v0 = df.iloc[:mid].reset_index(drop=True)
df_v1 = df.iloc[mid:].reset_index(drop=True)

df_v0.to_csv(out_v0/"transactions_2022.csv", index=False)
df_v1.to_csv(out_v1/"transactions_2023.csv", index=False)

print(f"Saved {len(df_v0)} rows to {out_v0/'transactions_2022.csv'}")
print(f"Saved {len(df_v1)} rows to {out_v1/'transactions_2023.csv'}")
