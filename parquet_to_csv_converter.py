import pandas as pd

input = "reduced_11-9-Hotfire-Attempts"
df = pd.read_parquet(f"data/{input}.parquet")
df.to_csv("reduced_11-9-Hotfire-Attempts.csv", index=False)