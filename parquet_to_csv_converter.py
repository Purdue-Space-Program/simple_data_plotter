import pandas as pd

input = "11-19-2025-hotfire-attempt"
df = pd.read_parquet(f"data/{input}.parquet")
df.to_csv(f"{input}.csv", index=False)