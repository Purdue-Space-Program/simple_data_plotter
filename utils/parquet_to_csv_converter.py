import pandas as pd

input_file = "11-22-2025-methane_fill_noah_parquet"
df = pd.read_parquet(f"data/{input_file}.parquet")
df.to_csv(f"{input_file}.csv", index=False)