import pandas as pd
from pathlib import Path

DATA_DIR = Path(".")
PARQUET_FILE = "p2p_data.parquet"

all_dfs = []

for file in DATA_DIR.glob("*.xlsx"):
    try:
        df = pd.read_excel(file)
        df.columns = [c.strip() for c in df.columns]
        for col in df.columns:
            if "date" in col.lower():
                df[col] = pd.to_datetime(df[col], errors="coerce")
        df["entity"] = file.stem.upper()
        all_dfs.append(df)
        print(f"Loaded {file.name} with {len(df)} rows")
    except Exception as e:
        print(f"Skipping {file.name}: {e}")

if not all_dfs:
    raise ValueError("No Excel files loaded")

final_df = pd.concat(all_dfs, ignore_index=True)
final_df.to_parquet(PARQUET_FILE, index=False)

print("Parquet rebuilt successfully")
print("Max date:", final_df.select_dtypes(include=["datetime"]).max())