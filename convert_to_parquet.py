import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PARQUET_PATH = BASE_DIR / "p2p_data.parquet"

SOURCE_FILES = [
    "MEPL.xlsx",
    "MLPL.xlsx",
    "MMPL.xlsx",
    "MMW.xlsx",
]

def update_parquet_if_needed(force: bool = False):
    if PARQUET_PATH.exists() and not force:
        return

    dfs = []
    for fname in SOURCE_FILES:
        fpath = BASE_DIR / fname
        if not fpath.exists():
            continue

        df = pd.read_excel(fpath)
        df.columns = [c.strip().lower() for c in df.columns]
        df["entity"] = fname.replace(".xlsx", "")
        dfs.append(df)

    if not dfs:
        raise RuntimeError("No source Excel files found")

    final_df = pd.concat(dfs, ignore_index=True)
    final_df.to_parquet(PARQUET_PATH, index=False)
