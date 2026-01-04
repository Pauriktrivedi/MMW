import pandas as pd
from pathlib import Path
import hashlib

FILES = {
    "MMW.xlsx": "MMW",
    "MMPL.xlsx": "MMPL",
    "MLPL.xlsx": "MLPL",
    "MEPL.xlsx": "MEPL",
}

PARQUET_FILE = "p2p_data.parquet"
HASH_FILE = ".data_hash"

def compute_hash():
    h = hashlib.md5()
    for f in FILES.keys():
        with open(f, "rb") as file:
            h.update(file.read())
    return h.hexdigest()

def rebuild_needed():
    if not Path(PARQUET_FILE).exists() or not Path(HASH_FILE).exists():
        return True
    return compute_hash() != Path(HASH_FILE).read_text()

def rebuild_parquet():
    dfs = []
    for file, entity in FILES.items():
        df = pd.read_excel(file)
        df["Posting Date"] = pd.to_datetime(df["Posting Date"], errors="coerce")
        df["Entity"] = entity
        dfs.append(df)

    final_df = (
        pd.concat(dfs, ignore_index=True)
          .drop_duplicates(subset=["PO Number", "Posting Date"])
    )

    final_df.to_parquet(PARQUET_FILE, index=False)
    Path(HASH_FILE).write_text(compute_hash())

if __name__ == "__main__":
    if rebuild_needed():
        print("Rebuilding parquet from Excel masters...")
        rebuild_parquet()
        print("Rebuild complete.")
    else:
        print("No data change detected. Using existing parquet.")