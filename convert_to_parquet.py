import pandas as pd
from pathlib import Path

# ---------- CONFIG ----------
DATA_DIR = Path(__file__).resolve().parent
RAW_FILES = [("MEPL.xlsx", "MEPL"), ("MLPL.xlsx", "MLPL"), ("mmw.xlsx", "MMW"), ("mmpl.xlsx", "MMPL")]

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized, robust column normalizer."""
    if df is None or df.empty:
        return df
    cols = list(df.columns)
    new = []
    for c in cols:
        s = str(c).strip()
        s = s.replace(chr(160), " ")
        s = s.replace(chr(92), "_").replace('/', '_')
        s = '_'.join(s.split())
        s = s.lower()
        s = ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in s)
        s = '_'.join([p for p in s.split('_') if p != ''])
        new.append(s)
    df.columns = new
    return df

def _resolve_path(fn: str) -> Path:
    path = Path(fn)
    if path.exists():
        return path
    candidate = DATA_DIR / fn
    return candidate

def _read_excel(path: Path, entity: str) -> pd.DataFrame:
    # skiprows=1 was in original — preserve
    df = pd.read_excel(path, skiprows=1, engine='openpyxl')
    df['entity_source_file'] = entity
    return df

def _finalize_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    x = pd.concat(frames, ignore_index=True)
    x = normalize_columns(x)
    # parse common date columns once
    for c in ['pr_date_submitted', 'po_create_date', 'po_delivery_date', 'po_approved_date']:
        if c in x.columns:
            x[c] = pd.to_datetime(x[c], errors='coerce')
    return x

def convert_all_to_parquet(file_list=None):
    if file_list is None:
        file_list = RAW_FILES
    frames = []
    for fn, ent in file_list:
        path = _resolve_path(fn)
        if not path.exists():
            print(f"File not found: {path}")
            continue
        try:
            frames.append(_read_excel(path, ent))
        except Exception as exc:
            print(f"Failed to read {path.name}: {exc}")
    
    df = _finalize_frames(frames)
    
    # Ensure all object columns are converted to strings to avoid Parquet errors
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str)

    # Save as a single parquet file
    output_path = DATA_DIR / "p2p_data.parquet"
    df.to_parquet(output_path)
    print(f"Successfully converted all Excel files to {output_path}")

if __name__ == "__main__":
    convert_all_to_parquet()
