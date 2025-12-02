# Optimized P2P Dashboard — Indirect
# - Index-based ultra-fast filtering
# - Safe handling of trend_date_col
# - Memoized heavy aggregations via session cache
# - Avoids expensive per-rerun string ops

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from plotly.subplots import make_subplots

DATA_DIR = Path(__file__).resolve().parent
RAW_FILES = [("MEPL.xlsx", "MEPL"), ("MLPL.xlsx", "MLPL"), ("mmw.xlsx", "MMW"), ("mmpl.xlsx", "MMPL")]
LOGO_PATH = DATA_DIR / "matter_logo.png"
INDIRECT_BUYERS = {'Aatish','Deepak','Deepakex','Dhruv','Dilip','Mukul','Nayan','Paurik','Kamlesh','Suresh','Priyam'}

st.set_page_config(page_title="P2P Dashboard — Indirect (Final)", layout="wide", initial_sidebar_state="expanded")

# ----------------- Helpers -----------------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    new_cols = []
    for c in cols:
        s = str(c).strip().replace(chr(160), ' ').replace(chr(92), '_').replace('/', '_')
        s = '_'.join(s.split()).lower()
        s = ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in s)
        s = '_'.join([p for p in s.split('_') if p])
        new_cols.append(s)
    df.columns = new_cols
    return df


def safe_col(df, candidates, default=None):
    for c in candidates:
        if c in df.columns:
            return c
    return default


def memoized_compute(namespace: str, signature: tuple, compute_fn):
    store = st.session_state.setdefault('_memo_cache', {})
    key = (namespace, signature)
    if key not in store:
        store[key] = compute_fn()
    return store[key]


def compute_buyer_type_vectorized(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=object)
    group_col = safe_col(df, ['buyer_group', 'buyer group', 'Buyer Group'])
    if not group_col:
        return pd.Series('Indirect', index=df.index)
    bg_raw = df[group_col].fillna('').astype(str).str.strip()
    code_series = pd.to_numeric(bg_raw.str.extract(r'(\d+)')[0], errors='coerce')
    bt = pd.Series('Direct', index=df.index)
    bt[bg_raw.eq('') | bg_raw.str.lower().isin(['not available','na','n/a'])] = 'Indirect'
    bt[(code_series >= 10) & (code_series <= 18)] = 'Indirect'
    bt[(code_series >= 1) & (code_series <= 9)] = 'Direct'
    bt[bg_raw.str.upper().isin({'ME_BG17','MLBG16'})] = 'Direct'
    return bt.fillna('Direct')


def compute_buyer_display_vectorized(df: pd.DataFrame, purchase_doc_col: str | None, requester_col: str | None) -> pd.Series:
    n = len(df)
    po_creator = df.get('po_creator', pd.Series(['']*n, index=df.index)).fillna('').astype(str).str.strip()
    requester = df.get(requester_col, pd.Series(['']*n, index=df.index)).fillna('').astype(str).str.strip() if requester_col else pd.Series(['']*n, index=df.index)
    has_po = pd.Series(False, index=df.index)
    if purchase_doc_col and purchase_doc_col in df.columns:
        has_po = df[purchase_doc_col].fillna('').astype(str).str.strip() != ''
    res = np.where(has_po & (po_creator != ''), po_creator, '')
    res = np.where((res == '') & (requester != ''), requester, res)
    res = np.where(res == '', 'PR only - Unassigned', res)
    return pd.Series(res, index=df.index)


def _resolve_path(fn: str) -> Path:
    p = Path(fn)
    if p.exists():
        return p
    return DATA_DIR / fn


def _read_excel(path: Path, entity: str) -> pd.DataFrame:
    return pd.read_excel(path, skiprows=1).assign(entity_source_file=entity)


def _finalize_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    x = pd.concat(frames, ignore_index=True)
    x = normalize_columns(x)
    for c in ['pr_date_submitted','po_create_date','po_delivery_date','po_approved_date']:
        if c in x.columns:
            x[c] = pd.to_datetime(x[c], errors='coerce')
    return x


@st.cache_data(show_spinner=False)
def load_all(file_list=None):
    if file_list is None:
        file_list = RAW_FILES
    frames = []
    for fn, ent in file_list:
        p = _resolve_path(fn)
        if not p.exists():
            continue
        try:
            frames.append(_read_excel(p, ent))
        except Exception as e:
            st.warning(f'Failed to read {p.name}: {e
