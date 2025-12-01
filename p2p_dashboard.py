# paste from line 1 into p2p_dashboard_indirect_final.py
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
INDIRECT_BUYERS = {
    'Aatish', 'Deepak', 'Deepakex', 'Dhruv', 'Dilip',
    'Mukul', 'Nayan', 'Paurik', 'Kamlesh', 'Suresh', 'Priyam'
}

# Run with: streamlit run p2p_dashboard_indirect_final.py
st.set_page_config(page_title="P2P Dashboard — Indirect (Final)", layout="wide", initial_sidebar_state="expanded")

# ----------------- Helpers -----------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    new = {}
    for c in df.columns:
        s = str(c).strip()
        s = s.replace(chr(160), " ")
        s = s.replace(chr(92), "_").replace('/', '_')
        s = "_".join(s.split())
        s = s.lower()
        s = ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in s)
        s = '_'.join([p for p in s.split('_') if p != ''])
        new[c] = s
    return df.rename(columns=new)

def safe_col(df, candidates, default=None):
    for c in candidates:
        if c in df.columns:
            return c
    return default

def memoized_compute(namespace: str, signature: tuple, compute_fn):
    """Simple session-state memoization keyed by the active filter tuple."""
    store = st.session_state.setdefault('_memo_cache', {})
    key = (namespace, signature)
    if key not in store:
        store[key] = compute_fn()
    return store[key]

def compute_buyer_type_vectorized(df: pd.DataFrame) -> pd.Series:
    """Classify PRs into Direct/Indirect using Buyer Group + numeric code."""
    if df.empty:
        return pd.Series(dtype=object)

    group_col = safe_col(df, ['buyer_group', 'Buyer Group', 'buyer group'])
    if not group_col:
        st.warning("⚠️ 'Buyer Group' column not found. Defaulting Buyer
