# complete_p2p_dashboard.py
import re
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="P2P Dashboard — Full", layout="wide", initial_sidebar_state="expanded")

# ---------------- Helpers ----------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = {}
    for c in df.columns:
        s = str(c).strip()
        s = s.replace("\xa0", " ").replace("\\", "_").replace("/", "_")
        s = "_".join(s.split())
        s = s.lower()
        s = ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in s)
        s = '_'.join([p for p in s.split('_') if p != ''])
        new_cols[c] = s
    return df.rename(columns=new_cols)

def find_col(df: pd.DataFrame, *variants):
    """Find first column name that matches any variant (case-insensitive substring)."""
    if df is None:
        return None
    cols = list(df.columns)
    low = {c.lower(): c for c in cols}
    for v in variants:
        if v is None: 
            continue
        key = str(v).lower()
        # exact or substring
        for lc, orig in low.items():
            if key == lc or key in lc or lc in key:
                return orig
    # fallback: try tokens
    for v in variants:
        if isinstance(v, (list, tuple)):
            for c in cols:
                lc = c.lower()
                if all(tok.lower() in lc for tok in v):
                    return c
    return None

@st.cache_data(show_spinner=False)
def load_default_files():
    files = [("MEPL.xlsx","MEPL"),("MLPL.xlsx","MLPL"),("mmw.xlsx","MMW"),("mmpl.xlsx","MMPL")]
    frames = []
    for fn, ent in files:
        try:
            df = pd.read_excel(fn, skiprows=1)
        except Exception:
            try:
                df = pd.read_excel(fn)
            except Exception:
                continue
        df
