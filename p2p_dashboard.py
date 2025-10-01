
# p2p_dashboard.py — FINAL VERSION
# Procure-to-Pay Dashboard with Budget-Code → Department/Subcategory mapping
# Includes Department-wise & Subcategory-wise spend charts with drill-down lists & CSV downloads

from __future__ import annotations
import re
from io import BytesIO
from datetime import date
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Procure-to-Pay Dashboard (Final)", layout="wide", initial_sidebar_state="expanded")

# -------------------------
# Helpers
# -------------------------
def normalize_code(x):
    if pd.isna(x):
        return None
    s = str(x).strip().upper()
    s = s.replace("&", "AND")
    s = re.sub(r"[\s\W_]+", "", s)
    return s if s else None

def pick_col(df: pd.DataFrame, patterns):
    for col in df.columns:
        name = str(col).strip().lower().replace("\xa0", " ")
        for pat in patterns:
            if re.search(pat, name, flags=re.I):
                return col
    return None

def to_datetime_safe(s):
    return pd.to_datetime(s, errors="coerce")

def download_csv_bytes(df):
    buf = BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()

def download_df_button(df, label, filename):
    st.download_button(label, download_csv_bytes(df), file_name=filename, mime="text/csv")

# -------------------------
# Load & combine data
# -------------------------
@st.cache_data(show_spinner=False)
def read_excel_first_sheet(path_or_buffer):
    if hasattr(path_or_buffer, "read"):
        try:
            return pd.read_excel(path_or_buffer)
        except Exception:
            path_or_buffer.seek(0)
            return pd.read_csv(path_or_buffer)
    else:
        try:
            return pd.read_excel(path_or_buffer)
        except Exception:
            return pd.read_csv(path_or_buffer)

@st.cache_data(show_spinner=False)
def load_and_combine(use_local: bool, uploads):
    frames = []
    if use_local:
        local_files = [
            ("MEPL1.xlsx", "MEPL"),
            ("MLPL1.xlsx", "MLPL"),
            ("mmw1.xlsx",  "MMW"),
            ("mmpl1.xlsx", "MMPL"),
        ]
        for p, tag in local_files:
            try:
                df = read_excel_first_sheet(p)
                df["Entity"] = tag
                frames.append(df)
            except Exception:
                continue
    else:
        if uploads:
            for up in uploads:
                try:
                    df = read_excel_first_sheet(up)
                    tag = re.sub(r"\W+", "", up.name.split(".")[0]).upper()[:6]
                    df["Entity"] = tag
                    frames.append(df)
                except Exception:
                    continue
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined.columns = (
        combined.columns.astype(str)
        .str.strip()
        .str.replace("\xa0", " ", regex=False)
        .str.replace(" +", " ", regex=True)
    )
    return combined

st.sidebar.header("📥 Data Source")
use_local = st.sidebar.checkbox("Use local repo files (MEPL1/MLPL1/mmw1/mmpl1)", value=True)
uploads = []
if not use_local:
    uploads = st.sidebar.file_uploader("Upload company files (multi-select)", type=["xlsx","xls","csv"], accept_multiple_files=True)

df = load_and_combine(use_local, uploads)
if df.empty:
    st.error("No data loaded.")
    st.stop()

# -------------------------
# Budget mapping
# -------------------------
st.sidebar.header("🧭 Budget Mapping")
map_file = st.sidebar.file_uploader("Upload mapping workbook", type=["xlsx","xls","csv"])
use_default_map = st.sidebar.checkbox("Use default mapping file", value=True)

@st.cache_data(show_spinner=False)
def load_mapping_df(map_file, use_default: bool):
    try:
        if map_file is not None:
            raw = read_excel_first_sheet(map_file)
        elif use_default:
            raw = read_excel_first_sheet("MainDept_SubCat_BudgetCodes_20250927_191809.xlsx")
        else:
            return None
    except Exception:
        return None
    dept_col = pick_col(raw, [r"department"])
    subcat_col = pick_col(raw, [r"sub"])
    code_cols = [c for c in raw.columns if "budget" in str(c).lower() or "code" in str(c).lower()]
    id_cols = [c for c in [dept_col, subcat_col] if c]
    tidy = raw.melt(id_vars=id_cols, value_vars=code_cols, value_name="BudgetCodeRaw")
    tidy["BudgetCode"] = tidy["BudgetCodeRaw"].map(normalize_code)
    tidy = tidy.dropna(subset=["BudgetCode"]).drop_duplicates("BudgetCode")
    tidy = tidy.rename(columns={dept_col:"Department", subcat_col:"Subcategory"})
    return tidy[["BudgetCode","Department","Subcategory"]]

mapping = load_mapping_df(map_file, use_default_map)
if mapping is not None:
    df["BudgetCode"] = df[pick_col(df,["budget","code"])].map(normalize_code)
    df = df.merge(mapping, on="BudgetCode", how="left")

# Ensure Net Amount numeric
if "Net Amount" in df.columns:
    df["Net Amount"] = pd.to_numeric(df["Net Amount"], errors="coerce").fillna(0.0)

# -------------------------
# Department-wise Spend
# -------------------------
st.subheader("🏢 Department-wise Spend")
if "Department" in df.columns and "Net Amount" in df.columns:
    dept_spend = df.groupby("Department", dropna=False)["Net Amount"].sum().reset_index().sort_values("Net Amount", ascending=False)
    dept_spend["Spend (Cr ₹)"] = dept_spend["Net Amount"] / 1e7
    fig_dept = px.bar(dept_spend, x="Department", y="Spend (Cr ₹)", text="Spend (Cr ₹)", title="Spend by Department")
    fig_dept.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    st.plotly_chart(fig_dept, use_container_width=True)
    selected = st.multiselect("Select Department(s)", dept_spend["Department"].dropna().tolist())
    if selected:
        cols = [c for c in ["Purchase Doc","PO Vendor","Product Name","Net Amount","Department","Subcategory"] if c in df.columns]
        detail = df[df["Department"].isin(selected)][cols]
        detail["Net Amount (Cr)"] = detail["Net Amount"] / 1e7
        st.dataframe(detail, use_container_width=True)
        download_df_button(detail, "⬇️ Download Department details", "dept_details.csv")

# -------------------------
# Subcategory-wise Spend
# -------------------------
st.subheader("📂 Subcategory-wise Spend")
if "Subcategory" in df.columns and "Net Amount" in df.columns:
    sub_spend = df.groupby("Subcategory", dropna=False)["Net Amount"].sum().reset_index().sort_values("Net Amount", ascending=False)
    sub_spend["Spend (Cr ₹)"] = sub_spend["Net Amount"] / 1e7
    fig_sub = px.bar(sub_spend, x="Subcategory", y="Spend (Cr ₹)", text="Spend (Cr ₹)", title="Spend by Subcategory")
    fig_sub.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    st.plotly_chart(fig_sub, use_container_width=True)
    selected_sub = st.multiselect("Select Subcategory(ies)", sub_spend["Subcategory"].dropna().tolist())
    if selected_sub:
        cols = [c for c in ["Purchase Doc","PO Vendor","Product Name","Net Amount","Department","Subcategory"] if c in df.columns]
        detail2 = df[df["Subcategory"].isin(selected_sub)][cols]
        detail2["Net Amount (Cr)"] = detail2["Net Amount"] / 1e7
        st.dataframe(detail2, use_container_width=True)
        download_df_button(detail2, "⬇️ Download Subcategory details", "subcat_details.csv")

# -------------------------
# Unmapped Codes
# -------------------------
if mapping is not None:
    unmapped = df[df["BudgetCode"].notna() & df["Department"].isna()]["BudgetCode"].drop_duplicates()
    st.subheader("⚠️ Unmapped Budget Codes")
    st.write(unmapped)
    download_df_button(unmapped.to_frame(), "⬇️ Download Unmapped Codes", "unmapped_codes.csv")
