from __future__ import annotations

import re
from io import BytesIO
from datetime import date
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Procure-to-Pay Dashboard", layout="wide", initial_sidebar_state="expanded")

# -------------------------
# Helpers
# -------------------------
def normalize_code(x: object) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip().upper()
    s = s.replace("&", "AND")
    s = re.sub(r"[\s\W_]+", "", s)
    return s if s else None

def pick_col(df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
    for col in df.columns:
        for pat in patterns:
            if re.search(pat, str(col), flags=re.I):
                return col
    return None

def to_datetime_safe(s):
    return pd.to_datetime(s, errors="coerce")

def read_any(path_or_buffer):
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

def download_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()

def download_df_button(df: pd.DataFrame, label: str, filename: str):
    st.download_button(label, download_csv_bytes(df), file_name=filename, mime="text/csv")

# -------------------------
# Load & combine files
# -------------------------
@st.cache_data(show_spinner=False)
def load_and_combine(use_local: bool = True, uploads: List = None) -> pd.DataFrame:
    frames = []
    if use_local:
        files = [
            ("MEPL1.xlsx", "MEPL"),
            ("MLPL1.xlsx", "MLPL"),
            ("mmw1.xlsx",  "MMW"),
            ("mmpl1.xlsx", "MMPL"),
        ]
        for path, tag in files:
            try:
                tmp = read_any(path)
                tmp["Entity"] = tag
                frames.append(tmp)
            except Exception:
                # ignore missing local files
                continue
    else:
        uploads = uploads or []
        for up in uploads:
            try:
                tmp = read_any(up)
                tag = re.sub(r"\W+", "", up.name.split(".")[0]).upper()[:6]
                tmp["Entity"] = tag
                frames.append(tmp)
            except Exception:
                continue
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    # Normalize column names
    combined.columns = [str(c).strip().replace("\xa0", " ") for c in combined.columns]
    return combined

# -------------------------
# Sidebar: data source
# -------------------------
st.sidebar.header("📥 Data Source")
use_local_files = st.sidebar.checkbox("Use local repo files (MEPL1/MLPL1/mmw1/mmpl1)", value=True, key="use_local_files")
uploads = []
if not use_local_files:
    uploads = st.sidebar.file_uploader("Upload files (xlsx/csv)", accept_multiple_files=True, type=["xlsx","xls","csv"], key="uploads")

df = load_and_combine(use_local=use_local_files, uploads=uploads)
if df.empty:
    st.error("No data loaded. Make sure files exist in repo or upload them in the sidebar.")
    st.stop()

# -------------------------
# Detect and canonicalize key columns
# -------------------------
def detect_amount_col(df: pd.DataFrame) -> Optional[str]:
    patterns = [r"\bnet\s*amount\b", r"\bnetamount\b", r"\bpr\s*value\b", r"\bpr\s*value\b", r"\bamount\b", r"\bvalue\b", r"\bnet\b"]
    for pat in patterns:
        col = pick_col(df, [pat])
        if col:
            return col
    # fallback: numeric column with most numeric entries
    best = None
    best_count = -1
    for c in df.columns:
        try:
            s = pd.to_numeric(df[c].astype(str).str.replace(r"[^\d\.\-]", "", regex=True), errors="coerce")
            cnt = int(s.notna().sum())
            if cnt > best_count:
                best_count = cnt
                best = c
        except Exception:
            continue
    return best

def detect_dept_col(df: pd.DataFrame) -> Optional[str]:
    patterns = [r"\bpo\s*department\b", r"\bpr\s*department\b", r"\bdepartment\b", r"\bdept\b"]
    for pat in patterns:
        col = pick_col(df, [pat])
        if col:
            return col
    # fallback heuristic
    for c in df.columns:
        vals = df[c].dropna().astype(str)
        if 2 < vals.nunique() < 500 and vals.str.len().median() < 60:
            sample = vals.sample(min(10, len(vals)))
            if any(re.search(r"[A-Za-z]", s) for s in sample):
                return c
    return None

amt_col = detect_amount_col(df)
dept_col = detect_dept_col(df)

# Canonical Net Amount
if amt_col:
    df["Net Amount"] = pd.to_numeric(df[amt_col], errors="coerce").fillna(0.0)
    st.sidebar.success(f"Using amount column: {amt_col} → 'Net Amount'")
else:
    df["Net Amount"] = 0.0
    st.sidebar.warning("No amount-like column detected; 'Net Amount' filled with zeros.")

# Canonical BudgetCode preference
for cand in ["PO Budget Code", "PR Budget Code", "Item Code", "PR BudgetCode", "PO BudgetCode"]:
    if cand in df.columns:
        df["BudgetCode"] = df[cand].map(normalize_code)
        break
if "BudgetCode" not in df.columns:
    fallback = next((c for c in df.columns if re.search(r"budget|code|item", c, flags=re.I)), None)
    df["BudgetCode"] = df[fallback].map(normalize_code) if fallback else None

# Canonical Department
if dept_col:
    df["Department"] = df[dept_col].astype(str).replace({"nan": pd.NA})
    st.sidebar.success(f"Using department column: {dept_col} → 'Department'")
else:
    if "PO Department" in df.columns:
        df["Department"] = df["PO Department"].astype(str).replace({"nan": pd.NA})
        st.sidebar.info("Using fallback 'PO Department' as Department.")
    elif "PR Department" in df.columns:
        df["Department"] = df["PR Department"].astype(str).replace({"nan": pd.NA})
        st.sidebar.info("Using fallback 'PR Department' as Department.")
    else:
        df["Department"] = pd.NA
        st.sidebar.warning("No department-like column detected. Department charts will be empty.")

# Ensure dates parsed
for date_col in ["PR Date Submitted", "Po create Date", "PO Delivery Date", "PO Approved Date", "Last PO Date"]:
    if date_col in df.columns:
        df[date_col] = to_datetime_safe(df[date_col])

# -------------------------
# Sidebar filters
# -------------------------
st.sidebar.header("🔍 Filters")
fy_options = {
    "All Years": (pd.to_datetime("2000-01-01"), pd.to_datetime("2100-01-01")),
    "2023": (pd.to_datetime("2023-04-01"), pd.to_datetime("2024-03-31")),
    "2024": (pd.to_datetime("2024-04-01"), pd.to_datetime("2025-03-31")),
    "2025": (pd.to_datetime("2025-04-01"), pd.to_datetime("2026-03-31")),
}
selected_fy = st.sidebar.selectbox("Financial Year", list(fy_options.keys()), index=0, key="fy_final")
pr_start, pr_end = fy_options[selected_fy]

def safe_unique(series):
    return sorted([str(x).strip() for x in series.dropna().unique() if str(x).strip() != ""])

buyer_options = safe_unique(df["Buyer.Type"]) if "Buyer.Type" in df.columns else []
entity_options = safe_unique(df["Entity"]) if "Entity" in df.columns else []
creator_options = safe_unique(df["PO.Creator"]) if "PO.Creator" in df.columns else []

buyer_filter = st.sidebar.multiselect("Buyer Type", options=buyer_options, default=buyer_options, key="ms_buyer_final")
entity_filter = st.sidebar.multiselect("Entity", options=entity_options, default=entity_options, key="ms_entity_final")
creator_filter = st.sidebar.multiselect("PO Creator", options=creator_options, default=creator_options, key="ms_creator_final")

date_col_for_filter = "PR Date Submitted" if "PR Date Submitted" in df.columns else ("Po create Date" if "Po create Date" in df.columns else None)
if date_col_for_filter:
    min_date = df[date_col_for_filter].min().date() if pd.notna(df[date_col_for_filter].min()) else date.today()
    max_date = df[date_col_for_filter].max().date() if pd.notna(df[date_col_for_filter].max()) else date.today()
    date_range = st.sidebar.date_input("Date Range", [min_date, max_date], key="date_range_final")
else:
    date_range = None

# -------------------------
# Apply filters
# -------------------------
filtered = df.copy()
if "PR Date Submitted" in filtered.columns:
    filtered = filtered[(filtered["PR Date Submitted"] >= pr_start) & (filtered["PR Date Submitted"] <= pr_end)]

if date_range and date_col_for_filter:
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        s_dt, e_dt = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        filtered = filtered[(filtered[date_col_for_filter] >= s_dt) & (filtered[date_col_for_filter] <= e_dt)]

if buyer_filter:
    filtered = filtered[filtered["Buyer.Type"].astype(str).str.strip().isin(buyer_filter)]
if entity_filter:
    filtered = filtered[filtered["Entity"].astype(str).str.strip().isin(entity_filter)]
if creator_filter:
    filtered = filtered[filtered["PO.Creator"].astype(str).str.strip().isin(creator_filter)]

st.sidebar.markdown("---")
st.sidebar.write("Rows after filters:", len(filtered))

# -------------------------
# Title & KPI row
# -------------------------
st.title("📊 Procure-to-Pay Dashboard")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total PRs", filtered.get("PR Number", pd.Series(dtype=object)).nunique())
c2.metric("Total POs", filtered.get("Purchase Doc", pd.Series(dtype=object)).nunique())
c3.metric("Line Items", len(filtered))
c4.metric("Entities", filtered.get("Entity", pd.Series(dtype=object)).nunique())
c5.metric("Spend (Cr ₹)", f"{filtered.get('Net Amount', pd.Series(dtype=float)).sum() / 1e7:,.2f}")

# -------------------------
# Department-wise Spend (requested)
# -------------------------
st.markdown("---")
st.subheader("🏢 Department-wise Spend (Drill-down)")

if filtered["Department"].notna().any() and filtered["Net Amount"].notna().any():
    dept_spend = (
        filtered.dropna(subset=["Department"])
        .groupby("Department", dropna=False)["Net Amount"]
        .sum()
        .reset_index()
        .sort_values("Net Amount", ascending=False)
    )
    dept_spend["Spend (Cr ₹)"] = dept_spend["Net Amount"] / 1e7

    fig_dept = px.bar(dept_spend, x="Department", y="Spend (Cr ₹)", title="Spend by Department (descending)", text="Spend (Cr ₹)")
    fig_dept.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_dept.update_layout(xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig_dept, use_container_width=True)

    dept_choices = dept_spend["Department"].dropna().tolist()
    selected_depts = st.multiselect("Select Department(s) to drill down", options=dept_choices, default=[], key="dept_multiselect")

    if selected_depts:
        show_cols = [c for c in ["Purchase Doc", "PO Vendor", "Product Name", "Net Amount", "Department", "BudgetCode"] if c in filtered.columns or c in ["Net Amount","Department","BudgetCode"]]
        detail = filtered[filtered["Department"].isin(selected_depts)][show_cols].copy()
        if "Net Amount" in detail.columns:
            detail["Net Amount (Cr)"] = detail["Net Amount"] / 1e7
            cols_order = [c for c in ["Purchase Doc", "PO Vendor", "Product Name", "BudgetCode", "Net Amount (Cr)"] if c in detail.columns]
            detail = detail[cols_order]
        st.markdown(f"#### 🔽 Detail rows: {len(detail)}")
        st.dataframe(detail, use_container_width=True)
        download_df_button(detail, "⬇️ Download selected department details (CSV)", "dept_details.csv")
    else:
        st.info("Select department(s) above to view PO-level details.")
else:
    st.info("No Department or Net Amount data available to render Department-wise spend.")

# -------------------------
# Monthly spend (kept but safe)
# -------------------------
st.markdown("---")
st.subheader("📊 Monthly Total Spend (with Cumulative)")

date_col = "Po create Date" if "Po create Date" in filtered.columns else ("PR Date Submitted" if "PR Date Submitted" in filtered.columns else None)

if date_col is None:
    st.info("No date column available ('Po create Date' or 'PR Date Submitted') to compute monthly spend.")
else:
    tmp = filtered.copy()
    tmp[date_col] = to_datetime_safe(tmp[date_col])
    tmp = tmp.dropna(subset=[date_col])
    if "Net Amount" not in tmp.columns or tmp["Net Amount"].sum() == 0:
        st.info("No 'Net Amount' data available to compute monthly spend.")
    else:
        tmp["PO_Month"] = tmp[date_col].dt.to_period("M").dt.to_timestamp()
        tmp["Month_Str"] = tmp["PO_Month"].dt.strftime("%b-%Y")
        monthly_total_spend = tmp.groupby(["PO_Month", "Month_Str"], as_index=False)["Net Amount"].sum().sort_values("PO_Month")
        monthly_total_spend["Spend (Cr ₹)"] = monthly_total_spend["Net Amount"] / 1e7
        monthly_total_spend["Cumulative Spend (Cr ₹)"] = monthly_total_spend["Spend (Cr ₹)"].cumsum()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=monthly_total_spend["Month_Str"], y=monthly_total_spend["Spend (Cr ₹)"], name="Monthly Spend (Cr ₹)"), secondary_y=False)
        fig.add_trace(go.Scatter(x=monthly_total_spend["Month_Str"], y=monthly_total_spend["Cumulative Spend (Cr ₹)"], mode="lines+markers", name="Cumulative Spend (Cr ₹)"), secondary_y=True)
        fig.update_layout(xaxis_tickangle=-45, height=450)
        fig.update_yaxes(title_text="Monthly Spend (Cr ₹)", secondary_y=False)
        fig.update_yaxes(title_text="Cumulative (Cr ₹)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

# -------------------------
# End
# -------------------------
st.markdown("---")
st.success("Dashboard loaded. Use the Department chart above to drill into PO-level rows.")
