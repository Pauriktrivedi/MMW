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
st.set_page_config(page_title="Procure-to-Pay Dashboard (With Budget Mapping)", layout="wide", initial_sidebar_state="expanded")

# -------------------------
# Helpers
# -------------------------
def normalize_code(x: object) -> Optional[str]:
    """Normalize budget code to uppercase no-spaces string for robust joins."""
    if pd.isna(x):
        return None
    s = str(x).strip().upper()
    # keep alphanum and dots/hyphens if you want, but remove stray whitespace/underscores
    s = s.replace("&", "AND")
    s = re.sub(r"[\s_]+", "", s)
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
# Load & combine company files
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
    combined.columns = [str(c).strip().replace("\xa0", " ") for c in combined.columns]
    return combined

# -------------------------
# Sidebar: data source + mapping upload
# -------------------------
st.sidebar.header("📥 Data Source & Mapping")

use_local_files = st.sidebar.checkbox("Use local repo files (MEPL1/MLPL1/mmw1/mmpl1)", value=True, key="use_local_files_map")
uploads = []
if not use_local_files:
    uploads = st.sidebar.file_uploader("Upload company files (multi)", type=["xlsx","xls","csv"], accept_multiple_files=True, key="uploads_map") or []

# mapping file uploader (explicit)
mapping_upload = st.sidebar.file_uploader("Upload Budget Mapping (Dept/Subcategory ↔ BudgetCode)", type=["xlsx","xls","csv"], key="mapping_upload")

# allow default mapping file name if present in repo
default_map_candidates = [
    "Final_Budget_Mapping_Completed_Verified.csv",
    "Final_Budget_Mapping_Completed_Verified.xlsx",
    "MainDept_SubCat_BudgetCodes_20250927_191809.xlsx"
]
use_default_mapping = any([st.sidebar.checkbox(f"Use default mapping file if '{name}' exists", value=(name.startswith("Final_")), key=name) for name in default_map_candidates])

df = load_and_combine(use_local=use_local_files, uploads=uploads)
if df.empty:
    st.error("No company data loaded. Upload files or enable local file usage.")
    st.stop()

# -------------------------
# Load mapping (cached)
# -------------------------
@st.cache_data(show_spinner=False)
def load_mapping_file(uploaded_file, default_candidates: List[str]) -> Optional[pd.DataFrame]:
    # Try uploaded file first
    if uploaded_file is not None:
        try:
            raw = read_any(uploaded_file)
        except Exception as e:
            st.warning(f"Couldn't read uploaded mapping file: {e}")
            return None
    else:
        # try defaults in repo
        raw = None
        for cand in default_candidates:
            try:
                raw = read_any(cand)
                if raw is not None:
                    break
            except Exception:
                raw = None
        if raw is None:
            return None

    # try to find Dept/Subcat and code columns
    dept_col = pick_col(raw, [r"\bdepartment\b", r"\bmain\s*dept\b", r"\bdept\b"]) or next((c for c in raw.columns if re.search(r"depart", str(c), flags=re.I)), None)
    subcat_col = pick_col(raw, [r"\bsub\s*cat", r"\bsubcategory\b", r"\bsub\s*category\b"]) or next((c for c in raw.columns if re.search(r"sub", str(c), flags=re.I)), None)

    # find columns that look like budget code columns (name contains 'budget' or 'code' or short-ish values)
    budget_cols = [c for c in raw.columns if re.search(r"budget.*code|^code$|^budget$|budget|code", str(c), flags=re.I)]
    if not budget_cols:
        # fallback: include columns with short-ish strings
        for c in raw.columns:
            if c in [dept_col, subcat_col]:
                continue
            vals = raw[c].dropna().astype(str)
            if len(vals) > 0 and (vals.str.len().median() <= 15 or vals.str.len().mean() <= 20):
                budget_cols.append(c)

    id_cols = [c for c in [dept_col, subcat_col] if c in raw.columns]
    if not id_cols:
        # can't tidy sensibly
        return None

    tidy = raw.melt(id_vars=id_cols, value_vars=budget_cols, var_name="_col", value_name="BudgetCodeRaw")
    tidy["BudgetCode"] = tidy["BudgetCodeRaw"].map(normalize_code)
    tidy = tidy.dropna(subset=["BudgetCode"]).drop_duplicates(subset=["BudgetCode"]).copy()

    rename_map = {}
    if dept_col and dept_col in tidy.columns:
        rename_map[dept_col] = "Department"
    if subcat_col and subcat_col in tidy.columns:
        rename_map[subcat_col] = "Subcategory"
    tidy = tidy.rename(columns=rename_map)
    for c in ["Department", "Subcategory"]:
        if c not in tidy.columns:
            tidy[c] = pd.NA
    tidy = tidy[["BudgetCode", "Department", "Subcategory"]]
    return tidy

mapping_df = load_mapping_file(mapping_upload, default_map_candidates)

if mapping_df is None or mapping_df.empty:
    st.sidebar.warning("No mapping loaded (upload mapping CSV/XLSX or ensure default mapping file exists). You can still view department spend based on existing department columns.")
else:
    st.sidebar.success(f"Mapping loaded — {len(mapping_df)} unique budget code mappings found.")

# -------------------------
# Normalize and canonicalize key columns in transactions
# -------------------------
# Detect amount column
def detect_amount_col(df: pd.DataFrame) -> Optional[str]:
    patterns = [r"\bnet\s*amount\b", r"\bnetamount\b", r"\bpr\s*value\b", r"\bamount\b", r"\bvalue\b"]
    for pat in patterns:
        col = pick_col(df, [pat])
        if col:
            return col
    # fallback: numeric-like column with many numeric values
    best = None
    best_cnt = -1
    for c in df.columns:
        try:
            s = pd.to_numeric(df[c].astype(str).str.replace(r"[^\d\.\-]", "", regex=True), errors="coerce")
            cnt = int(s.notna().sum())
            if cnt > best_cnt:
                best_cnt = cnt
                best = c
        except Exception:
            continue
    return best

amt_col = detect_amount_col(df)
if amt_col:
    df["Net Amount"] = pd.to_numeric(df[amt_col], errors="coerce").fillna(0.0)
    st.sidebar.info(f"Using amount column: {amt_col} → 'Net Amount'")
else:
    df["Net Amount"] = 0.0
    st.sidebar.warning("No amount-like column detected; 'Net Amount' set to zeros.")

# BudgetCode canonicalization: favor PO Budget Code -> PR Budget Code -> Item Code
budget_candidates = ["PO Budget Code", "PR Budget Code", "Item Code", "PR BudgetCode", "PO BudgetCode"]
found_budget_col = next((c for c in budget_candidates if c in df.columns), None)
if not found_budget_col:
    # fallback spare: any column with "budget" or "code" in its name
    found_budget_col = next((c for c in df.columns if re.search(r"budget|code|item", str(c), flags=re.I)), None)

if found_budget_col:
    df["BudgetCodeRaw"] = df[found_budget_col]
    df["BudgetCode"] = df["BudgetCodeRaw"].map(normalize_code)
    st.sidebar.info(f"Using budget column: {found_budget_col} → canonicalized to 'BudgetCode'")
else:
    df["BudgetCodeRaw"] = pd.NA
    df["BudgetCode"] = None
    st.sidebar.info("No budget-like column detected in transactions.")

# Normalize Department column if exists
dept_col_in_df = pick_col(df, [r"\bpo\s*department\b", r"\bpr\s*department\b", r"\bdepartment\b", r"\bdept\b"])
if dept_col_in_df:
    df["Department_orig"] = df[dept_col_in_df].astype(str).replace({"nan": pd.NA})
    st.sidebar.info(f"Found department column: {dept_col_in_df} → 'Department_orig'")
else:
    df["Department_orig"] = pd.NA

# -------------------------
# Merge mapping into transactions (if mapping exists)
# -------------------------
if mapping_df is not None and not mapping_df.empty:
    mapping_df = mapping_df.copy()
    mapping_df["BudgetCode"] = mapping_df["BudgetCode"].astype(str)
    # ensure types
    df["BudgetCode"] = df["BudgetCode"].astype(object)
    merged = df.merge(mapping_df, on="BudgetCode", how="left", suffixes=("", "_map"))
    # Use Department_orig if present; else use mapping Department
    merged["Department_mapped"] = merged["Department"].where(merged.get("Department").notna(), pd.NA) if "Department" in merged.columns else pd.NA
    # create final Department column preference: existing dept -> mapping -> NaN
    merged["Department_final"] = merged["Department_orig"].fillna(merged["Department"])
    # if still null, use mapping
    merged["Department_final"] = merged["Department_final"].fillna(merged["Department"].fillna(merged.get("Department_map", pd.NA)))
    # subcategory from mapping (if any)
    merged["Subcategory"] = merged.get("Subcategory", merged.get("Subcategory_map", pd.NA))
    # finally rename DataFrame used below
    full_df = merged.copy()
else:
    # mapping not available: fall back to using df and Department_orig as Department_final
    full_df = df.copy()
    full_df["Department_final"] = full_df["Department_orig"]
    full_df["Subcategory"] = pd.NA

# canonicalize date fields used later
for date_col in ["PR Date Submitted", "Po create Date", "PO Delivery Date", "PO Approved Date"]:
    if date_col in full_df.columns:
        full_df[date_col] = to_datetime_safe(full_df[date_col])

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
selected_fy = st.sidebar.selectbox("Financial Year", list(fy_options.keys()), index=0, key="fy_map")
pr_start, pr_end = fy_options[selected_fy]

def safe_unique(series):
    return sorted([str(x).strip() for x in series.dropna().unique() if str(x).strip() != ""])

buyer_options = safe_unique(full_df["Buyer.Type"]) if "Buyer.Type" in full_df.columns else []
entity_options = safe_unique(full_df["Entity"]) if "Entity" in full_df.columns else []
creator_options = safe_unique(full_df["PO.Creator"]) if "PO.Creator" in full_df.columns else []

buyer_filter = st.sidebar.multiselect("Buyer Type", options=buyer_options, default=buyer_options, key="ms_buyer_map")
entity_filter = st.sidebar.multiselect("Entity", options=entity_options, default=entity_options, key="ms_entity_map")
creator_filter = st.sidebar.multiselect("PO Creator", options=creator_options, default=creator_options, key="ms_creator_map")

date_col_for_filter = "PR Date Submitted" if "PR Date Submitted" in full_df.columns else ("Po create Date" if "Po create Date" in full_df.columns else None)
if date_col_for_filter:
    min_date = full_df[date_col_for_filter].min().date() if pd.notna(full_df[date_col_for_filter].min()) else date.today()
    max_date = full_df[date_col_for_filter].max().date() if pd.notna(full_df[date_col_for_filter].max()) else date.today()
    date_range = st.sidebar.date_input("Date Range", [min_date, max_date], key="date_range_map")
else:
    date_range = None

# -------------------------
# Apply filters to full_df
# -------------------------
filtered = full_df.copy()
# FY filter
if "PR Date Submitted" in filtered.columns:
    filtered = filtered[(filtered["PR Date Submitted"] >= pr_start) & (filtered["PR Date Submitted"] <= pr_end)]

# date range
if date_range and date_col_for_filter:
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        s_dt, e_dt = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        filtered = filtered[(filtered[date_col_for_filter] >= s_dt) & (filtered[date_col_for_filter] <= e_dt)]

# string filters
if buyer_filter:
    filtered = filtered[filtered["Buyer.Type"].astype(str).str.strip().isin(buyer_filter)]
if entity_filter:
    filtered = filtered[filtered["Entity"].astype(str).str.strip().isin(entity_filter)]
if creator_filter:
    filtered = filtered[filtered["PO.Creator"].astype(str).str.strip().isin(creator_filter)]

st.sidebar.markdown("---")
st.sidebar.write("Rows after filters:", len(filtered))

# -------------------------
# KPIs
# -------------------------
st.title("📊 Procure-to-Pay Dashboard — with BudgetCode → Dept/Subcategory Mapping")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total PRs", filtered.get("PR Number", pd.Series(dtype=object)).nunique())
c2.metric("Total POs", filtered.get("Purchase Doc", pd.Series(dtype=object)).nunique())
c3.metric("Line Items", len(filtered))
c4.metric("Entities", filtered.get("Entity", pd.Series(dtype=object)).nunique())
c5.metric("Spend (Cr ₹)", f"{filtered.get('Net Amount', pd.Series(dtype=float)).sum() / 1e7:,.2f}")

# -------------------------
# Department-wise spend (uses Department_final)
# -------------------------
st.markdown("---")
st.subheader("🏢 Department-wise Spend (using mapping + original)")

if filtered["Department_final"].notna().any() and filtered["Net Amount"].notna().any() and filtered["Net Amount"].sum() != 0:
    dept_spend = (
        filtered.dropna(subset=["Department_final"])
        .groupby("Department_final", dropna=False)["Net Amount"]
        .sum()
        .reset_index()
        .sort_values("Net Amount", ascending=False)
    )
    dept_spend = dept_spend.rename(columns={"Department_final": "Department"})
    dept_spend["Spend (Cr ₹)"] = dept_spend["Net Amount"] / 1e7

    fig_dept = px.bar(dept_spend, x="Department", y="Spend (Cr ₹)", text="Spend (Cr ₹)", title="Spend by Department (descending)")
    fig_dept.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_dept.update_layout(xaxis_tickangle=-45, height=520)
    st.plotly_chart(fig_dept, use_container_width=True)

    dept_choices = dept_spend["Department"].dropna().tolist()
    selected_depts = st.multiselect("Select Department(s) to drill down", options=dept_choices, default=[], key="dept_map_multiselect")

    if selected_depts:
        show_cols = [c for c in ["Purchase Doc", "PO Vendor", "Product Name", "BudgetCode", "Department_final", "Subcategory", "Net Amount"] if c in filtered.columns or c in ["BudgetCode", "Department_final", "Subcategory", "Net Amount"]]
        detail = filtered[filtered["Department_final"].isin(selected_depts)][show_cols].copy()
        # normalize column names for presentation
        if "Department_final" in detail.columns:
            detail = detail.rename(columns={"Department_final": "Department"})
        if "Net Amount" in detail.columns:
            detail["Net Amount (Cr)"] = detail["Net Amount"] / 1e7
            # present select ordering
            cols_order = [c for c in ["Purchase Doc", "PO Vendor", "Product Name", "BudgetCode", "Department", "Subcategory", "Net Amount (Cr)"] if c in detail.columns]
            detail = detail[cols_order]
        st.markdown(f"#### 🔽 PO-level rows: {len(detail)}")
        st.dataframe(detail, use_container_width=True)
        download_df_button(detail, "⬇️ Download selected department details (CSV)", "dept_details_mapped.csv")
    else:
        st.info("Select department(s) above to view PO-level details.")
else:
    st.info("No Department or Net Amount data available to render Department-wise spend. If department is empty, upload a mapping file or ensure department columns exist in the source files.")

# -------------------------
# Unmapped budget codes summary
# -------------------------
st.markdown("---")
st.subheader("🧾 Budget Codes: mapping coverage")

if "BudgetCode" in filtered.columns:
    all_codes = filtered["BudgetCode"].dropna().astype(str).str.strip().unique()
    mapped_codes = mapping_df["BudgetCode"].dropna().astype(str).unique() if mapping_df is not None else []
    unmapped_codes = sorted([c for c in all_codes if c not in mapped_codes])
    st.write(f"Total distinct budget codes in filtered data: {len(all_codes)}")
    st.write(f"Mapped codes available in mapping: {len(mapped_codes)}" if mapping_df is not None else "No mapping loaded")
    st.write(f"Unmapped budget codes (count: {len(unmapped_codes)}):")
    if unmapped_codes:
        # show up to 500 codes
        st.dataframe(pd.DataFrame({"UnmappedBudgetCode": unmapped_codes}), use_container_width=True)
        download_df_button(pd.DataFrame({"UnmappedBudgetCode": unmapped_codes}), "⬇️ Download Unmapped Budget Codes (CSV)", "unmapped_budget_codes.csv")
    else:
        st.success("All budget codes in filtered data are mapped (or no budget codes present).")
else:
    st.info("No BudgetCode column present to check mapping coverage.")

# -------------------------
# Keep monthly spend & other charts (safe) - optional, minimal
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
        fig.update_layout(xaxis_tickangle=-45, height=420)
        fig.update_yaxes(title_text="Monthly Spend (Cr ₹)", secondary_y=False)
        fig.update_yaxes(title_text="Cumulative (Cr ₹)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

# -------------------------
# End
# -------------------------
st.markdown("---")
st.success("Done — budget mapping merged. Use the Department chart above to drill into PO-level rows. If mapping looks off, upload the mapping file in the sidebar (CSV/XLSX).")
