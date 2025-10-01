# p2p_dashboard_optimized.py — clean, mapped, bug‑fixed
# Streamlit Procure‑to‑Pay Dashboard with Budget Code → Department/Subcategory mapping
# Works on Streamlit Cloud with repo files OR via file uploader fallback.
# Drop-in replacement for your current app.

from __future__ import annotations

import io
from datetime import date
from typing import List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ===============================
# Streamlit Page Configuration
# ===============================
st.set_page_config(
    page_title="Procure-to-Pay Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===============
# File Constants
# ===============
ENTITY_FILES_DEFAULT = [
    ("MEPL1.xlsx", "MEPL"),
    ("MLPL1.xlsx", "MLPL"),
    ("mmw1.xlsx",  "MMW"),
    ("mmpl1.xlsx", "MMPL"),
]
BUDGET_MAP_FILE_DEFAULT = "Final_Budget_Mapping_Completed_Verified.xlsx"

# ==================
# Helper Functions
# ==================
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Trim & unify column names; add friendly aliases; standardize key names used downstream."""
    if df is None or df.empty:
        return df
    # Clean headers
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.replace("\xa0", " ", regex=False)
        .str.replace(" +", " ", regex=True)
    )
    # Standardize some known variants
    rename_map = {
        "ReceivedQTY": "Received Qty",
        "Pending QTY": "Pending Qty",
        "PO Quantity": "PO Qty",
        "Po create Date": "PO Create Date",
        "PR Bussiness Unit": "PR Business Unit",
        "PO Business Unit": "PO Business Unit",
        "PR Department": "PR Dept",
        "PO Department": "PO Dept",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df

@st.cache_data(show_spinner=False)
def load_entity_frames(files_with_entity: List[tuple[str, str]]) -> pd.DataFrame:
    frames = []
    for fname, entity in files_with_entity:
        try:
            _df = pd.read_excel(fname, skiprows=1)
        except Exception:
            # Some sheets don’t have an extra header row
            _df = pd.read_excel(fname)
        _df = _normalize_columns(_df)
        _df["Entity"] = entity
        frames.append(_df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

@st.cache_data(show_spinner=False)
def load_budget_mapping(path: str) -> pd.DataFrame:
    try:
        bm = pd.read_excel(path)
    except Exception:
        bm = pd.DataFrame()
    bm = _normalize_columns(bm)
    # Expecting columns like: Budget Code, Department, Subcategory, (optionals)
    # Make a defensive rename set so downstream joins are stable
    candidate_map = {
        "Budget Code": "Budget Code",
        "Budget_Code": "Budget Code",
        "Code": "Budget Code",
        "Department": "Department",
        "Dept": "Department",
        "Subcategory": "Subcategory",
        "Sub Category": "Subcategory",
        "Sub-Category": "Subcategory",
    }
    bm = bm.rename(columns={k: v for k, v in candidate_map.items() if k in bm.columns})
    # Keep only distinct mapping rows
    if {"Budget Code", "Department", "Subcategory"}.issubset(bm.columns):
        bm = bm[["Budget Code", "Department", "Subcategory"]].drop_duplicates()
    return bm

def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def coerce_datetime(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

# =========================
# Data Loading UI / Fallback
# =========================
st.sidebar.subheader("📂 Data Sources")
use_uploads = st.sidebar.checkbox("Use file uploader (override repo files)", value=False)

if use_uploads:
    up_entities = st.sidebar.file_uploader(
        "Upload 1–4 entity Excel files (PR/PO lines)",
        type=["xlsx", "xls"],
        accept_multiple_files=True,
    )
    up_budget = st.sidebar.file_uploader(
        "Upload Budget Mapping Excel (Final_Budget_Mapping_Completed_Verified.xlsx)",
        type=["xlsx", "xls"],
        accept_multiple_files=False,
    )
    if up_entities:
        frames = []
        for f in up_entities:
            try:
                _df = pd.read_excel(f, skiprows=1)
            except Exception:
                f.seek(0)
                _df = pd.read_excel(f)
            _df = _normalize_columns(_df)
            # Try to infer Entity from filename; default Unknown
            inferred = getattr(f, "name", "").split(".")[0].upper()[:4]
            _df["Entity"] = inferred or "Unknown"
            frames.append(_df)
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        df = pd.DataFrame()
    bm = load_budget_mapping(up_budget) if up_budget else pd.DataFrame()
else:
    # Repo files path
    df = load_entity_frames(ENTITY_FILES_DEFAULT)
    bm = load_budget_mapping(BUDGET_MAP_FILE_DEFAULT)

if df.empty:
    st.error("No entity data loaded. Upload files or place MEPL1.xlsx, MLPL1.xlsx, mmw1.xlsx, mmpl1.xlsx in the repo.")
    st.stop()

# =================
# Core Preparation
# =================
# Normalize columns and dtypes
_df = _normalize_columns(df).copy()

# Coerce dates (keep as Timestamp for robust math)
_df = coerce_datetime(
    _df,
    [
        "PR Date Submitted",
        "PO Create Date",
        "PO Delivery Date",
        "PO Approved Date",
        "Last PO Date",
    ],
)

# Coerce numerics used in metrics & charts
_df = coerce_numeric(
    _df,
    [
        "PR Quantity", "Unit Rate", "PR Value",
        "PO Qty", "PO Unit Rate", "Net Amount",
        "Received Qty", "Pending Qty",
    ],
)

# ===========================
# Buyer Group & PO Creator Map
# ===========================
o_created_by_map = {
    "MMW2324030": "Dhruv",
    "MMW2324062": "Deepak",
    "MMW2425154": "Mukul",
    "MMW2223104": "Paurik",
    "MMW2021181": "Nayan",
    "MMW2223014": "Aatish",
    "MMW_EXT_002": "Deepakex",
    "MMW2425024": "Kamlesh",
    "MMW2021184": "Suresh",
    "N/A": "Dilip",
}

if "PO Orderer" in _df.columns:
    _df["PO Orderer"] = _df["PO Orderer"].fillna("N/A").astype(str).str.strip()
    _df["PO.Creator"] = _df["PO Orderer"].map(o_created_by_map).fillna(_df["PO Orderer"])
    _df["PO.Creator"] = _df["PO.Creator"].replace({"N/A": "Dilip"})
else:
    _df["PO.Creator"] = "Unknown"

# Buyer type rule
indirect_buyers = {"Aatish", "Deepak", "Deepakex", "Dhruv", "Dilip", "Mukul", "Nayan", "Paurik", "Kamlesh", "Suresh"}
_df["PO.BuyerType"] = _df["PO.Creator"].apply(lambda x: "Indirect" if x in indirect_buyers else "Direct")

# ==============================
# Budget Mapping (PR & PO sides)
# ==============================
if not bm.empty and "Budget Code" in bm.columns:
    # Attach mapping for PR & PO budget codes separately, suffixing columns
    if "PR Budget Code" in _df.columns:
        _df = _df.merge(bm.add_suffix(" (PR)"), left_on="PR Budget Code", right_on="Budget Code (PR)", how="left")
    if "PO Budget Code" in _df.columns:
        _df = _df.merge(bm.add_suffix(" (PO)"), left_on="PO Budget Code", right_on="Budget Code (PO)", how="left")

    # Prefer PO-side mapping for realized spend visuals; fallback to in-file departments
    _df["Dept.Final"] = (
        _df.get("Department (PO)")
        .fillna(_df.get("PO Dept"))
        .fillna(_df.get("Department (PR)"))
        .fillna(_df.get("PR Dept"))
    )
    _df["Subcat.Final"] = (
        _df.get("Subcategory (PO)")
        .fillna(_df.get("Subcategory (PR)"))
    )
else:
    # No mapping present – use existing department columns if any
    _df["Dept.Final"] = _df.get("PO Dept").fillna(_df.get("PR Dept"))
    _df["Subcat.Final"] = None

# ==================
# Sidebar Filters
# ==================
st.sidebar.header("🔍 Filters")

fy_options = {
    "All Years": (pd.Timestamp("2023-04-01"), pd.Timestamp("2026-03-31")),
    "2023": (pd.Timestamp("2023-04-01"), pd.Timestamp("2024-03-31")),
    "2024": (pd.Timestamp("2024-04-01"), pd.Timestamp("2025-03-31")),
    "2025": (pd.Timestamp("2025-04-01"), pd.Timestamp("2026-03-31")),
    "2026": (pd.Timestamp("2026-04-01"), pd.Timestamp("2027-03-31")),
}

selected_fy = st.sidebar.selectbox("Financial Year", options=list(fy_options.keys()), index=0)
pr_start, pr_end = fy_options[selected_fy]

# Choose date column for filtering priority: PR Submitted → PO Create
date_col_for_filter = "PR Date Submitted" if "PR Date Submitted" in _df.columns else ("PO Create Date" if "PO Create Date" in _df.columns else None)

# Build select options (safe)
get_opts = lambda s: sorted(pd.Series(s).dropna().astype(str).str.strip().unique().tolist())

buyer_types = get_opts(_df.get("PO.BuyerType", []))
entities = get_opts(_df.get("Entity", []))
creators = get_opts(_df.get("PO.Creator", []))
departments = get_opts(_df.get("Dept.Final", []))

buyer_sel = st.sidebar.multiselect("PO Buyer Type", buyer_types, default=buyer_types)
entity_sel = st.sidebar.multiselect("Entity", entities, default=entities)
creator_sel = st.sidebar.multiselect("PO Creator", creators, default=creators)
dept_sel = st.sidebar.multiselect("Department", departments, default=departments)

# Optional custom date range
if date_col_for_filter:
    min_dt = pd.to_datetime(_df[date_col_for_filter]).min()
    max_dt = pd.to_datetime(_df[date_col_for_filter]).max()
    default_range = [min_dt.date() if pd.notna(min_dt) else date.today(), max_dt.date() if pd.notna(max_dt) else date.today()]
    date_range = st.sidebar.date_input("Date Range", value=default_range)
else:
    date_range = None

# Apply filters
f = _df.copy()

# FY filter (on PR date if present)
if "PR Date Submitted" in f.columns:
    f = f[(f["PR Date Submitted"] >= pr_start) & (f["PR Date Submitted"] <= pr_end)]

# date_range explicit
if date_range and date_col_for_filter:
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        s_dt, e_dt = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        f = f[(f[date_col_for_filter] >= s_dt) & (f[date_col_for_filter] <= e_dt)]

# categorical filters
if buyer_sel:
    f = f[f["PO.BuyerType"].astype(str).str.strip().isin(buyer_sel)]
if entity_sel:
    f = f[f["Entity"].astype(str).str.strip().isin(entity_sel)]
if creator_sel:
    f = f[f["PO.Creator"].astype(str).str.strip().isin(creator_sel)]
if dept_sel:
    f = f[f["Dept.Final"].astype(str).str.strip().isin(dept_sel)]

st.sidebar.caption(f"Rows after filters: {len(f):,}")

# =============
# KPI Summary
# =============
st.title("📊 Procure‑to‑Pay Dashboard")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total PRs", int(f["PR Number"].nunique()) if "PR Number" in f.columns else 0)
c2.metric("Total POs", int(f["Purchase Doc"].nunique()) if "Purchase Doc" in f.columns else 0)
c3.metric("Line Items", len(f))
c4.metric("Entities", int(f["Entity"].nunique()) if "Entity" in f.columns else 0)
spend_cr = (f.get("Net Amount", pd.Series(dtype=float)).sum() / 1e7) if "Net Amount" in f.columns else 0
c5.metric("Spend (Cr ₹)", f"{spend_cr:,.2f}")

# =========================================
# SLA Gauge: PR → PO lead time (days ≤ 7)
# =========================================
st.subheader("🎯 SLA Compliance (PR → PO ≤ 7 days)")
lead_df = f.dropna(subset=["PO Create Date"]).copy() if "PO Create Date" in f.columns else pd.DataFrame()
if not lead_df.empty and "PR Date Submitted" in lead_df.columns:
    lead_df["Lead Time (Days)"] = (lead_df["PO Create Date"] - lead_df["PR Date Submitted"]).dt.days
    SLA_DAYS = 7
    avg_lead = float(lead_df["Lead Time (Days)"].mean()) if not lead_df.empty else 0.0
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_lead if pd.notna(avg_lead) else 0.0,
        number={"suffix": " days"},
        gauge={
            "axis": {"range": [0, max(14, (avg_lead or 0) * 1.2)]},
            "bar": {"color": "darkblue"},
            "steps": [
                {"range": [0, SLA_DAYS], "color": "lightgreen"},
                {"range": [SLA_DAYS, max(14, (avg_lead or 0) * 1.2)], "color": "lightcoral"},
            ],
            "threshold": {"line": {"color": "red", "width": 4}, "value": SLA_DAYS},
        },
        title={"text": "Average Lead Time"},
    ))
    st.plotly_chart(gauge, use_container_width=True)
    st.caption(f"Current Avg Lead Time: {avg_lead:.1f} days • Target ≤ {SLA_DAYS} days")
else:
    st.info("Not enough data to compute PR→PO lead time.")

# =======================================
# Monthly Spend: bars + cumulative line
# =======================================
st.subheader("📊 Monthly Total Spend (with Cumulative Line)")
_date_col = "PO Create Date" if "PO Create Date" in f.columns else ("PR Date Submitted" if "PR Date Submitted" in f.columns else None)
if _date_col and "Net Amount" in f.columns:
    tmp = f.dropna(subset=[_date_col]).copy()
    tmp["PO_Month"] = tmp[_date_col].dt.to_period("M").dt.to_timestamp()
    monthly = tmp.groupby("PO_Month", as_index=False)["Net Amount"].sum()
    monthly["Spend (Cr ₹)"] = monthly["Net Amount"]/1e7
    monthly = monthly.sort_values("PO_Month")
    monthly["Cumulative (Cr ₹)"] = monthly["Spend (Cr ₹)"].cumsum()
    monthly["Month_Str"] = monthly["PO_Month"].dt.strftime("%b-%Y")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(
        x=monthly["Month_Str"],
        y=monthly["Spend (Cr ₹)"],
        name="Monthly Spend (Cr ₹)",
        text=[f"{x:.2f}" for x in monthly["Spend (Cr ₹)"]],
        textposition="outside",
    )
    fig.add_scatter(
        x=monthly["Month_Str"], y=monthly["Cumulative (Cr ₹)"], name="Cumulative (Cr ₹)", mode="lines+markers"
    )
    fig.update_layout(margin=dict(t=60, b=110), xaxis_tickangle=-45)
    fig.update_yaxes(title_text="Monthly Spend (Cr ₹)", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative (Cr ₹)", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Need date & Net Amount columns to chart monthly spend.")

# ======================================================
# Department‑wise Spend (bar) + Drill‑down details table
# ======================================================
st.subheader("🏢 Department‑wise Spend (with details)")
if "Dept.Final" in f.columns and "Net Amount" in f.columns:
    dept_spend = (
        f.groupby("Dept.Final", dropna=False)["Net Amount"].sum().reset_index()
        .sort_values("Net Amount", ascending=False)
    )
    dept_spend["Spend (Cr ₹)"] = dept_spend["Net Amount"]/1e7
    dept_sel_single = st.selectbox("Select Department to view details", ["— All —"] + dept_spend["Dept.Final"].astype(str).tolist(), index=0)

    fig_dept = px.bar(
        dept_spend,
        x="Dept.Final",
        y="Spend (Cr ₹)",
        labels={"Dept.Final": "Department"},
        title="Spend by Department",
    )
    fig_dept.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_dept, use_container_width=True)

    detail_df = f.copy()
    if dept_sel_single != "— All —":
        detail_df = detail_df[detail_df["Dept.Final"].astype(str) == dept_sel_single]

    cols_detail = [
        c for c in [
            "Purchase Doc", "PO Vendor", "Product Name", "Item Description",
            "PO Qty", "PO Unit Rate", "Net Amount", "PO Budget Code", "Budget Code (PO)", "Subcategory (PO)",
        ] if c in detail_df.columns
    ]
    if cols_detail:
        detail_df = detail_df[cols_detail].copy()
        # Formatting
        if "Net Amount" in detail_df.columns:
            detail_df["Spend (₹)"] = detail_df["Net Amount"].round(2)
        st.dataframe(detail_df, use_container_width=True)
        st.download_button(
            "⬇️ Download Department Details (CSV)",
            data=detail_df.to_csv(index=False),
            file_name="department_details.csv",
            mime="text/csv",
        )
    else:
        st.info("No detail columns available to display.")
else:
    st.info("Department mapping not available. Ensure budget mapping file is present.")

# =============================================
# Budget Code Analysis (PO side; realized spend)
# =============================================
st.subheader("🧾 Budget Code Analysis (PO)")
if "PO Budget Code" in f.columns and "Net Amount" in f.columns:
    bc = (
        f.groupby("PO Budget Code", dropna=False)["Net Amount"].sum().reset_index()
        .sort_values("Net Amount", ascending=False)
    )
    bc["Spend (Cr ₹)"] = bc["Net Amount"]/1e7
    fig_bc = px.bar(bc.head(50), x="PO Budget Code", y="Spend (Cr ₹)", title="Top Budget Codes by Spend (PO)")
    fig_bc.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_bc, use_container_width=True)

    # Join mapping for context
    if not bm.empty and "Budget Code" in bm.columns:
        bc_map = bc.merge(bm, left_on="PO Budget Code", right_on="Budget Code", how="left")
        st.dataframe(bc_map.head(200), use_container_width=True)
        st.download_button(
            "⬇️ Download Budget Code Spend (with mapping)",
            data=bc_map.to_csv(index=False),
            file_name="budget_code_spend_mapped.csv",
            mime="text/csv",
        )
else:
    st.info("Budget code or Net Amount not available.")

# =======================================
# PR & PO Monthly Counts (activity view)
# =======================================
st.subheader("📅 Monthly PR & PO Counts")
if {"PR Date Submitted", "Purchase Doc"}.issubset(f.columns):
    tmp = f.copy()
    tmp["PR Month"] = tmp["PR Date Submitted"].dt.to_period("M").astype(str)
    # PO month based on PO Create Date if present
    if "PO Create Date" in tmp.columns:
        tmp["PO Month"] = tmp["PO Create Date"].dt.to_period("M").astype(str)
    else:
        tmp["PO Month"] = tmp["PR Month"]

    monthly_counts = (
        tmp.groupby("PR Month").agg(
            PR_Count=("PR Number", "count"),
            PO_Count=("Purchase Doc", "count"),
        ).reset_index().rename(columns={"PR Month": "Month"})
    )
    st.line_chart(monthly_counts.set_index("Month"), use_container_width=True)
else:
    st.info("Missing PR/PO columns for activity trend.")

# ==================================
# Open PRs (Approved/InReview only)
# ==================================
st.subheader("⚠️ Open PRs (Approved / InReview)")
if {"PR Status", "PR Date Submitted"}.issubset(f.columns):
    open_df = f[f["PR Status"].isin(["Approved", "InReview"])].copy()
    if not open_df.empty:
        open_df["Pending Age (Days)"] = (pd.Timestamp.today().normalize() - open_df["PR Date Submitted"]).dt.days
        open_summary = (
            open_df.groupby("PR Number", dropna=False)
            .agg(
                PR_Date=("PR Date Submitted", "first"),
                Pending_Age=("Pending Age (Days)", "first"),
                Category=("Procurement Category", "first"),
                Product=("Product Name", "first"),
                Value=("Net Amount", "sum"),
                PO_Budget_Code=("PO Budget Code", "first"),
                Status=("PR Status", "first"),
                Buyer_Group=("Buyer Group", "first"),
                Buyer_Type=("PO.BuyerType", "first"),
                Entity=("Entity", "first"),
                Creator=("PO.Creator", "first"),
                PO=("Purchase Doc", "first"),
            ).reset_index()
        )
        st.metric("Open PRs", int(open_summary["PR Number"].nunique()))
        st.dataframe(open_summary.sort_values("Pending_Age", ascending=False), use_container_width=True)
    else:
        st.success("No open PRs in current filters ✨")
else:
    st.info("PR status/date columns missing.")

# ==================================
# Vendor Delivery Performance
# ==================================
st.subheader("🚚 Vendor Delivery Performance")
if {"PO Vendor", "Purchase Doc", "PO Delivery Date"}.issubset(f.columns):
    perf = f.copy()
    perf["Pending Qty Filled"] = perf.get("Pending Qty", pd.Series(0)).fillna(0).astype(float)
    perf["Is_Fully_Delivered"] = perf["Pending Qty Filled"] == 0
    perf["Is_Late"] = (
        perf["PO Delivery Date"].notna()
        & (perf["PO Delivery Date"].dt.date < date.today())
        & (perf["Pending Qty Filled"] > 0)
    )
    vendor_perf = (
        perf.groupby("PO Vendor", dropna=False)
        .agg(
            Total_PO=("Purchase Doc", "nunique"),
            Fully_Delivered=("Is_Fully_Delivered", "sum"),
            Late=("Is_Late", "sum"),
            Spend=("Net Amount", "sum"),
        )
        .reset_index()
    )
    vendor_perf["Spend (Cr ₹)"] = vendor_perf["Spend"]/1e7
    vendor_perf["% Fully Delivered"] = (vendor_perf["Fully_Delivered"] / vendor_perf["Total_PO"] * 100).round(1)
    vendor_perf["% Late"] = (vendor_perf["Late"] / vendor_perf["Total_PO"] * 100).round(1)

    top = vendor_perf.sort_values(["Spend"], ascending=False).head(10)
    st.dataframe(top[["PO Vendor", "Total_PO", "Fully_Delivered", "Late", "% Fully Delivered", "% Late", "Spend (Cr ₹)"]], use_container_width=True)

    melted = top.melt(id_vars=["PO Vendor"], value_vars=["% Fully Delivered", "% Late"], var_name="Metric", value_name="Percentage")
    fig_v = px.bar(melted, x="PO Vendor", y="Percentage", color="Metric", barmode="group", title="% Fully Delivered vs % Late (Top 10 by Spend)")
    st.plotly_chart(fig_v, use_container_width=True)
else:
    st.info("Vendor performance needs PO Vendor, Purchase Doc, and PO Delivery Date columns.")

# =================
# PO Status Pie
# =================
st.subheader("📊 PO Status Breakdown")
if "PO Status" in f.columns:
    po_status = f["PO Status"].value_counts(dropna=False).reset_index()
    po_status.columns = ["PO Status", "Count"]
    c1, c2 = st.columns([2, 3])
    c1.dataframe(po_status, use_container_width=True)
    fig_status = px.pie(po_status, names="PO Status", values="Count", title="PO Status Distribution", hole=0.35)
    c2.plotly_chart(fig_status, use_container_width=True)
else:
    st.info("PO Status column missing.")

# =============================
# Monthly Unique PO Generation
# =============================
st.subheader("🗓️ Monthly Unique PO Generation")
if {"Purchase Doc", "PO Create Date"}.issubset(f.columns):
    pm = f.dropna(subset=["Purchase Doc", "PO Create Date"]).copy()
    pm["PO Month"] = pm["PO Create Date"].dt.to_period("M").astype(str)
    monthly_po = pm.groupby("PO Month")["Purchase Doc"].nunique().reset_index(name="Unique PO Count")
    fig_monthly_po = px.bar(monthly_po, x="PO Month", y="Unique PO Count", title="Monthly Unique PO Generation", text="Unique PO Count")
    fig_monthly_po.update_traces(textposition="outside")
    st.plotly_chart(fig_monthly_po, use_container_width=True)
else:
    st.info("Need Purchase Doc & PO Create Date for monthly unique POs.")

# ===== End =====
