# p2p_dashboard_final.py — full modules kept + optimizations + new insights
# Ready for Streamlit Cloud. Uses repo files by default, with optional upload override.

from __future__ import annotations

from datetime import date
from typing import List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ---------------------------------
# Streamlit Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="Procure-to-Pay Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------
# Constants
# ---------------------------------
ENTITY_FILES_DEFAULT = [
    ("MEPL1.xlsx", "MEPL"),
    ("MLPL1.xlsx", "MLPL"),
    ("mmw1.xlsx",  "MMW"),
    ("mmpl1.xlsx", "MMPL"),
]
BUDGET_MAP_FILE_DEFAULT = "Final_Budget_Mapping_Completed_Verified.xlsx"

# ---------------------------------
# Helpers
# ---------------------------------
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace("\xa0", " ", regex=False)
        .str.replace(" +", " ", regex=True)
    )
    ren = {
        "ReceivedQTY": "Received Qty",
        "Pending QTY": "Pending Qty",
        "PO Quantity": "PO Qty",
        "Po create Date": "PO Create Date",
        "PR Bussiness Unit": "PR Business Unit",
        "PO Business Unit": "PO Business Unit",
        "PR Department": "PR Dept",
        "PO Department": "PO Dept",
    }
    return df.rename(columns={k: v for k, v in ren.items() if k in df.columns})

@st.cache_data(show_spinner=False)
def load_entity_frames(files_with_entity: List[tuple[str, str]]) -> pd.DataFrame:
    frames = []
    for fname, entity in files_with_entity:
        try:
            _df = pd.read_excel(fname, skiprows=1)
        except Exception:
            _df = pd.read_excel(fname)
        _df = _normalize_columns(_df)
        _df["Entity"] = entity
        frames.append(_df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_budget_mapping(path: str) -> pd.DataFrame:
    try:
        bm = pd.read_excel(path)
    except Exception:
        return pd.DataFrame()
    bm = _normalize_columns(bm)
    cand = {
        "Budget Code": "Budget Code",
        "Budget_Code": "Budget Code",
        "Code": "Budget Code",
        "Department": "Department",
        "Dept": "Department",
        "Subcategory": "Subcategory",
        "Sub Category": "Subcategory",
        "Sub-Category": "Subcategory",
    }
    bm = bm.rename(columns={k: v for k, v in cand.items() if k in bm.columns})
    keep = [c for c in ["Budget Code", "Department", "Subcategory"] if c in bm.columns]
    return bm[keep].drop_duplicates() if keep else bm

# dtype coercers

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

# ---------------------------------
# Data Source UI
# ---------------------------------
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
            inferred = getattr(f, "name", "").split(".")[0].upper()[:4]
            _df["Entity"] = inferred or "Unknown"
            frames.append(_df)
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.DataFrame()
    bm = load_budget_mapping(up_budget) if up_budget else pd.DataFrame()
else:
    df = load_entity_frames(ENTITY_FILES_DEFAULT)
    bm = load_budget_mapping(BUDGET_MAP_FILE_DEFAULT)

if df.empty:
    st.error("No entity data loaded. Upload files or ensure repo files are present.")
    st.stop()

# ---------------------------------
# Core prep
# ---------------------------------
_df = _normalize_columns(df).copy()
_df = coerce_datetime(
    _df,
    ["PR Date Submitted", "PO Create Date", "PO Delivery Date", "PO Approved Date", "Last PO Date"],
)
_df = coerce_numeric(
    _df,
    ["PR Quantity", "Unit Rate", "PR Value", "PO Qty", "PO Unit Rate", "Net Amount", "Received Qty", "Pending Qty"],
)

# Buyer mapping
CREATOR_MAP = {
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
    _df["PO.Creator"] = _df["PO Orderer"].map(CREATOR_MAP).fillna(_df["PO Orderer"])
    _df["PO.Creator"] = _df["PO.Creator"].replace({"N/A": "Dilip"})
else:
    _df["PO.Creator"] = "Unknown"

INDIRECT = {"Aatish", "Deepak", "Deepakex", "Dhruv", "Dilip", "Mukul", "Nayan", "Paurik", "Kamlesh", "Suresh"}
_df["PO.BuyerType"] = _df["PO.Creator"].apply(lambda x: "Indirect" if x in INDIRECT else "Direct")

# Budget mapping merges (PR and PO)
# Use dictionary .map to prevent row multiplication and double counting.
if not bm.empty and "Budget Code" in bm.columns:
    bm_unique = bm.copy()
    bm_unique = bm_unique.dropna(subset=["Budget Code"]) if "Budget Code" in bm_unique.columns else bm_unique
    # one row per Budget Code (first occurrence wins)
    if "Budget Code" in bm_unique.columns:
        bm_unique = bm_unique.drop_duplicates(subset=["Budget Code"], keep="first")

    dept_map = bm_unique.set_index("Budget Code")["Department"].to_dict() if "Department" in bm_unique.columns else {}
    subc_map = bm_unique.set_index("Budget Code")["Subcategory"].to_dict() if "Subcategory" in bm_unique.columns else {}

    # Map (no merges) on PO and PR sides
    _df["Dept.from.POCode"] = _df.get("PO Budget Code").map(dept_map) if "PO Budget Code" in _df.columns else pd.Series(pd.NA, index=_df.index)
    _df["Subcat.from.POCode"] = _df.get("PO Budget Code").map(subc_map) if "PO Budget Code" in _df.columns else pd.Series(pd.NA, index=_df.index)
    _df["Dept.from.PRCode"] = _df.get("PR Budget Code").map(dept_map) if "PR Budget Code" in _df.columns else pd.Series(pd.NA, index=_df.index)
    _df["Subcat.from.PRCode"] = _df.get("PR Budget Code").map(subc_map) if "PR Budget Code" in _df.columns else pd.Series(pd.NA, index=_df.index)

    def _ser(col: str) -> pd.Series:
        return _df[col] if col in _df.columns else pd.Series(pd.NA, index=_df.index)

    _df["Dept.Final"] = (
        _ser("Dept.from.POCode")
        .combine_first(_ser("PO Dept"))
        .combine_first(_ser("Dept.from.PRCode"))
        .combine_first(_ser("PR Dept"))
    )

    _df["Subcat.Final"] = _ser("Subcat.from.POCode").combine_first(_ser("Subcat.from.PRCode"))
else:
    # No mapping file: fallback to in-file departments
    _df["Dept.Final"] = _df["PO Dept"] if "PO Dept" in _df.columns else (_df["PR Dept"] if "PR Dept" in _df.columns else pd.Series(pd.NA, index=_df.index))
    _df["Subcat.Final"] = pd.Series(pd.NA, index=_df.index)

# ---------------------------------
# Filters
# ---------------------------------
st.sidebar.header("🔍 Filters")
FY = {
    "All Years": (pd.Timestamp("2023-04-01"), pd.Timestamp("2026-03-31")),
    "2023": (pd.Timestamp("2023-04-01"), pd.Timestamp("2024-03-31")),
    "2024": (pd.Timestamp("2024-04-01"), pd.Timestamp("2025-03-31")),
    "2025": (pd.Timestamp("2025-04-01"), pd.Timestamp("2026-03-31")),
    "2026": (pd.Timestamp("2026-04-01"), pd.Timestamp("2027-03-31")),
}
fy_key = st.sidebar.selectbox("Financial Year", list(FY.keys()), index=0)
pr_start, pr_end = FY[fy_key]

sel = lambda s: sorted(pd.Series(s).dropna().astype(str).str.strip().unique().tolist())

buyer_opts = sel(_df.get("PO.BuyerType", []))
entity_opts = sel(_df.get("Entity", []))
creator_opts = sel(_df.get("PO.Creator", []))
dept_opts = sel(_df.get("Dept.Final", []))

buyer_sel = st.sidebar.multiselect("PO Buyer Type", buyer_opts, default=buyer_opts)
entity_sel = st.sidebar.multiselect("Entity", entity_opts, default=entity_opts)
creator_sel = st.sidebar.multiselect("PO Creator", creator_opts, default=creator_opts)
dept_sel = st.sidebar.multiselect("Department", dept_opts, default=dept_opts)

# choose date column for explicit range
# Date basis selector for consistent filtering — default to PR date to match your original app
_date_basis_opts = []
if "PR Date Submitted" in _df.columns:
    _date_basis_opts.append("PR Date Submitted")
if "PO Create Date" in _df.columns:
    _date_basis_opts.append("PO Create Date")
_date_col = st.sidebar.radio(
    "Filter by date",
    options=_date_basis_opts,
    index=0  # PR Date Submitted first to mirror p2p_dashboard
) if _date_basis_opts else None

if _date_col:
    col_non_null = _df[_date_col].dropna()
    min_dt = pd.to_datetime(col_non_null).min() if not col_non_null.empty else None
    max_dt = pd.to_datetime(col_non_null).max() if not col_non_null.empty else None
    default_range = [
        (min_dt.date() if pd.notna(min_dt) else date.today()),
        (max_dt.date() if pd.notna(max_dt) else date.today()),
    ]
    date_range = st.sidebar.date_input("Date Range", value=default_range)
else:
    date_range = None

# Apply filters
f = _df.copy()
if _date_col and _date_col in f.columns:
    f = f[(f[_date_col] >= pr_start) & (f[_date_col] <= pr_end)]
if date_range and _date_col:
    s_dt, e_dt = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    f = f[(f[_date_col] >= s_dt) & (f[_date_col] <= e_dt)]
if buyer_sel:
    f = f[f["PO.BuyerType"].astype(str).isin(buyer_sel)]
if entity_sel:
    f = f[f["Entity"].astype(str).isin(entity_sel)]
if creator_sel:
    f = f[f["PO.Creator"].astype(str).isin(creator_sel)]
if dept_sel:
    f = f[f["Dept.Final"].astype(str).isin(dept_sel)]

st.sidebar.caption(f"Rows after filters: {len(f):,}")

# ---------------------------------
# Title + KPIs
# ---------------------------------
st.title("📊 Procure‑to‑Pay Dashboard")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total PRs", int(f["PR Number"].nunique()) if "PR Number" in f.columns else 0)
c2.metric("Total POs", int(f["Purchase Doc"].nunique()) if "Purchase Doc" in f.columns else 0)
c3.metric("Line Items", len(f))
c4.metric("Entities", int(f["Entity"].nunique()) if "Entity" in f.columns else 0)
spend_cr = (f.get("Net Amount", pd.Series(dtype=float)).sum() / 1e7) if "Net Amount" in f.columns else 0
c5.metric("Spend (Cr ₹)", f"{spend_cr:,.2f}")

# ---------------------------------
# Keyword Search (kept)
# ---------------------------------
st.markdown("## 🔍 Keyword Search")
valid_cols = [c for c in ["PR Number", "Purchase Doc", "Product Name", "PO Vendor"] if c in f.columns]
if valid_cols:
    user_query = st.text_input("Search PR/PO/Vendor/Product")
    filt = f.copy()
    if user_query:
        q = str(user_query).strip().lower()
        mask = (
            filt[valid_cols].fillna("").astype(str).apply(lambda col: col.str.lower().str.contains(q))
        ).any(axis=1)
        results = filt[mask]
        st.write(f"Found {len(results):,} rows")
        st.dataframe(results, use_container_width=True)
        st.download_button("⬇️ Download Search Results (CSV)", results.to_csv(index=False), "search_results.csv", "text/csv")
else:
    st.info("Columns needed for search not found.")

# ---------------------------------
# SLA Gauge PR→PO (kept)
# ---------------------------------
st.subheader("🎯 SLA Compliance (PR → PO ≤ 7 days)")
lead_df = f.dropna(subset=["PO Create Date"]).copy() if "PO Create Date" in f.columns else pd.DataFrame()
if not lead_df.empty and "PR Date Submitted" in lead_df.columns:
    lead_df["Lead Time (Days)"] = (lead_df["PO Create Date"] - lead_df["PR Date Submitted"]).dt.days
    SLA_DAYS = 7
    avg_lead = float(lead_df["Lead Time (Days)"].mean()) if not lead_df.empty else 0.0
    fig_g = go.Figure(go.Indicator(
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
    st.plotly_chart(fig_g, use_container_width=True)
else:
    st.info("Not enough data to compute PR→PO lead time.")

# ---------------------------------
# Monthly Spend (bars + cumulative) (kept)
# ---------------------------------
st.subheader("📊 Monthly Total Spend (with Cumulative Line)")
_date_for_spend = "PO Create Date" if "PO Create Date" in f.columns else ("PR Date Submitted" if "PR Date Submitted" in f.columns else None)
if _date_for_spend and "Net Amount" in f.columns:
    tmp = f.dropna(subset=[_date_for_spend]).copy()
    tmp["PO_Month"] = tmp[_date_for_spend].dt.to_period("M").dt.to_timestamp()
    monthly = tmp.groupby("PO_Month", as_index=False)["Net Amount"].sum()
    monthly["Spend (Cr ₹)"] = monthly["Net Amount"]/1e7
    monthly = monthly.sort_values("PO_Month")
    monthly["Cumulative (Cr ₹)"] = monthly["Spend (Cr ₹)"].cumsum()
    monthly["Month_Str"] = monthly["PO_Month"].dt.strftime("%b-%Y")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=monthly["Month_Str"], y=monthly["Spend (Cr ₹)"], name="Monthly Spend (Cr ₹)", text=[f"{x:.2f}" for x in monthly["Spend (Cr ₹)"]], textposition="outside")
    fig.add_scatter(x=monthly["Month_Str"], y=monthly["Cumulative (Cr ₹)"], name="Cumulative (Cr ₹)", mode="lines+markers")
    fig.update_layout(margin=dict(t=60, b=110), xaxis_tickangle=-45)
    fig.update_yaxes(title_text="Monthly Spend (Cr ₹)", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative (Cr ₹)", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Need date & Net Amount columns to chart monthly spend.")

# ---------------------------------
# Monthly Spend Trend by Entity (kept)
# ---------------------------------
st.subheader("💹 Monthly Spend Trend by Entity")
if {"PO Create Date", "Entity", "Net Amount"}.issubset(f.columns):
    s_df = f.copy()
    s_df["PO Month"] = s_df["PO Create Date"].dt.to_period("M").dt.to_timestamp()
    ms = s_df.dropna(subset=["PO Month"]).groupby(["PO Month", "Entity"], as_index=False)["Net Amount"].sum()
    ms["Spend (Cr ₹)"] = ms["Net Amount"]/1e7
    ms["Month_Str"] = ms["PO Month"].dt.strftime("%b-%Y")
    fig_spend = px.line(ms, x="Month_Str", y="Spend (Cr ₹)", color="Entity", markers=True)
    fig_spend.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_spend, use_container_width=True)
else:
    st.info("Need PO Create Date, Entity, Net Amount for this chart.")

# ---------------------------------
# Lead time tables (kept)
# ---------------------------------
st.subheader("⏱️ PR→PO Lead Time by Buyer Type & by Buyer")
if not lead_df.empty:
    c1, c2 = st.columns(2)
    t1 = lead_df.groupby("PO.BuyerType")["Lead Time (Days)"].mean().round(0).reset_index()
    t2 = lead_df.groupby("PO.Creator")["Lead Time (Days)"].mean().round(0).reset_index()
    c1.dataframe(t1, use_container_width=True)
    c2.dataframe(t2, use_container_width=True)

# ---------------------------------
# Monthly PR & PO trends (kept)
# ---------------------------------
st.subheader("📅 Monthly PR & PO Trends")
if {"PR Date Submitted", "Purchase Doc"}.issubset(f.columns):
    g = f.copy()
    g["PR Month"] = g["PR Date Submitted"].dt.to_period("M")
    g["PO Month"] = (g["PO Create Date"].dt.to_period("M") if "PO Create Date" in g.columns else g["PR Month"])  # fallback
    monthly_summary = g.groupby("PR Month").agg({"PR Number": "count", "Purchase Doc": "count"}).reset_index()
    monthly_summary.columns = ["Month", "PR Count", "PO Count"]
    monthly_summary["Month"] = monthly_summary["Month"].astype(str)
    st.line_chart(monthly_summary.set_index("Month"), use_container_width=True)

# ---------------------------------
# Aging buckets (kept)
# ---------------------------------
st.subheader("🧮 PR→PO Aging Buckets")
if not lead_df.empty:
    bins = [0, 7, 15, 30, 60, 90, 999]
    labels = ["0-7", "8-15", "16-30", "31-60", "61-90", "90+"]
    aging = pd.cut(lead_df["Lead Time (Days)"], bins=bins, labels=labels)
    age_summary = aging.value_counts(normalize=True).sort_index().reset_index()
    age_summary.columns = ["Aging Bucket", "Percentage"]
    age_summary["Percentage"] *= 100
    fig_aging = px.bar(age_summary, x="Aging Bucket", y="Percentage", text="Percentage", labels={"Percentage": "Percentage (%)"})
    fig_aging.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    st.plotly_chart(fig_aging, use_container_width=True)

# ---------------------------------
# Weekday PR & PO (kept)
# ---------------------------------
st.subheader("📆 PRs & POs by Weekday")
if {"PR Date Submitted"}.issubset(f.columns):
    wd = f.copy()
    wd["PR Weekday"] = wd["PR Date Submitted"].dt.day_name()
    if "PO Create Date" in wd.columns:
        wd["PO Weekday"] = wd["PO Create Date"].dt.day_name()
    pr_counts = wd["PR Weekday"].value_counts().reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"], fill_value=0)
    st.bar_chart(pr_counts, use_container_width=True)
    if "PO Weekday" in wd.columns:
        po_counts = wd["PO Weekday"].value_counts().reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"], fill_value=0)
        st.bar_chart(po_counts, use_container_width=True)

# ---------------------------------
# Open PRs (kept)
# ---------------------------------
st.subheader("⚠️ Open PRs (Approved/InReview)")
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
            )
            .reset_index()
        )
        st.metric("Open PRs", int(open_summary["PR Number"].nunique()))
        st.dataframe(open_summary.sort_values("Pending_Age", ascending=False), use_container_width=True)

# ---------------------------------
# Daily PR trend (kept)
# ---------------------------------
st.subheader("📅 Daily PR Trends")
if "PR Date Submitted" in f.columns:
    ddf = f.copy()
    ddf["PR Date"] = ddf["PR Date Submitted"]
    daily = ddf.groupby("PR Date").size().reset_index(name="PR Count")
    st.line_chart(daily.set_index("PR Date"), use_container_width=True)

# ---------------------------------
# Buyer-wise Spend (kept)
# ---------------------------------
st.subheader("💰 Buyer-wise Spend (Cr ₹)")
if {"PO.Creator", "Net Amount"}.issubset(f.columns):
    bs = f.groupby("PO.Creator")["Net Amount"].sum().sort_values(ascending=False).reset_index()
    bs["Net Amount (Cr)"] = bs["Net Amount"]/1e7
    fig_buyer = px.bar(bs, x="PO.Creator", y="Net Amount (Cr)", text="Net Amount (Cr)")
    fig_buyer.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    st.plotly_chart(fig_buyer, use_container_width=True)

# ---------------------------------
# Category Spend (kept)
# ---------------------------------
st.subheader("🗂️ Spend by Procurement Category")
if {"Procurement Category", "Net Amount"}.issubset(f.columns):
    cs = f.groupby("Procurement Category")["Net Amount"].sum().sort_values(ascending=False).reset_index()
    cs["Spend (Cr ₹)"] = cs["Net Amount"]/1e7
    fig_cat = px.bar(cs, x="Procurement Category", y="Spend (Cr ₹)")
    fig_cat.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_cat, use_container_width=True)

# ---------------------------------
# PO Approval Summary (kept)
# ---------------------------------
if "PO Approved Date" in f.columns and "PO Create Date" in f.columns:
    st.subheader("📋 PO Approval Summary")
    po_app = f.dropna(subset=["PO Create Date"]).copy()
    po_app["PO Approval Lead Time"] = (pd.to_datetime(po_app["PO Approved Date"]) - po_app["PO Create Date"]).dt.days
    total_pos = po_app["Purchase Doc"].nunique() if "Purchase Doc" in po_app.columns else 0
    approved_pos = po_app[po_app["PO Approved Date"].notna()]["Purchase Doc"].nunique() if "Purchase Doc" in po_app.columns else 0
    pending_pos = total_pos - approved_pos
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Total POs", total_pos)
    a2.metric("Approved POs", approved_pos)
    a3.metric("Pending Approval", pending_pos)
    a4.metric("Avg Approval Lead Time", round(po_app["PO Approval Lead Time"].mean(),1) if not po_app.empty else 0)
    keep_cols = [c for c in ["PO.Creator", "Purchase Doc", "PO Create Date", "PO Approved Date", "PO Approval Lead Time"] if c in po_app.columns]
    if keep_cols:
        st.dataframe(po_app[keep_cols].sort_values(by="PO Approval Lead Time", ascending=False), use_container_width=True)

# ---------------------------------
# PO Status Pie (kept)
# ---------------------------------
st.subheader("📊 PO Status Breakdown")
if "PO Status" in f.columns:
    pstat = f["PO Status"].value_counts(dropna=False).reset_index()
    pstat.columns = ["PO Status", "Count"]
    c1, c2 = st.columns([2,3])
    c1.dataframe(pstat, use_container_width=True)
    c2.plotly_chart(px.pie(pstat, names="PO Status", values="Count", hole=0.35), use_container_width=True)

# ---------------------------------
# Delivery Summary + Pending Top 20 (kept)
# ---------------------------------
st.subheader("🚚 PO Delivery Summary & Top Pending")
if {"Purchase Doc", "PO Vendor", "Product Name", "Item Description"}.issubset(f.columns):
    d = f.rename(columns={"PO Qty": "PO Qty"}).copy()
    if "PO Qty" in d.columns and "Received Qty" in d.columns:
        d["% Received"] = (d["Received Qty"] / d["PO Qty"]).replace([pd.NA, pd.NaT], 0)*100
    d["% Received"] = d.get("% Received", pd.Series(0)).fillna(0).round(1)
    grp = d.groupby(["Purchase Doc", "PO Vendor", "Product Name", "Item Description"], dropna=False).agg({
        "PO Qty": "sum",
        "Received Qty": "sum",
        "Pending Qty": "sum",
        "% Received": "mean",
    }).reset_index()
    st.dataframe(grp.sort_values(by="Pending Qty", ascending=False), use_container_width=True)
    if "Pending Qty" in grp.columns:
        st.plotly_chart(px.bar(grp.sort_values(by="Pending Qty", ascending=False).head(20), x="Purchase Doc", y="Pending Qty", color="PO Vendor", text="Pending Qty"), use_container_width=True)

# ---------------------------------
# NEW: Top 50 Pending Lines by Value (restored)
# ---------------------------------
st.subheader("📋 Top 50 Pending Lines (by Value)")
if {"Pending Qty", "PO Unit Rate"}.issubset(f.columns):
    pend = f[f["Pending Qty"] > 0].copy()
    pend["Pending Value"] = pend["Pending Qty"] * pend["PO Unit Rate"]
    cols = [c for c in ["PR Number","Purchase Doc","Procurement Category","Buying legal entity","PR Budget description","Product Name","Item Description","Pending Qty","Pending Value"] if c in pend.columns]
    if cols:
        top50 = pend.sort_values("Pending Value", ascending=False).head(50)[cols]
        st.dataframe(top50.style.format({"Pending Qty": "{:,.0f}", "Pending Value": "₹ {:,.2f}"}), use_container_width=True)

# ---------------------------------
# Top 10 Vendors by Spend (restored)
# ---------------------------------
st.subheader("🏆 Top 10 Vendors by Spend (Cr ₹)")
if {"PO Vendor", "Purchase Doc", "Net Amount"}.issubset(f.columns):
    vs = (
        f.groupby("PO Vendor", dropna=False)
        .agg(Vendor_PO_Count=("Purchase Doc", "nunique"), Total_Spend_Cr=("Net Amount", lambda x: (x.sum()/1e7).round(2)))
        .reset_index()
        .sort_values("Total_Spend_Cr", ascending=False)
    )
    top10 = vs.head(10)
    st.dataframe(top10, use_container_width=True)
    st.plotly_chart(px.bar(top10, x="PO Vendor", y="Total_Spend_Cr", text="Total_Spend_Cr", title="Top 10 Vendors by Spend (Cr ₹)").update_traces(texttemplate="%{text:.2f}", textposition="outside"), use_container_width=True)

# ---------------------------------
# Budget Code Analysis (kept)
# ---------------------------------
st.subheader("🧾 Budget Code Analysis (PO)")
if {"PO Budget Code", "Net Amount"}.issubset(f.columns):
    bc = f.groupby("PO Budget Code", dropna=False)["Net Amount"].sum().reset_index().sort_values("Net Amount", ascending=False)
    bc["Spend (Cr ₹)"] = bc["Net Amount"]/1e7
    st.plotly_chart(px.bar(bc.head(50), x="PO Budget Code", y="Spend (Cr ₹)", title="Top Budget Codes by Spend (PO)").update_layout(xaxis_tickangle=-45), use_container_width=True)
    if not bm.empty and "Budget Code" in bm.columns:
        bc_map = bc.merge(bm, left_on="PO Budget Code", right_on="Budget Code", how="left")
        st.dataframe(bc_map.head(200), use_container_width=True)
        st.download_button("⬇️ Download Budget Code Spend (mapped)", bc_map.to_csv(index=False), "budget_code_spend_mapped.csv", "text/csv")

# ---------------------------------
# NEW: Entity-wise Procurement Scorecards
# ---------------------------------
st.subheader("🏷️ Entity-wise Procurement Scorecards")
if {"Entity"}.issubset(f.columns):
    sc = f.copy()
    # compute metrics per entity
    ent = sc.groupby("Entity").agg(
        PRs=("PR Number", "nunique"),
        POs=("Purchase Doc", "nunique"),
        Spend=("Net Amount", "sum")
    ).reset_index()
    if not lead_df.empty:
        lead_by_ent = lead_df.groupby("Entity")["Lead Time (Days)"].mean().reset_index()
        ent = ent.merge(lead_by_ent, on="Entity", how="left")
    ent["Spend (Cr ₹)"] = ent["Spend"]/1e7
    ent = ent.rename(columns={"Lead Time (Days)": "Avg Lead Days"})
    st.dataframe(ent[["Entity","PRs","POs","Spend (Cr ₹)","Avg Lead Days"]], use_container_width=True)

# ---------------------------------
# NEW: Vendor Concentration Risk (HHI)
# ---------------------------------
st.subheader("⚖️ Vendor Concentration Risk (HHI)")
if {"PO Vendor", "Net Amount"}.issubset(f.columns):
    v = f.groupby("PO Vendor")["Net Amount"].sum()
    total = v.sum()
    if total > 0:
        shares = (v/total)
        hhi = float((shares**2).sum())
        st.metric("Herfindahl–Hirschman Index", f"{hhi:.3f}")
        st.caption("Closer to 1 = highly concentrated. Under 0.15 = unconcentrated; 0.15–0.25 = moderate; >0.25 = high.")

# ---------------------------------
# NEW: On‑Time Delivery & Overdue Analysis
# ---------------------------------
st.subheader("⏳ Delivery Timeliness — Overdue by Vendor")
if {"PO Delivery Date"}.issubset(f.columns):
    today = pd.Timestamp.today().normalize()
    d2 = f.copy()
    d2["Pending Qty Filled"] = d2.get("Pending Qty", pd.Series(0)).fillna(0)
    d2["Overdue Days"] = (today - d2["PO Delivery Date"]).dt.days
    d2["Is Overdue"] = (d2["PO Delivery Date"].notna()) & (d2["Overdue Days"] > 0) & (d2["Pending Qty Filled"] > 0)
    od = d2[d2["Is Overdue"]]
    if not od.empty and "PO Vendor" in od.columns:
        over = od.groupby("PO Vendor").agg(
            Overdue_Lines=("Purchase Doc", "count"),
            Avg_Overdue_Days=("Overdue Days", "mean"),
            Pending_Qty=("Pending Qty Filled", "sum"),
            Pending_Value=("Net Amount", "sum")
        ).reset_index().sort_values(["Overdue_Lines","Avg_Overdue_Days"], ascending=[False, False])
        st.dataframe(over, use_container_width=True)

# ---------------------------------
# NEW: PR→PO Conversion Rate
# ---------------------------------
st.subheader("🔁 PR→PO Conversion Rate")
if {"PR Number","Purchase Doc"}.issubset(f.columns):
    pr_total = f["PR Number"].nunique()
    pr_with_po = f[f["Purchase Doc"].notna()]["PR Number"].nunique()
    rate = (pr_with_po/pr_total*100) if pr_total else 0
    st.metric("Conversion Rate", f"{rate:.1f}%")

# ------------- END -------------
