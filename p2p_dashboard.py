# p2p_dashboard.py — CLEAN FINAL
# Procure-to-Pay Dashboard with Budget-Code → Department/Subcategory mapping
# Drop this into your repo and run:  streamlit run p2p_dashboard.py

from __future__ import annotations

import re
from io import BytesIO
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# =============================
# 0) Page Configuration
# =============================
st.set_page_config(
    page_title="Procure-to-Pay Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================
# Helpers
# =============================

def normalize_code(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    s = re.sub(r"[\s\-_/]", "", s)
    return s.upper() if s else np.nan


def pick_col(df: pd.DataFrame, patterns: list[str]):
    for col in df.columns:
        name = str(col).strip().lower()
        for pat in patterns:
            if re.search(pat, name, flags=re.I):
                return col
    return None


def to_datetime_safe(s: pd.Series):
    return pd.to_datetime(s, errors="coerce")


def download_df(df: pd.DataFrame, filename: str, label: str):
    buf = BytesIO()
    df.to_csv(buf, index=False)
    st.download_button(label, buf.getvalue(), file_name=filename, mime="text/csv")


# =============================
# 1) Load & Combine Source Data
# =============================
@st.cache_data(show_spinner=False)
def load_local_excel(path: str, skiprows: int = 1) -> pd.DataFrame:
    return pd.read_excel(path, skiprows=skiprows)


@st.cache_data(show_spinner=False)
def load_and_combine_data(use_local: bool = True, uploads: list | None = None) -> pd.DataFrame:
    """Load MEPL/MLPL/MMW/MMPL either from local files or from uploaded files.
    Normalizes columns by stripping whitespace & collapsing spaces.
    Adds an Entity column per file.
    """
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
                df = load_local_excel(path, skiprows=1)
                df["Entity"] = tag
                frames.append(df)
            except Exception as e:
                st.warning(f"Couldn't load {path}: {e}")
    else:
        if not uploads:
            return pd.DataFrame()
        for up in uploads:
            try:
                df = pd.read_excel(up, skiprows=1) if up.name.lower().endswith((".xlsx", ".xls")) else pd.read_csv(up)
                tag = re.sub(r"\W+", "", up.name.split(".")[0]).upper()[:6]
                df["Entity"] = tag
                frames.append(df)
            except Exception as e:
                st.warning(f"Couldn't read {up.name}: {e}")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Clean up column names: strip whitespace, replace NBSP, collapse multiple spaces
    combined.columns = (
        combined.columns
        .astype(str)
        .str.strip()
        .str.replace("\xa0", " ", regex=False)
        .str.replace(" +", " ", regex=True)
    )

    return combined


# Sidebar source selection
st.sidebar.header("📥 Data Source")
use_local_files = st.sidebar.toggle("Use local repo files (MEPL1/MLPL1/mmw1/mmpl1)", value=True)
uploads = []
if not use_local_files:
    uploads = st.sidebar.file_uploader("Upload company workbooks (multi-select)", type=["xlsx", "xls", "csv"], accept_multiple_files=True) or []

df = load_and_combine_data(use_local=use_local_files, uploads=uploads)
if df.empty:
    st.stop()

# =============================
# 2) Clean & Prepare Date Columns
# =============================
for date_col in ["PR Date Submitted", "Po create Date"]:
    if date_col in df.columns:
        df[date_col] = to_datetime_safe(df[date_col]).dt.date
    else:
        st.error(f"❌ Column '{date_col}' not found. Please verify your Excel sheets.")

# =============================
# 3) Buyer Group Classification
# =============================
if "Buyer Group" in df.columns:
    df["Buyer Group Code"] = df["Buyer Group"].astype(str).str.extract(r"(\d+)").astype(float)

    def classify_buyer_group(row):
        bg = row["Buyer Group"]
        code = row["Buyer Group Code"]
        if bg in ["ME_BG17", "MLBG16"]:
            return "Direct"
        elif bg in ["Not Available"] or pd.isna(bg):
            return "Indirect"
        elif pd.notna(code) and 1 <= code <= 9:
            return "Direct"
        elif pd.notna(code) and 10 <= code <= 18:
            return "Indirect"
        else:
            return "Other"

    df["Buyer.Type"] = df.apply(classify_buyer_group, axis=1)
else:
    df["Buyer.Type"] = "Unknown"
    st.warning("⚠️ 'Buyer Group' column not found. All Buyer.Type set to 'Unknown'.")

# =============================
# 4) PO Orderer → PO.Creator Mapping
# =============================
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

df["PO Orderer"] = df.get("PO Orderer", pd.Series(index=df.index)).fillna("N/A").astype(str).str.strip()
df["PO.Creator"] = df["PO Orderer"].map(o_created_by_map).fillna(df["PO Orderer"]).replace({"N/A": "Dilip"})

indirect_buyers = ["Aatish", "Deepak", "Deepakex", "Dhruv", "Dilip", "Mukul", "Nayan", "Paurik", "Kamlesh", "Suresh"]
df["PO.BuyerType"] = df["PO.Creator"].apply(lambda x: "Indirect" if x in indirect_buyers else "Direct")

# =============================
# 5) Budget Code → Dept/Subcategory Mapping (+ gap list)
# =============================
st.sidebar.header("🧭 Budget Mapping")
map_file = st.sidebar.file_uploader("Upload mapping workbook (Department/Subcategory ⇄ Budget Codes)", type=["xlsx", "xls", "csv"])
use_default_map = st.sidebar.toggle("Use default mapping: MainDept_SubCat_BudgetCodes_20250927_191809.xlsx", value=True)

@st.cache_data(show_spinner=False)
def load_mapping(map_file, use_default: bool) -> pd.DataFrame | None:
    try:
        if map_file is not None:
            raw = pd.read_csv(map_file) if map_file.name.lower().endswith(".csv") else pd.read_excel(map_file)
        elif use_default:
            raw = pd.read_excel("MainDept_SubCat_BudgetCodes_20250927_191809.xlsx")
        else:
            return None
    except Exception as e:
        st.warning(f"Couldn't load mapping: {e}")
        return None

    # Identify columns
    dept_col = pick_col(raw, [r"\bdepartment\b", r"\bmain\s*dept", r"\bmain\s*department\b", r"\bdept\b"]) or "Department"
    subcat_col = pick_col(raw, [r"\bsub\s*cat", r"\bsub\s*category", r"\bsubcategory\b", r"\bsub\s*classification\b"]) or "Subcategory"
    budget_cols = [c for c in raw.columns if re.search(r"budget.*code|^code$|^budget$", str(c), flags=re.I)]
    if not budget_cols:
        # fallback: treat short-ish columns as possible codes
        for c in raw.columns:
            if c in [dept_col, subcat_col]:
                continue
            vals = raw[c].dropna().astype(str).str.len()
            if len(vals) > 0 and (vals.median() <= 10 or vals.mean() <= 12):
                budget_cols.append(c)

    id_cols = [c for c in [dept_col, subcat_col] if c in raw.columns]
    tidy = raw.melt(id_vars=id_cols, value_vars=budget_cols, var_name="_c", value_name="BudgetCodeRaw")
    tidy["BudgetCode"] = tidy["BudgetCodeRaw"].map(normalize_code)
    tidy = tidy.dropna(subset=["BudgetCode"]).drop_duplicates(subset=["BudgetCode"]).copy()

    rename_map = {}
    if dept_col in tidy.columns:
        rename_map[dept_col] = "Department"
    if subcat_col in tidy.columns:
        rename_map[subcat_col] = "Subcategory"
    tidy = tidy.rename(columns=rename_map)
    for c in ["Department", "Subcategory"]:
        if c not in tidy.columns:
            tidy[c] = np.nan
    tidy = tidy[["BudgetCode", "Department", "Subcategory"]]
    return tidy

mapping = load_mapping(map_file, use_default_map)

# Determine which column in the fact data holds the budget code
budget_col_in_df = pick_col(df, [r"\bpo\s*budget\s*code\b", r"\bbudget\s*code\b", r"^code$", r"^budget$"])
if not budget_col_in_df:
    # try a soft fallback
    budget_col_in_df = pick_col(df, [r"\bbudg\w*", r"\bcode\b"])

if mapping is not None and budget_col_in_df:
    df["BudgetCode"] = df[budget_col_in_df].map(normalize_code)
    mapping["BudgetCode"] = mapping["BudgetCode"].astype("object")
    df["BudgetCode"] = df["BudgetCode"].astype("object")

    mapped_df = df.merge(mapping, on="BudgetCode", how="left")

    # gaps
    unmapped = (
        mapped_df[mapped_df["BudgetCode"].notna() & mapped_df["Department"].isna()]["BudgetCode"]
        .dropna().astype(str).drop_duplicates().sort_values().to_frame(name="UnmappedBudgetCode")
    )
    by_entity_gap = (
        mapped_df[mapped_df["BudgetCode"].notna() & mapped_df["Department"].isna()]
        .groupby(["Entity", "BudgetCode"]).size().reset_index(name="Count")
        .sort_values(["Entity", "Count"], ascending=[True, False])
    )
else:
    mapped_df = df.copy()
    mapped_df["Department"] = np.nan
    mapped_df["Subcategory"] = np.nan
    unmapped = pd.DataFrame(columns=["UnmappedBudgetCode"])
    by_entity_gap = pd.DataFrame(columns=["Entity", "BudgetCode", "Count"])

# =============================
# 6) Sidebar Filters (robust)
# =============================
st.sidebar.header("🔍 Filters")

for col in ["Buyer.Type", "Entity", "PO.Creator", "PO.BuyerType", "PR Date Submitted"]:
    if col not in mapped_df.columns:
        mapped_df[col] = pd.NA

for col in ["Buyer.Type", "Entity", "PO.Creator", "PO.BuyerType"]:
    mapped_df[col] = mapped_df[col].astype(str).fillna("").str.strip()

buyer_options = sorted(mapped_df["Buyer.Type"].dropna().unique().tolist())
entity_options = sorted(mapped_df["Entity"].dropna().unique().tolist())
orderer_options = sorted(mapped_df["PO.Creator"].dropna().unique().tolist())
po_buyer_type_options = sorted(mapped_df["PO.BuyerType"].dropna().unique().tolist())

fy_options = {
    "All Years": (pd.to_datetime("2023-04-01"), pd.to_datetime("2026-03-31")),
    "2023": (pd.to_datetime("2023-04-01"), pd.to_datetime("2024-03-31")),
    "2024": (pd.to_datetime("2024-04-01"), pd.to_datetime("2025-03-31")),
    "2025": (pd.to_datetime("2025-04-01"), pd.to_datetime("2026-03-31")),
    "2026": (pd.to_datetime("2026-04-01"), pd.to_datetime("2027-03-31")),
}
selected_fy = st.sidebar.selectbox("Select Financial Year", options=list(fy_options.keys()), index=0)
pr_start, pr_end = fy_options[selected_fy]

buyer_filter = st.sidebar.multiselect("Buyer Type", options=buyer_options, default=buyer_options)
entity_filter = st.sidebar.multiselect("Entity", options=entity_options, default=entity_options)
orderer_filter = st.sidebar.multiselect("PO Ordered By", options=orderer_options, default=orderer_options)
po_buyer_type_filter = st.sidebar.multiselect("PO Buyer Type", options=po_buyer_type_options, default=po_buyer_type_options)

# choose date column for free-range filter
date_col_for_filter = "PR Date Submitted" if "PR Date Submitted" in mapped_df.columns else ("Po create Date" if "Po create Date" in mapped_df.columns else None)
if date_col_for_filter:
    mapped_df[date_col_for_filter] = pd.to_datetime(mapped_df[date_col_for_filter], errors="coerce")
    min_date = mapped_df[date_col_for_filter].min().date() if pd.notna(mapped_df[date_col_for_filter].min()) else date.today()
    max_date = mapped_df[date_col_for_filter].max().date() if pd.notna(mapped_df[date_col_for_filter].max()) else date.today()
    date_range = st.sidebar.date_input("Date Range", [min_date, max_date])
else:
    date_range = None

filtered_df = mapped_df.copy()

# Apply FY filter on PR Date Submitted if present
if "PR Date Submitted" in filtered_df.columns:
    filtered_df["PR Date Submitted"] = pd.to_datetime(filtered_df["PR Date Submitted"], errors="coerce")
    filtered_df = filtered_df[(filtered_df["PR Date Submitted"] >= pr_start) & (filtered_df["PR Date Submitted"] <= pr_end)]

# Apply date range
if date_range and date_col_for_filter:
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        s_dt, e_dt = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        filtered_df = filtered_df[(filtered_df[date_col_for_filter] >= s_dt) & (filtered_df[date_col_for_filter] <= e_dt)]

# Multiselect filters
if buyer_filter:
    filtered_df = filtered_df[filtered_df["Buyer.Type"].astype(str).str.strip().isin(buyer_filter)]
if entity_filter:
    filtered_df = filtered_df[filtered_df["Entity"].astype(str).str.strip().isin(entity_filter)]
if orderer_filter:
    filtered_df = filtered_df[filtered_df["PO.Creator"].astype(str).str.strip().isin(orderer_filter)]
if po_buyer_type_filter:
    filtered_df = filtered_df[filtered_df["PO.BuyerType"].astype(str).str.strip().isin(po_buyer_type_filter)]

st.sidebar.markdown("----")
st.sidebar.write("Selected FY:", selected_fy)
st.sidebar.write("Rows after filters:", len(filtered_df))

# =============================
# 7) Keyword Search (simple)
# =============================
st.markdown("## 🔍 Keyword Search")
valid_columns = [c for c in ["PR Number", "Purchase Doc", "Product Name", "PO Vendor"] if c in filtered_df.columns]
user_query = st.text_input("Type a keyword (vendor/product/PO/PR)", "")
if user_query:
    mask = pd.Series(False, index=filtered_df.index)
    for c in valid_columns:
        mask = mask | filtered_df[c].astype(str).str.contains(user_query, case=False, na=False)
    result_df = filtered_df[mask]
    st.markdown(f"### 🔎 Found {len(result_df)} matching rows")
    st.dataframe(result_df, use_container_width=True)
    download_df(result_df, "search_results.csv", "⬇️ Download Search Results (CSV)")
else:
    st.info("Start typing a keyword to search…")

# =============================
# 8) Top KPI Row
# =============================
st.title("📊 Procure-to-Pay Dashboard")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total PRs",        filtered_df.get("PR Number", pd.Series(dtype=object)).nunique())
col2.metric("Total POs",        filtered_df.get("Purchase Doc", pd.Series(dtype=object)).nunique())
col3.metric("Line Items",       len(filtered_df))
col4.metric("Entities",         filtered_df.get("Entity", pd.Series(dtype=object)).nunique())
col5.metric("Spend (Cr ₹)",     f"{(filtered_df.get('Net Amount', pd.Series(dtype=float)).sum() / 1e7):,.2f}")

# =============================
# 9) SLA Compliance Gauge (PR → PO ≤ 7 days)
# =============================
st.subheader("🎯 SLA Compliance (PR → PO ≤ 7 days)")
lead_df = filtered_df[filtered_df.get("Po create Date").notna()] if "Po create Date" in filtered_df.columns else pd.DataFrame()
if not lead_df.empty and "PR Date Submitted" in lead_df.columns:
    lead_df = lead_df.copy()
    lead_df["Lead Time (Days)"] = (pd.to_datetime(lead_df["Po create Date"]) - pd.to_datetime(lead_df["PR Date Submitted"]).dt.tz_localize(None)).dt.days
    SLA_DAYS = 7
    avg_lead = float(pd.to_numeric(lead_df["Lead Time (Days)"], errors="coerce").dropna().mean()) if not lead_df.empty else np.nan
    if pd.notna(avg_lead):
        avg_lead = round(avg_lead, 1)
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_lead,
            number={"suffix": " days"},
            gauge={
                "axis": {"range": [0, max(14, avg_lead * 1.2)]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, SLA_DAYS], "color": "lightgreen"},
                    {"range": [SLA_DAYS, max(14, avg_lead * 1.2)], "color": "lightcoral"},
                ],
                "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": SLA_DAYS},
            },
            title={"text": "Average Lead Time"},
        ))
        st.plotly_chart(gauge_fig, use_container_width=True)
        st.caption(f"Current Avg Lead Time: {avg_lead:.1f} days   •   Target ≤ {SLA_DAYS} days")
    else:
        st.info("Not enough valid dates to compute SLA.")
else:
    st.info("Need 'PR Date Submitted' and 'Po create Date' to compute SLA.")

# =============================
# 10) Monthly Total Spend + Cumulative
# =============================
st.subheader("📊 Monthly Total Spend (with Cumulative Value Line)")

date_col = "Po create Date" if "Po create Date" in filtered_df.columns else ("PR Date Submitted" if "PR Date Submitted" in filtered_df.columns else None)
if date_col and "Net Amount" in filtered_df.columns:
    temp = filtered_df.copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp = temp.dropna(subset=[date_col])
    temp["PO_Month"] = temp[date_col].dt.to_period("M").dt.to_timestamp()
    temp["Month_Str"] = temp["PO_Month"].dt.strftime("%b-%Y")

    monthly_total_spend = temp.groupby(["PO_Month", "Month_Str"], as_index=False)["Net Amount"].sum()
    monthly_total_spend["Spend (Cr ₹)"] = monthly_total_spend["Net Amount"] / 1e7
    monthly_total_spend = monthly_total_spend.sort_values("PO_Month").reset_index(drop=True)
    monthly_total_spend["Cumulative Spend (Cr ₹)"] = monthly_total_spend["Spend (Cr ₹)"].cumsum()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=monthly_total_spend["Month_Str"],
        y=monthly_total_spend["Spend (Cr ₹)"],
        name="Monthly Spend (Cr ₹)",
        text=monthly_total_spend["Spend (Cr ₹)"].map("{:.2f}".format),
        textposition="outside",
        marker=dict(opacity=0.85),
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=monthly_total_spend["Month_Str"],
        y=monthly_total_spend["Cumulative Spend (Cr ₹)"],
        mode="lines+markers",
        name="Cumulative Spend (Cr ₹)",
        line=dict(width=3, color="darkblue"),
        marker=dict(size=8),
        hovertemplate="₹ %{y:.2f} Cr<br>%{x}<extra></extra>",
    ), secondary_y=True)

    fig.update_layout(title="Monthly Total Spend with Cumulative Value", xaxis=dict(tickangle=-45), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(t=70, b=120))
    fig.update_yaxes(title_text="Monthly Spend (Cr ₹)", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Spend (Cr ₹)", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Need a date column and 'Net Amount' for monthly spend chart.")

# =============================
# 11) Monthly Spend Trend by Entity
# =============================
st.subheader("💹 Monthly Spend Trend by Entity")
if "Po create Date" in filtered_df.columns and "Net Amount" in filtered_df.columns:
    spend_df = filtered_df.copy()
    spend_df["PO Month"] = pd.to_datetime(spend_df["Po create Date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    monthly_spend = spend_df.dropna(subset=["PO Month"]).groupby(["PO Month", "Entity"], as_index=False)["Net Amount"].sum()
    monthly_spend["Spend (Cr ₹)"] = monthly_spend["Net Amount"] / 1e7
    monthly_spend["Month_Str"] = monthly_spend["PO Month"].dt.strftime("%b-%Y")
    fig_spend = px.line(monthly_spend, x="Month_Str", y="Spend (Cr ₹)", color="Entity", markers=True, title="Monthly Spend Trend by Entity", labels={"Month_Str": "Month", "Spend (Cr ₹)": "Spend (Cr ₹)"})
    fig_spend.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_spend, use_container_width=True)
else:
    st.info("Need 'Po create Date' and 'Net Amount' for entity trend chart.")

# =============================
# 12) Lead Time by Buyer Type & Buyer
# =============================
st.subheader("⏱️ PR to PO Lead Time by Buyer Type & by Buyer")
if not lead_df.empty:
    lt = lead_df.copy()
    lt["Lead Time (Days)"] = pd.to_numeric(lt["Lead Time (Days)"], errors="coerce")
    lead_avg_by_type = lt.groupby("Buyer.Type")["Lead Time (Days)"].mean().round(0).reset_index()
    lead_avg_by_buyer = lt.groupby("PO.Creator")["Lead Time (Days)"].mean().round(0).reset_index()
    c1, c2 = st.columns(2)
    c1.dataframe(lead_avg_by_type, use_container_width=True)
    c2.dataframe(lead_avg_by_buyer, use_container_width=True)
else:
    st.info("Lead time table needs valid PR & PO dates.")

# =============================
# 13) Monthly PR & PO Trends
# =============================
st.subheader("📅 Monthly PR & PO Trends")
if "PR Date Submitted" in filtered_df.columns and "Po create Date" in filtered_df.columns:
    tmp = filtered_df.copy()
    tmp["PR Month"] = pd.to_datetime(tmp["PR Date Submitted"], errors="coerce").dt.to_period("M")
    tmp["PO Month"] = pd.to_datetime(tmp["Po create Date"], errors="coerce").dt.to_period("M")
    monthly_summary = tmp.groupby("PR Month").agg({"PR Number": "count", "Purchase Doc": "count"}).reset_index()
    monthly_summary.columns = ["Month", "PR Count", "PO Count"]
    monthly_summary["Month"] = monthly_summary["Month"].astype(str)
    st.line_chart(monthly_summary.set_index("Month"), use_container_width=True)
else:
    st.info("Need 'PR Date Submitted' & 'Po create Date' for PR/PO trend.")

# =============================
# 14) PR → PO Aging Buckets
# =============================
st.subheader("🧮 PR to PO Aging Buckets")
if not lead_df.empty and "Lead Time (Days)" in lead_df.columns:
    aging_vals = pd.to_numeric(lead_df["Lead Time (Days)"], errors="coerce").dropna()
    if not aging_vals.empty:
        bins = [0, 7, 15, 30, 60, 90, 999]
        labels = ["0-7", "8-15", "16-30", "31-60", "61-90", "90+"]
        aging_buckets = pd.cut(aging_vals, bins=bins, labels=labels)
        age_summary = aging_buckets.value_counts(normalize=True).sort_index().reset_index()
        age_summary.columns = ["Aging Bucket", "Percentage"]
        age_summary["Percentage"] *= 100
        fig_aging = px.bar(age_summary, x="Aging Bucket", y="Percentage", text="Percentage", title="PR to PO Aging Bucket Distribution (%)", labels={"Percentage": "Percentage (%)"})
        fig_aging.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(fig_aging, use_container_width=True)
    else:
        st.info("No numeric lead time values to bucket.")
else:
    st.info("Lead time not available for aging buckets.")

# =============================
# 15) PRs & POs by Weekday
# =============================
st.subheader("📆 PRs and POs by Weekday")
df_wd = filtered_df.copy()
df_wd["PR Weekday"] = pd.to_datetime(df_wd.get("PR Date Submitted"), errors="coerce").dt.day_name()
df_wd["PO Weekday"] = pd.to_datetime(df_wd.get("Po create Date"), errors="coerce").dt.day_name()

order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
pr_counts = df_wd["PR Weekday"].value_counts().reindex(order, fill_value=0)
po_counts = df_wd["PO Weekday"].value_counts().reindex(order, fill_value=0)

c1, c2 = st.columns(2)
c1.bar_chart(pr_counts, use_container_width=True)
c2.bar_chart(po_counts, use_container_width=True)

# =============================
# 16) Open PRs (Approved / InReview)
# =============================
st.subheader("⚠️ Open PRs (Approved/InReview)")
if "PR Status" in filtered_df.columns:
    open_df = filtered_df[filtered_df["PR Status"].isin(["Approved", "InReview"])].copy()
    if not open_df.empty and "PR Date Submitted" in open_df.columns:
        open_df["Pending Age (Days)"] = (pd.Timestamp.today().normalize() - pd.to_datetime(open_df["PR Date Submitted"], errors="coerce")).dt.days
        open_summary = open_df.groupby("PR Number").agg({
            "PR Date Submitted": "first",
            "Pending Age (Days)": "first",
            "Procurement Category": "first",
            "Product Name": "first",
            "Net Amount": "sum",
            "PO Budget Code": "first",
            "PR Status": "first",
            "Buyer Group": "first",
            "Buyer.Type": "first",
            "Entity": "first",
            "PO.Creator": "first",
            "Purchase Doc": "first",
        }).reset_index()
        st.metric("🔢 Open PRs", open_summary["PR Number"].nunique())
        open_monthly_counts = pd.to_datetime(open_summary["PR Date Submitted"], errors="coerce").dt.to_period("M").value_counts().sort_index()
        st.bar_chart(open_monthly_counts, use_container_width=True)
        def highlight_age(val):
            return "background-color: red" if pd.notna(val) and val > 30 else ""
        st.dataframe(open_summary.style.applymap(highlight_age, subset=["Pending Age (Days)"]), use_container_width=True)
        st.subheader("🏢 Open PRs by Entity")
        ent_counts = open_summary["Entity"].value_counts().reset_index()
        ent_counts.columns = ["Entity", "Count"]
        st.bar_chart(ent_counts.set_index("Entity"), use_container_width=True)
    else:
        st.warning("⚠️ No open PRs match the current filters or missing dates.")
else:
    st.info("ℹ️ 'PR Status' column not found.")

# =============================
# 17) Daily PR Trends
# =============================
st.subheader("📅 Daily PR Trends")
if "PR Date Submitted" in filtered_df.columns:
    daily_df = filtered_df.copy()
    daily_df["PR Date"] = pd.to_datetime(daily_df["PR Date Submitted"], errors="coerce")
    daily_trend = daily_df.groupby("PR Date").size().reset_index(name="PR Count")
    fig_daily = px.line(daily_trend, x="PR Date", y="PR Count", title="Daily PR Submissions", labels={"PR Count": "PR Count"})
    st.plotly_chart(fig_daily, use_container_width=True)
else:
    st.info("Need 'PR Date Submitted' for daily PR trend.")

# =============================
# 18) Buyer-wise Spend
# =============================
st.subheader("💰 Buyer-wise Spend (Cr ₹)")
if all(c in filtered_df.columns for c in ["PO.Creator", "Net Amount"]):
    buyer_spend = filtered_df.groupby("PO.Creator")["Net Amount"].sum().sort_values(ascending=False).reset_index()
    buyer_spend["Net Amount (Cr)"] = buyer_spend["Net Amount"] / 1e7
    fig_buyer = px.bar(buyer_spend, x="PO.Creator", y="Net Amount (Cr)", title="Spend by Buyer", labels={"Net Amount (Cr)": "Spend (Cr ₹)", "PO.Creator": "Buyer"}, text="Net Amount (Cr)")
    fig_buyer.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    st.plotly_chart(fig_buyer, use_container_width=True)
else:
    st.info("Need 'PO.Creator' and 'Net Amount' for buyer spend chart.")

# =============================
# 19) Spend by Category
# =============================
if "Procurement Category" in filtered_df.columns and "Net Amount" in filtered_df.columns:
    st.subheader("🧭 Spend by Category (Descending)")
    cat_spend = filtered_df.groupby("Procurement Category")["Net Amount"].sum().sort_values(ascending=False).reset_index()
    cat_spend["Spend (Cr ₹)"] = cat_spend["Net Amount"] / 1e7
    fig_cat = px.bar(cat_spend, x="Procurement Category", y="Spend (Cr ₹)", title="Spend by Category (Descending)", labels={"Spend (Cr ₹)": "Spend (Cr ₹)", "Procurement Category": "Category"})
    fig_cat.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_cat, use_container_width=True)

# =============================
# 20) PO Approval Summary & Details
# =============================
if "PO Approved Date" in filtered_df.columns and "Po create Date" in filtered_df.columns and "Purchase Doc" in filtered_df.columns:
    st.subheader("📋 PO Approval Summary")
    po_app_df = filtered_df[filtered_df["Po create Date"].notna()].copy()
    po_app_df["PO Approved Date"] = pd.to_datetime(po_app_df["PO Approved Date"], errors="coerce")
    total_pos = po_app_df["Purchase Doc"].nunique()
    approved_pos = po_app_df[po_app_df["PO Approved Date"].notna()]["Purchase Doc"].nunique()
    pending_pos = total_pos - approved_pos
    po_app_df["PO Approval Lead Time"] = (po_app_df["PO Approved Date"] - pd.to_datetime(po_app_df["Po create Date"])) .dt.days
    avg_approval = float(pd.to_numeric(po_app_df["PO Approval Lead Time"], errors="coerce").dropna().mean()) if total_pos else np.nan
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📦 Total POs", total_pos)
    c2.metric("✅ Approved POs", approved_pos)
    c3.metric("⏳ Pending Approval", pending_pos)
    c4.metric("⏱️ Avg Approval Lead Time (days)", f"{avg_approval:.1f}" if pd.notna(avg_approval) else "-")
    st.subheader("📄 Detailed PO Approval Aging List")
    approval_detail = po_app_df[["PO.Creator", "Purchase Doc", "Po create Date", "PO Approved Date", "PO Approval Lead Time"]].sort_values(by="PO Approval Lead Time", ascending=False)
    st.dataframe(approval_detail, use_container_width=True)

# =============================
# 21) PO Delivery Summary: Received vs Pending
# =============================
st.subheader("🚚 PO Delivery Summary: Received vs Pending")
# Normalize column names used downstream
delivery_df = filtered_df.rename(columns={"PO Quantity": "PO Qty", "ReceivedQTY": "Received Qty", "Pending QTY": "Pending Qty"}).copy()
if set(["PO Qty", "Received Qty", "Pending Qty"]).issubset(delivery_df.columns):
    delivery_df["% Received"] = (pd.to_numeric(delivery_df["Received Qty"], errors="coerce") / pd.to_numeric(delivery_df["PO Qty"], errors="coerce")) * 100
    delivery_df["% Received"] = delivery_df["% Received"].fillna(0).round(1)
    po_delivery_summary = delivery_df.groupby(["Purchase Doc", "PO Vendor", "Product Name", "Item Description"], dropna=False).agg({"PO Qty": "sum", "Received Qty": "sum", "Pending Qty": "sum", "% Received": "mean"}).reset_index()
    st.dataframe(po_delivery_summary.sort_values(by="Pending Qty", ascending=False), use_container_width=True)
    fig_pending = px.bar(po_delivery_summary.sort_values(by="Pending Qty", ascending=False).head(20), x="Purchase Doc", y="Pending Qty", color="PO Vendor", hover_data=["Product Name", "Item Description"], title="Top 20 POs Awaiting Delivery (Pending Qty)", text="Pending Qty")
    fig_pending.update_traces(textposition="outside")
    st.plotly_chart(fig_pending, use_container_width=True)
    total_po_lines = len(delivery_df)
    fully_received = (pd.to_numeric(delivery_df["Pending Qty"], errors="coerce") == 0).sum()
    partially_pending = (pd.to_numeric(delivery_df["Pending Qty"], errors="coerce") > 0).sum()
    avg_receipt_pct = float(pd.to_numeric(delivery_df["% Received"], errors="coerce").dropna().mean()) if total_po_lines else 0
    st.markdown("### 📋 Delivery Performance Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PO Lines", total_po_lines)
    c2.metric("Fully Delivered", fully_received)
    c3.metric("Pending Delivery", partially_pending)
    c4.metric("Avg. Receipt %", f"{avg_receipt_pct:.1f}%")
    st.download_button("📥 Download Delivery Status", data=po_delivery_summary.to_csv(index=False), file_name="PO_Delivery_Status.csv", mime="text/csv")
else:
    st.info("Missing quantity columns for delivery summary.")

# =============================
# 22) Top 50 Pending Delivery Lines by Value
# =============================
st.subheader("📋 Top 50 Pending Lines (by Value)")
if set(["Pending Qty", "PO Unit Rate"]).issubset(delivery_df.columns):
    pending_items = delivery_df[pd.to_numeric(delivery_df["Pending Qty"], errors="coerce") > 0].copy()
    pending_items["Pending Value"] = pd.to_numeric(pending_items["Pending Qty"], errors="coerce") * pd.to_numeric(pending_items["PO Unit Rate"], errors="coerce")
    cols_present = [c for c in ["PR Number", "Purchase Doc", "Procurement Category", "Buying legal entity", "PR Budget description", "Product Name", "Item Description", "Pending Qty", "Pending Value"] if c in pending_items.columns]
    top_pending_items = pending_items.sort_values(by="Pending Value", ascending=False).head(50)[cols_present].reset_index(drop=True)
    st.dataframe(top_pending_items.style.format({"Pending Qty": "{:,.0f}", "Pending Value": "₹ {:,.2f}"}), use_container_width=True)
else:
    st.info("Missing 'Pending Qty' or 'PO Unit Rate' to compute pending value.")

# =============================
# 23) Top 10 Vendors by Spend
# =============================
st.subheader("🏆 Top 10 Vendors by Spend (Cr ₹)")
if all(c in filtered_df.columns for c in ["PO Vendor", "Purchase Doc", "Net Amount"]):
    vendor_spend = filtered_df.groupby("PO Vendor", dropna=False).agg(Vendor_PO_Count=("Purchase Doc", "nunique"), Total_Spend_Cr=("Net Amount", lambda x: (x.sum() / 1e7).round(2))).reset_index().sort_values(by="Total_Spend_Cr", ascending=False)
    top10_spend = vendor_spend.head(10).copy()
    st.dataframe(top10_spend, use_container_width=True)
    fig_top_vendors = px.bar(top10_spend, x="PO Vendor", y="Total_Spend_Cr", title="Top 10 Vendors by Spend (Cr ₹)", labels={"Total_Spend_Cr": "Spend (Cr ₹)", "PO Vendor": "Vendor"}, text="Total_Spend_Cr")
    fig_top_vendors.update_traces(textposition="outside", texttemplate="%{text:.2f}")
    st.plotly_chart(fig_top_vendors, use_container_width=True)

# =============================
# 24) Vendor Delivery Performance
# =============================
st.subheader("📊 Vendor Delivery Performance (Top 10 by Spend)")
if all(c in filtered_df.columns for c in ["PO Vendor", "Purchase Doc"]) and "Pending Qty" in delivery_df.columns:
    today = pd.Timestamp.today().normalize().date()
    df_vp = filtered_df.copy()
    df_vp["Pending Qty Filled"] = pd.to_numeric(delivery_df["Pending Qty"], errors="coerce").fillna(0)
    df_vp["Is_Fully_Delivered"] = df_vp["Pending Qty Filled"] == 0
    if "PO Delivery Date" in df_vp.columns:
        df_vp["PO Delivery Date"] = pd.to_datetime(df_vp["PO Delivery Date"], errors="coerce")
        df_vp["Is_Late"] = df_vp["PO Delivery Date"].dt.date.notna() & (df_vp["PO Delivery Date"].dt.date < today) & (df_vp["Pending Qty Filled"] > 0)
    else:
        df_vp["Is_Late"] = False
    vendor_perf = df_vp.groupby("PO Vendor", dropna=False).agg(Total_PO_Count=("Purchase Doc", "nunique"), Fully_Delivered_PO_Count=("Is_Fully_Delivered", "sum"), Late_PO_Count=("Is_Late", "sum")).reset_index()
    vendor_perf["Pct_Fully_Delivered"] = (vendor_perf["Fully_Delivered_PO_Count"] / vendor_perf["Total_PO_Count"] * 100).round(1)
    vendor_perf["Pct_Late"] = (vendor_perf["Late_PO_Count"] / vendor_perf["Total_PO_Count"] * 100).round(1)
    if 'vendor_spend' in locals():
        vendor_perf = vendor_perf.merge(vendor_spend[["PO Vendor", "Total_Spend_Cr"]], on="PO Vendor", how="left")
        top10_vendor_perf = vendor_perf.sort_values("Total_Spend_Cr", ascending=False).head(10)
    else:
        top10_vendor_perf = vendor_perf.sort_values("Total_PO_Count", ascending=False).head(10)
        top10_vendor_perf["Total_Spend_Cr"] = None
    st.dataframe(top10_vendor_perf[["PO Vendor", "Total_PO_Count", "Fully_Delivered_PO_Count", "Late_PO_Count", "Pct_Fully_Delivered", "Pct_Late", "Total_Spend_Cr"]], use_container_width=True)
    melted_perf = top10_vendor_perf.melt(id_vars=["PO Vendor"], value_vars=["Pct_Fully_Delivered", "Pct_Late"], var_name="Metric", value_name="Percentage")
    fig_vendor_perf = px.bar(melted_perf, x="PO Vendor", y="Percentage", color="Metric", barmode="group", title="% Fully Delivered vs % Late (Top 10 Vendors by Spend)", labels={"Percentage": "% of POs", "PO Vendor": "Vendor"})
    st.plotly_chart(fig_vendor_perf, use_container_width=True)

# =============================
# 25) Monthly Unique PO Generation
# =============================
st.subheader("🗓️ Monthly Unique PO Generation")
if "Purchase Doc" in filtered_df.columns and "Po create Date" in filtered_df.columns:
    po_monthly = filtered_df[filtered_df["Purchase Doc"].notna()].copy()
    po_monthly["PO Month"] = pd.to_datetime(po_monthly["Po create Date"], errors="coerce").dt.to_period("M")
    monthly_po_counts = po_monthly.groupby("PO Month")["Purchase Doc"].nunique().reset_index(name="Unique PO Count")
    monthly_po_counts["PO Month"] = monthly_po_counts["PO Month"].astype(str)
    fig_monthly_po = px.bar(monthly_po_counts, x="PO Month", y="Unique PO Count", title="Monthly Unique PO Generation", labels={"PO Month": "Month", "Unique PO Count": "Number of Unique POs"}, text="Unique PO Count")
    fig_monthly_po.update_traces(textposition="outside")
    st.plotly_chart(fig_monthly_po, use_container_width=True)

# =============================
# 26) Mapping Deliverables
# =============================
st.subheader("🧾 Mapping Deliverables")
col_a, col_b, col_c = st.columns([2, 1, 1])
with col_a:
    st.markdown("**All Transactions + Mapping (sample)**")
    st.dataframe(filtered_df.head(5000), use_container_width=True)
with col_b:
    st.markdown("**Unmapped Budget Codes**")
    st.metric("Unique unmapped codes", len(unmapped))
    st.dataframe(unmapped, use_container_width=True, height=280)
with col_c:
    st.markdown("**Unmapped by Entity**")
    st.dataframe(by_entity_gap, use_container_width=True, height=280)

with st.expander("⬇️ Download CSVs"):
    download_df(mapping if mapping is not None else pd.DataFrame(), "Master_BudgetCode_Department_Subcategory_Mapping.csv", "Master Mapping")
    download_df(filtered_df, "AllCompany_Transactions_WithMapping.csv", "All Transactions + Mapping")
    download_df(unmapped, "Unmapped_BudgetCodes_List.csv", "Unmapped Budget Codes")
    download_df(by_entity_gap, "Unmapped_BudgetCodes_ByEntity.csv", "Unmapped by Entity")

st.success("Done. Budget codes are mapped to Department/Subcategory. Any codes without a Department appear in the Unmapped list.")
