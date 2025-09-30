# p2p_dashboard.py — CLEAN FINAL (with Dept/Subcategory Spend drilldowns)
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
_o_map = {
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
[df.__setitem__("PO.Creator", df["PO Orderer"].map(_o_map).fillna(df["PO Orderer"]).replace({"N/A": "Dilip"}))]

_indirect = ["Aatish", "Deepak", "Deepakex", "Dhruv", "Dilip", "Mukul", "Nayan", "Paurik", "Kamlesh", "Suresh"]
df["PO.BuyerType"] = df["PO.Creator"].apply(lambda x: "Indirect" if x in _indirect else "Direct")

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
    temp["PO_Month"] = temp[date_col].dt.to_period("M").dt.to_timestamp
