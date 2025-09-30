# p2p_dashboard.py

import re
from datetime import date
import io

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ====================================
#  Procure-to-Pay Dashboard (Streamlit)
#  + Budget Code → Department/Subcategory mapping
# ====================================

# --- 0) Page Configuration ---
st.set_page_config(
    page_title="Procure-to-Pay Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------
# Helpers
# ------------------------------
def _normalize_code(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    s = re.sub(r"[\s\-_\/]", "", s, flags=re.I)
    return s.upper() or None

def _pick_col(df: pd.DataFrame, patterns):
    for col in df.columns:
        name = str(col).strip().lower().replace("\xa0", " ")
        for pat in patterns:
            if re.search(pat, name, flags=re.I):
                return col
    return None

def _safe_unique_vals(series):
    if series is None:
        return []
    return sorted([str(x).strip() for x in series.dropna().unique()])

@st.cache_data(show_spinner=False)
def _load_excel_first_sheet(f):
    """Load first sheet from an uploaded file or path-like, ensure string column names."""
    xls = pd.ExcelFile(f)
    df = xls.parse(xls.sheet_names[0])
    df.columns = [str(c) for c in df.columns]
    return df

def _build_tidy_mapping(map_df_raw: pd.DataFrame) -> pd.DataFrame:
    """Convert mapping workbook into a tidy: BudgetCode, Department, Subcategory."""
    dept_col = _pick_col(map_df_raw, [r"\bdepartment\b", r"\bmain\s*dept", r"\bdept\b"])
    subcat_col = _pick_col(map_df_raw, [r"\bsub\s*cat", r"\bsub\s*category", r"\bsubcategory\b"])

    # likely budget code columns
    budget_cols = [c for c in map_df_raw.columns if re.search(r"budget.*code|^code$|^budget$", str(c), flags=re.I)]
    if not budget_cols:
        # fallback: columns that look like short codes (median length <= 10)
        for c in map_df_raw.columns:
            if c in {dept_col, subcat_col}:
                continue
            vals = map_df_raw[c].dropna().astype(str).str.len()
            if len(vals) > 0 and (vals.median() <= 10 or vals.mean() <= 12):
                budget_cols.append(c)

    id_cols = [c for c in [dept_col, subcat_col] if c]
    tidy = map_df_raw.melt(id_vars=id_cols, value_vars=budget_cols, var_name="_col", value_name="BudgetCodeRaw")
    tidy["BudgetCode"] = tidy["BudgetCodeRaw"].map(_normalize_code)
    tidy = tidy.dropna(subset=["BudgetCode"]).drop_duplicates(subset=["BudgetCode"]).copy()

    tidy = tidy.rename(columns={
        (dept_col or "Department"): "Department",
        (subcat_col or "Subcategory"): "Subcategory"
    })
    for c in ["Department", "Subcategory"]:
        if c not in tidy.columns:
            tidy[c] = np.nan
    tidy = tidy[["BudgetCode", "Department", "Subcategory"]]
    return tidy

# ------------------------------------
#  1) Load & Combine Source Data
# ------------------------------------
@st.cache_data(show_spinner=False)
def load_and_combine_data():
    """
    Reads the four Excel files from the current directory:
       - MEPL1.xlsx
       - MLPL1.xlsx
       - mmw1.xlsx
       - mmpl1.xlsx
    Tags each with an "Entity" column, concatenates them, and
    normalizes column names by stripping whitespace.
    """
    def _read(path):
        try:
            return pd.read_excel(path, skiprows=1)
        except Exception:
            # fallback: try without skiprows
            return pd.read_excel(path)

    mepl_df = _read("MEPL1.xlsx"); mepl_df["Entity"] = "MEPL"
    mlpl_df = _read("MLPL1.xlsx"); mlpl_df["Entity"] = "MLPL"
    mmw_df  = _read("mmw1.xlsx");  mmw_df["Entity"]  = "MMW"
    mmpl_df = _read("mmpl1.xlsx"); mmpl_df["Entity"] = "MMPL"

    combined = pd.concat([mepl_df, mlpl_df, mmw_df, mmpl_df], ignore_index=True)

    # Clean up column names: strip whitespace, replace non-breaking spaces, collapse multiple spaces
    combined.columns = (
        combined.columns
        .str.strip()
        .str.replace("\xa0", " ", regex=False)
        .str.replace(" +", " ", regex=True)
    )
    combined.rename(columns=lambda c: c.strip(), inplace=True)
    return combined

# Load (and cache) the combined DataFrame
df = load_and_combine_data()

# ------------------------------------
#  1b) Mapping Workbook (upload or default on disk)
# ------------------------------------
with st.sidebar:
    st.header("🧭 Mapping Source")
    map_file = st.file_uploader("Upload mapping workbook (Dept/Subcategory ⇄ Budget Codes)", type=["xlsx", "xls", "csv"])
    use_default_map = st.checkbox("Use default on-disk mapping (MainDept_SubCat_BudgetCodes_...xlsx)", value=True)

if map_file is not None:
    if map_file.name.lower().endswith(".csv"):
        map_raw = pd.read_csv(map_file)
    else:
        map_raw = _load_excel_first_sheet(map_file)
elif use_default_map:
    # try default file name from your share; adjust if you change the file name
    try:
        map_raw = _load_excel_first_sheet("MainDept_SubCat_BudgetCodes_20250927_191809.xlsx")
    except Exception as e:
        map_raw = None
        st.sidebar.warning(f"Couldn't load default mapping: {e}")
else:
    map_raw = None

if map_raw is None:
    st.warning("Upload a mapping workbook or enable the default mapping to proceed with Department/Subcategory mapping.")
else:
    mapping_tidy = _build_tidy_mapping(map_raw)
    st.sidebar.success(f"Loaded mapping: {len(mapping_tidy):,} unique Budget Codes")

# ------------------------------------
#  2) Clean & Prepare Date Columns
# ------------------------------------
for date_col in ["PR Date Submitted", "Po create Date"]:
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.date
    else:
        st.error(f"❌ Column '{date_col}' not found. Please verify your Excel sheets.")

# ------------------------------------
#  3) Buyer Group Classification
# ------------------------------------
if "Buyer Group" in df.columns:
    df["Buyer Group Code"] = (
        df["Buyer Group"].astype(str).str.extract(r"(\d+)").astype(float)
    )
    def classify_buyer_group(row):
        bg   = row["Buyer Group"]
        code = row["Buyer Group Code"]
        if bg in ["ME_BG17", "MLBG16"]:
            return "Direct"
        elif bg in ["Not Available"] or pd.isna(bg):
            return "Indirect"
        elif pd.notna(code) and (1 <= code <= 9):
            return "Direct"
        elif pd.notna(code) and (10 <= code <= 18):
            return "Indirect"
        else:
            return "Other"
    df["Buyer.Type"] = df.apply(classify_buyer_group, axis=1)
else:
    df["Buyer.Type"] = "Unknown"
    st.warning("⚠️ 'Buyer Group' column not found. All Buyer.Type set to 'Unknown'.")

# ------------------------------------
#  4) PO Orderer → PO.Creator Mapping
# ------------------------------------
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
    "N/A": "Dilip"
}
df["PO Orderer"] = df.get("PO Orderer", pd.Series(index=df.index)).fillna("N/A").astype(str).str.strip()
df["PO.Creator"] = df["PO Orderer"].map(o_created_by_map).fillna(df["PO Orderer"])
df["PO.Creator"] = df["PO.Creator"].replace({"N/A": "Dilip"})

indirect_buyers = ["Aatish", "Deepak", "Deepakex", "Dhruv", "Dilip", "Mukul", "Nayan", "Paurik", "Kamlesh", "Suresh"]
df["PO.BuyerType"] = df["PO.Creator"].apply(lambda x: "Indirect" if x in indirect_buyers else "Direct")

# ------------------------------------
#  5) Keyword Search Suggestions List
# ------------------------------------
all_suggestions = []
if "PR Number" in df.columns:
    all_suggestions.extend(df["PR Number"].dropna().astype(str).unique().tolist())
if "Purchase Doc" in df.columns:
    all_suggestions.extend(df["Purchase Doc"].dropna().astype(str).unique().tolist())
if "Product Name" in df.columns:
    all_suggestions.extend(df["Product Name"].dropna().astype(str).unique().tolist())
all_suggestions = list(dict.fromkeys(all_suggestions))

# ------------------------------------
#  6) Budget Code Mapping (Department/Subcategory)  **NEW**
# ------------------------------------
# Detect budget code column in transaction data
budget_code_col = _pick_col(df, [r"\bpo\s*budget\s*code\b", r"\bbudget\s*code\b", r"^code$", r"^budget$"]) or "PO Budget Code"
if budget_code_col not in df.columns:
    df["BudgetCode"] = np.nan
else:
    df["BudgetCode"] = df[budget_code_col].map(_normalize_code)

if map_raw is not None:
    mapping_tidy["BudgetCode"] = mapping_tidy["BudgetCode"].astype("object")
    df["BudgetCode"] = df["BudgetCode"].astype("object")
    df = df.merge(mapping_tidy, on="BudgetCode", how="left")  # adds Department, Subcategory

    # Unmapped list (present in data but no Department in mapping)
    unmapped_mask = df["BudgetCode"].notna() & df["Department"].isna()
    unmapped_codes = (
        df.loc[unmapped_mask, "BudgetCode"]
        .dropna().astype(str).drop_duplicates().sort_values()
        .to_frame(name="UnmappedBudgetCode")
    )
else:
    df["Department"] = np.nan
    df["Subcategory"] = np.nan
    unmapped_codes = pd.DataFrame(columns=["UnmappedBudgetCode"])

# ------------------------------------
#  7) Sidebar Filters (robust)
# ------------------------------------
st.sidebar.header("🔍 Filters (robust)")

# Ensure key columns exist
for col in ["Buyer.Type", "Entity", "PO.Creator", "PO.BuyerType", "PR Date Submitted"]:
    if col not in df.columns:
        df[col] = pd.NA

for col in ["Buyer.Type", "Entity", "PO.Creator", "PO.BuyerType"]:
    df[col] = df[col].astype(str).fillna("").apply(lambda x: x.strip() if x is not None else "")

buyer_options = _safe_unique_vals(df["Buyer.Type"]) if "Buyer.Type" in df.columns else []
entity_options = _safe_unique_vals(df["Entity"]) if "Entity" in df.columns else []
orderer_options = _safe_unique_vals(df["PO.Creator"]) if "PO.Creator" in df.columns else []
po_buyer_type_options = _safe_unique_vals(df["PO.BuyerType"]) if "PO.BuyerType" in df.columns else []

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

# Date preference for filtering
date_col_for_filter = "PR Date Submitted" if "PR Date Submitted" in df.columns else ("Po create Date" if "Po create Date" in df.columns else None)
if date_col_for_filter:
    df[date_col_for_filter] = pd.to_datetime(df[date_col_for_filter], errors="coerce")
    min_date = df[date_col_for_filter].min().date() if pd.notna(df[date_col_for_filter].min()) else date.today()
    max_date = df[date_col_for_filter].max().date() if pd.notna(df[date_col_for_filter].max()) else date.today()
    date_range = st.sidebar.date_input("Date Range", [min_date, max_date])
else:
    date_range = None

# Apply filters
filtered_df = df.copy()
if "PR Date Submitted" in filtered_df.columns:
    filtered_df["PR Date Submitted"] = pd.to_datetime(filtered_df["PR Date Submitted"], errors="coerce")
    filtered_df = filtered_df[(filtered_df["PR Date Submitted"] >= pr_start) & (filtered_df["PR Date Submitted"] <= pr_end)]

if date_range and date_col_for_filter:
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        s_dt, e_dt = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        filtered_df = filtered_df[(filtered_df[date_col_for_filter] >= s_dt) & (filtered_df[date_col_for_filter] <= e_dt)]

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
st.sidebar.write("Row count after filters:", len(filtered_df))

# ------------------------------------
#  8) Keyword Search (simple contains)
# ------------------------------------
st.markdown("## 🔍 Keyword Search")
valid_columns = [c for c in ["PR Number", "Purchase Doc", "Product Name", "PO Vendor"] if c in df.columns]
search_pool = []
if valid_columns:
    tmp = df[valid_columns].fillna("").astype(str)
    search_pool = (tmp.apply(lambda r: " | ".join(r.values), axis=1).str.lower())

if "search_history" not in st.session_state:
    st.session_state.search_history = []
user_query = st.text_input("Start typing a keyword (e.g., vendor, product, PO, PR...)", "")

if user_query and user_query not in st.session_state.search_history:
    st.session_state.search_history.append(user_query)
if st.session_state.search_history:
    with st.expander("🕘 Search History"):
        st.write(st.session_state.search_history[-10:])

if user_query and len(search_pool):
    idx = search_pool[search_pool.str.contains(user_query.lower(), na=False)].index
    result_df = df.loc[idx]
    st.markdown(f"### 🔎 Found {len(result_df)} matching results:")
    st.dataframe(result_df, use_container_width=True)

    def _csv(dataframe): return dataframe.to_csv(index=False).encode("utf-8")
    def _xlsx(dataframe):
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
            dataframe.to_excel(writer, index=False, sheet_name="Search_Results")
        return out.getvalue()
    st.download_button("⬇️ Download CSV", _csv(result_df), file_name="search_results.csv", mime="text/csv")
    st.download_button("⬇️ Download Excel", _xlsx(result_df), file_name="search_results.xlsx")

# ------------------------------------
#  9) Top KPI Row
# ------------------------------------
st.title("📊 Procure-to-Pay Dashboard")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total PRs",        filtered_df.get("PR Number", pd.Series(dtype=object)).nunique())
col2.metric("Total POs",        filtered_df.get("Purchase Doc", pd.Series(dtype=object)).nunique())
col3.metric("Line Items",       len(filtered_df))
col4.metric("Entities",         filtered_df.get("Entity", pd.Series(dtype=object)).nunique())
col5.metric("Spend (Cr ₹)",     f"{filtered_df.get('Net Amount', pd.Series(dtype=float)).sum() / 1e7:,.2f}")

# ------------------------------------
# 10) SLA Compliance Gauge (PR → PO ≤ 7 days)
# ------------------------------------
st.subheader("🎯 SLA Compliance (PR → PO ≤ 7 days)")
lead_df = filtered_df[filtered_df.get("Po create Date").notna()] if "Po create Date" in filtered_df.columns else filtered_df.iloc[0:0].copy()
if not lead_df.empty and "PR Date Submitted" in lead_df.columns:
    lead_df = lead_df.copy()
    lead_df["Lead Time (Days)"] = (
        pd.to_datetime(lead_df["Po create Date"]) - pd.to_datetime(lead_df["PR Date Submitted"])
    ).dt.days
    SLA_DAYS = 7
    avg_lead = float(np.nanmean(lead_df["Lead Time (Days)"])) if len(lead_df) else 0.0
    avg_lead = round(avg_lead, 1)

    gauge_fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=avg_lead,
            number={"suffix": " days"},
            gauge={
                "axis": {"range": [0, max(14, avg_lead * 1.2 if avg_lead > 0 else 14)]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, SLA_DAYS], "color": "lightgreen"},
                    {"range": [SLA_DAYS, max(14, avg_lead * 1.2 if avg_lead > 0 else 14)], "color": "lightcoral"},
                ],
                "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": SLA_DAYS},
            },
            title={"text": "Average Lead Time"},
        )
    )
    st.plotly_chart(gauge_fig, use_container_width=True)
    st.caption(f"Current Avg Lead Time: {avg_lead:.1f} days   •   Target ≤ {SLA_DAYS} days")
else:
    st.info("Not enough data to compute PR→PO lead time.")

# ---------- Monthly Total Spend (bars) + Cumulative Spend line ----------
st.subheader("📊 Monthly Total Spend (with Cumulative Value Line)")
date_col = "Po create Date" if "Po create Date" in filtered_df.columns else ("PR Date Submitted" if "PR Date Submitted" in filtered_df.columns else None)
if date_col is None:
    st.info("No date column available ('Po create Date' or 'PR Date Submitted') to compute monthly spend.")
else:
    filtered_df[date_col] = pd.to_datetime(filtered_df[date_col], errors="coerce")
    temp = filtered_df.dropna(subset=[date_col])
    if "Net Amount" not in temp.columns:
        st.info("No 'Net Amount' column found to compute spend.")
    else:
        temp = temp.copy()
        temp["PO_Month"] = temp[date_col].dt.to_period("M").dt.to_timestamp()
        temp["Month_Str"] = temp["PO_Month"].dt.strftime("%b-%Y")

        monthly_total_spend = temp.groupby(["PO_Month", "Month_Str"], as_index=False)["Net Amount"].sum()
        monthly_total_spend["Spend (Cr ₹)"] = monthly_total_spend["Net Amount"] / 1e7
        monthly_total_spend = monthly_total_spend.sort_values("PO_Month").reset_index(drop=True)
        monthly_total_spend["Cumulative Spend (Cr ₹)"] = monthly_total_spend["Spend (Cr ₹)"].cumsum()
        month_order = monthly_total_spend["Month_Str"].tolist()
        monthly_total_spend["Month_Str"] = pd.Categorical(monthly_total_spend["Month_Str"], categories=month_order, ordered=True)

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(
                x=monthly_total_spend["Month_Str"],
                y=monthly_total_spend["Spend (Cr ₹)"],
                name="Monthly Spend (Cr ₹)",
                text=monthly_total_spend["Spend (Cr ₹)"].map("{:.2f}".format),
                textposition="outside",
                marker=dict(opacity=0.85),
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=monthly_total_spend["Month_Str"],
                y=monthly_total_spend["Cumulative Spend (Cr ₹)"],
                mode="lines+markers",
                name="Cumulative Spend (Cr ₹)",
                line=dict(width=3, color="darkblue"),
                marker=dict(size=8),
                hovertemplate="₹ %{y:.2f} Cr<br>%{x}<extra></extra>",
            ),
            secondary_y=True,
        )
        fig.update_layout(
            title="Monthly Total Spend with Cumulative Value",
            xaxis=dict(tickangle=-45),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=70, b=120),
        )
        fig.update_yaxes(title_text="Monthly Spend (Cr ₹)", secondary_y=False)
        fig.update_yaxes(title_text="Cumulative Spend (Cr ₹)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------
# Monthly Spend Trend by Entity
# ------------------------------------
st.subheader("💹 Monthly Spend Trend by Entity")
spend_df = filtered_df.copy()
if "Po create Date" in spend_df.columns:
    spend_df["PO Month"] = pd.to_datetime(spend_df["Po create Date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    monthly_spend = (
        spend_df.dropna(subset=["PO Month"])
        .groupby(["PO Month", "Entity"], as_index=False)["Net Amount"]
        .sum()
    )
    monthly_spend["Spend (Cr ₹)"] = monthly_spend["Net Amount"] / 1e7
    monthly_spend["Month_Str"] = monthly_spend["PO Month"].dt.strftime("%b-%Y")
    fig_spend = px.line(
        monthly_spend, x="Month_Str", y="Spend (Cr ₹)", color="Entity",
        markers=True, title="Monthly Spend Trend by Entity",
        labels={"Month_Str": "Month", "Spend (Cr ₹)": "Spend (Cr ₹)"},
    )
    fig_spend.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_spend, use_container_width=True)

# ------------------------------------
# PR → PO Lead Time by Buyer Type & Buyer
# ------------------------------------
st.subheader("⏱️ PR to PO Lead Time by Buyer Type & by Buyer")
if "Lead Time (Days)" not in lead_df.columns and not lead_df.empty and "PR Date Submitted" in lead_df.columns:
    lead_df = lead_df.copy()
    lead_df["Lead Time (Days)"] = (
        pd.to_datetime(lead_df["Po create Date"]) - pd.to_datetime(lead_df["PR Date Submitted"])
    ).dt.days

if not lead_df.empty and "Lead Time (Days)" in lead_df.columns:
    lead_avg_by_type = lead_df.groupby("Buyer.Type")["Lead Time (Days)"].mean().round(0).reset_index()
    lead_avg_by_buyer = lead_df.groupby("PO.Creator")["Lead Time (Days)"].mean().round(0).reset_index()
    c1, c2 = st.columns(2)
    c1.dataframe(lead_avg_by_type, use_container_width=True)
    c2.dataframe(lead_avg_by_buyer, use_container_width=True)

# ------------------------------------
# Monthly PR & PO Trends
# ------------------------------------
st.subheader("📅 Monthly PR & PO Trends")
if "PR Date Submitted" in filtered_df.columns and "Po create Date" in filtered_df.columns:
    filtered_df["PR Month"] = pd.to_datetime(filtered_df["PR Date Submitted"]).dt.to_period("M")
    filtered_df["PO Month"] = pd.to_datetime(filtered_df["Po create Date"]).dt.to_period("M")

    monthly_summary = (
        filtered_df.groupby("PR Month").agg({"PR Number": "count", "Purchase Doc": "count"}).reset_index()
    )
    monthly_summary.columns = ["Month", "PR Count", "PO Count"]
    monthly_summary["Month"] = monthly_summary["Month"].astype(str)
    st.line_chart(monthly_summary.set_index("Month"), use_container_width=True)

# ------------------------------------
# PR → PO Aging Buckets
# ------------------------------------
st.subheader("🧮 PR to PO Aging Buckets")
if not lead_df.empty and "Lead Time (Days)" in lead_df.columns:
    bins = [0, 7, 15, 30, 60, 90, 999]
    labels = ["0-7", "8-15", "16-30", "31-60", "61-90", "90+"]
    aging_buckets = pd.cut(lead_df["Lead Time (Days)"], bins=bins, labels=labels)
    age_summary = (aging_buckets.value_counts(normalize=True).sort_index().reset_index())
    age_summary.columns = ["Aging Bucket", "Percentage"]
    age_summary["Percentage"] *= 100
    fig_aging = px.bar(
        age_summary, x="Aging Bucket", y="Percentage", text="Percentage",
        title="PR to PO Aging Bucket Distribution (%)",
        labels={"Percentage": "Percentage (%)"},
    )
    fig_aging.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    st.plotly_chart(fig_aging, use_container_width=True)

# ------------------------------------
# PRs & POs by Weekday
# ------------------------------------
st.subheader("📆 PRs and POs by Weekday")
df_wd = filtered_df.copy()
if "PR Date Submitted" in df_wd.columns:
    df_wd["PR Weekday"] = pd.to_datetime(df_wd["PR Date Submitted"]).dt.day_name()
if "Po create Date" in df_wd.columns:
    df_wd["PO Weekday"] = pd.to_datetime(df_wd["Po create Date"]).dt.day_name()

if "PR Weekday" in df_wd.columns:
    pr_counts = df_wd["PR Weekday"].value_counts().reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], fill_value=0,
    )
    st.bar_chart(pr_counts, use_container_width=True)
if "PO Weekday" in df_wd.columns:
    po_counts = df_wd["PO Weekday"].value_counts().reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], fill_value=0,
    )
    st.bar_chart(po_counts, use_container_width=True)

# ------------------------------------
# Open PRs (Approved / InReview)
# ------------------------------------
st.subheader("⚠️ Open PRs (Approved/InReview)")
if "PR Status" in filtered_df.columns and "PR Date Submitted" in filtered_df.columns:
    open_df = filtered_df[filtered_df["PR Status"].isin(["Approved", "InReview"])].copy()
    if not open_df.empty:
        open_df["Pending Age (Days)"] = (pd.to_datetime(pd.Timestamp.today().date()) - pd.to_datetime(open_df["PR Date Submitted"])).dt.days
        open_summary = (
            open_df.groupby("PR Number")
            .agg({
                "PR Date Submitted":   "first",
                "Pending Age (Days)":  "first",
                "Procurement Category":"first",
                "Product Name":        "first",
                "Net Amount":          "sum",
                "PO Budget Code":      "first" if "PO Budget Code" in open_df.columns else "first",
                "PR Status":           "first",
                "Buyer Group":         "first",
                "Buyer.Type":          "first",
                "Entity":              "first",
                "PO.Creator":          "first",
                "Purchase Doc":        "first",
            })
            .reset_index()
        )
        st.metric("🔢 Open PRs", open_summary["PR Number"].nunique())
        open_monthly_counts = pd.to_datetime(open_summary["PR Date Submitted"]).dt.to_period("M").value_counts().sort_index()
        st.bar_chart(open_monthly_counts, use_container_width=True)

        def _hl_age(v): return "background-color: red" if v > 30 else ""
        st.dataframe(open_summary.style.applymap(_hl_age, subset=["Pending Age (Days)"]), use_container_width=True)

        st.subheader("🏢 Open PRs by Entity")
        ent_counts = open_summary["Entity"].value_counts().reset_index()
        ent_counts.columns = ["Entity", "Count"]
        st.bar_chart(ent_counts.set_index("Entity"), use_container_width=True)

# ------------------------------------
# Daily PR Trends
# ------------------------------------
st.subheader("📅 Daily PR Trends")
if "PR Date Submitted" in filtered_df.columns:
    daily_df = filtered_df.copy()
    daily_df["PR Date"] = pd.to_datetime(daily_df["PR Date Submitted"])
    daily_trend = daily_df.groupby("PR Date").size().reset_index(name="PR Count")
    fig_daily = px.line(daily_trend, x="PR Date", y="PR Count", title="Daily PR Submissions", labels={"PR Count": "PR Count"})
    st.plotly_chart(fig_daily, use_container_width=True)

# ------------------------------------
# Buyer-wise Spend
# ------------------------------------
st.subheader("💰 Buyer-wise Spend (Cr ₹)")
if all(c in filtered_df.columns for c in ["PO.Creator", "Net Amount"]):
    buyer_spend = (
        filtered_df.groupby("PO.Creator")["Net Amount"].sum().sort_values(ascending=False).reset_index()
    )
    buyer_spend["Net Amount (Cr)"] = buyer_spend["Net Amount"] / 1e7
    fig_buyer = px.bar(
        buyer_spend, x="PO.Creator", y="Net Amount (Cr)", title="Spend by Buyer",
        labels={"Net Amount (Cr)": "Spend (Cr ₹)", "PO.Creator": "Buyer"}, text="Net Amount (Cr)"
    )
    fig_buyer.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    st.plotly_chart(fig_buyer, use_container_width=True)

# ------------------------------------
# Category Spend
# ------------------------------------
if "Procurement Category" in filtered_df.columns and "Net Amount" in filtered_df.columns:
    st.subheader("🗂️ Spend by Category (Descending)")
    cat_spend = filtered_df.groupby("Procurement Category")["Net Amount"].sum().sort_values(ascending=False).reset_index()
    cat_spend["Spend (Cr ₹)"] = cat_spend["Net Amount"] / 1e7
    fig_cat = px.bar(
        cat_spend, x="Procurement Category", y="Spend (Cr ₹)",
        title="Spend by Category (Descending)",
        labels={"Spend (Cr ₹)": "Spend (Cr ₹)", "Procurement Category": "Category"},
    )
    fig_cat.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_cat, use_container_width=True)

# ------------------------------------
# PO Approval Summary
# ------------------------------------
if "PO Approved Date" in filtered_df.columns and "Po create Date" in filtered_df.columns and "Purchase Doc" in filtered_df.columns:
    st.subheader("📋 PO Approval Summary")
    po_app_df = filtered_df[filtered_df["Po create Date"].notna()].copy()
    po_app_df["PO Approved Date"] = pd.to_datetime(po_app_df["PO Approved Date"], errors="coerce")
    total_pos    = po_app_df["Purchase Doc"].nunique()
    approved_pos = po_app_df[po_app_df["PO Approved Date"].notna()]["Purchase Doc"].nunique()
    pending_pos  = total_pos - approved_pos
    po_app_df["PO Approval Lead Time"] = (po_app_df["PO Approved Date"] - pd.to_datetime(po_app_df["Po create Date"])).dt.days
    avg_approval = float(np.nanmean(po_app_df["PO Approval Lead Time"])) if len(po_app_df) else np.nan
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📦 Total POs", total_pos)
    c2.metric("✅ Approved POs", approved_pos)
    c3.metric("⏳ Pending Approval", pending_pos)
    c4.metric("⏱️ Avg Approval Lead Time (days)", None if np.isnan(avg_approval) else round(avg_approval, 1))
    st.subheader("📄 Detailed PO Approval Aging List")
    approval_detail = po_app_df[["PO.Creator", "Purchase Doc", "Po create Date", "PO Approved Date", "PO Approval Lead Time"]].sort_values(by="PO Approval Lead Time", ascending=False)
    st.dataframe(approval_detail, use_container_width=True)

# ------------------------------------
# PO Status Breakdown
# ------------------------------------
if "PO Status" in filtered_df.columns:
    st.subheader("📊 PO Status Breakdown")
    po_status_summary = filtered_df["PO Status"].value_counts().reset_index()
    po_status_summary.columns = ["PO Status", "Count"]
    c1, c2 = st.columns([2, 3])
    with c1:
        st.dataframe(po_status_summary, use_container_width=True)
    with c2:
        fig_status = px.pie(po_status_summary, names="PO Status", values="Count", title="PO Status Distribution", hole=0.3)
        fig_status.update_traces(textinfo="percent+label")
        st.plotly_chart(fig_status, use_container_width=True)

# ------------------------------------
# PO Delivery Summary: Received vs Pending
# ------------------------------------
st.subheader("🚚 PO Delivery Summary: Received vs Pending")
delivery_df = filtered_df.rename(columns={
    "PO Quantity": "PO Qty",
    "ReceivedQTY": "Received Qty",
    "Pending QTY": "Pending Qty"
}).copy()
if "PO Qty" in delivery_df.columns and "Received Qty" in delivery_df.columns:
    delivery_df["% Received"] = (delivery_df["Received Qty"] / delivery_df["PO Qty"]) * 100
    delivery_df["% Received"] = delivery_df["% Received"].fillna(0).round(1)
    po_delivery_summary = (
        delivery_df.groupby(["Purchase Doc", "PO Vendor", "Product Name", "Item Description"], dropna=False)
        .agg({"PO Qty":"sum","Received Qty":"sum","Pending Qty":"sum","% Received":"mean"}).reset_index()
    )
    st.dataframe(po_delivery_summary.sort_values(by="Pending Qty", ascending=False), use_container_width=True)
    fig_pending = px.bar(
        po_delivery_summary.sort_values(by="Pending Qty", ascending=False).head(20),
        x="Purchase Doc", y="Pending Qty", color="PO Vendor",
        hover_data=["Product Name", "Item Description"],
        title="Top 20 POs Awaiting Delivery (Pending Qty)", text="Pending Qty",
    )
    fig_pending.update_traces(textposition="outside")
    st.plotly_chart(fig_pending, use_container_width=True)

    total_po_lines    = len(delivery_df)
    fully_received    = (delivery_df.get("Pending Qty", pd.Series(0, index=delivery_df.index)) == 0).sum()
    partially_pending = (delivery_df.get("Pending Qty", pd.Series(0, index=delivery_df.index)) > 0).sum()
    avg_receipt_pct   = delivery_df["% Received"].mean().round(1)
    st.markdown("### 📋 Delivery Performance Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PO Lines",        total_po_lines)
    c2.metric("Fully Delivered", int(fully_received))
    c3.metric("Pending Delivery", int(partially_pending))
    c4.metric("Avg. Receipt %",   f"{avg_receipt_pct}%")

    st.download_button(
        "📥 Download Delivery Status",
        data=po_delivery_summary.to_csv(index=False),
        file_name="PO_Delivery_Status.csv",
        mime="text/csv"
    )

# ------------------------------------
# Top 50 Pending Delivery Lines by Value
# ------------------------------------
st.subheader("📋 Top 50 Pending Lines (by Value)")
if "Pending Qty" in delivery_df.columns and "PO Unit Rate" in delivery_df.columns:
    pending_items = delivery_df[delivery_df["Pending Qty"] > 0].copy()
    pending_items["Pending Value"] = pending_items["Pending Qty"] * pending_items["PO Unit Rate"]
    top_pending_items = (
        pending_items.sort_values(by="Pending Value", ascending=False)
        .head(50)[["PR Number","Purchase Doc","Procurement Category","Buying legal entity","PR Budget description",
                   "Product Name","Item Description","Pending Qty","Pending Value"]]
        .reset_index(drop=True)
    )
    st.dataframe(top_pending_items.style.format({"Pending Qty":"{:,.0f}","Pending Value":"₹ {:,.2f}"}), use_container_width=True)

# ------------------------------------
# Top 10 Vendors by Spend
# ------------------------------------
st.subheader("🏆 Top 10 Vendors by Spend (Cr ₹)")
if all(c in filtered_df.columns for c in ["PO Vendor", "Purchase Doc", "Net Amount"]):
    vendor_spend = (
        filtered_df.groupby("PO Vendor", dropna=False)
        .agg(Vendor_PO_Count=("Purchase Doc","nunique"),
             Total_Spend_Cr=("Net Amount", lambda x: (x.sum()/1e7).round(2)))
        .reset_index()
        .sort_values(by="Total_Spend_Cr", ascending=False)
    )
    top10_spend = vendor_spend.head(10).copy()
    st.dataframe(top10_spend, use_container_width=True)
    fig_top_vendors = px.bar(
        top10_spend, x="PO Vendor", y="Total_Spend_Cr",
        title="Top 10 Vendors by Spend (Cr ₹)", labels={"Total_Spend_Cr":"Spend (Cr ₹)","PO Vendor":"Vendor"},
        text="Total_Spend_Cr",
    )
    fig_top_vendors.update_traces(textposition="outside", texttemplate="%{text:.2f}")
    st.plotly_chart(fig_top_vendors, use_container_width=True)

# ------------------------------------
# Vendor Delivery Performance
# ------------------------------------
st.subheader("📊 Vendor Delivery Performance (Top 10 by Spend)")
if all(c in filtered_df.columns for c in ["PO Vendor", "Purchase Doc"]) and "Pending QTY" in df.columns or "Pending Qty" in df.columns:
    df_vp = filtered_df.copy()
    pend_col = "Pending QTY" if "Pending QTY" in df_vp.columns else ("Pending Qty" if "Pending Qty" in df_vp.columns else None)
    if pend_col:
        df_vp["Pending Qty Filled"] = df_vp[pend_col].fillna(0).astype(float)
        df_vp["Is_Fully_Delivered"] = df_vp["Pending Qty Filled"] == 0
        if "PO Delivery Date" in df_vp.columns:
            today = pd.Timestamp.today().normalize().date()
            df_vp["PO Delivery Date"] = pd.to_datetime(df_vp["PO Delivery Date"], errors="coerce")
            df_vp["Is_Late"] = (df_vp["PO Delivery Date"].dt.date.notna()
                                & (df_vp["PO Delivery Date"].dt.date < today)
                                & (df_vp["Pending Qty Filled"] > 0))
        else:
            df_vp["Is_Late"] = False

        vendor_perf = (
            df_vp.groupby("PO Vendor", dropna=False)
            .agg(Total_PO_Count=("Purchase Doc","nunique"),
                 Fully_Delivered_PO_Count=("Is_Fully_Delivered","sum"),
                 Late_PO_Count=("Is_Late","sum"))
            .reset_index()
        )
        vendor_perf["Pct_Fully_Delivered"] = (vendor_perf["Fully_Delivered_PO_Count"]/vendor_perf["Total_PO_Count"]*100).round(1)
        vendor_perf["Pct_Late"] = (vendor_perf["Late_PO_Count"]/vendor_perf["Total_PO_Count"]*100).round(1)

        # Merge spend if available
        try:
            vendor_perf = vendor_perf.merge(vendor_spend[["PO Vendor","Total_Spend_Cr"]], on="PO Vendor", how="left")
            top10_vendor_perf = vendor_perf.sort_values("Total_Spend_Cr", ascending=False).head(10)
        except Exception:
            top10_vendor_perf = vendor_perf.sort_values("Total_PO_Count", ascending=False).head(10)
            top10_vendor_perf["Total_Spend_Cr"] = None

        st.dataframe(
            top10_vendor_perf[["PO Vendor","Total_PO_Count","Fully_Delivered_PO_Count","Late_PO_Count","Pct_Fully_Delivered","Pct_Late","Total_Spend_Cr"]],
            use_container_width=True,
        )
        melted_perf = top10_vendor_perf.melt(
            id_vars=["PO Vendor"], value_vars=["Pct_Fully_Delivered","Pct_Late"],
            var_name="Metric", value_name="Percentage",
        )
        fig_vendor_perf = px.bar(
            melted_perf, x="PO Vendor", y="Percentage", color="Metric", barmode="group",
            title="% Fully Delivered vs % Late (Top 10 Vendors by Spend)", labels={"Percentage":"% of POs","PO Vendor":"Vendor"},
        )
        st.plotly_chart(fig_vendor_perf, use_container_width=True)

# ------------------------------------
# Monthly Unique PO Generation
# ------------------------------------
st.subheader("🗓️ Monthly Unique PO Generation")
if "Purchase Doc" in filtered_df.columns and "Po create Date" in filtered_df.columns:
    po_monthly = filtered_df[filtered_df["Purchase Doc"].notna()].copy()
    po_monthly["PO Month"] = pd.to_datetime(po_monthly["Po create Date"]).dt.to_period("M")
    monthly_po_counts = po_monthly.groupby("PO Month")["Purchase Doc"].nunique().reset_index(name="Unique PO Count")
    monthly_po_counts["PO Month"] = monthly_po_counts["PO Month"].astype(str)
    fig_monthly_po = px.bar(
        monthly_po_counts, x="PO Month", y="Unique PO Count",
        title="Monthly Unique PO Generation",
        labels={"PO Month":"Month","Unique PO Count":"Number of Unique POs"},
        text="Unique PO Count",
    )
    fig_monthly_po.update_traces(textposition="outside")
    st.plotly_chart(fig_monthly_po, use_container_width=True)

# ------------------------------------
#  Budget Mapping Outputs / Downloads
# ------------------------------------
st.markdown("---")
st.subheader("🧾 Budget Code Mapping Outputs")

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Unique Budget Codes in Data", int(df["BudgetCode"].dropna().nunique()))
with c2:
    st.metric("Mapped Codes", int(df.loc[df["Department"].notna() & df["BudgetCode"].notna(), "BudgetCode"].nunique()))
with c3:
    st.metric("Unmapped Codes", int(unmapped_codes.shape[0]))

# download buttons
def _dl_csv_button(label, df_, fname):
    st.download_button(
        label, df_.to_csv(index=False).encode("utf-8"),
        file_name=fname, mime="text/csv"
    )

dl_col1, dl_col2, dl_col3 = st.columns(3)
with dl_col1:
    if map_raw is not None:
        _dl_csv_button("⬇️ Master Mapping (CSV)", mapping_tidy, "Master_BudgetCode_Department_Subcategory_Mapping.csv")
with dl_col2:
    _dl_csv_button("⬇️ All Transactions + Mapping (CSV)", df, "AllCompany_Transactions_WithMapping.csv")
with dl_col3:
    _dl_csv_button("⬇️ Unmapped Budget Codes (CSV)", unmapped_codes, "Unmapped_BudgetCodes_List.csv")

with st.expander("🧩 Unmapped Budget Codes (by Entity)"):
    if not unmapped_codes.empty:
        by_company = (
            df[df["BudgetCode"].isin(unmapped_codes["UnmappedBudgetCode"])]
            .groupby(["Entity","BudgetCode"])
            .size().reset_index(name="Count")
            .sort_values(["Entity","Count"], ascending=[True, False])
        )
        st.dataframe(by_company, use_container_width=True)
        _dl_csv_button("⬇️ Unmapped by Entity (CSV)", by_company, "Unmapped_BudgetCodes_ByEntity.csv")
    else:
        st.success("All budget codes in data are mapped. Nice.")

st.success("Done. Use the download buttons above to export clean CSVs.")
