# p2p_dashboard.py — polished & bug-fixed
# Streamlit Procure-to-Pay Dashboard

from __future__ import annotations

import io
import re
import base64
from datetime import date, datetime

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ====================================
#  Procure-to-Pay Dashboard (Streamlit)
# ====================================

# --- 0) Page Configuration ---
st.set_page_config(
    page_title="Procure-to-Pay Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
    # NOTE: Filenames must match exactly (case-sensitive on some platforms).
    mepl_df = pd.read_excel("MEPL1.xlsx", skiprows=1)
    mlpl_df = pd.read_excel("MLPL1.xlsx", skiprows=1)
    mmw_df  = pd.read_excel("mmw1.xlsx",  skiprows=1)
    mmpl_df = pd.read_excel("mmpl1.xlsx", skiprows=1)

    # Tag each sheet with an "Entity" column
    mepl_df["Entity"] = "MEPL"
    mlpl_df["Entity"] = "MLPL"
    mmw_df["Entity"]  = "MMW"
    mmpl_df["Entity"] = "MMPL"

    # Concatenate into a single DataFrame
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
#  1b) Load Budget → Department/Subcategory mapping & merge
# ------------------------------------
@st.cache_data(show_spinner=False)
def load_budget_mapping(github_url: str | None = None) -> pd.DataFrame | None:
    """Loads the budget mapping file and returns a normalized mapping.
    Tries (in order): user-uploaded file, local working dir, /mnt/data, and optional GitHub URL.
    Expected columns (case-insensitive variants handled):
      - Budget Code (or PO Budget Code)
      - Department (or Dept)
      - Subcategory (optional)
    """
    # 1) Uploader (user can override at runtime)
    up = st.sidebar.file_uploader("📄 Upload Budget Mapping (.xlsx)", type=["xlsx"], key="budget_map")
    if up is not None:
        try:
            m = pd.read_excel(up)
            return m
        except Exception as e:
            st.sidebar.error(f"Failed to read uploaded mapping: {e}")

    # 2) Local common paths
    local_candidates = [
        "Final_Budget_Mapping_Completed_Verified.xlsx",
        "/mnt/data/Final_Budget_Mapping_Completed_Verified.xlsx",
    ]
    for p in local_candidates:
        try:
            return pd.read_excel(p)
        except Exception:
            pass

    # 3) GitHub URL if provided (we'll auto-convert blob → raw)
    if github_url:
        try:
            raw_url = github_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
            return pd.read_excel(raw_url)
        except Exception as e:
            st.sidebar.warning(f"Could not load mapping from GitHub: {e}")

    return None


def _norm_code(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.strip()
         .str.replace(" ", " ", regex=False)
         .str.replace(" ", "", regex=False)   # drop spaces
         .str.upper()
    )

mapping_df = load_budget_mapping("https://github.com/Pauriktrivedi/MMW/blob/main/Final_Budget_Mapping_Completed_Verified.xlsx")

if mapping_df is not None:
    # Normalize column names
    mapping_df.columns = (
        mapping_df.columns.str.strip().str.replace(" ", " ", regex=False).str.replace(" +", " ", regex=True)
    )
    # Try to identify standard columns
    col_map = {}
    for c in mapping_df.columns:
        lc = c.lower()
        if lc in ["budget code", "po budget code", "budget_code", "code", "po budgetcode"]:
            col_map[c] = "PO Budget Code"
        elif lc in ["department", "dept", "department name", "budget department", "pr department"]:
            col_map[c] = "Department"
        elif lc in ["subcategory", "sub-category", "sub cat", "sub catg", "sub dept", "subdepartment"]:
            col_map[c] = "Subcategory"
    if col_map:
        mapping_df = mapping_df.rename(columns=col_map)

    if "PO Budget Code" in mapping_df.columns:
        mapping_df["PO Budget Code Norm"] = _norm_code(mapping_df["PO Budget Code"]) 
        # Prepare main df code
        if "PO Budget Code" in df.columns:
            df["PO Budget Code Norm"] = _norm_code(df["PO Budget Code"])
            # Left merge
            df = df.merge(
                mapping_df[[c for c in ["PO Budget Code Norm", "Department", "Subcategory"] if c in mapping_df.columns]],
                on="PO Budget Code Norm",
                how="left",
                suffixes=("", ".map"),
            )
            # Place mapped fields with clear names
            if "Department" in df.columns:
                df.rename(columns={"Department": "Department (Mapped)"}, inplace=True)
            if "Subcategory" in df.columns:
                df.rename(columns={"Subcategory": "Subcategory (Mapped)"}, inplace=True)

            # If an in-file department already exists, fill missing from mapped
            for dcol in ["Department", "PR Department", "Budget Department", "Department Name"]:
                if dcol in df.columns and "Department (Mapped)" in df.columns:
                    df[dcol] = df[dcol].fillna(df["Department (Mapped)"])

            # Coverage metric in sidebar
            matched = int(df["PO Budget Code Norm"].notna().sum())
            mapped  = int(df.get("Department (Mapped)", pd.Series(dtype=object)).notna().sum())
            pct = round(mapped * 100 / matched, 1) if matched else 0.0
            st.sidebar.metric("🔗 Budget→Dept mapped", f"{pct}%")
        else:
            st.sidebar.warning("'PO Budget Code' column not found in transaction data. Mapping skipped.")
    else:
        st.sidebar.warning("Mapping file lacks a 'Budget Code' / 'PO Budget Code' column.")
else:
    st.sidebar.info("Upload or place 'Final_Budget_Mapping_Completed_Verified.xlsx' to enable Budget→Dept mapping.")

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
    # Extract numeric code (if present) from strings like "ME_BG17" → 17
    code_series = (
        df["Buyer Group"].astype(str).str.extract(r"(\d+)").squeeze()
    )
    df["Buyer Group Code"] = pd.to_numeric(code_series, errors="coerce")

    def classify_buyer_group(row):
        bg   = row.get("Buyer Group")
        code = row.get("Buyer Group Code")
        if bg in ["ME_BG17", "MLBG16"]:
            return "Direct"
        if (bg == "Not Available") or (pd.isna(bg)):
            return "Indirect"
        if pd.notna(code) and 1 <= code <= 9:
            return "Direct"
        if pd.notna(code) and 10 <= code <= 18:
            return "Indirect"
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
    "N/A": "Dilip",
}

if "PO Orderer" not in df.columns:
    df["PO Orderer"] = "N/A"

df["PO Orderer"] = df["PO Orderer"].fillna("N/A").astype(str).str.strip()
df["PO.Creator"] = df["PO Orderer"].map(o_created_by_map).fillna(df["PO Orderer"])
df["PO.Creator"] = df["PO.Creator"].replace({"N/A": "Dilip"})

indirect_buyers = [
    "Aatish", "Deepak", "Deepakex", "Dhruv", "Dilip",
    "Mukul", "Nayan", "Paurik", "Kamlesh", "Suresh",
]
df["PO.BuyerType"] = df["PO.Creator"].apply(lambda x: "Indirect" if x in indirect_buyers else "Direct")

# ------------------------------------
#  5) Build Keyword Search Suggestions List
# ------------------------------------
all_suggestions: list[str] = []
if "PR Number" in df.columns:
    all_suggestions.extend(df["PR Number"].dropna().astype(str).unique().tolist())
if "Purchase Doc" in df.columns:
    all_suggestions.extend(df["Purchase Doc"].dropna().astype(str).unique().tolist())
if "Product Name" in df.columns:
    all_suggestions.extend(df["Product Name"].dropna().astype(str).unique().tolist())
# Deduplicate:
all_suggestions = list(dict.fromkeys(all_suggestions))

# ------------------------------------
#  7) Sidebar Filters (FY-Based) — robust
# ------------------------------------
st.sidebar.header("🔍 Filters")

# Ensure key columns exist
for col in ["Buyer.Type", "Entity", "PO.Creator", "PO.BuyerType", "PR Date Submitted"]:
    if col not in df.columns:
        df[col] = pd.NA

# Normalize string columns
for col in ["Buyer.Type", "Entity", "PO.Creator", "PO.BuyerType"]:
    df[col] = df[col].astype(str).fillna("").str.strip()

safe_unique = lambda s: sorted([str(x).strip() for x in s.dropna().unique()]) if s is not None else []

buyer_options = safe_unique(df["Buyer.Type"]) if "Buyer.Type" in df.columns else []
entity_options = safe_unique(df["Entity"]) if "Entity" in df.columns else []
orderer_options = safe_unique(df["PO.Creator"]) if "PO.Creator" in df.columns else []
po_buyer_type_options = safe_unique(df["PO.BuyerType"]) if "PO.BuyerType" in df.columns else []

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

# Date range source
date_col_for_filter = (
    "PR Date Submitted" if "PR Date Submitted" in df.columns else (
        "Po create Date" if "Po create Date" in df.columns else None
    )
)
if date_col_for_filter:
    df[date_col_for_filter] = pd.to_datetime(df[date_col_for_filter], errors="coerce")
    min_date = df[date_col_for_filter].min().date() if pd.notna(df[date_col_for_filter].min()) else date.today()
    max_date = df[date_col_for_filter].max().date() if pd.notna(df[date_col_for_filter].max()) else date.today()
    date_range = st.sidebar.date_input("Date Range", [min_date, max_date])
else:
    date_range = None

# Apply filters
filtered_df = df.copy()

# FY filter
if "PR Date Submitted" in filtered_df.columns:
    filtered_df["PR Date Submitted"] = pd.to_datetime(filtered_df["PR Date Submitted"], errors="coerce")
    filtered_df = filtered_df[(filtered_df["PR Date Submitted"] >= pr_start) & (filtered_df["PR Date Submitted"] <= pr_end)]

# Date range filter
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

# Debug: show applied filters & counts
st.sidebar.markdown("----")
st.sidebar.write("Selected FY:", selected_fy)
st.sidebar.write("Selected Buyer Types:", buyer_filter if buyer_filter else "ALL")
st.sidebar.write("Selected Entities:", entity_filter if entity_filter else "ALL")
st.sidebar.write("Row count after filters:", len(filtered_df))

# ------------------------------------
#  8b) Keyword Search with History + tag filters
# ------------------------------------
st.markdown("## 🔍 Keyword Search")

valid_columns = [c for c in ["PR Number", "Purchase Doc", "Product Name", "PO Vendor"] if c in df.columns]
search_data: list[tuple[str,int]] = []
if valid_columns:
    for idx, row in df[valid_columns].fillna("").astype(str).iterrows():
        combined = " | ".join(row[c] for c in valid_columns)
        search_data.append((combined.lower(), idx))

# Memory persistence
if "search_history" not in st.session_state:
    st.session_state.search_history = []

user_query = st.text_input("Start typing a keyword (e.g., vendor, product, PO, PR...)", "")

if user_query and user_query not in st.session_state.search_history:
    st.session_state.search_history.append(user_query)

if st.session_state.search_history:
    with st.expander("🕘 Search History"):
        st.write(st.session_state.search_history[-10:])

with st.expander("🏷️ Filter by Tags"):
    selected_categories = st.multiselect("Procurement Category", sorted(filtered_df.get("Procurement Category", pd.Series(dtype=str)).dropna().unique().tolist()))
    selected_vendors = st.multiselect("PO Vendor", sorted(filtered_df.get("PO Vendor", pd.Series(dtype=str)).dropna().unique().tolist()))

if user_query:
    matches = [idx for text, idx in search_data if user_query.lower() in text]
    result_df = df.loc[matches]

    if selected_categories:
        result_df = result_df[result_df.get("Procurement Category").isin(selected_categories)]
    if selected_vendors:
        result_df = result_df[result_df.get("PO Vendor").isin(selected_vendors)]

    if not result_df.empty:
        st.markdown(f"### 🔎 Found {len(result_df)} matching results:")
        st.dataframe(result_df, use_container_width=True)

        def convert_df_to_csv(_df: pd.DataFrame) -> bytes:
            return _df.to_csv(index=False).encode("utf-8")

        def convert_df_to_excel(_df: pd.DataFrame) -> bytes:
            from io import BytesIO
            output = BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                _df.to_excel(writer, index=False, sheet_name="Search_Results")
            return output.getvalue()

        st.download_button("⬇️ Download CSV", convert_df_to_csv(result_df), file_name="search_results.csv", mime="text/csv")
        st.download_button("⬇️ Download Excel", convert_df_to_excel(result_df), file_name="search_results.xlsx")
    else:
        st.warning("No matching results found.")
else:
    st.info("Start typing a keyword to search...")

# ------------------------------------
#  9) KPI Row
# ------------------------------------
st.title("📊 Procure-to-Pay Dashboard")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total PRs",        int(filtered_df.get("PR Number", pd.Series(dtype=object)).nunique()))
col2.metric("Total POs",        int(filtered_df.get("Purchase Doc", pd.Series(dtype=object)).nunique()))
col3.metric("Line Items",       int(len(filtered_df)))
col4.metric("Entities",         int(filtered_df.get("Entity", pd.Series(dtype=object)).nunique()))
col5.metric("Spend (Cr ₹)",     f"{filtered_df.get('Net Amount', pd.Series(dtype=float)).sum() / 1e7:,.2f}")

# ------------------------------------
# 10) SLA Compliance Gauge (PR → PO ≤ 7 days)
# ------------------------------------
st.subheader("🎯 SLA Compliance (PR → PO ≤ 7 days)")
lead_df = filtered_df[filtered_df.get("Po create Date").notna()] if "Po create Date" in filtered_df.columns else pd.DataFrame()
if not lead_df.empty:
    lead_df = lead_df.copy()
    lead_df["Lead Time (Days)"] = (
        pd.to_datetime(lead_df.get("Po create Date")) - pd.to_datetime(lead_df.get("PR Date Submitted"))
    ).dt.days
    SLA_DAYS = 7
    avg_lead = float(pd.to_numeric(lead_df["Lead Time (Days)"], errors="coerce").mean()) if not lead_df.empty else 0.0
    avg_lead = 0.0 if pd.isna(avg_lead) else round(avg_lead, 1)

    gauge_fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=avg_lead,
            number={"suffix": " days"},
            gauge={
                "axis": {"range": [0, max(14, (avg_lead or 0) * 1.2)]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, SLA_DAYS], "color": "lightgreen"},
                    {"range": [SLA_DAYS, max(14, (avg_lead or 0) * 1.2)], "color": "lightcoral"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": SLA_DAYS,
                },
            },
            title={"text": "Average Lead Time"},
        )
    )
    st.plotly_chart(gauge_fig, use_container_width=True)
    st.caption(f"Current Avg Lead Time: {avg_lead:.1f} days   •   Target ≤ {SLA_DAYS} days")
else:
    st.info("No PO/PR date data available to compute lead time.")

# ---------- Monthly Total Spend (bars) + Cumulative Spend line ----------
st.subheader("📊 Monthly Total Spend (with Cumulative Value Line)")

# pick date column
_date_col = "Po create Date" if "Po create Date" in filtered_df.columns else ("PR Date Submitted" if "PR Date Submitted" in filtered_df.columns else None)
if _date_col is None:
    st.info("No date column available ('Po create Date' or 'PR Date Submitted') to compute monthly spend.")
else:
    filtered_df[_date_col] = pd.to_datetime(filtered_df[_date_col], errors="coerce")
    temp = filtered_df.dropna(subset=[_date_col])
    if "Net Amount" not in temp.columns:
        st.info("No 'Net Amount' column found to compute spend.")
    else:
        temp = temp.copy()
        temp["PO_Month"] = temp[_date_col].dt.to_period("M").dt.to_timestamp()
        temp["Month_Str"] = temp["PO_Month"].dt.strftime("%b-%Y")

        monthly_total_spend = (
            temp.groupby(["PO_Month", "Month_Str"], as_index=False)["Net Amount"].sum()
        )

        # convert to Cr
        monthly_total_spend["Spend (Cr ₹)"] = monthly_total_spend["Net Amount"] / 1e7
        monthly_total_spend = monthly_total_spend.sort_values("PO_Month").reset_index(drop=True)

        # cumulative spend in Cr
        monthly_total_spend["Cumulative Spend (Cr ₹)"] = monthly_total_spend["Spend (Cr ₹)"].cumsum()

        # keep month order
        month_order = monthly_total_spend["Month_Str"].tolist()
        monthly_total_spend["Month_Str"] = pd.Categorical(monthly_total_spend["Month_Str"], categories=month_order, ordered=True)

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        # bars: monthly spend (Cr)
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

        # line: cumulative spend (Cr)
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
# 26) Monthly Spend Trend by Entity
# ------------------------------------
st.subheader("💹 Monthly Spend Trend by Entity")
spend_df = filtered_df.copy()
if "Po create Date" in spend_df.columns:
    spend_df["PO Month"] = pd.to_datetime(spend_df["Po create Date"], errors="coerce").dt.to_period("M").dt.to_timestamp()

    monthly_spend = (
        spend_df.dropna(subset=["PO Month"]).groupby(["PO Month", "Entity"], as_index=False)["Net Amount"].sum()
    )
    monthly_spend["Spend (Cr ₹)"] = monthly_spend["Net Amount"] / 1e7
    monthly_spend["Month_Str"] = monthly_spend["PO Month"].dt.strftime("%b-%Y")

    fig_spend = px.line(
        monthly_spend,
        x="Month_Str",
        y="Spend (Cr ₹)",
        color="Entity",
        markers=True,
        title="Monthly Spend Trend by Entity",
        labels={"Month_Str": "Month", "Spend (Cr ₹)": "Spend (Cr ₹)"},
    )
    fig_spend.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_spend, use_container_width=True)
else:
    st.info("'Po create Date' column not found for Entity trend.")

# ------------------------------------
# NEW) Department-wise Spend + Details (as requested)
# ------------------------------------
st.subheader("🏢 Department-wise Spend + Details")
if all(c in filtered_df.columns for c in ["Net Amount"]):
    dept_col = None
    for candidate in ["Department (Mapped)", "Department", "PR Department", "Budget Department", "Budget Dept", "Department Name"]:
        if candidate in filtered_df.columns:
            dept_col = candidate
            break

    if dept_col is not None:
        dept_spend = (
            filtered_df.groupby(dept_col, dropna=False)["Net Amount"].sum().reset_index()
        )
        dept_spend["Spend (Cr ₹)"] = dept_spend["Net Amount"] / 1e7
        dept_spend = dept_spend.sort_values("Spend (Cr ₹)", ascending=False)

        fig_dept = px.bar(
            dept_spend,
            x=dept_col,
            y="Spend (Cr ₹)",
            title="Department-wise Spend (Cr ₹)",
            labels={dept_col: "Department", "Spend (Cr ₹)": "Spend (Cr ₹)"},
            text="Spend (Cr ₹)",
        )
        fig_dept.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig_dept.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_dept, use_container_width=True)

        st.markdown("#### 📋 Line-item details for selected departments")
        selected_depts = st.multiselect(
            "Choose departments to view details",
            options=dept_spend[dept_col].astype(str).tolist(),
            default=dept_spend[dept_col].astype(str).head(5).tolist(),
        )

        details = filtered_df.copy()
        details[dept_col] = details[dept_col].astype(str)
        if selected_depts:
            details = details[details[dept_col].isin(selected_depts)]

        # Columns to show (existence-checked)
        cols_pref = [
            "Purchase Doc", "PO Vendor", "Product Name", "Item Description",
            "Net Amount", dept_col, "PO Budget Code", "Procurement Category", "PO Unit Rate", "Pending QTY"
        ]
        cols = [c for c in cols_pref if c in details.columns]
        if cols:
            st.dataframe(
                details[cols].sort_values(by="Net Amount", ascending=False).style.format({
                    "Net Amount": "₹ {:,.2f}",
                    "PO Unit Rate": "₹ {:,.2f}",
                    "Pending QTY": "{:,.0f}",
                }),
                use_container_width=True,
            )

            st.download_button(
                "📥 Download Department Details (CSV)",
                data=details[cols].to_csv(index=False),
                file_name="Department_Details.csv",
                mime="text/csv",
            )
        else:
            st.info("No standard detail columns found to display.")
    else:
        st.info("No department column found. Add a 'Department' or similar column to enable this section.")
else:
    st.info("Net Amount not found; cannot compute Department spend.")

# ------------------------------------
# 11) PR → PO Lead Time by Buyer Type & Buyer
# ------------------------------------
st.subheader("⏱️ PR to PO Lead Time by Buyer Type & by Buyer")
if not lead_df.empty:
    lead_avg_by_type = (
        lead_df.groupby("Buyer.Type")["Lead Time (Days)"].mean().round(0).reset_index()
    )
    lead_avg_by_buyer = (
        lead_df.groupby("PO.Creator")["Lead Time (Days)"].mean().round(0).reset_index()
    )
    c1, c2 = st.columns(2)
    c1.dataframe(lead_avg_by_type, use_container_width=True)
    c2.dataframe(lead_avg_by_buyer, use_container_width=True)
else:
    st.info("No lead-time rows to summarize by Buyer.")

# ------------------------------------
# 12) Monthly PR & PO Trends
# ------------------------------------
st.subheader("📅 Monthly PR & PO Trends")
if all(c in filtered_df.columns for c in ["PR Date Submitted", "Po create Date", "PR Number", "Purchase Doc"]):
    filtered_df["PR Month"] = pd.to_datetime(filtered_df["PR Date Submitted"]).dt.to_period("M")
    filtered_df["PO Month"] = pd.to_datetime(filtered_df["Po create Date"]).dt.to_period("M")

    monthly_summary = (
        filtered_df.groupby("PR Month").agg({"PR Number": "count", "Purchase Doc": "count"}).reset_index()
    )
    monthly_summary.columns = ["Month", "PR Count", "PO Count"]
    monthly_summary["Month"] = monthly_summary["Month"].astype(str)

    st.line_chart(monthly_summary.set_index("Month"), use_container_width=True)
else:
    st.info("Missing columns for Monthly PR & PO Trends.")

# ------------------------------------
# 14) PR → PO Aging Buckets
# ------------------------------------
st.subheader("🧮 PR to PO Aging Buckets")
if not lead_df.empty and "Lead Time (Days)" in lead_df.columns:
    bins = [0, 7, 15, 30, 60, 90, 999]
    labels = ["0-7", "8-15", "16-30", "31-60", "61-90", "90+"]

    aging_buckets = pd.cut(lead_df["Lead Time (Days)"], bins=bins, labels=labels)
    age_summary = (
        aging_buckets.value_counts(normalize=True).sort_index().reset_index()
    )
    age_summary.columns = ["Aging Bucket", "Percentage"]
    age_summary["Percentage"] *= 100

    fig_aging = px.bar(
        age_summary,
        x="Aging Bucket",
        y="Percentage",
        text="Percentage",
        title="PR to PO Aging Bucket Distribution (%)",
        labels={"Percentage": "Percentage (%)"},
    )
    fig_aging.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    st.plotly_chart(fig_aging, use_container_width=True)
else:
    st.info("Not enough data to compute Aging buckets.")

# ------------------------------------
# 15) PRs & POs by Weekday
# ------------------------------------
st.subheader("📆 PRs and POs by Weekday")
df_wd = filtered_df.copy()
if "PR Date Submitted" in df_wd.columns:
    df_wd["PR Weekday"] = pd.to_datetime(df_wd["PR Date Submitted"]).dt.day_name()
if "Po create Date" in df_wd.columns:
    df_wd["PO Weekday"] = pd.to_datetime(df_wd["Po create Date"]).dt.day_name()

if "PR Weekday" in df_wd.columns and "PO Weekday" in df_wd.columns:
    pr_counts = df_wd["PR Weekday"].value_counts().reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        fill_value=0,
    )
    po_counts = df_wd["PO Weekday"].value_counts().reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        fill_value=0,
    )

    c1, c2 = st.columns(2)
    c1.bar_chart(pr_counts, use_container_width=True)
    c2.bar_chart(po_counts, use_container_width=True)
else:
    st.info("Missing PR/PO date data for weekday analysis.")

# ------------------------------------
# 16) Open PRs (Approved / InReview)
# ------------------------------------
st.subheader("⚠️ Open PRs (Approved/InReview)")
if "PR Status" in filtered_df.columns and "PR Date Submitted" in filtered_df.columns:
    open_df = filtered_df[filtered_df["PR Status"].isin(["Approved", "InReview"])].copy()
    if not open_df.empty:
        open_df["Pending Age (Days)"] = (
            pd.to_datetime(pd.Timestamp.today().date()) - pd.to_datetime(open_df["PR Date Submitted"])  # type: ignore
        ).dt.days

        open_summary = (
            open_df.groupby("PR Number")
            .agg({
                "PR Date Submitted":   "first",
                "Pending Age (Days)":  "first",
                "Procurement Category":"first",
                "Product Name":        "first",
                "Net Amount":          "sum",
                "PO Budget Code":      "first",
                "PR Status":           "first",
                "Buyer Group":         "first",
                "Buyer.Type":          "first",
                "Entity":              "first",
                "PO.Creator":          "first",
                "Purchase Doc":        "first",
            })
            .reset_index()
        )

        st.metric("🔢 Open PRs", int(open_summary["PR Number"].nunique()))

        open_monthly_counts = (
            pd.to_datetime(open_summary["PR Date Submitted"]).dt.to_period("M").value_counts().sort_index()
        )
        st.bar_chart(open_monthly_counts, use_container_width=True)

        def highlight_age(val):
            return "background-color: red" if (pd.notna(val) and val > 30) else ""

        st.dataframe(open_summary.style.applymap(highlight_age, subset=["Pending Age (Days)"]), use_container_width=True)

        st.subheader("🏢 Open PRs by Entity")
        ent_counts = open_summary["Entity"].value_counts().reset_index()
        ent_counts.columns = ["Entity", "Count"]
        st.bar_chart(ent_counts.set_index("Entity"), use_container_width=True)
    else:
        st.warning("⚠️ No open PRs match the current filters.")
else:
    st.info("ℹ️ 'PR Status' or 'PR Date Submitted' column not found.")

# ------------------------------------
# 17) Daily PR Submissions Trend
# ------------------------------------
st.subheader("📅 Daily PR Trends")
if "PR Date Submitted" in filtered_df.columns:
    daily_df = filtered_df.copy()
    daily_df["PR Date"] = pd.to_datetime(daily_df["PR Date Submitted"])  # type: ignore
    daily_trend = daily_df.groupby("PR Date").size().reset_index(name="PR Count")

    fig_daily = px.line(
        daily_trend,
        x="PR Date",
        y="PR Count",
        title="Daily PR Submissions",
        labels={"PR Count": "PR Count"},
    )
    st.plotly_chart(fig_daily, use_container_width=True)
else:
    st.info("PR Date Submitted not available for daily trend.")

# ------------------------------------
# 18) Buyer-wise Spend
# ------------------------------------
st.subheader("💰 Buyer-wise Spend (Cr ₹)")
if all(c in filtered_df.columns for c in ["PO.Creator", "Net Amount"]):
    buyer_spend = (
        filtered_df.groupby("PO.Creator")["Net Amount"].sum().sort_values(ascending=False).reset_index()
    )
    buyer_spend["Net Amount (Cr)"] = buyer_spend["Net Amount"] / 1e7

    fig_buyer = px.bar(
        buyer_spend,
        x="PO.Creator",
        y="Net Amount (Cr)",
        title="Spend by Buyer",
        labels={"Net Amount (Cr)": "Spend (Cr ₹)", "PO.Creator": "Buyer"},
        text="Net Amount (Cr)",
    )
    fig_buyer.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    st.plotly_chart(fig_buyer, use_container_width=True)
else:
    st.info("Missing PO.Creator/Net Amount for buyer spend.")

# ------------------------------------
#  Category Spend Chart Sorted Descending
# ------------------------------------
if all(c in filtered_df.columns for c in ["Procurement Category", "Net Amount"]):
    cat_spend = (
        filtered_df.groupby("Procurement Category")["Net Amount"].sum().sort_values(ascending=False).reset_index()
    )
    cat_spend["Spend (Cr ₹)"] = cat_spend["Net Amount"] / 1e7

    fig_cat = px.bar(
        cat_spend,
        x="Procurement Category",
        y="Spend (Cr ₹)",
        title="Spend by Category (Descending)",
        labels={"Spend (Cr ₹)": "Spend (Cr ₹)", "Procurement Category": "Category"},
    )
    fig_cat.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_cat, use_container_width=True)
else:
    st.warning("'Procurement Category' or 'Net Amount' column missing from data.")

# ------------------------------------
# 19) PO Approval Summary & Details
# ------------------------------------
if "PO Approved Date" in filtered_df.columns and "Po create Date" in filtered_df.columns and "Purchase Doc" in filtered_df.columns:
    st.subheader("📋 PO Approval Summary")
    po_app_df = filtered_df[filtered_df["Po create Date"].notna()].copy()
    po_app_df["PO Approved Date"] = pd.to_datetime(po_app_df["PO Approved Date"], errors="coerce")

    total_pos    = po_app_df["Purchase Doc"].nunique()
    approved_pos = po_app_df[po_app_df["PO Approved Date"].notna()]["Purchase Doc"].nunique()
    pending_pos  = int(total_pos - approved_pos)

    po_app_df["PO Approval Lead Time"] = (
        po_app_df["PO Approved Date"] - pd.to_datetime(po_app_df["Po create Date"])  # type: ignore
    ).dt.days
    avg_approval = float(pd.to_numeric(po_app_df["PO Approval Lead Time"], errors="coerce").mean())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📦 Total POs",           int(total_pos))
    c2.metric("✅ Approved POs",        int(approved_pos))
    c3.metric("⏳ Pending Approval",    int(pending_pos))
    c4.metric("⏱️ Avg Approval Lead Time (days)", round(avg_approval, 1) if pd.notna(avg_approval) else 0)

    st.subheader("📄 Detailed PO Approval Aging List")
    approval_detail = po_app_df[["PO.Creator", "Purchase Doc", "Po create Date", "PO Approved Date", "PO Approval Lead Time"]].sort_values(by="PO Approval Lead Time", ascending=False)
    st.dataframe(approval_detail, use_container_width=True)
else:
    st.info("ℹ️ 'PO Approved Date' or 'Po create Date' or 'Purchase Doc' column not found.")

# ------------------------------------
# 20) PO Status Breakdown
# ------------------------------------
if "PO Status" in filtered_df.columns:
    st.subheader("📊 PO Status Breakdown")
    po_status_summary = filtered_df["PO Status"].value_counts().reset_index()
    po_status_summary.columns = ["PO Status", "Count"]

    c1, c2 = st.columns([2, 3])
    with c1:
        st.dataframe(po_status_summary, use_container_width=True)
    with c2:
        fig_status = px.pie(
            po_status_summary,
            names="PO Status",
            values="Count",
            title="PO Status Distribution",
            hole=0.3,
        )
        fig_status.update_traces(textinfo="percent+label")
        st.plotly_chart(fig_status, use_container_width=True)
else:
    st.info("ℹ️ 'PO Status' column not found.")

# ------------------------------------
# 21) PO Delivery Summary: Received vs Pending
# ------------------------------------
st.subheader("🚚 PO Delivery Summary: Received vs Pending")
delivery_df = filtered_df.rename(columns={
    "PO Quantity": "PO Qty",
    "ReceivedQTY": "Received Qty",
    "Pending QTY": "Pending Qty",
}).copy()

if all(c in delivery_df.columns for c in ["PO Qty", "Received Qty"]):
    delivery_df["% Received"] = (delivery_df["Received Qty"] / delivery_df["PO Qty"]) * 100
    delivery_df["% Received"] = delivery_df["% Received"].fillna(0).round(1)

    group_cols = [c for c in ["Purchase Doc", "PO Vendor", "Product Name", "Item Description"] if c in delivery_df.columns]
    agg_df = (
        delivery_df.groupby(group_cols, dropna=False)
        .agg({
            "PO Qty":       "sum",
            "Received Qty": "sum",
            "Pending Qty":  "sum",
            "% Received":   "mean",
        })
        .reset_index()
    )

    st.dataframe(agg_df.sort_values(by="Pending Qty", ascending=False), use_container_width=True)

    if "Pending Qty" in agg_df.columns and "Purchase Doc" in agg_df.columns:
        fig_pending = px.bar(
            agg_df.sort_values(by="Pending Qty", ascending=False).head(20),
            x="Purchase Doc",
            y="Pending Qty",
            color=group_cols[1] if len(group_cols) > 1 else None,
            hover_data=[c for c in ["Product Name", "Item Description"] if c in agg_df.columns],
            title="Top 20 POs Awaiting Delivery (Pending Qty)",
            text="Pending Qty",
        )
        fig_pending.update_traces(textposition="outside")
        st.plotly_chart(fig_pending, use_container_width=True)

    # Delivery Performance Summary Metrics
    total_po_lines    = int(len(delivery_df))
    fully_received    = int((delivery_df.get("Pending Qty", pd.Series(0)).fillna(0) == 0).sum())
    partially_pending = int((delivery_df.get("Pending Qty", pd.Series(0)).fillna(0) > 0).sum())
    avg_receipt_pct   = float(delivery_df.get("% Received", pd.Series(0)).mean())

    st.markdown("### 📋 Delivery Performance Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PO Lines",        total_po_lines)
    c2.metric("Fully Delivered", fully_received)
    c3.metric("Pending Delivery", partially_pending)
    c4.metric("Avg. Receipt %",   f"{avg_receipt_pct:.1f}%")

    st.download_button(
        "📥 Download Delivery Status",
        data=agg_df.to_csv(index=False),
        file_name="PO_Delivery_Status.csv",
        mime="text/csv",
    )
else:
    st.info("Required columns for delivery summary not found.")

# ------------------------------------
# 22) Top 50 Pending Delivery Lines by Value
# ------------------------------------
st.subheader("📋 Top 50 Pending Lines (by Value)")
if all(c in filtered_df.columns for c in ["Pending QTY", "PO Unit Rate"]):
    pending_items = filtered_df[filtered_df["Pending QTY"].fillna(0).astype(float) > 0].copy()
    pending_items["Pending Value"] = pending_items["Pending QTY"].astype(float) * pending_items["PO Unit Rate"].astype(float)

    cols_to_show = [
        "PR Number", "Purchase Doc", "Procurement Category", "Buying legal entity",
        "PR Budget description", "Product Name", "Item Description", "Pending QTY", "Pending Value"
    ]
    cols_to_show = [c for c in cols_to_show if c in pending_items.columns]

    top_pending_items = pending_items.sort_values(by="Pending Value", ascending=False).head(50)[cols_to_show].reset_index(drop=True)

    st.dataframe(
        top_pending_items.style.format({
            "Pending QTY":   "{:,.0f}",
            "Pending Value": "₹ {:,.2f}",
        }),
        use_container_width=True,
    )
else:
    st.info("Missing 'Pending QTY' or 'PO Unit Rate' for pending lines by value.")

# ------------------------------------
# 23) Top 10 Vendors by Spend
# ------------------------------------
st.subheader("🏆 Top 10 Vendors by Spend (Cr ₹)")
if all(c in filtered_df.columns for c in ["PO Vendor", "Purchase Doc", "Net Amount"]):
    vendor_spend = (
        filtered_df.groupby("PO Vendor", dropna=False)
        .agg(
            Vendor_PO_Count=("Purchase Doc", "nunique"),
            Total_Spend_Cr=("Net Amount", lambda x: (x.sum() / 1e7).round(2)),
        )
        .reset_index()
        .sort_values(by="Total_Spend_Cr", ascending=False)
    )

    top10_spend = vendor_spend.head(10).copy()
    st.dataframe(top10_spend, use_container_width=True)

    fig_top_vendors = px.bar(
        top10_spend,
        x="PO Vendor",
        y="Total_Spend_Cr",
        title="Top 10 Vendors by Spend (Cr ₹)",
        labels={"Total_Spend_Cr": "Spend (Cr ₹)", "PO Vendor": "Vendor"},
        text="Total_Spend_Cr",
    )
    fig_top_vendors.update_traces(textposition="outside", texttemplate="%{text:.2f}")
    st.plotly_chart(fig_top_vendors, use_container_width=True)
else:
    st.info("ℹ️ Cannot compute Top 10 Vendors – required columns missing.")

# ------------------------------------
# 24) Vendor Delivery Performance
# ------------------------------------
st.subheader("📊 Vendor Delivery Performance (Top 10 by Spend)")
if all(c in filtered_df.columns for c in ["PO Vendor", "Purchase Doc", "PO Delivery Date", "Pending QTY"]):
    today = pd.Timestamp.today().normalize().date()
    df_vp = filtered_df.copy()
    df_vp["Pending Qty Filled"] = df_vp["Pending QTY"].fillna(0).astype(float)
    df_vp["Is_Fully_Delivered"] = df_vp["Pending Qty Filled"] == 0
    df_vp["PO Delivery Date"] = pd.to_datetime(df_vp["PO Delivery Date"], errors="coerce")

    df_vp["Is_Late"] = (
        df_vp["PO Delivery Date"].dt.date.notna()
        & (df_vp["PO Delivery Date"].dt.date < today)
        & (df_vp["Pending Qty Filled"] > 0)
    )

    vendor_perf = (
        df_vp.groupby("PO Vendor", dropna=False)
        .agg(
            Total_PO_Count=("Purchase Doc", "nunique"),
            Fully_Delivered_PO_Count=("Is_Fully_Delivered", "sum"),
            Late_PO_Count=("Is_Late", "sum"),
        )
        .reset_index()
    )
    vendor_perf["Pct_Fully_Delivered"] = (
        (vendor_perf["Fully_Delivered_PO_Count"] / vendor_perf["Total_PO_Count"] * 100).round(1)
    )
    vendor_perf["Pct_Late"] = (
        (vendor_perf["Late_PO_Count"] / vendor_perf["Total_PO_Count"] * 100).round(1)
    )

    # Merge in “Total_Spend_Cr” if available
    try:
        vendor_perf = vendor_perf.merge(
            vendor_spend[["PO Vendor", "Total_Spend_Cr"]], on="PO Vendor", how="left"
        )
    except Exception:
        vendor_perf["Total_Spend_Cr"] = None

    top10_vendor_perf = vendor_perf.sort_values(["Total_Spend_Cr", "Total_PO_Count"], ascending=False).head(10)

    st.dataframe(
        top10_vendor_perf[[
            "PO Vendor", "Total_PO_Count", "Fully_Delivered_PO_Count",
            "Late_PO_Count", "Pct_Fully_Delivered", "Pct_Late", "Total_Spend_Cr"
        ]],
        use_container_width=True,
    )

    melted_perf = top10_vendor_perf.melt(
        id_vars=["PO Vendor"],
        value_vars=["Pct_Fully_Delivered", "Pct_Late"],
        var_name="Metric",
        value_name="Percentage",
    )
    fig_vendor_perf = px.bar(
        melted_perf,
        x="PO Vendor",
        y="Percentage",
        color="Metric",
        barmode="group",
        title="% Fully Delivered vs % Late (Top 10 Vendors by Spend)",
        labels={"Percentage": "% of POs", "PO Vendor": "Vendor"},
    )
    st.plotly_chart(fig_vendor_perf, use_container_width=True)
else:
    st.info("ℹ️ Cannot compute Vendor Delivery Performance – required columns missing.")

# ------------------------------------
# 25) Monthly Unique PO Generation
# ------------------------------------
st.subheader("🗓️ Monthly Unique PO Generation")
if all(c in filtered_df.columns for c in ["Purchase Doc", "Po create Date"]):
    po_monthly = filtered_df[filtered_df["Purchase Doc"].notna()].copy()
    po_monthly["PO Month"] = pd.to_datetime(po_monthly["Po create Date"]).dt.to_period("M")

    monthly_po_counts = (
        po_monthly.groupby("PO Month")["Purchase Doc"].nunique().reset_index(name="Unique PO Count")
    )
    monthly_po_counts["PO Month"] = monthly_po_counts["PO Month"].astype(str)

    fig_monthly_po = px.bar(
        monthly_po_counts,
        x="PO Month",
        y="Unique PO Count",
        title="Monthly Unique PO Generation",
        labels={"PO Month": "Month", "Unique PO Count": "Number of Unique POs"},
        text="Unique PO Count",
    )
    fig_monthly_po.update_traces(textposition="outside")
    st.plotly_chart(fig_monthly_po, use_container_width=True)
else:
    st.info("Missing columns for Monthly Unique PO Generation.")

# ------------------------------------
# 31) End of Dashboard
# ------------------------------------
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

