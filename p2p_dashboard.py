# p2p_dashboard.py

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

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
    df["Buyer Group Code"] = (
        df["Buyer Group"]
        .astype(str)
        .str.extract(r"(\d+)")
        .astype(float)
    )
    def classify_buyer_group(row):
        bg   = row["Buyer Group"]
        code = row["Buyer Group Code"]
        if bg in ["ME_BG17", "MLBG16"]:
            return "Direct"
        elif bg in ["Not Available"] or pd.isna(bg):
            return "Indirect"
        elif (code >= 1) & (code <= 9):
            return "Direct"
        elif (code >= 10) & (code <= 18):
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

df["PO Orderer"] = df["PO Orderer"].fillna("N/A").astype(str).str.strip()
df["PO.Creator"] = df["PO Orderer"].map(o_created_by_map).fillna(df["PO Orderer"])
df["PO.Creator"] = df["PO.Creator"].replace({"N/A": "Dilip"})

indirect_buyers = [
    "Aatish", "Deepak", "Deepakex", "Dhruv", "Dilip",
    "Mukul", "Nayan", "Paurik", "Kamlesh", "Suresh"
]
df["PO.BuyerType"] = df["PO.Creator"].apply(lambda x: "Indirect" if x in indirect_buyers else "Direct")

# ------------------------------------
#  5) Build Keyword Search Suggestions List
# ------------------------------------
# Collect every unique PR Number, Purchase Doc, and Product Name as strings.
# We won’t use `autocomplete=` in this Streamlit version, but having this list
# means you could adapt to a newer Streamlit that supports autocomplete later.
all_suggestions = []
if "PR Number" in df.columns:
    all_suggestions.extend(df["PR Number"].dropna().astype(str).unique().tolist())
if "Purchase Doc" in df.columns:
    all_suggestions.extend(df["Purchase Doc"].dropna().astype(str).unique().tolist())
if "Product Name" in df.columns:
    all_suggestions.extend(df["Product Name"].dropna().astype(str).unique().tolist())
# Deduplicate:
all_suggestions = list(dict.fromkeys(all_suggestions))

# ------------------------------------
#  7) Sidebar Filters (FY-Based)
# ------------------------------------
# ---------------------------
# Sidebar Filters (robust)
# ---------------------------
st.sidebar.header("🔍 Filters (robust)")

# Ensure key columns exist and normalize dtype/strings to avoid mismatches
def safe_unique_vals(series):
    if series is None:
        return []
    # cast to string, strip, drop null-like values, give sorted unique
    return sorted([str(x).strip() for x in series.dropna().unique()])

# Make sure these columns exist (create dummy empty columns if missing)
for col in ["Buyer.Type", "Entity", "PO.Creator", "PO.BuyerType", "PR Date Submitted"]:
    if col not in df.columns:
        df[col] = pd.NA

# Convert important string-like columns to stripped strings to avoid whitespace mismatches
for col in ["Buyer.Type", "Entity", "PO.Creator", "PO.BuyerType"]:
    df[col] = df[col].astype(str).fillna("").apply(lambda x: x.strip() if x is not None else "")

# Prepare options
buyer_options = safe_unique_vals(df["Buyer.Type"]) if "Buyer.Type" in df.columns else []
entity_options = safe_unique_vals(df["Entity"]) if "Entity" in df.columns else []
orderer_options = safe_unique_vals(df["PO.Creator"]) if "PO.Creator" in df.columns else []
po_buyer_type_options = safe_unique_vals(df["PO.BuyerType"]) if "PO.BuyerType" in df.columns else []

# Financial Year select (same as before)
fy_options = {
    "All Years": (pd.to_datetime("2023-04-01"), pd.to_datetime("2026-03-31")),
    "2023": (pd.to_datetime("2023-04-01"), pd.to_datetime("2024-03-31")),
    "2024": (pd.to_datetime("2024-04-01"), pd.to_datetime("2025-03-31")),
    "2025": (pd.to_datetime("2025-04-01"), pd.to_datetime("2026-03-31")),
    "2026": (pd.to_datetime("2026-04-01"), pd.to_datetime("2027-03-31")),
}
selected_fy = st.sidebar.selectbox("Select Financial Year", options=list(fy_options.keys()), index=0)
pr_start, pr_end = fy_options[selected_fy]

# Multiselects with safe defaults
buyer_filter = st.sidebar.multiselect("Buyer Type", options=buyer_options, default=buyer_options)
entity_filter = st.sidebar.multiselect("Entity", options=entity_options, default=entity_options)
orderer_filter = st.sidebar.multiselect("PO Ordered By", options=orderer_options, default=orderer_options)
po_buyer_type_filter = st.sidebar.multiselect("PO Buyer Type", options=po_buyer_type_options, default=po_buyer_type_options)

# Date range filter: use PR Date Submitted if available, else use Po create Date
date_col_for_filter = "PR Date Submitted" if "PR Date Submitted" in df.columns else ("Po create Date" if "Po create Date" in df.columns else None)
if date_col_for_filter:
    # ensure datetime dtype
    df[date_col_for_filter] = pd.to_datetime(df[date_col_for_filter], errors="coerce")
    min_date = df[date_col_for_filter].min().date() if pd.notna(df[date_col_for_filter].min()) else date.today()
    max_date = df[date_col_for_filter].max().date() if pd.notna(df[date_col_for_filter].max()) else date.today()
    date_range = st.sidebar.date_input("Date Range", [min_date, max_date])
else:
    date_range = None

# ---------------------------
# Apply filters (all safe)
# ---------------------------
filtered_df = df.copy()

# Apply FY filter using PR Date Submitted if present
if "PR Date Submitted" in filtered_df.columns:
    filtered_df["PR Date Submitted"] = pd.to_datetime(filtered_df["PR Date Submitted"], errors="coerce")
    # pr_start, pr_end are Timestamps; compare accordingly
    filtered_df = filtered_df[(filtered_df["PR Date Submitted"] >= pr_start) & (filtered_df["PR Date Submitted"] <= pr_end)]

# Apply date_range if chosen
if date_range and date_col_for_filter:
    # date_range may be (start, end) or a single date on some UI states
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        s_dt, e_dt = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        filtered_df = filtered_df[(filtered_df[date_col_for_filter] >= s_dt) & (filtered_df[date_col_for_filter] <= e_dt)]

# Apply string-based multiselect filters (coerce to str before matching)
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
#  8b) Keyword Search with Chips and Autocomplete + Memory
# ------------------------------------
from rapidfuzz import process, fuzz
import re
from datetime import datetime
import streamlit as st
import pandas as pd
import base64

st.markdown("## 🔍 Keyword Search")

valid_columns = [col for col in ["PR Number", "Purchase Doc", "Product Name", "PO Vendor"] if col in df.columns]
search_data = []
row_lookup = []

if valid_columns:
    for idx, row in df[valid_columns].fillna("").astype(str).iterrows():
        combined = " | ".join(row[col] for col in valid_columns)
        search_data.append((combined.lower(), idx))

# Memory persistence
if "search_history" not in st.session_state:
    st.session_state.search_history = []

# UI Input
user_query = st.text_input("Start typing a keyword (e.g., vendor, product, PO, PR...)", "")

if user_query and user_query not in st.session_state.search_history:
    st.session_state.search_history.append(user_query)

if st.session_state.search_history:
    with st.expander("🕘 Search History"):
        st.write(st.session_state.search_history[-10:])

# Filter Chips
with st.expander("🏷️ Filter by Tags"):
    selected_categories = st.multiselect("Procurement Category", sorted(df["Procurement Category"].dropna().unique())) if "Procurement Category" in df.columns else []
    selected_vendors = st.multiselect("PO Vendor", sorted(df["PO Vendor"].dropna().unique())) if "PO Vendor" in df.columns else []

# Matching Logic
if user_query:
    matches = [idx for text, idx in search_data if user_query.lower() in text]
    result_df = df.loc[matches]

    if selected_categories:
        result_df = result_df[result_df["Procurement Category"].isin(selected_categories)]
    if selected_vendors:
        result_df = result_df[result_df["PO Vendor"].isin(selected_vendors)]

    if not result_df.empty:
        st.markdown(f"### 🔎 Found {len(result_df)} matching results:")
        st.dataframe(result_df, use_container_width=True)

        # ----------------------
        # Download Buttons
        # ----------------------
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')

        def convert_df_to_excel(df):
            from io import BytesIO
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Search_Results')
            return output.getvalue()

        st.download_button("⬇️ Download CSV", convert_df_to_csv(result_df), file_name="search_results.csv", mime='text/csv')
        st.download_button("⬇️ Download Excel", convert_df_to_excel(result_df), file_name="search_results.xlsx")
    else:
        st.warning("No matching results found.")
else:
    st.info("Start typing a keyword to search...")




# ------------------------------------
#  9) Top KPI Row (Total PRs, POs, Line Items, Entities, Spend)
# ------------------------------------
st.title("📊 Procure-to-Pay Dashboard")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total PRs",        filtered_df["PR Number"].nunique())
col2.metric("Total POs",        filtered_df["Purchase Doc"].nunique())
col3.metric("Line Items",       len(filtered_df))
col4.metric("Entities",         filtered_df["Entity"].nunique())
col5.metric("Spend (Cr ₹)",     f"{filtered_df['Net Amount'].sum() / 1e7:,.2f}")

# ------------------------------------
# 10) SLA Compliance Gauge (PR → PO ≤ 7 days)
# ------------------------------------
st.subheader("🎯 SLA Compliance (PR → PO ≤ 7 days)")
lead_df = filtered_df[filtered_df["Po create Date"].notna()].copy()
lead_df["Lead Time (Days)"] = (
    pd.to_datetime(lead_df["Po create Date"]) 
    - pd.to_datetime(lead_df["PR Date Submitted"]) 
).dt.days

SLA_DAYS = 7
avg_lead = lead_df["Lead Time (Days)"].mean().round(1)

gauge_fig = go.Figure(
    go.Indicator(
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

# ---------- Monthly Total Spend (bars) + Cumulative Spend line ----------
from plotly.subplots import make_subplots

st.subheader("📊 Monthly Total Spend (with Cumulative Value Line)")

# pick date column
date_col = "Po create Date" if "Po create Date" in filtered_df.columns else (
    "PR Date Submitted" if "PR Date Submitted" in filtered_df.columns else None
)

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

        # build figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        # bars: monthly spend (Cr)
        fig.add_trace(
            go.Bar(
                x=monthly_total_spend["Month_Str"],
                y=monthly_total_spend["Spend (Cr ₹)"],
                name="Monthly Spend (Cr ₹)",
                text=monthly_total_spend["Spend (Cr ₹)"].map("{:.2f}".format),
                textposition="outside",
                marker=dict(opacity=0.85)
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
                hovertemplate="₹ %{y:.2f} Cr<br>%{x}<extra></extra>"
            ),
            secondary_y=True,
        )

        # layout and axis titles
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
spend_df["PO Month"] = (
    pd.to_datetime(spend_df["Po create Date"], errors="coerce")
    .dt.to_period("M")
    .dt.to_timestamp()
)

monthly_spend = (
    spend_df.dropna(subset=["PO Month"]) 
    .groupby(["PO Month", "Entity"], as_index=False)["Net Amount"]
    .sum()
)
monthly_spend["Spend (Cr ₹)"] = monthly_spend["Net Amount"] / 1e7

# Convert timestamp to string like "Apr-2023", "May-2023", etc.
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


# ------------------------------------
# 11) PR → PO Lead Time by Buyer Type & Buyer
# ------------------------------------
st.subheader("⏱️ PR to PO Lead Time by Buyer Type & by Buyer")
lead_avg_by_type = (
    lead_df.groupby("Buyer.Type")["Lead Time (Days)"]
    .mean()
    .round(0)
    .reset_index()
)
lead_avg_by_buyer = (
    lead_df.groupby("PO.Creator")["Lead Time (Days)"]
    .mean()
    .round(0)
    .reset_index()
)
c1, c2 = st.columns(2)
c1.dataframe(lead_avg_by_type, use_container_width=True)
c2.dataframe(lead_avg_by_buyer, use_container_width=True)

# ------------------------------------
# 12) Monthly PR & PO Trends
# ------------------------------------
st.subheader("📅 Monthly PR & PO Trends")
filtered_df["PR Month"] = pd.to_datetime(filtered_df["PR Date Submitted"]).dt.to_period("M")
filtered_df["PO Month"] = pd.to_datetime(filtered_df["Po create Date"]).dt.to_period("M")

monthly_summary = (
    filtered_df.groupby("PR Month")
    .agg({"PR Number": "count", "Purchase Doc": "count"})
    .reset_index()
)
monthly_summary.columns = ["Month", "PR Count", "PO Count"]
monthly_summary["Month"] = monthly_summary["Month"].astype(str)

st.line_chart(monthly_summary.set_index("Month"), use_container_width=True)


# ------------------------------------
# 13) Procurement Category Spend (Top 15, Descending, Excluding <0)
# ------------------------------------
st.subheader("📦 Top 15 Procurement Categories by Spend (Descending)")

if "Procurement Category" in filtered_df.columns and "Net Amount" in filtered_df.columns:
    cat_spend = (
        filtered_df.groupby("Procurement Category")["Net Amount"]
        .sum()
        .reset_index()
    )

    # Exclude negative spend
    cat_spend = cat_spend[cat_spend["Net Amount"] > 0]

    # Sort descending and keep Top 15
    cat_spend = cat_spend.sort_values(by="Net Amount", ascending=False).head(15)

    # Add Spend in Cr ₹
    cat_spend["Spend (Cr ₹)"] = cat_spend["Net Amount"] / 1e7

    # Plot
    fig_cat = px.bar(
        cat_spend,
        x="Procurement Category",
        y="Spend (Cr ₹)",
        title="Top 15 Procurement Categories by Spend",
        labels={"Spend (Cr ₹)": "Spend (Cr ₹)", "Procurement Category": "Category"},
        text="Spend (Cr ₹)"
    )
    fig_cat.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_cat.update_layout(xaxis_tickangle=-45)

    st.plotly_chart(fig_cat, use_container_width=True)
else:
    st.info("ℹ️ No 'Procurement Category' or 'Net Amount' column found.")


# ------------------------------------
#  Category Spend Chart Sorted Descending
# ------------------------------------
if "Procurement Category" in filtered_df.columns and "Net Amount" in filtered_df.columns:
    cat_spend = (
        filtered_df.groupby("Procurement Category")["Net Amount"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
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
#  PR Budget Code Spend (Top 15, Descending, Excluding <0)
# ------------------------------------
st.subheader("🏷️ Top 15 PR Budget Codes by Spend (Descending)")

if "PR Budget Code" in filtered_df.columns and "Net Amount" in filtered_df.columns:
    budget_spend = (
        filtered_df.groupby("PR Budget Code")["Net Amount"]
        .sum()
        .reset_index()
    )

    # Exclude negative/zero spend
    budget_spend = budget_spend[budget_spend["Net Amount"] > 0]

    # Sort descending and keep Top 15
    budget_spend = budget_spend.sort_values(by="Net Amount", ascending=False).head(15)

    # Add Spend in Cr ₹
    budget_spend["Spend (Cr ₹)"] = budget_spend["Net Amount"] / 1e7

    # Plot
    fig_budget = px.bar(
        budget_spend,
        x="PR Budget Code",
        y="Spend (Cr ₹)",
        title="Top 15 PR Budget Codes by Spend",
        labels={"Spend (Cr ₹)": "Spend (Cr ₹)", "PR Budget Code": "PR Budget Code"},
        text="Spend (Cr ₹)"
    )
    fig_budget.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_budget.update_layout(xaxis_tickangle=-45)

    st.plotly_chart(fig_budget, use_container_width=True)
else:
    st.info("ℹ️ No 'PR Budget Code' or 'Net Amount' column found.")

def compute_and_plot_department_spend(dframe: pd.DataFrame, top_n: int = 15):
    """
    Replacement function (no streamlit-plotly-events dependency).
    - Maps PR Budget Code -> Main Department / Sub Category using _mapping_df
    - Aggregates positive Net Amount only
    - Renders stacked bar chart (sub-categories) with a Department Total line
    - Provides an interactive selectbox (simulates clicking a bar) that shows underlying rows
    - Offers CSV download for selected department or all top departments
    """
    import io
    import plotly.graph_objects as go
    import pandas as pd
    import streamlit as st

    df = dframe.copy()

    # Validate required columns (accept small variations)
    if 'PR Budget Code' not in df.columns and 'PR Budget code' not in df.columns:
        st.error("Column 'PR Budget Code' not found in data. Please check your DataFrame column names.")
        return
    if 'Net Amount' not in df.columns:
        st.error("Column 'Net Amount' not found in data.")
        return

    code_col = 'PR Budget Code' if 'PR Budget Code' in df.columns else 'PR Budget code'

    # Normalize the budget code column and merge with mapping
    df['PR_Budget_Code_norm'] = df[code_col].astype(str).str.strip()
    map_df_local = _mapping_df[['Main Department', 'Sub Category', 'Budget Code']].copy()
    map_df_local['Budget Code'] = map_df_local['Budget Code'].astype(str).str.strip()

    merged = df.merge(map_df_local, left_on='PR_Budget_Code_norm', right_on='Budget Code', how='left')

    merged['Main Department'] = merged['Main Department'].fillna('Unmapped')
    merged['Sub Category'] = merged['Sub Category'].fillna('Unspecified')

    # Filter: keep only positive Net Amount
    merged = merged[merged['Net Amount'].notna()]
    merged = merged[merged['Net Amount'] > 0]

    if merged.empty:
        st.warning("No positive 'Net Amount' rows available after filtering/mapping.")
        return

    # Aggregate sub-category spend (INR)
    subcat = (
        merged.groupby(['Main Department', 'Sub Category'], as_index=False)['Net Amount']
        .sum()
    )
    subcat = subcat.sort_values(['Main Department', 'Net Amount'], ascending=[True, False])

    # Department totals
    dept_totals = (
        subcat.groupby('Main Department', as_index=False)['Net Amount']
        .sum()
        .sort_values('Net Amount', ascending=False)
        .reset_index(drop=True)
    )
    dept_totals['Spend (Cr ₹)'] = dept_totals['Net Amount'] / 1e7

    # Top N departments
    top_depts = dept_totals.head(top_n)['Main Department'].tolist()

    st.markdown(f"### Top {top_n} Departments by Spend (positive Net Amount only)")
    st.dataframe(dept_totals.head(top_n)[['Main Department', 'Spend (Cr ₹)']], use_container_width=True)

    # Prepare pivot for stacked bars: rows = main dept, cols = sub category, values = Net Amount
    pivot = subcat[subcat['Main Department'].isin(top_depts)].pivot_table(
        index='Main Department', columns='Sub Category', values='Net Amount', aggfunc='sum'
    ).fillna(0)
    pivot = pivot.reindex(top_depts)  # ensure order matches dept_totals

    # Build figure: stacked bars for sub-categories
    fig = go.Figure()
    for col in pivot.columns:
        fig.add_trace(
            go.Bar(
                x=pivot.index,
                y=pivot[col] / 1e7,  # convert to Cr
                name=str(col),
                hovertemplate='%{x}<br>%{y:.2f} Cr<extra></extra>',
            )
        )

    # Overlay department totals as a line (value in Cr)
    totals_cr = dept_totals.set_index('Main Department').reindex(top_depts)['Net Amount'].fillna(0) / 1e7
    fig.add_trace(
        go.Scatter(
            x=top_depts,
            y=totals_cr,
            name='Department Total (Cr ₹)',
            mode='lines+markers',
            marker=dict(symbol='diamond', size=9),
            line=dict(color='black', width=2),
            hovertemplate='%{x}<br>Total: %{y:.2f} Cr<extra></extra>',
        )
    )

    fig.update_layout(
        barmode='stack',
        title=f'Top {top_n} Departments — Sub-Category stacked spend (Cr ₹) with Department Total line',
        xaxis_title='Main Department',
        yaxis_title='Spend (Cr ₹)',
        legend_title='Sub Category / Total',
        xaxis_tickangle=-45,
        height=650,
    )

    # Render the chart
    st.plotly_chart(fig, use_container_width=True)

    # --------------------------
    # INTERACTIVE SELECTBOX FALLBACK (no external dep)
    # --------------------------
    st.markdown("### 🔎 Select a Department to view underlying rows (simulates clicking a bar)")
    dept_options = ['(All Top Departments)'] + top_depts
    selected = st.selectbox("Choose department", options=dept_options, index=0, help="Select a department to see its rows below")

    if selected and selected != '(All Top Departments)':
        dept = selected
        st.markdown(f"#### Details for **{dept}**")
        details = merged[merged['Main Department'] == dept].copy()
        # columns to show (adjust as needed)
        show_cols = ['PR Number', 'Purchase Doc', 'PR Date Submitted', 'Po create Date',
                     'PR Budget Code', 'PO Vendor', 'Product Name', 'Net Amount', 'Main Department', 'Sub Category']
        # adapt to columns actually present
        show_cols = [c for c in show_cols if c in details.columns]
        if details.empty:
            st.info("No rows available for this department.")
        else:
            # format Net Amount for readability
            if 'Net Amount' in details.columns:
                details = details.assign(**{'Net Amount (₹)': details['Net Amount'].map(lambda v: f"₹{v:,.2f}")})
            st.dataframe(details[show_cols], use_container_width=True)
            # CSV download
            csv_buf = io.StringIO()
            details.to_csv(csv_buf, index=False)
            st.download_button(f"⬇️ Download {dept} details (CSV)", csv_buf.getvalue(),
                               file_name=f"{dept.replace(' ', '_')}_details.csv", mime="text/csv")

    elif selected == '(All Top Departments)':
        st.markdown("#### Showing results for all top departments combined")
        details = merged[merged['Main Department'].isin(top_depts)].copy()
        show_cols = ['PR Number', 'Purchase Doc', 'PR Date Submitted', 'Po create Date',
                     'PR Budget Code', 'PO Vendor', 'Product Name', 'Net Amount', 'Main Department', 'Sub Category']
        show_cols = [c for c in show_cols if c in details.columns]
        if details.empty:
            st.info("No rows available.")
        else:
            if 'Net Amount' in details.columns:
                details = details.assign(**{'Net Amount (₹)': details['Net Amount'].map(lambda v: f"₹{v:,.2f}")})
            st.dataframe(details[show_cols], use_container_width=True)
            csv_buf = io.StringIO()
            details.to_csv(csv_buf, index=False)
            st.download_button("⬇️ Download All top-dept details (CSV)", csv_buf.getvalue(),
                               file_name="top_departments_details.csv", mime="text/csv")
    else:
        st.info("Select a department from the dropdown above to view details here.")


# ------------------------------------
# 15) PRs & POs by Weekday
# ------------------------------------
st.subheader("📆 PRs and POs by Weekday")
df_wd = filtered_df.copy()
df_wd["PR Weekday"] = pd.to_datetime(df_wd["PR Date Submitted"]).dt.day_name()
df_wd["PO Weekday"] = pd.to_datetime(df_wd["Po create Date"]).dt.day_name()

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

# ------------------------------------
# 16) Open PRs (Approved / InReview)
# ------------------------------------
st.subheader("⚠️ Open PRs (Approved/InReview)")
if "PR Status" in filtered_df.columns:
    open_df = filtered_df[filtered_df["PR Status"].isin(["Approved", "InReview"])].copy()
    if not open_df.empty:
        open_df["Pending Age (Days)"] = (
            pd.to_datetime(pd.Timestamp.today().date())
            - pd.to_datetime(open_df["PR Date Submitted"])
        ).dt.days

        open_summary = (
            open_df.groupby("PR Number")
            .agg(
                {
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
                }
            )
            .reset_index()
        )

        st.metric("🔢 Open PRs", open_summary["PR Number"].nunique())

        open_monthly_counts = (
            pd.to_datetime(open_summary["PR Date Submitted"]) 
            .dt.to_period("M")
            .value_counts()
            .sort_index()
        )
        st.bar_chart(open_monthly_counts, use_container_width=True)

        def highlight_age(val):
            return "background-color: red" if val > 30 else ""

        st.dataframe(
            open_summary.style.applymap(
                highlight_age, subset=["Pending Age (Days)"]
            ),
            use_container_width=True,
        )

        st.subheader("🏢 Open PRs by Entity")
        ent_counts = open_summary["Entity"].value_counts().reset_index()
        ent_counts.columns = ["Entity", "Count"]
        st.bar_chart(ent_counts.set_index("Entity"), use_container_width=True)
    else:
        st.warning("⚠️ No open PRs match the current filters.")
else:
    st.info("ℹ️ 'PR Status' column not found.")

# ------------------------------------
# 17) Daily PR Submissions Trend
# ------------------------------------
st.subheader("📅 Daily PR Trends")
daily_df = filtered_df.copy()
daily_df["PR Date"] = pd.to_datetime(daily_df["PR Date Submitted"])
daily_trend = daily_df.groupby("PR Date").size().reset_index(name="PR Count")

fig_daily = px.line(
    daily_trend,
    x="PR Date",
    y="PR Count",
    title="Daily PR Submissions",
    labels={"PR Count": "PR Count"},
)
st.plotly_chart(fig_daily, use_container_width=True)

# ------------------------------------
# 18) Buyer-wise Spend
# ------------------------------------
st.subheader("💰 Buyer-wise Spend (Cr ₹)")
buyer_spend = (
    filtered_df.groupby("PO.Creator")["Net Amount"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
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


# ------------------------------------
# 19) PO Approval Summary & Details
# ------------------------------------
if "PO Approved Date" in filtered_df.columns:
    st.subheader("📋 PO Approval Summary")
    po_app_df = filtered_df[filtered_df["Po create Date"].notna()].copy()
    po_app_df["PO Approved Date"] = pd.to_datetime(po_app_df["PO Approved Date"], errors="coerce")

    total_pos    = po_app_df["Purchase Doc"].nunique()
    approved_pos = po_app_df[po_app_df["PO Approved Date"].notna()]["Purchase Doc"].nunique()
    pending_pos  = total_pos - approved_pos

    po_app_df["PO Approval Lead Time"] = (
        po_app_df["PO Approved Date"] - pd.to_datetime(po_app_df["Po create Date"])
    ).dt.days
    avg_approval = po_app_df["PO Approval Lead Time"].mean().round(1)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📦 Total POs",           total_pos)
    c2.metric("✅ Approved POs",        approved_pos)
    c3.metric("⏳ Pending Approval",    pending_pos)
    c4.metric("⏱️ Avg Approval Lead Time (days)", avg_approval)

    st.subheader("📄 Detailed PO Approval Aging List")
    approval_detail = po_app_df[
        ["PO.Creator", "Purchase Doc", "Po create Date", "PO Approved Date", "PO Approval Lead Time"]
    ].sort_values(by="PO Approval Lead Time", ascending=False)
    st.dataframe(approval_detail, use_container_width=True)
else:
    st.info("ℹ️ 'PO Approved Date' column not found.")

# ------------------------------------
# 20) PO Status Breakdown
# ------------------------------------
if "PO Status" in filtered_df.columns:
    st.subheader("📊 PO Status Breakdown")
    po_status_summary = (
        filtered_df["PO Status"].value_counts().reset_index()
    )
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
    "ReceivedQTY":   "Received Qty",
    "Pending QTY":   "Pending Qty"
}).copy()

delivery_df["% Received"] = (delivery_df["Received Qty"] / delivery_df["PO Qty"]) * 100
delivery_df["% Received"] = delivery_df["% Received"].fillna(0).round(1)

po_delivery_summary = (
    delivery_df.groupby(
        ["Purchase Doc", "PO Vendor", "Product Name", "Item Description"],
        dropna=False
    )
    .agg({
        "PO Qty":       "sum",
        "Received Qty": "sum",
        "Pending Qty":  "sum",
        "% Received":   "mean",
    })
    .reset_index()
)

st.dataframe(po_delivery_summary.sort_values(by="Pending Qty", ascending=False), use_container_width=True)

fig_pending = px.bar(
    po_delivery_summary.sort_values(by="Pending Qty", ascending=False).head(20),
    x="Purchase Doc",
    y="Pending Qty",
    color="PO Vendor",
    hover_data=["Product Name", "Item Description"],
    title="Top 20 POs Awaiting Delivery (Pending Qty)",
    text="Pending Qty",
)
fig_pending.update_traces(textposition="outside")
st.plotly_chart(fig_pending, use_container_width=True)

# Delivery Performance Summary Metrics
total_po_lines    = len(delivery_df)
fully_received    = (delivery_df["Pending Qty"] == 0).sum()
partially_pending = (delivery_df["Pending Qty"] > 0).sum()
avg_receipt_pct   = delivery_df["% Received"].mean().round(1)

st.markdown("### 📋 Delivery Performance Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("PO Lines",        total_po_lines)
c2.metric("Fully Delivered", fully_received)
c3.metric("Pending Delivery", partially_pending)
c4.metric("Avg. Receipt %",   f"{avg_receipt_pct}%")

st.download_button(
    "📥 Download Delivery Status",
    data=po_delivery_summary.to_csv(index=False),
    file_name="PO_Delivery_Status.csv",
    mime="text/csv"
)

# ------------------------------------
# 22) Top 50 Pending Delivery Lines by Value
# ------------------------------------
st.subheader("📋 Top 50 Pending Lines (by Value)")
pending_items = delivery_df[delivery_df["Pending Qty"] > 0].copy()
pending_items["Pending Value"] = pending_items["Pending Qty"] * pending_items["PO Unit Rate"]

top_pending_items = (
    pending_items.sort_values(by="Pending Value", ascending=False)
    .head(50)[
        [
            "PR Number",
            "Purchase Doc",
            "Procurement Category",
            "Buying legal entity",
            "PR Budget description",
            "Product Name",
            "Item Description",
            "Pending Qty",
            "Pending Value",
        ]
    ]
    .reset_index(drop=True)
)

st.dataframe(
    top_pending_items.style.format({
        "Pending Qty":   "{:,.0f}",
        "Pending Value": "₹ {:,.2f}"
    }),
    use_container_width=True
)

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
    if "vendor_spend" in locals():
        vendor_perf = vendor_perf.merge(
            vendor_spend[["PO Vendor", "Total_Spend_Cr"]], on="PO Vendor", how="left"
        )
        top10_vendor_perf = vendor_perf.sort_values("Total_Spend_Cr", ascending=False).head(10)
    else:
        top10_vendor_perf = vendor_perf.sort_values("Total_PO_Count", ascending=False).head(10)
        top10_vendor_perf["Total_Spend_Cr"] = None

    st.dataframe(
        top10_vendor_perf[
            ["PO Vendor", "Total_PO_Count", "Fully_Delivered_PO_Count",
             "Late_PO_Count", "Pct_Fully_Delivered", "Pct_Late", "Total_Spend_Cr"]
        ],
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
po_monthly = filtered_df[filtered_df["Purchase Doc"].notna()].copy()
po_monthly["PO Month"] = pd.to_datetime(po_monthly["Po create Date"]).dt.to_period("M")

monthly_po_counts = (
    po_monthly.groupby("PO Month")["Purchase Doc"]
    .nunique()
    .reset_index(name="Unique PO Count")
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

# ------------------------------------
# Department-wise Spend Bar Chart
# ------------------------------------

def plot_department_bar(df, mapping_df=None, top_n=20):
    """
    Simple department-wise spend bar chart.
    - df: main data (must contain 'PR Budget Code' and 'Net Amount')
    - mapping_df: DataFrame with columns ['Main Department','Sub Category','Budget Code']
      If None, the function will attempt to use a global _mapping_df variable.
    - top_n: show top N departments by spend
    """
    import pandas as pd
    import plotly.express as px
    import streamlit as st

    if mapping_df is None:
        try:
            mapping_df = _mapping_df
        except NameError:
            st.error("Department mapping (_mapping_df) not found. Please provide a mapping DataFrame with columns: Main Department, Sub Category, Budget Code.")
            return

    # Validate columns
    if 'Net Amount' not in df.columns:
        st.error("'Net Amount' column missing from data.")
        return
    if 'PR Budget Code' not in df.columns and 'PR Budget code' not in df.columns:
        st.error("'PR Budget Code' column missing from data.")
        return

    code_col = 'PR Budget Code' if 'PR Budget Code' in df.columns else 'PR Budget code'

    # Normalize codes
    temp = df.copy()
    temp[code_col] = temp[code_col].astype(str).str.strip()
    mapping_df['Budget Code'] = mapping_df['Budget Code'].astype(str).str.strip()

    # Merge to get department for each row
    merged = temp.merge(mapping_df[['Main Department','Budget Code']], left_on=code_col, right_on='Budget Code', how='left')
    merged['Main Department'] = merged['Main Department'].fillna('Unmapped')

    # Keep only positive spend
    merged = merged[merged['Net Amount'].notna()]
    merged = merged[merged['Net Amount'] > 0]

    if merged.empty:
        st.info('No positive Net Amount rows to plot after applying mapping/filtering.')
        return

    dept_totals = merged.groupby('Main Department', as_index=False)['Net Amount'].sum()
    dept_totals = dept_totals.sort_values('Net Amount', ascending=False).reset_index(drop=True)
    dept_totals['Spend (Cr ₹)'] = dept_totals['Net Amount'] / 1e7

    # Limit to top_n
    plot_df = dept_totals.head(top_n).copy()

    fig = px.bar(
        plot_df,
        x='Main Department',
        y='Spend (Cr ₹)',
        title=f'Top {min(top_n,len(plot_df))} Departments by Spend (Cr ₹)',
        text='Spend (Cr ₹)'
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(xaxis_tickangle=-45, yaxis_title='Spend (Cr ₹)')

    st.plotly_chart(fig, use_container_width=True)

    # interactive details: select a department to see rows
    dept_list = ['(All shown)'] + plot_df['Main Department'].tolist()
    sel = st.selectbox('Show rows for department', dept_list)
    if sel:
        if sel == '(All shown)':
            rows = merged[merged['Main Department'].isin(plot_df['Main Department'])]
        else:
            rows = merged[merged['Main Department']==sel]

        if rows.empty:
            st.info('No rows for selection.')
        else:
            show_cols = [c for c in ['PR Number','Purchase Doc','PR Date Submitted','Po create Date','PR Budget Code','PO Vendor','Product Name','Net Amount','Main Department'] if c in rows.columns]
            st.dataframe(rows[show_cols], use_container_width=True)
            csv = rows.to_csv(index=False)
            st.download_button('⬇️ Download rows (CSV)', csv, file_name=f"{sel.replace(' ','_')}_rows.csv", mime='text/csv')


# Call the department plot function here (you can move this call anywhere appropriate)
try:
    plot_department_bar(filtered_df, mapping_df=_mapping_df, top_n=20)
except Exception as e:
    st.error(f"Department plot failed: {e}")


# Enhanced interactive department-spend helper for your Streamlit dashboard
# Save this as a separate file or paste into your main app after you have loaded df and _mapping_df.
# This tries to capture Plotly click events using streamlit-plotly-events if available.
# If the package is not installed it falls back to a selectbox which simulates click behaviour.

import io
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from streamlit_plotly_events import plotly_events
    HAS_PLOTLY_EVENTS = True
except Exception:
    HAS_PLOTLY_EVENTS = False


def plot_department_bar_interactive(df: pd.DataFrame, mapping_df: pd.DataFrame, top_n: int = 20):
    """Plot top departments by spend and allow clicking a bar (if supported) to show underlying rows.

    - df: filtered dataframe with at least 'Net Amount' and 'PR Budget Code' columns
    - mapping_df: DataFrame with columns ['Main Department','Sub Category','Budget Code'] mapping budget codes
    - top_n: number of top departments to show
    """

    # validate
    if 'Net Amount' not in df.columns:
        st.error("'Net Amount' column missing from dataframe")
        return
    if 'PR Budget Code' not in df.columns and 'PR Budget code' not in df.columns:
        st.error("'PR Budget Code' column missing from dataframe")
        return

    code_col = 'PR Budget Code' if 'PR Budget Code' in df.columns else 'PR Budget code'

    temp = df.copy()
    temp[code_col] = temp[code_col].astype(str).str.strip()
    mapping_df = mapping_df.copy()
    mapping_df['Budget Code'] = mapping_df['Budget Code'].astype(str).str.strip()

    merged = temp.merge(mapping_df[['Main Department','Sub Category','Budget Code']], left_on=code_col, right_on='Budget Code', how='left')
    merged['Main Department'] = merged['Main Department'].fillna('Unmapped')
    merged['Sub Category'] = merged['Sub Category'].fillna('Unspecified')

    # positive spend only
    merged = merged[merged['Net Amount'].notna()]
    merged = merged[merged['Net Amount'] > 0]

    if merged.empty:
        st.info('No positive Net Amount rows to display for departments.')
        return

    dept_totals = merged.groupby('Main Department', as_index=False)['Net Amount'].sum()
    dept_totals = dept_totals.sort_values('Net Amount', ascending=False).reset_index(drop=True)
    dept_totals['Spend (Cr ₹)'] = dept_totals['Net Amount'] / 1e7

    top_depts = dept_totals.head(top_n)

    # build plotly bar
    fig = go.Figure([go.Bar(x=top_depts['Main Department'], y=top_depts['Spend (Cr ₹)'], text=top_depts['Spend (Cr ₹)'].map('{:.2f}'.format), textposition='outside')])
    fig.update_layout(title=f'Top {len(top_depts)} Departments by Spend (Cr ₹)', xaxis_tickangle=-45, height=600)

    st.plotly_chart(fig, use_container_width=True)

    clicked_dept = None
    if HAS_PLOTLY_EVENTS:
        # capture click (user needs to install streamlit-plotly-events)
        st.write('Click a bar to load details — using streamlit-plotly-events')
        clicks = plotly_events(fig, click_event=True)
        if clicks and isinstance(clicks, list) and len(clicks) > 0:
            payload = clicks[0]
            # payload may have 'x' or nested points
            clicked_dept = payload.get('x') or payload.get('points', [{}])[0].get('x')
            if isinstance(clicked_dept, (list, tuple)):
                clicked_dept = clicked_dept[0] if clicked_dept else None
    else:
        st.warning('Package `streamlit-plotly-events` not installed. Select a department from dropdown below.')

    # fallback or show details
    dept_list = ['(All top departments)'] + top_depts['Main Department'].tolist()
    sel = st.selectbox('Or pick department to view details', dept_list, index=0)

    # prioritize clicked value if exists
    if clicked_dept:
        sel = clicked_dept

    if sel and sel != '(All top departments)':
        details = merged[merged['Main Department'] == sel].copy()
        show_cols = [c for c in ['PR Number','Purchase Doc','PR Date Submitted','Po create Date','PR Budget Code','PO Vendor','Product Name','Net Amount','Main Department','Sub Category'] if c in details.columns]
        if 'Net Amount' in details.columns:
            details = details.assign(**{'Net Amount (₹)': details['Net Amount'].map(lambda v: f"₹{v:,.2f}")})
        st.markdown(f"#### Details for: **{sel}** ({len(details)} rows)")
        st.dataframe(details[show_cols], use_container_width=True)
        csv_buf = io.StringIO()
        details.to_csv(csv_buf, index=False)
        st.download_button('⬇️ Download details (CSV)', csv_buf.getvalue(), file_name=f"{sel.replace(' ','_')}_details.csv", mime='text/csv')
    else:
        # all top depts combined
        details = merged[merged['Main Department'].isin(top_depts['Main Department'].tolist())].copy()
        show_cols = [c for c in ['PR Number','Purchase Doc','PR Date Submitted','Po create Date','PR Budget Code','PO Vendor','Product Name','Net Amount','Main Department','Sub Category'] if c in details.columns]
        if 'Net Amount' in details.columns:
            details = details.assign(**{'Net Amount (₹)': details['Net Amount'].map(lambda v: f"₹{v:,.2f}")})
        st.markdown(f"#### Combined details for top {len(top_depts)} departments ({len(details)} rows)")
        st.dataframe(details[show_cols], use_container_width=True)
        csv_buf = io.StringIO()
        details.to_csv(csv_buf, index=False)
        st.download_button('⬇️ Download combined details (CSV)', csv_buf.getvalue(), file_name="top_departments_details.csv", mime='text/csv')


# Example usage (call this after you've loaded `df` (filtered) and your mapping df as `_mapping_df`):
# plot_department_bar_interactive(filtered_df, mapping_df=_mapping_df, top_n=20)

# Note: To enable click-to-filter functionality install the package:
# pip install streamlit-plotly-events
# and redeploy the app.

# --------------------
# Department mapping loader (drop this right after df = load_and_combine_data())
# --------------------
import io
import os

# Helper to clean budget-code strings
def _norm_code(s):
    if pd.isna(s):
        return ""
    return str(s).strip()

# Attempt 1: load a CSV file present in the working directory named 'department_mapping.csv'
_mapping_df = None
mapping_path = "department_mapping.csv"
if os.path.exists(mapping_path):
    try:
        _mapping_df = pd.read_csv(mapping_path, dtype=str)
        st.sidebar.success(f"Loaded department mapping from {mapping_path}")
    except Exception as e:
        st.sidebar.error(f"Failed to read {mapping_path}: {e}")

# UI option: allow user to upload mapping file
st.sidebar.markdown("### Department → BudgetCode mapping")
uploaded = st.sidebar.file_uploader("Upload mapping CSV (columns: Main Department, Sub Category, Budget Code)", type=["csv", "txt", "tsv"])
if uploaded is not None:
    try:
        # try comma then tab
        try:
            tmp = pd.read_csv(uploaded, dtype=str)
        except Exception:
            uploaded.seek(0)
            tmp = pd.read_csv(uploaded, sep="\t", dtype=str)
        _mapping_df = tmp.copy()
        st.sidebar.success("Mapping uploaded and parsed.")
    except Exception as e:
        st.sidebar.error(f"Could not parse uploaded mapping file: {e}")

# UI fallback: paste mapping text (3 columns) - useful for quick paste of the table
mapping_text = st.sidebar.text_area(
    "Or paste mapping text (3 columns: Main Department, Sub Category, Budget Code). Paste rows newline-separated.",
    height=1400,
    help="If you paste the mapping table here, the app will parse it. Columns should be comma/tab separated."
)

if mapping_text and (not uploaded and _mapping_df is None):
    try:
        # try comma first, then tab
        try:
            _mapping_df = pd.read_csv(io.StringIO(mapping_text), dtype=str)
        except Exception:
            _mapping_df = pd.read_csv(io.StringIO(mapping_text), sep="\t", dtype=str)
        st.sidebar.success("Mapping parsed from pasted text.")
    except Exception as e:
        st.sidebar.error(f"Failed to parse pasted mapping: {e}")

# Validate mapping: ensure required columns exist (allow flexible column names)
if _mapping_df is not None:
    # normalize column names to expected ones
    cols = {c.lower().strip(): c for c in _mapping_df.columns}
    # map possible variants to canonical names
    rename_map = {}
    if "main department" in cols:
        rename_map[cols["main department"]] = "Main Department"
    if "sub category" in cols:
        rename_map[cols["sub category"]] = "Sub Category"
    if "subcatogery" in cols:  # user typo in provided mapping
        rename_map[cols["subcatogery"]] = "Sub Category"
    if "budget code" in cols:
        rename_map[cols["budget code"]] = "Budget Code"
    # apply rename
    _mapping_df = _mapping_df.rename(columns=rename_map)
    # if Budget Code is missing but present in a third unnamed column, try to salvage
    if "Budget Code" not in _mapping_df.columns and len(_mapping_df.columns) >= 3:
        # assume third column is the code
        _mapping_df = _mapping_df.rename(columns={_mapping_df.columns[2]: "Budget Code"})
    # now keep only the 3 columns we need (drop empties)
    needed = ["Main Department", "Sub Category", "Budget Code"]
    if not all(c in _mapping_df.columns for c in needed):
        st.sidebar.warning("Mapping file parsed but required columns not found. Please ensure CSV has columns: Main Department, Sub Category, Budget Code")
    else:
        # Normalize Budget Code strings
        _mapping_df["Budget Code"] = _mapping_df["Budget Code"].astype(str).apply(_norm_code)
        _mapping_df["Main Department"] = _mapping_df["Main Department"].astype(str).fillna("Unmapped").str.strip()
        _mapping_df["Sub Category"] = _mapping_df["Sub Category"].astype(str).fillna("Unspecified").str.strip()
        # Drop exact duplicate mappings
        _mapping_df = _mapping_df[["Main Department", "Sub Category", "Budget Code"]].drop_duplicates().reset_index(drop=True)
        st.sidebar.write(f"Mapping rows: {_mapping_df.shape[0]}")
else:
    st.sidebar.info("No mapping provided yet. Please upload 'department_mapping.csv' or paste mapping in sidebar.")
