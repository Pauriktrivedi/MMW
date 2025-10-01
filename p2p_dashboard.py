# p2p_dashboard.py

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import io

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
# 14) PR → PO Aging Buckets
# ------------------------------------
st.subheader("🧮 PR to PO Aging Buckets")
bins = [0, 7, 15, 30, 60, 90, 999]
labels = ["0-7", "8-15", "16-30", "31-60", "61-90", "90+"]

aging_buckets = pd.cut(lead_df["Lead Time (Days)"], bins=bins, labels=labels)
age_summary = (
    aging_buckets.value_counts(normalize=True)
    .sort_index()
    .reset_index()
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
# 26b) Sanity Checks (quick debug)
# ------------------------------------
with st.expander("🔎 Sanity Checks (debug)", expanded=False):
    total_rows = len(df)
    filtered_rows = len(filtered_df)
    total_spend_cr = (df["Net Amount"].sum() / 1e7) if "Net Amount" in df.columns else 0
    filt_spend_cr = (filtered_df["Net Amount"].sum() / 1e7) if "Net Amount" in filtered_df.columns else 0
    st.write({
        "Rows: filtered / total": f"{filtered_rows} / {total_rows}",
        "Spend (Cr) raw": round(total_spend_cr, 2),
        "Spend (Cr) filtered": round(filt_spend_cr, 2),
    })

# ------------------------------------
# 27) PR → PO Conversion Rate
# ------------------------------------
st.subheader("🔁 PR → PO Conversion Rate")
if all(c in filtered_df.columns for c in ["PR Number", "Purchase Doc"]):
    pr_total = filtered_df["PR Number"].nunique()
    pr_with_po = filtered_df[filtered_df["Purchase Doc"].notna()]["PR Number"].nunique()
    rate = (pr_with_po / pr_total * 100) if pr_total else 0
    st.metric("Conversion Rate", f"{rate:.1f}%")
else:
    st.info("Need PR Number and Purchase Doc columns for conversion rate.")

# ------------------------------------
# 28) Vendor Concentration Risk (HHI)
# ------------------------------------
st.subheader("⚖️ Vendor Concentration Risk (HHI)")
if all(c in filtered_df.columns for c in ["PO Vendor", "Net Amount"]):
    v = filtered_df.groupby("PO Vendor")["Net Amount"].sum()
    tot = v.sum()
    if tot > 0:
        hhi = float(((v / tot) ** 2).sum())
        st.metric("Herfindahl–Hirschman Index", f"{hhi:.3f}")
        st.caption("Closer to 1 = highly concentrated. <0.15 unconcentrated, 0.15–0.25 moderate, >0.25 high.")

# ------------------------------------
# 29) Overdue Deliveries by Vendor
# ------------------------------------
st.subheader("⏳ Overdue Deliveries by Vendor")
pending_col = None
for cand in ["Pending Qty", "Pending QTY"]:
    if cand in filtered_df.columns:
        pending_col = cand
        break
if "PO Delivery Date" in filtered_df.columns and pending_col:
    today = pd.Timestamp.today().normalize()
    d2 = filtered_df.copy()
    d2["PO Delivery Date"] = pd.to_datetime(d2["PO Delivery Date"], errors="coerce")
    d2["PendingQtyFilled"] = d2[pending_col].fillna(0).astype(float)
    d2["Overdue Days"] = (today - d2["PO Delivery Date"]).dt.days
    d2["Is Overdue"] = d2["PO Delivery Date"].notna() & (d2["Overdue Days"] > 0) & (d2["PendingQtyFilled"] > 0)
    od = d2[d2["Is Overdue"]]
    if not od.empty and "PO Vendor" in od.columns:
        over = (
            od.groupby("PO Vendor")
            .agg(
                Overdue_Lines=("Purchase Doc", "count"),
                Avg_Overdue_Days=("Overdue Days", "mean"),
                Pending_Qty=("PendingQtyFilled", "sum"),
                Pending_Value=("Net Amount", "sum"),
            )
            .reset_index()
            .sort_values(["Overdue_Lines", "Avg_Overdue_Days"], ascending=[False, False])
        )
        st.dataframe(over, use_container_width=True)
        st.plotly_chart(
            px.bar(over.head(15), x="PO Vendor", y="Overdue_Lines", text="Overdue_Lines", title="Top Overdue Vendors (by lines)"),
            use_container_width=True,
        )
else:
    st.info("Need PO Delivery Date and Pending Qty columns for overdue analysis.")

# ------------------------------------
# 30) Buyer Scorecards (POs, Spend, Lead Time)
# ------------------------------------
st.subheader("🏷️ Buyer Scorecards")
if "PO.Creator" in filtered_df.columns:
    score = filtered_df.groupby("PO.Creator").agg(
        POs=("Purchase Doc", "nunique"),
        Spend=("Net Amount", "sum"),
    ).reset_index()
    score["Spend (Cr ₹)"] = score["Spend"] / 1e7
    # add avg lead time from lead_df if available
    if 'lead_df' in globals() and not lead_df.empty and "PO.Creator" in lead_df.columns:
        lt = lead_df.groupby("PO.Creator")["Lead Time (Days)"].mean().reset_index()
        score = score.merge(lt, on="PO.Creator", how="left").rename(columns={"Lead Time (Days)": "Avg Lead Days"})
    else:
        score["Avg Lead Days"] = None
    st.dataframe(score.sort_values("Spend", ascending=False)[["PO.Creator", "POs", "Spend (Cr ₹)", "Avg Lead Days"]], use_container_width=True)
else:
    st.info("No PO.Creator column available for buyer scorecards.")

# ------------------------------------
# 30b) Unit-Rate Outliers vs Historical Median
# ------------------------------------
st.subheader("🚩 Unit-Rate Outliers vs Historical Median")
key_item_col = "Product Name" if "Product Name" in filtered_df.columns else ("Item Description" if "Item Description" in filtered_df.columns else None)
if key_item_col and "PO Unit Rate" in filtered_df.columns:
    hist = filtered_df.dropna(subset=[key_item_col, "PO Unit Rate"]).copy()
    # ensure numeric
    hist["PO Unit Rate"] = pd.to_numeric(hist["PO Unit Rate"], errors="coerce")
    grp = hist.groupby(key_item_col)
    stats = grp["PO Unit Rate"].agg(['count','median']).rename(columns={'median':'Median'}).reset_index()
    # MAD for robustness
    def mad(x):
        med = x.median()
        return (x - med).abs().median()
    mad_df = grp["PO Unit Rate"].apply(mad).reset_index(name='MAD')
    stats = stats.merge(mad_df, on=key_item_col, how='left')
    # keep only items with enough history
    stats = stats[stats['count'] >= 5]
    if not stats.empty:
        hist = hist.merge(stats[[key_item_col,'Median','MAD']], on=key_item_col, how='inner')
        # robust z-score using MAD: z = 0.6745*(x - median)/MAD ; flag |z|>3
        hist['robust_z'] = 0.6745 * (hist['PO Unit Rate'] - hist['Median']) / hist['MAD'].replace(0, pd.NA)
        outliers = hist[hist['robust_z'].abs() > 3].copy()
        outliers['Deviation %'] = ((hist['PO Unit Rate'] - hist['Median']) / hist['Median'] * 100).round(1)
        keep = [c for c in [key_item_col, 'PO Vendor', 'Purchase Doc', 'PO Unit Rate', 'Median', 'robust_z', 'Deviation %'] if c in outliers.columns]
        if not outliers.empty and keep:
            st.dataframe(outliers.sort_values('robust_z', key=lambda s: s.abs(), ascending=False)[keep].head(200), use_container_width=True)
            st.caption("Rule: |robust z| > 3 using MAD. Focus on large absolute deviations.")
        else:
            st.info("No significant outliers found with current filters.")
    else:
        st.info("Need at least 5 historical PO lines per item to detect outliers.")
else:
    st.info("Need item column and 'PO Unit Rate' to compute price outliers.")

# ------------------------------------
# 31) Forecast Next Month's Spend (SMA with confidence band)
# ------------------------------------
st.subheader("📈 Forecast Next Month's Spend (SMA + CI)")
forecast_date_col = "Po create Date" if "Po create Date" in filtered_df.columns else ("PR Date Submitted" if "PR Date Submitted" in filtered_df.columns else None)
if forecast_date_col and "Net Amount" in filtered_df.columns:
    fc = filtered_df.dropna(subset=[forecast_date_col]).copy()
    fc[forecast_date_col] = pd.to_datetime(fc[forecast_date_col], errors='coerce')
    fc = fc.dropna(subset=[forecast_date_col])
    if not fc.empty:
        m = fc.groupby(fc[forecast_date_col].dt.to_period('M'))['Net Amount'].sum().to_timestamp().sort_index()
        m_cr = m / 1e7
        # 3-month simple moving average
        sma3 = m_cr.rolling(3).mean()
        # residuals where SMA defined
        resid = (m_cr - sma3).dropna()
        sigma = resid.std() if len(resid) > 1 else 0
        next_month = (m_cr.index.max() + pd.offsets.MonthBegin(1))
        fcst = float(sma3.iloc[-1]) if not pd.isna(sma3.iloc[-1]) else float(m_cr.iloc[-1])
        lower = fcst - 1.96 * sigma
        upper = fcst + 1.96 * sigma
        # Plot
        figf = make_subplots(specs=[[{"secondary_y": False}]])
        figf.add_bar(x=m_cr.index.strftime('%b-%Y'), y=m_cr.values, name='Actual Spend (Cr)')
        figf.add_scatter(x=m_cr.index.strftime('%b-%Y'), y=sma3.values, mode='lines', name='SMA-3')
        figf.add_scatter(x=[next_month.strftime('%b-%Y')], y=[fcst], mode='markers', name='Forecast')
        figf.add_scatter(x=[next_month.strftime('%b-%Y'), next_month.strftime('%b-%Y')], y=[lower, upper], mode='lines', name='95% CI')
        figf.update_layout(title='Monthly Spend with SMA-3 and Next-Month Forecast', xaxis_tickangle=-45)
        st.plotly_chart(figf, use_container_width=True)
        c1,c2,c3 = st.columns(3)
        c1.metric('Forecast (Cr ₹)', f"{fcst:,.2f}")
        c2.metric('Lower 95% CI', f"{lower:,.2f}")
        c3.metric('Upper 95% CI', f"{upper:,.2f}")
    else:
        st.info("Not enough dated rows to forecast.")
else:
    st.info("Need date and Net Amount columns for forecasting.")

# ------------------------------------
# 32) Drillable Vendor Scorecards
# ------------------------------------
st.subheader("📒 Vendor Scorecard — Drilldown")
if 'PO Vendor' in filtered_df.columns:
    vendor_order = (
        filtered_df.groupby('PO Vendor')['Net Amount'].sum().sort_values(ascending=False)
        if 'Net Amount' in filtered_df.columns else filtered_df['PO Vendor'].value_counts()
    )
    vendors = vendor_order.index.tolist()
    if vendors:
        vsel = st.selectbox('Pick a vendor', vendors)
        vdf = filtered_df[filtered_df['PO Vendor'] == vsel].copy()
        spend_cr = vdf['Net Amount'].sum()/1e7 if 'Net Amount' in vdf.columns else 0
        po_count = vdf['Purchase Doc'].nunique() if 'Purchase Doc' in vdf.columns else len(vdf)
        pending_qty = vdf.get('Pending Qty', pd.Series(0)).fillna(0).sum()
        # late calc
        if 'PO Delivery Date' in vdf.columns:
            today = pd.Timestamp.today().normalize()
            vdf['PO Delivery Date'] = pd.to_datetime(vdf['PO Delivery Date'], errors='coerce')
            vdf['PendingFilled'] = vdf.get('Pending Qty', pd.Series(0)).fillna(0)
            vdf['Is_Late'] = vdf['PO Delivery Date'].notna() & (vdf['PO Delivery Date'] < today) & (vdf['PendingFilled'] > 0)
            late_pct = (vdf['Is_Late'].sum()/len(vdf)*100) if len(vdf) else 0
        else:
            late_pct = None
        k1,k2,k3,k4 = st.columns(4)
        k1.metric('POs', po_count)
        k2.metric('Spend (Cr ₹)', f"{spend_cr:,.2f}")
        k3.metric('Pending Qty', f"{pending_qty:,.0f}")
        k4.metric('% Late Lines', f"{late_pct:.1f}%" if late_pct is not None else '—')
        # trend
        dcol = 'Po create Date' if 'Po create Date' in vdf.columns else ('PR Date Submitted' if 'PR Date Submitted' in vdf.columns else None)
        if dcol and 'Net Amount' in vdf.columns:
            vdf[dcol] = pd.to_datetime(vdf[dcol], errors='coerce')
            vm = vdf.dropna(subset=[dcol]).groupby(vdf[dcol].dt.to_period('M'))['Net Amount'].sum().to_timestamp()
            st.plotly_chart(px.line(vm / 1e7, labels={'value':'Spend (Cr ₹)','index':'Month'}, title=f"{vsel} — Spend Trend"), use_container_width=True)
        # price variance
        if key_item_col and 'PO Unit Rate' in vdf.columns:
            pv = (
                vdf.dropna(subset=['PO Unit Rate']).groupby(key_item_col)['PO Unit Rate']
                .agg(['count','mean','std']).reset_index()
            )
            pv = pv[pv['count']>=3]
            if not pv.empty:
                pv['CoV %'] = (pv['std']/pv['mean']*100).round(1)
                st.dataframe(pv.sort_values('CoV %', ascending=False).head(30), use_container_width=True)
else:
    st.info('No vendor column present for scorecards.')

# ------------------------------------
# 33) Department-wise Spend — robust (mapped if possible)
# ------------------------------------
st.subheader("🏢 Department-wise Spend")

# Always create a working column inside filtered_df
filtered_df = filtered_df.copy()
filtered_df["Dept.Chart"] = pd.NA

# Try mapping via Budget Code → Department from the mapping file
_dept_map = None
try:
    bm = pd.read_excel("Final_Budget_Mapping_Completed_Verified.xlsx")
    bm.columns = bm.columns.astype(str).str.strip()
    if "Budget Code" in bm.columns and "PO Budget Code" in filtered_df.columns:
        bm_u = bm.dropna(subset=["Budget Code"]).drop_duplicates(subset=["Budget Code"], keep="first")
        dept_cols = [c for c in bm_u.columns if ("dept" in c.lower()) or ("department" in c.lower())]
        dept_col = dept_cols[0] if dept_cols else None
        if dept_col:
            _dept_map = bm_u.set_index("Budget Code")[dept_col].to_dict()
            mapped = filtered_df["PO Budget Code"].map(_dept_map)
            if mapped.notna().any():
                filtered_df["Dept.Chart"] = mapped
except Exception:
    pass

# Fallback chain from in-file columns if mapping is missing or empty
if filtered_df["Dept.Chart"].isna().all():
    for cand in ["PO Department", "PO Dept", "PR Department", "PR Dept", "Dept.Final", "Department"]:
        if cand in filtered_df.columns:
            filtered_df["Dept.Chart"] = filtered_df["Dept.Chart"].combine_first(filtered_df[cand])

# Final safety: if still all NA, label as "Unmapped"
filtered_df["Dept.Chart"].fillna("Unmapped / Missing", inplace=True)

if "Net Amount" in filtered_df.columns:
    dept_spend = (
        filtered_df.groupby("Dept.Chart", dropna=False)["Net Amount"].sum().reset_index()
        .sort_values("Net Amount", ascending=False)
    )
    dept_spend["Spend (Cr ₹)"] = dept_spend["Net Amount"] / 1e7
    st.plotly_chart(
        px.bar(dept_spend.head(30), x="Dept.Chart", y="Spend (Cr ₹)", title="Department-wise Spend (Top 30)")
        .update_layout(xaxis_tickangle=-45),
        use_container_width=True,
    )

    # --- Drilldown selector ---
    dd_col1, dd_col2 = st.columns([2,1])
    with dd_col1:
        dept_pick = st.selectbox("Drill down: choose a department", dept_spend["Dept.Chart"].tolist())
    with dd_col2:
        topn = st.number_input("Show top N vendors/items", min_value=5, max_value=100, value=20, step=5)

    detail = filtered_df[filtered_df["Dept.Chart"].astype(str) == str(dept_pick)].copy()
    st.markdown(f"### 🔎 Details for **{dept_pick}**")

    # KPIs
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Lines", len(detail))
    k2.metric("PRs", int(detail["PR Number"].nunique()) if "PR Number" in detail.columns else 0)
    k3.metric("POs", int(detail["Purchase Doc"].nunique()) if "Purchase Doc" in detail.columns else 0)
    k4.metric("Spend (Cr ₹)", f"{(detail.get('Net Amount', pd.Series(0)).sum()/1e7):,.2f}")

    # Top vendors & items
    c1,c2 = st.columns(2)
    if {"PO Vendor","Net Amount"}.issubset(detail.columns):
        top_v = (
            detail.groupby("PO Vendor", dropna=False)["Net Amount"].sum().sort_values(ascending=False).head(int(topn)).reset_index()
        )
        top_v["Spend (Cr ₹)"] = top_v["Net Amount"]/1e7
        c1.plotly_chart(px.bar(top_v, x="PO Vendor", y="Spend (Cr ₹)", title="Top Vendors (Cr ₹)").update_layout(xaxis_tickangle=-45), use_container_width=True)
    if {"Product Name","Net Amount"}.issubset(detail.columns):
        top_i = (
            detail.groupby("Product Name", dropna=False)["Net Amount"].sum().sort_values(ascending=False).head(int(topn)).reset_index()
        )
        top_i["Spend (Cr ₹)"] = top_i["Net Amount"]/1e7
        c2.plotly_chart(px.bar(top_i, x="Product Name", y="Spend (Cr ₹)", title="Top Items (Cr ₹)").update_layout(xaxis_tickangle=-45), use_container_width=True)

    # Optional time trend for the picked department
    dcol = "Po create Date" if "Po create Date" in detail.columns else ("PR Date Submitted" if "PR Date Submitted" in detail.columns else None)
    if dcol and "Net Amount" in detail.columns:
        detail[dcol] = pd.to_datetime(detail[dcol], errors="coerce")
        m = detail.dropna(subset=[dcol]).groupby(detail[dcol].dt.to_period('M'))['Net Amount'].sum().to_timestamp()
        st.plotly_chart(px.line(m/1e7, labels={'value':'Spend (Cr ₹)','index':'Month'}, title=f"{dept_pick} — Monthly Spend"), use_container_width=True)

    st.markdown("#### 📄 Line-level detail")
    show_cols = [c for c in ["PR Number","Purchase Doc","PO Vendor","Procurement Category","Product Name","Item Description","PO Qty","PO Unit Rate","Net Amount","PO Create Date","PO Delivery Date","PR Date Submitted"] if c in detail.columns]
    st.dataframe(detail[show_cols] if show_cols else detail, use_container_width=True)
    st.download_button(
        "⬇️ Download Department Lines (CSV)",
        (detail[show_cols] if show_cols else detail).to_csv(index=False),
        file_name=f"dept_drilldown_{str(dept_pick).replace(' ','_')}.csv",
        mime="text/csv",
    )

    with st.expander("View table / download"):
        st.dataframe(dept_spend, use_container_width=True)
        st.download_button(
            "⬇️ Download Department Spend (CSV)",
            dept_spend.to_csv(index=False),
            "department_spend.csv",
            "text/csv",
        )
        st.download_button(
            "⬇️ Download Department Spend (CSV)",
            dept_spend.to_csv(index=False),
            "department_spend.csv",
            "text/csv",
        )
else:
    st.info("Net Amount column not present, cannot compute department spend.")

# ------------------------------------
# 34) End of Dashboard
# ------------------------------------
# ------------------------------------
