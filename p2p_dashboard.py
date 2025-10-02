# Write the consolidated Streamlit dashboard with Smart Budget Mapper to a file
from textwrap import dedent
code = dedent('''
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date
import re

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
    mepl_df = pd.read_excel("MEPL1.xlsx", skiprows=1)
    mlpl_df = pd.read_excel("MLPL1.xlsx", skiprows=1)
    mmw_df  = pd.read_excel("mmw1.xlsx",  skiprows=1)
    mmpl_df = pd.read_excel("mmpl1.xlsx", skiprows=1)

    mepl_df["Entity"] = "MEPL"
    mlpl_df["Entity"] = "MLPL"
    mmw_df["Entity"]  = "MMW"
    mmpl_df["Entity"] = "MMPL"

    combined = pd.concat([mepl_df, mlpl_df, mmw_df, mmpl_df], ignore_index=True)

    combined.columns = (
        combined.columns
        .str.strip()
        .str.replace("\\xa0", " ", regex=False)
        .str.replace(" +", " ", regex=True)
    )
    combined.rename(columns=lambda c: c.strip(), inplace=True)

    return combined

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
    df["Buyer Group Code"] = (
        df["Buyer Group"]
        .astype(str)
        .str.extract(r"(\\d+)")
        .astype(float)
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

df["PO Orderer"] = df.get("PO Orderer", pd.Series([None]*len(df))).fillna("N/A").astype(str).str.strip()
df["PO.Creator"] = df["PO Orderer"].map(o_created_by_map).fillna(df["PO Orderer"])
df["PO.Creator"] = df["PO.Creator"].replace({"N/A": "Dilip"})

indirect_buyers = [
    "Aatish", "Deepak", "Deepakex", "Dhruv", "Dilip",
    "Mukul", "Nayan", "Paurik", "Kamlesh", "Suresh"
]
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
#  7) Sidebar Filters (FY-Based)
# ------------------------------------
st.sidebar.header("🔍 Filters (robust)")

def safe_unique_vals(series):
    if series is None:
        return []
    return sorted([str(x).strip() for x in series.dropna().unique()])

for col in ["Buyer.Type", "Entity", "PO.Creator", "PO.BuyerType", "PR Date Submitted"]:
    if col not in df.columns:
        df[col] = pd.NA

for col in ["Buyer.Type", "Entity", "PO.Creator", "PO.BuyerType"]:
    df[col] = df[col].astype(str).fillna("").apply(lambda x: x.strip() if x is not None else "")

buyer_options = safe_unique_vals(df["Buyer.Type"]) if "Buyer.Type" in df.columns else []
entity_options = safe_unique_vals(df["Entity"]) if "Entity" in df.columns else []
orderer_options = safe_unique_vals(df["PO.Creator"]) if "PO.Creator" in df.columns else []
po_buyer_type_options = safe_unique_vals(df["PO.BuyerType"]) if "PO.BuyerType" in df.columns else []

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

date_col_for_filter = "PR Date Submitted" if "PR Date Submitted" in df.columns else ("Po create Date" if "Po create Date" in df.columns else None)
if date_col_for_filter:
    df[date_col_for_filter] = pd.to_datetime(df[date_col_for_filter], errors="coerce")
    min_date = df[date_col_for_filter].min().date() if pd.notna(df[date_col_for_filter].min()) else date.today()
    max_date = df[date_col_for_filter].max().date() if pd.notna(df[date_col_for_filter].max()) else date.today()
    date_range = st.sidebar.date_input("Date Range", [min_date, max_date])
else:
    date_range = None

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
st.sidebar.write("Selected Buyer Types:", buyer_filter if buyer_filter else "ALL")
st.sidebar.write("Selected Entities:", entity_filter if entity_filter else "ALL")
st.sidebar.write("Row count after filters:", len(filtered_df))

# ------------------------------------
#  8b) Keyword Search with History
# ------------------------------------
try:
    from rapidfuzz import process, fuzz  # optional
except Exception:
    process = None
    fuzz = None
st.markdown("## 🔍 Keyword Search")
valid_columns = [col for col in ["PR Number", "Purchase Doc", "Product Name", "PO Vendor"] if col in df.columns]
search_data = []
if valid_columns:
    for idx, row in df[valid_columns].fillna("").astype(str).iterrows():
        combined = " | ".join(row[col] for col in valid_columns)
        search_data.append((combined.lower(), idx))
if "search_history" not in st.session_state:
    st.session_state.search_history = []
user_query = st.text_input("Start typing a keyword (e.g., vendor, product, PO, PR...)", "")
if user_query and user_query not in st.session_state.search_history:
    st.session_state.search_history.append(user_query)
if st.session_state.search_history:
    with st.expander("🕘 Search History"):
        st.write(st.session_state.search_history[-10:])
with st.expander("🏷️ Filter by Tags"):
    selected_categories = st.multiselect("Procurement Category", sorted(df["Procurement Category"].dropna().unique())) if "Procurement Category" in df.columns else []
    selected_vendors = st.multiselect("PO Vendor", sorted(df["PO Vendor"].dropna().unique())) if "PO Vendor" in df.columns else []
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
        def convert_df_to_csv(df_): return df_.to_csv(index=False).encode('utf-8')
        def convert_df_to_excel(df_):
            from io import BytesIO
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_.to_excel(writer, index=False, sheet_name='Search_Results')
            return output.getvalue()
        st.download_button("⬇️ Download CSV", convert_df_to_csv(result_df), file_name="search_results.csv", mime='text/csv')
        st.download_button("⬇️ Download Excel", convert_df_to_excel(result_df), file_name="search_results.xlsx")
    else:
        st.warning("No matching results found.")
else:
    st.info("Start typing a keyword to search...")

# ------------------------------------
#  9) Top KPI Row
# ------------------------------------
st.title("📊 Procure-to-Pay Dashboard")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total PRs",        filtered_df.get("PR Number", pd.Series(dtype=object)).nunique())
col2.metric("Total POs",        filtered_df.get("Purchase Doc", pd.Series(dtype=object)).nunique())
col3.metric("Line Items",       len(filtered_df))
col4.metric("Entities",         filtered_df.get("Entity", pd.Series(dtype=object)).nunique())
col5.metric("Spend (Cr ₹)",     f"{filtered_df.get('Net Amount', pd.Series(0)).sum() / 1e7:,.2f}")

# ------------------------------------
# 10) SLA Compliance Gauge (PR → PO ≤ 7 days)
# ------------------------------------
st.subheader("🎯 SLA Compliance (PR → PO ≤ 7 days)")
lead_df = filtered_df[filtered_df.get("Po create Date").notna()] if "Po create Date" in filtered_df.columns else filtered_df.copy()
if "Po create Date" in filtered_df.columns and "PR Date Submitted" in filtered_df.columns:
    lead_df = lead_df.copy()
    lead_df["Lead Time (Days)"] = (
        pd.to_datetime(lead_df["Po create Date"]) - pd.to_datetime(lead_df["PR Date Submitted"])
    ).dt.days
    SLA_DAYS = 7
    avg_lead = float(lead_df["Lead Time (Days)"].mean()) if not lead_df.empty else 0.0
    gauge_fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=avg_lead,
            number={"suffix": " days"},
            gauge={
                "axis": {"range": [0, max(14, avg_lead * 1.2 if avg_lead else 14)]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, SLA_DAYS], "color": "lightgreen"},
                    {"range": [SLA_DAYS, max(14, avg_lead * 1.2 if avg_lead else 14)], "color": "lightcoral"},
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
    st.info("Need both 'Po create Date' and 'PR Date Submitted' to compute lead time.")

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
                marker=dict(opacity=0.85)
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
                hovertemplate="₹ %{y:.2f} Cr<br>%{x}<extra></extra>"
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
spend_df["PO Month"] = (
    pd.to_datetime(spend_df.get("Po create Date"), errors="coerce")
    .dt.to_period("M")
    .dt.to_timestamp()
)
if "Net Amount" in spend_df.columns:
    monthly_spend = (
        spend_df.dropna(subset=["PO Month"])
        .groupby(["PO Month", "Entity"], as_index=False)["Net Amount"]
        .sum()
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

# ------------------------------------
# 11) PR → PO Lead Time by Buyer Type & Buyer
# ------------------------------------
st.subheader("⏱️ PR to PO Lead Time by Buyer Type & by Buyer")
if "Lead Time (Days)" not in lead_df.columns and "Po create Date" in filtered_df.columns and "PR Date Submitted" in filtered_df.columns:
    lead_df = filtered_df.copy()
    lead_df["Lead Time (Days)"] = (pd.to_datetime(lead_df["Po create Date"]) - pd.to_datetime(lead_df["PR Date Submitted"])).dt.days
lead_avg_by_type = (
    lead_df.groupby("Buyer.Type")["Lead Time (Days)"]
    .mean()
    .round(0)
    .reset_index()
) if "Lead Time (Days)" in lead_df.columns else pd.DataFrame()
lead_avg_by_buyer = (
    lead_df.groupby("PO.Creator")["Lead Time (Days)"]
    .mean()
    .round(0)
    .reset_index()
) if "Lead Time (Days)" in lead_df.columns else pd.DataFrame()
c1, c2 = st.columns(2)
c1.dataframe(lead_avg_by_type, use_container_width=True)
c2.dataframe(lead_avg_by_buyer, use_container_width=True)

# ------------------------------------
# 12) Monthly PR & PO Trends
# ------------------------------------
st.subheader("📅 Monthly PR & PO Trends")
if "PR Date Submitted" in filtered_df.columns:
    filtered_df["PR Month"] = pd.to_datetime(filtered_df["PR Date Submitted"]).dt.to_period("M")
if "Po create Date" in filtered_df.columns:
    filtered_df["PO Month"] = pd.to_datetime(filtered_df["Po create Date"]).dt.to_period("M")
monthly_summary = (
    filtered_df.groupby("PR Month")
    .agg({"PR Number": "count", "Purchase Doc": "count"})
    .reset_index()
) if "PR Month" in filtered_df.columns else pd.DataFrame(columns=["Month","PR Count","PO Count"])
if not monthly_summary.empty:
    monthly_summary.columns = ["Month", "PR Count", "PO Count"]
    monthly_summary["Month"] = monthly_summary["Month"].astype(str)
    st.line_chart(monthly_summary.set_index("Month"), use_container_width=True)

# ------------------------------------
# 14) PR → PO Aging Buckets
# ------------------------------------
st.subheader("🧮 PR to PO Aging Buckets")
if "Lead Time (Days)" in lead_df.columns:
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
        age_summary, x="Aging Bucket", y="Percentage", text="Percentage",
        title="PR to PO Aging Bucket Distribution (%)", labels={"Percentage": "Percentage (%)"},
    )
    fig_aging.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    st.plotly_chart(fig_aging, use_container_width=True)

# ------------------------------------
# 15) PRs & POs by Weekday
# ------------------------------------
st.subheader("📆 PRs and POs by Weekday")
df_wd = filtered_df.copy()
if "PR Date Submitted" in df_wd.columns:
    df_wd["PR Weekday"] = pd.to_datetime(df_wd["PR Date Submitted"]).dt.day_name()
if "Po create Date" in df_wd.columns:
    df_wd["PO Weekday"] = pd.to_datetime(df_wd["Po create Date"]).dt.day_name()
pr_counts = df_wd.get("PR Weekday", pd.Series(dtype=object)).value_counts().reindex(
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], fill_value=0,
)
po_counts = df_wd.get("PO Weekday", pd.Series(dtype=object)).value_counts().reindex(
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], fill_value=0,
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
    if not open_df.empty and "PR Date Submitted" in open_df.columns:
        open_df["Pending Age (Days)"] = (
            pd.to_datetime(pd.Timestamp.today().date()) - pd.to_datetime(open_df["PR Date Submitted"])
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
            }).reset_index()
        )
        st.metric("🔢 Open PRs", open_summary["PR Number"].nunique())
        open_monthly_counts = pd.to_datetime(open_summary["PR Date Submitted"]).dt.to_period("M").value_counts().sort_index()
        st.bar_chart(open_monthly_counts, use_container_width=True)
        def highlight_age(val): return "background-color: red" if val > 30 else ""
        st.dataframe(open_summary.style.applymap(highlight_age, subset=["Pending Age (Days)"]), use_container_width=True)
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
if "PR Date Submitted" in filtered_df.columns:
    daily_df = filtered_df.copy()
    daily_df["PR Date"] = pd.to_datetime(daily_df["PR Date Submitted"])
    daily_trend = daily_df.groupby("PR Date").size().reset_index(name="PR Count")
    fig_daily = px.line(daily_trend, x="PR Date", y="PR Count", title="Daily PR Submissions", labels={"PR Count": "PR Count"})
    st.plotly_chart(fig_daily, use_container_width=True)

# ------------------------------------
# 18) Buyer-wise Spend
# ------------------------------------
st.subheader("💰 Buyer-wise Spend (Cr ₹)")
if "Net Amount" in filtered_df.columns and "PO.Creator" in filtered_df.columns:
    buyer_spend = (
        filtered_df.groupby("PO.Creator")["Net Amount"].sum().sort_values(ascending=False).reset_index()
    )
    buyer_spend["Net Amount (Cr)"] = buyer_spend["Net Amount"] / 1e7
    fig_buyer = px.bar(
        buyer_spend, x="PO.Creator", y="Net Amount (Cr)", title="Spend by Buyer",
        labels={"Net Amount (Cr)": "Spend (Cr ₹)", "PO.Creator": "Buyer"}, text="Net Amount (Cr)",
    )
    fig_buyer.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    st.plotly_chart(fig_buyer, use_container_width=True)

# ------------------------------------
#  Category Spend Chart Sorted Descending
# ------------------------------------
st.subheader("🧭 Spend by Category (Descending)")
if "Procurement Category" in filtered_df.columns and "Net Amount" in filtered_df.columns:
    cat_spend = (
        filtered_df.groupby("Procurement Category")["Net Amount"]
        .sum().sort_values(ascending=False).reset_index()
    )
    cat_spend["Spend (Cr ₹)"] = cat_spend["Net Amount"] / 1e7
    fig_cat = px.bar(
        cat_spend, x="Procurement Category", y="Spend (Cr ₹)",
        title="Spend by Category (Descending)", labels={"Spend (Cr ₹)": "Spend (Cr ₹)", "Procurement Category": "Category"},
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
    po_app_df = filtered_df[filtered_df.get("Po create Date").notna()].copy()
    po_app_df["PO Approved Date"] = pd.to_datetime(po_app_df["PO Approved Date"], errors="coerce")
    total_pos    = po_app_df.get("Purchase Doc", pd.Series(dtype=object)).nunique()
    approved_pos = po_app_df[po_app_df["PO Approved Date"].notna()].get("Purchase Doc", pd.Series(dtype=object)).nunique()
    pending_pos  = total_pos - approved_pos
    po_app_df["PO Approval Lead Time"] = (po_app_df["PO Approved Date"] - pd.to_datetime(po_app_df["Po create Date"])).dt.days
    avg_approval = float(po_app_df["PO Approval Lead Time"].mean()) if not po_app_df.empty else 0.0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📦 Total POs",           total_pos)
    c2.metric("✅ Approved POs",        approved_pos)
    c3.metric("⏳ Pending Approval",    pending_pos)
    c4.metric("⏱️ Avg Approval Lead Time (days)", avg_approval)
    st.subheader("📄 Detailed PO Approval Aging List")
    approval_detail = po_app_df[["PO.Creator", "Purchase Doc", "Po create Date", "PO Approved Date", "PO Approval Lead Time"]].sort_values(by="PO Approval Lead Time", ascending=False)
    st.dataframe(approval_detail, use_container_width=True)
else:
    st.info("ℹ️ 'PO Approved Date' column not found.")

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
        fig_status = px.pie(po_status_summary, names="PO Status", values="Count", title="PO Status Distribution", hole=0.3)
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
if "PO Qty" in delivery_df.columns and "Received Qty" in delivery_df.columns:
    delivery_df["% Received"] = (delivery_df["Received Qty"] / delivery_df["PO Qty"]) * 100
    delivery_df["% Received"] = delivery_df["% Received"].fillna(0).round(1)
    po_delivery_summary = (
        delivery_df.groupby(["Purchase Doc", "PO Vendor", "Product Name", "Item Description"], dropna=False)
        .agg({"PO Qty":"sum","Received Qty":"sum","Pending Qty":"sum","% Received":"mean"})
        .reset_index()
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
    fully_received    = (delivery_df["Pending Qty"] == 0).sum()
    partially_pending = (delivery_df["Pending Qty"] > 0).sum()
    avg_receipt_pct   = float(delivery_df["% Received"].mean())
    st.markdown("### 📋 Delivery Performance Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PO Lines",        total_po_lines)
    c2.metric("Fully Delivered", fully_received)
    c3.metric("Pending Delivery", partially_pending)
    c4.metric("Avg. Receipt %",   f"{avg_receipt_pct:.1f}%")
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
if "Pending Qty" in delivery_df.columns and "PO Unit Rate" in delivery_df.columns:
    pending_items = delivery_df[delivery_df["Pending Qty"] > 0].copy()
    pending_items["Pending Value"] = pending_items["Pending Qty"] * pending_items["PO Unit Rate"]
    top_pending_items = (
        pending_items.sort_values(by="Pending Value", ascending=False)
        .head(50)[["PR Number","Purchase Doc","Procurement Category","Buying legal entity","PR Budget description",
                   "Product Name","Item Description","Pending Qty","Pending Value"]]
        .reset_index(drop=True)
    )
    st.dataframe(
        top_pending_items.style.format({"Pending Qty": "{:,.0f}", "Pending Value": "₹ {:,.2f}"}),
        use_container_width=True
    )

# ------------------------------------
# 23) Top 10 Vendors by Spend
# ------------------------------------
st.subheader("🏆 Top 10 Vendors by Spend (Cr ₹)")
if all(c in filtered_df.columns for c in ["PO Vendor", "Purchase Doc", "Net Amount"]):
    vendor_spend = (
        filtered_df.groupby("PO Vendor", dropna=False)
        .agg(Vendor_PO_Count=("Purchase Doc", "nunique"),
             Total_Spend_Cr=("Net Amount", lambda x: (x.sum() / 1e7).round(2)))
        .reset_index()
        .sort_values(by="Total_Spend_Cr", ascending=False)
    )
    top10_spend = vendor_spend.head(10).copy()
    st.dataframe(top10_spend, use_container_width=True)
    fig_top_vendors = px.bar(
        top10_spend, x="PO Vendor", y="Total_Spend_Cr",
        title="Top 10 Vendors by Spend (Cr ₹)",
        labels={"Total_Spend_Cr": "Spend (Cr ₹)", "PO Vendor": "Vendor"}, text="Total_Spend_Cr",
    )
    fig_top_vendors.update_traces(textposition="outside", texttemplate="%{text:.2f}")
    st.plotly_chart(fig_top_vendors, use_container_width=True)

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
        .agg(Total_PO_Count=("Purchase Doc", "nunique"),
             Fully_Delivered_PO_Count=("Is_Fully_Delivered", "sum"),
             Late_PO_Count=("Is_Late", "sum"))
        .reset_index()
    )
    vendor_perf["Pct_Fully_Delivered"] = (vendor_perf["Fully_Delivered_PO_Count"]/vendor_perf["Total_PO_Count"]*100).round(1)
    vendor_perf["Pct_Late"] = (vendor_perf["Late_PO_Count"]/vendor_perf["Total_PO_Count"]*100).round(1)
    if "vendor_spend" in locals():
        vendor_perf = vendor_perf.merge(vendor_spend[["PO Vendor","Total_Spend_Cr"]], on="PO Vendor", how="left")
        top10_vendor_perf = vendor_perf.sort_values("Total_Spend_Cr", ascending=False).head(10)
    else:
        top10_vendor_perf = vendor_perf.sort_values("Total_PO_Count", ascending=False).head(10)
        top10_vendor_perf["Total_Spend_Cr"] = None
    st.dataframe(top10_vendor_perf[
        ["PO Vendor","Total_PO_Count","Fully_Delivered_PO_Count","Late_PO_Count","Pct_Fully_Delivered","Pct_Late","Total_Spend_Cr"]
    ], use_container_width=True)
    melted_perf = top10_vendor_perf.melt(
        id_vars=["PO Vendor"], value_vars=["Pct_Fully_Delivered", "Pct_Late"],
        var_name="Metric", value_name="Percentage",
    )
    fig_vendor_perf = px.bar(
        melted_perf, x="PO Vendor", y="Percentage", color="Metric", barmode="group",
        title="% Fully Delivered vs % Late (Top 10 Vendors by Spend)",
        labels={"Percentage": "% of POs", "PO Vendor": "Vendor"},
    )
    st.plotly_chart(fig_vendor_perf, use_container_width=True)

# ------------------------------------
# 25) Monthly Unique PO Generation
# ------------------------------------
st.subheader("🗓️ Monthly Unique PO Generation")
if "Purchase Doc" in filtered_df.columns and "Po create Date" in filtered_df.columns:
    po_monthly = filtered_df[filtered_df["Purchase Doc"].notna()].copy()
    po_monthly["PO Month"] = pd.to_datetime(po_monthly["Po create Date"]).dt.to_period("M")
    monthly_po_counts = po_monthly.groupby("PO Month")["Purchase Doc"].nunique().reset_index(name="Unique PO Count")
    monthly_po_counts["PO Month"] = monthly_po_counts["PO Month"].astype(str)
    fig_monthly_po = px.bar(
        monthly_po_counts, x="PO Month", y="Unique PO Count",
        title="Monthly Unique PO Generation", labels={"PO Month": "Month", "Unique PO Count": "Number of Unique POs"},
        text="Unique PO Count",
    )
    fig_monthly_po.update_traces(textposition="outside")
    st.plotly_chart(fig_monthly_po, use_container_width=True)

# ------------------------------------
# 33) Department-wise Spend — Smart Budget Mapper (Unified)
# ------------------------------------
st.subheader("🏢 Department-wise Spend — Smart Mapper")

# Work on a copy
smart_df = filtered_df.copy()
smart_df["Dept.Chart"], smart_df["Subcat.Chart"] = pd.NA, pd.NA
smart_df["__Dept.MapSrc"] = "UNMAPPED"  # EXACT | HIER | ENTITY_PFX | PFX3 | KEYWORD | FALLBACK | UNMAPPED

def _norm_code_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip().str.upper()
    s = s.str.replace("\\xa0", " ", regex=False)
    s = s.str.replace("&", "AND", regex=False)
    s = s.str.replace(r"\\s+", " ", regex=True)
    s = s.str.replace(r"\\.+$", "", regex=True)  # remove trailing dots
    s = s.str.replace(r"\\.{2,}", ".", regex=True)  # collapse multiple dots
    s = s.str.replace(" ", "")  # remove internal spaces
    return s

# Load expanded mapping
expanded = None
try:
    expanded = pd.read_excel("Expanded_Budget_Code_Mapping.xlsx")
except Exception:
    try:
        expanded = pd.read_excel("Final_Budget_Mapping_Completed_Verified.xlsx")
    except Exception:
        expanded = None

exact_map = {}
exact_map_sub = {}
entity_pfx_map = {}
entity_pfx_map_sub = {}
pfx_map = {}
pfx_map_sub = {}

if expanded is not None:
    exp = expanded.copy()
    exp.columns = exp.columns.astype(str).str.strip()
    code_col = next((c for c in exp.columns if c.lower().strip() in ["budget code","code","budget_code"]), None)
    dept_col = next((c for c in exp.columns if "department" in c.lower()), None)
    subc_col = next((c for c in exp.columns if ("subcat" in c.lower()) or ("sub category" in c.lower()) or ("subcategory" in c.lower())), None)
    p3_col   = next((c for c in exp.columns if "prefix_3" in c.lower()), None)
    ent_col  = next((c for c in exp.columns if c.lower().strip() in ["entity","domain","company","prefix_1","brand"]), None)

    if code_col and dept_col:
        exp["__code_norm"] = _norm_code_series(exp[code_col])
        tmp = exp.dropna(subset=["__code_norm"]).drop_duplicates(subset=["__code_norm"], keep="first")
        exact_map = dict(zip(tmp["__code_norm"], tmp[dept_col].astype(str).str.strip()))
        if subc_col:
            exact_map_sub = dict(zip(tmp["__code_norm"], tmp[subc_col].astype(str).str.strip()))
    if p3_col:
        exp["__p3_norm"] = _norm_code_series(exp[p3_col])
    if ent_col:
        exp["__ent_norm"] = _norm_code_series(exp[ent_col])

    if p3_col and dept_col and ent_col and not exp.empty:
        grp = exp.dropna(subset=["__p3_norm","__ent_norm"]).copy()
        if not grp.empty:
            mode_dept = grp.groupby(["__ent_norm","__p3_norm"])[dept_col].agg(lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0])
            entity_pfx_map = mode_dept.to_dict()
            if subc_col:
                mode_sub = grp.groupby(["__ent_norm","__p3_norm"])[subc_col].agg(lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0])
                entity_pfx_map_sub = mode_sub.to_dict()

    if p3_col and dept_col and not exp.empty:
        grp2 = exp.dropna(subset=["__p3_norm"]).copy()
        if not grp2.empty:
            mode_dept2 = grp2.groupby("__p3_norm")[dept_col].agg(lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0])
            pfx_map = mode_dept2.to_dict()
            if subc_col:
                mode_sub2 = grp2.groupby("__p3_norm")[subc_col].agg(lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0])
                pfx_map_sub = mode_sub2.to_dict()

def _norm_one(x):
    if pd.isna(x):
        return ""
    return _norm_code_series(pd.Series([x])).iloc[0]

entity_series = smart_df.get("Entity", pd.Series([pd.NA]*len(smart_df)))
entity_norm = entity_series.apply(_norm_one)

p3_keys = set(pfx_map.keys())

def _map_single(code_raw, ent_raw):
    code = _norm_one(code_raw)
    ent  = _norm_one(ent_raw)
    if not code:
        return (pd.NA, pd.NA, "UNMAPPED")
    # Tier 1: EXACT
    if code in exact_map:
        return (exact_map.get(code), exact_map_sub.get(code, pd.NA), "EXACT")
    # Tier 2: HIER (drop left segments)
    parts = code.split('.')
    if len(parts) > 1:
        for i in range(1, len(parts)):
            suf = '.'.join(parts[i:])
            if suf in exact_map:
                return (exact_map.get(suf), exact_map_sub.get(suf, pd.NA), "HIER")
    # pick Prefix_3 candidate from right-to-left
    def _pick_p3(c):
        segs = c.split('.')
        for j in range(len(segs)-1, -1, -1):
            seg = segs[j]
            if seg in p3_keys:
                return seg
        return None
    p3 = _pick_p3(code)
    # Tier 3: ENTITY + PFX3
    if p3 and ent and (ent, p3) in entity_pfx_map:
        return (entity_pfx_map.get((ent,p3)), entity_pfx_map_sub.get((ent,p3), pd.NA), "ENTITY_PFX")
    # Tier 4: PFX3 only
    if p3 and (p3 in pfx_map):
        return (pfx_map.get(p3), pfx_map_sub.get(p3, pd.NA), "PFX3")
    # Tier 5: KEYWORD
    tokens = set(parts)
    if {"MFG"} & tokens: return ("Manufacturing", pd.NA, "KEYWORD")
    if {"R&D","RANDD","PRDDEV","TV","VIC","PT"} & tokens: return ("R&D", pd.NA, "KEYWORD")
    if {"HR","HRP","ADM","OTHADM","PLANT","PLNT"} & tokens: return ("HR & Admin", pd.NA, "KEYWORD")
    if {"MKT","A&M"} & tokens: return ("Marketing", pd.NA, "KEYWORD")
    if {"INF","PUNE","CONCOR","SAF"} & tokens: return ("Infra", pd.NA, "KEYWORD")
    if {"CNST","PM","PRJ"} & tokens: return ("Program", pd.NA, "KEYWORD")
    if {"FIN"} & tokens: return ("Finance", pd.NA, "KEYWORD")
    if {"LGL","LGLF","IP","PRT"} & tokens: return ("Legal & IP", pd.NA, "KEYWORD")
    if {"SS","COG","TLG"} & tokens: return ("SS & SCM", pd.NA, "KEYWORD")
    if {"SLS"} & tokens: return ("Sales", pd.NA, "KEYWORD")
    if {"RENT","COUR"} & tokens: return ("Rental Offices", pd.NA, "KEYWORD")
    return (pd.NA, pd.NA, "UNMAPPED")

code_cols_avail = [c for c in ["PO Budget Code","PR Budget Code"] if c in smart_df.columns]
if code_cols_avail:
    base = smart_df[code_cols_avail[0]]
    mapped = pd.DataFrame([_map_single(c,e) for c,e in zip(base.tolist(), entity_norm.tolist())],
                          columns=["__dept","__subc","__src"], index=smart_df.index)
    smart_df["Dept.Chart"] = smart_df["Dept.Chart"].combine_first(mapped["__dept"]) if "Dept.Chart" in smart_df.columns else mapped["__dept"]
    smart_df["Subcat.Chart"] = smart_df["Subcat.Chart"].combine_first(mapped["__subc"]) if "Subcat.Chart" in smart_df.columns else mapped["__subc"]
    need_src = smart_df["__Dept.MapSrc"].isin(["UNMAPPED", pd.NA, None])
    smart_df.loc[need_src, "__Dept.MapSrc"] = mapped.loc[need_src, "__src"].fillna("UNMAPPED")

pre_fallback_na = smart_df["Dept.Chart"].isna()
if pre_fallback_na.all():
    for cand in ["PO Department", "PO Dept", "PR Department", "PR Dept", "Dept.Final", "Department"]:
        if cand in smart_df.columns:
            smart_df["Dept.Chart"] = smart_df["Dept.Chart"].combine_first(smart_df[cand])
smart_df.loc[smart_df["__Dept.MapSrc"].eq("UNMAPPED") & smart_df["Dept.Chart"].notna(), "__Dept.MapSrc"] = "FALLBACK"
smart_df["Dept.Chart"].fillna("Unmapped / Missing", inplace=True)

with st.expander("🧪 Smart Mapper QA", expanded=False):
    st.write({"counts": smart_df["__Dept.MapSrc"].value_counts(dropna=False).to_dict()})
    if code_cols_avail:
        unm = smart_df[smart_df["__Dept.MapSrc"] == "UNMAPPED"].copy()
        if not unm.empty:
            unm["Budget Code (normalized)"] = _norm_code_series(unm[code_cols_avail[0]])
            summary_unm = unm["Budget Code (normalized)"].value_counts().reset_index()
            summary_unm.columns = ["Budget Code (normalized)", "Lines"]
            st.dataframe(summary_unm.head(200), use_container_width=True)
            st.download_button("⬇️ Download Unmapped Budget Codes", summary_unm.to_csv(index=False),
                               "unmapped_budget_codes.csv", "text/csv", key="dl_unmapped_codes_smart")
        else:
            st.caption("All lines mapped by Smart Mapper.")

if "Net Amount" in smart_df.columns:
    dept_spend = smart_df.groupby("Dept.Chart", dropna=False)["Net Amount"].sum().reset_index().sort_values("Net Amount", ascending=False)
    dept_spend["Spend (Cr ₹)"] = dept_spend["Net Amount"] / 1e7
    st.plotly_chart(
        px.bar(dept_spend.head(30), x="Dept.Chart", y="Spend (Cr ₹)", title="Department-wise Spend (Top 30) — Smart Mapper")
        .update_layout(xaxis_tickangle=-45),
        use_container_width=True,
    )
    dd1, dd2 = st.columns([2,1])
    with dd1:
        dept_pick = st.selectbox("Drill down: choose a department (Smart)", dept_spend["Dept.Chart"].tolist())
    with dd2:
        topn = st.number_input("Show top N vendors/items", min_value=5, max_value=100, value=20, step=5, key="smart_topn")
    detail = smart_df[smart_df["Dept.Chart"].astype(str) == str(dept_pick)].copy()
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Lines", len(detail))
    k2.metric("PRs", int(detail.get("PR Number", pd.Series(dtype=object)).nunique()))
    k3.metric("POs", int(detail.get("Purchase Doc", pd.Series(dtype=object)).nunique()))
    k4.metric("Spend (Cr ₹)", f"{(detail.get('Net Amount', pd.Series(0)).sum()/1e7):,.2f}")
    c3,c4 = st.columns(2)
    if {"PO Vendor","Net Amount"}.issubset(detail.columns):
        top_v = detail.groupby("PO Vendor", dropna=False)["Net Amount"].sum().sort_values(ascending=False).head(int(topn)).reset_index()
        top_v["Spend (Cr ₹)"] = top_v["Net Amount"]/1e7
        c3.plotly_chart(px.bar(top_v, x="PO Vendor", y="Spend (Cr ₹)", title="Top Vendors (Cr ₹) — Smart").update_layout(xaxis_tickangle=-45), use_container_width=True)
    if {"Product Name","Net Amount"}.issubset(detail.columns):
        top_i = detail.groupby("Product Name", dropna=False)["Net Amount"].sum().sort_values(ascending=False).head(int(topn)).reset_index()
        top_i["Spend (Cr ₹)"] = top_i["Net Amount"]/1e7
        c4.plotly_chart(px.bar(top_i, x="Product Name", y="Spend (Cr ₹)", title="Top Items (Cr ₹) — Smart").update_layout(xaxis_tickangle=-45), use_container_width=True)
    dcol = "Po create Date" if "Po create Date" in detail.columns else ("PR Date Submitted" if "PR Date Submitted" in detail.columns else None)
    if dcol and "Net Amount" in detail.columns:
        detail[dcol] = pd.to_datetime(detail[dcol], errors="coerce")
        m = detail.dropna(subset=[dcol]).groupby(detail[dcol].dt.to_period('M'))['Net Amount'].sum().to_timestamp()
        st.plotly_chart(px.line(m/1e7, labels={'value':'Spend (Cr ₹)','index':'Month'}, title=f"{dept_pick} — Monthly Spend — Smart"), use_container_width=True)
    desired_order = [
        "PO Budget Code", "Subcat.Chart", "Dept.Chart", "Purchase Doc", "PR Number",
        "Procurement Category", "Product Name", "Item Description",
    ]
    show_cols = [c for c in desired_order if c in detail.columns]
    st.dataframe(detail[show_cols], use_container_width=True)
    st.download_button(
        "⬇️ Download Department Lines (CSV) — Smart",
        detail[show_cols].to_csv(index=False),
        file_name=f"dept_drilldown_smart_{str(dept_pick).replace(' ','_')}.csv",
        mime="text/csv",
        key=f"dl_dept_lines_smart_{str(dept_pick)}",
    )
    with st.expander("View table / download — Smart"):
        st.dataframe(dept_spend, use_container_width=True)
        st.download_button(
            "⬇️ Download Department Spend (CSV) — Smart",
            dept_spend.to_csv(index=False),
            "department_spend_smart.csv",
            "text/csv",
            key="dl_dept_spend_smart_csv",
        )

# ------------------------------------
# 34) End of Dashboard
# ------------------------------------
''')

path = "/mnt/data/p2p_dashboard_final.py"
with open(path, "w", encoding="utf-8") as f:
    f.write(code)

print("Wrote file:", path)
