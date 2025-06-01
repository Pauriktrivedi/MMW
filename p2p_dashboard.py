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
    initial_sidebar_state="expanded"
)

# ------------------------------------
#  1) Load & Combine Source Data
# ------------------------------------
@st.cache_data(show_spinner=False)
def load_and_combine_data():
    """
    Reads the four Excel files (from the same folder):
      - MEPL.xlsx
      - MLPL.xlsx
      - mmw.xlsx
      - mmpl.xlsx

    Tags each sheet with an "Entity" column, concatenates them, and 
    cleans up column names.
    """
    # NOTE: Filenames are case‐sensitive. In this repo, they must be:
    #       MEPL.xlsx   (uppercase)
    #       MLPL.xlsx   (uppercase)
    #       mmw.xlsx    (lowercase)
    #       mmpl.xlsx   (lowercase)
    mepl_df = pd.read_excel("MEPL.xlsx", skiprows=1)
    mlpl_df = pd.read_excel("MLPL.xlsx", skiprows=1)
    mmw_df  = pd.read_excel("mmw.xlsx",  skiprows=1)
    mmpl_df = pd.read_excel("mmpl.xlsx", skiprows=1)

    # Tag each with an "Entity" column
    mepl_df["Entity"] = "MEPL"
    mlpl_df["Entity"] = "MLPL"
    mmw_df["Entity"]  = "MMW"
    mmpl_df["Entity"] = "MMPL"

    # Concatenate into one DataFrame
    combined = pd.concat([mepl_df, mlpl_df, mmw_df, mmpl_df], ignore_index=True)

    # Clean column names: strip whitespace, replace non-breaking spaces
    combined.columns = (
        combined.columns
        .str.strip()
        .str.replace("\xa0", " ", regex=False)
        .str.replace(" +", " ", regex=True)
    )
    combined.rename(columns=lambda c: c.strip(), inplace=True)

    return combined

# Actually load the data
df = load_and_combine_data()

# ------------------------------------
#  2) Clean & Prepare Date Columns
# ------------------------------------
for date_col in ["PR Date Submitted", "Po create Date"]:
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.date
    else:
        st.error(f"❌ Column '{date_col}' not found in the data.")

# ------------------------------------
#  3) Buyer Group Classification
# ------------------------------------
if "Buyer Group" in df.columns:
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
df["PO.Creator"]  = df["PO Orderer"].map(o_created_by_map).fillna(df["PO Orderer"])
df["PO.Creator"]  = df["PO.Creator"].replace({"N/A": "Dilip"})

indirect_buyers = ["Aatish","Deepak","Deepakex","Dhruv","Dilip","Mukul","Nayan","Paurik","Kamlesh","Suresh"]
df["PO.BuyerType"] = df["PO.Creator"].apply(lambda x: "Indirect" if x in indirect_buyers else "Direct")

# ------------------------------------
#  5) Sidebar Filters & Keyword Search
# ------------------------------------
st.sidebar.header("🔍 Filters")

# Determine min/max for date sliders
pr_min = pd.to_datetime(df["PR Date Submitted"]).min()
pr_max = pd.to_datetime(df["PR Date Submitted"]).max()
po_min = pd.to_datetime(df["Po create Date"]).min()
po_max = pd.to_datetime(df["Po create Date"]).max()

pr_range = st.sidebar.date_input(
    "PR Date Range",
    value=[pr_min, pr_max],
    key="pr_range"
)
po_range = st.sidebar.date_input(
    "PO Date Range",
    value=[po_min, po_max],
    key="po_range"
)

buyer_filter = st.sidebar.multiselect(
    "Buyer Type",
    options=df["Buyer.Type"].unique(),
    default=list(df["Buyer.Type"].unique()),
    key="buyer_filter"
)
entity_filter = st.sidebar.multiselect(
    "Entity",
    options=df["Entity"].unique(),
    default=list(df["Entity"].unique()),
    key="entity_filter"
)
orderer_filter = st.sidebar.multiselect(
    "PO Ordered By",
    options=df["PO.Creator"].unique(),
    default=list(df["PO.Creator"].unique()),
    key="orderer_filter"
)
po_buyer_type_filter = st.sidebar.multiselect(
    "PO Buyer Type",
    options=df["PO.BuyerType"].unique(),
    default=list(df["PO.BuyerType"].unique()),
    key="po_buyer_type_filter"
)

st.sidebar.header("🔎 Keyword Search")
search_term = st.sidebar.text_input("Search PR/PO/Product", key="search_term")

# ------------------------------------
#  6) Apply Filters → filtered_df
# ------------------------------------
filtered_df = df.copy()

# 6a) Filter by PR Date Submitted
filtered_df = filtered_df[
    filtered_df["PR Date Submitted"].between(pr_range[0], pr_range[1])
]

# 6b) Filter by PO create Date (allow NA through)
po_mask = (
    filtered_df["Po create Date"].notna() 
    & filtered_df["Po create Date"].between(po_range[0], po_range[1])
)
filtered_df = filtered_df[po_mask | filtered_df["Po create Date"].isna()]

# 6c) Filter by Buyer.Type, Entity, PO.Creator, PO.BuyerType
filtered_df = filtered_df[
    (filtered_df["Buyer.Type"].isin(buyer_filter))
    & (filtered_df["Entity"].isin(entity_filter))
    & (filtered_df["PO.Creator"].isin(orderer_filter))
    & (filtered_df["PO.BuyerType"].isin(po_buyer_type_filter))
]

# 6d) Keyword Search (PR Number, Purchase Doc, Product Name)
if search_term:
    mask_search = (
        filtered_df["PR Number"].astype(str).str.contains(search_term, case=False, na=False)
        | filtered_df["Purchase Doc"].astype(str).str.contains(search_term, case=False, na=False)
        | filtered_df["Product Name"].astype(str).str.contains(search_term, case=False, na=False)
    )
    search_results = filtered_df[mask_search]
    st.subheader(f"🔍 Search Results for '{search_term}'")
    st.dataframe(search_results, use_container_width=True)

# ------------------------------------
#  7) Top KPI Row (Total PRs, POs, Line Items, Entities, Spend)
# ------------------------------------
st.title("📊 Procure-to-Pay Dashboard")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total PRs", filtered_df["PR Number"].nunique())
c2.metric("Total POs", filtered_df["Purchase Doc"].nunique())
c3.metric("Line Items", len(filtered_df))
c4.metric("Entities", filtered_df["Entity"].nunique())
c5.metric(
    "Spend (Cr ₹)", 
    f"{filtered_df['Net Amount'].sum() / 1e7:,.2f}"
)

# ------------------------------------
#  8) SLA Compliance Gauge (PR → PO ≤ 7 days)
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
                {"range": [0, SLA_DAYS],               "color": "lightgreen"},
                {"range": [SLA_DAYS, max(14, avg_lead * 1.2)], "color": "lightcoral"}
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": SLA_DAYS
            }
        },
        title={"text": "Average Lead Time"}
    )
)
st.plotly_chart(gauge_fig, use_container_width=True)
st.caption(f"Current Avg Lead Time: {avg_lead:.1f} days   •   Target ≤ {SLA_DAYS} days")

# ------------------------------------
#  9) PR → PO Lead Time by Buyer Type & Buyer
# ------------------------------------
st.subheader("⏱️ PR to PO Lead Time by Buyer Type & by Buyer")
lead_avg_by_type  = lead_df.groupby("Buyer.Type")["Lead Time (Days)"].mean().round(0).reset_index()
lead_avg_by_buyer = lead_df.groupby("PO.Creator")["Lead Time (Days)"].mean().round(0).reset_index()

col1, col2 = st.columns(2)
col1.dataframe(lead_avg_by_type,  use_container_width=True)
col2.dataframe(lead_avg_by_buyer, use_container_width=True)

# ------------------------------------
# 10) Monthly PR & PO Trends
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
# 11) Procurement Category Spend
# ------------------------------------
st.subheader("📦 Procurement Category Spend")
if "Procurement Category" in filtered_df.columns:
    cat_spend = (
        filtered_df.groupby("Procurement Category")["Net Amount"]
        .sum()
        .reset_index()
    )
    cat_spend["Spend (Cr ₹)"] = cat_spend["Net Amount"] / 1e7

    fig_cat = px.bar(
        cat_spend,
        x="Procurement Category",
        y="Spend (Cr ₹)",
        title="Spend by Category",
        labels={"Spend (Cr ₹)": "Spend (Cr ₹)", "Procurement Category": "Category"}
    )
    fig_cat.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_cat, use_container_width=True)
else:
    st.info("ℹ️ No 'Procurement Category' column found.")

# ------------------------------------
# 12) PR → PO Aging Buckets
# ------------------------------------
st.subheader("🧮 PR to PO Aging Buckets")
bins   = [0, 7, 15, 30, 60, 90, 999]
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
    labels={"Percentage": "Percentage (%)"}
)
fig_aging.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
st.plotly_chart(fig_aging, use_container_width=True)

# ------------------------------------
# 13) PRs & POs by Weekday
# ------------------------------------
st.subheader("📆 PRs and POs by Weekday")
df_wd = filtered_df.copy()
df_wd["PR Weekday"] = pd.to_datetime(df_wd["PR Date Submitted"]).dt.day_name()
df_wd["PO Weekday"] = pd.to_datetime(df_wd["Po create Date"]).dt.day_name()

pr_counts = df_wd["PR Weekday"].value_counts().reindex(
    ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"], fill_value=0
)
po_counts = df_wd["PO Weekday"].value_counts().reindex(
    ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"], fill_value=0
)

col1, col2 = st.columns(2)
col1.bar_chart(pr_counts, use_container_width=True)
col2.bar_chart(po_counts, use_container_width=True)

# ------------------------------------
# 14) Open PRs (Approved / InReview)
# ------------------------------------
st.subheader("⚠️ Open PRs (Approved/InReview)")
if "PR Status" in filtered_df.columns:
    open_df = filtered_df[filtered_df["PR Status"].isin(["Approved","InReview"])].copy()
    if not open_df.empty:
        open_df["Pending Age (Days)"] = (
            pd.to_datetime(pd.Timestamp.today().date())
            - pd.to_datetime(open_df["PR Date Submitted"])
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
                "Purchase Doc":        "first"
            })
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
            open_summary.style.applymap(highlight_age, subset=["Pending Age (Days)"]),
            use_container_width=True
        )

        st.subheader("🏢 Open PRs by Entity")
        ent_counts = open_summary["Entity"].value_counts().reset_index()
        ent_counts.columns = ["Entity", "Count"]
        st.bar_chart(ent_counts.set_index("Entity"), use_container_width=True)
    else:
        st.warning("⚠️ No open PRs match the current filters.")
else:
    st.info("ℹ️ 'PR Status' column not found in the dataset.")

# ------------------------------------
# 15) Daily PR Submissions Trend
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
    labels={"PR Count": "PR Count"}
)
st.plotly_chart(fig_daily, use_container_width=True)

# ------------------------------------
# 16) Buyer-wise Spend
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
    text="Net Amount (Cr)"
)
fig_buyer.update_traces(texttemplate="%{text:.2f}", textposition="outside")
st.plotly_chart(fig_buyer, use_container_width=True)

# ------------------------------------
# 17) PO Approval Summary & Details
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
    c1.metric("📦 Total POs", total_pos)
    c2.metric("✅ Approved POs", approved_pos)
    c3.metric("⏳ Pending Approval", pending_pos)
    c4.metric("⏱️ Avg Approval Lead Time (days)", avg_approval)

    st.subheader("📄 Detailed PO Approval Aging List")
    approval_detail = po_app_df[
        ["PO.Creator", "Purchase Doc", "Po create Date", "PO Approved Date", "PO Approval Lead Time"]
    ].sort_values(by="PO Approval Lead Time", ascending=False)
    st.dataframe(approval_detail, use_container_width=True)
else:
    st.info("ℹ️ 'PO Approved Date' column not found.")

# ------------------------------------
# 18) PO Status Breakdown
# ------------------------------------
if "PO Status" in filtered_df.columns:
    st.subheader("📊 PO Status Breakdown")
    po_status_summary = (
        filtered_df["PO Status"]
        .value_counts()
        .reset_index()
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
            hole=0.3
        )
        fig_status.update_traces(textinfo="percent+label")
        st.plotly_chart(fig_status, use_container_width=True)
else:
    st.info("ℹ️ 'PO Status' column not found.")

# ------------------------------------
# 19) PO Delivery Summary: Received vs Pending
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
        "% Received":   "mean"
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
    text="Pending Qty"
)
fig_pending.update_traces(textposition="outside")
st.plotly_chart(fig_pending, use_container_width=True)

total_po_lines    = len(delivery_df)
fully_received    = (delivery_df["Pending Qty"] == 0).sum()
partially_pending = (delivery_df["Pending Qty"] > 0).sum()
avg_receipt_pct   = delivery_df["% Received"].mean().round(1)

st.markdown("### 📋 Delivery Performance Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("PO Lines", total_po_lines)
c2.metric("Fully Delivered", fully_received)
c3.metric("Pending Delivery", partially_pending)
c4.metric("Avg. Receipt %", f"{avg_receipt_pct}%")

st.download_button(
    "📥 Download Delivery Status",
    data=po_delivery_summary.to_csv(index=False),
    file_name="PO_Delivery_Status.csv",
    mime="text/csv"
)

# ------------------------------------
# 20) Top 50 Pending Delivery Lines by Value
# ------------------------------------
st.subheader("📋 Top 50 Pending Lines (by Value)")
pending_items = delivery_df[delivery_df["Pending Qty"] > 0].copy()
pending_items["Pending Value"] = pending_items["Pending Qty"] * pending_items["PO Unit Rate"]

top_pending_items = (
    pending_items.sort_values(by="Pending Value", ascending=False)
    .head(50)[[
        "PR Number",
        "Purchase Doc",
        "Procurement Category",
        "Buying legal entity",
        "PR Budget description",
        "Product Name",
        "Item Description",
        "Pending Qty",
        "Pending Value"
    ]]
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
# 21) Top 10 Vendors by Spend
# ------------------------------------
st.subheader("🏆 Top 10 Vendors by Spend (Cr ₹)")
if all(c in filtered_df.columns for c in ["PO Vendor", "Purchase Doc", "Net Amount"]):
    vendor_spend = (
        filtered_df.groupby("PO Vendor", dropna=False)
        .agg(
            Vendor_PO_Count=("Purchase Doc", "nunique"),
            Total_Spend_Cr=("Net Amount", lambda x: (x.sum() / 1e7).round(2))
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
        text="Total_Spend_Cr"
    )
    fig_top_vendors.update_traces(textposition="outside", texttemplate="%{text:.2f}")
    st.plotly_chart(fig_top_vendors, use_container_width=True)
else:
    st.info("ℹ️ Cannot compute Top 10 Vendors – required columns missing.")

# ------------------------------------
# 22) Vendor Delivery Performance
# ------------------------------------
st.subheader("📊 Vendor Delivery Performance (Top 10 by Spend)")
if all(c in filtered_df.columns for c in ["PO Vendor", "Purchase Doc", "PO Delivery Date", "Pending QTY"]):
    today = pd.Timestamp.today().normalize().date()
    df_vp = filtered_df.copy()
    df_vp["Pending Qty Filled"] = df_vp["Pending QTY"].fillna(0).astype(float)
    df_vp["Is_Fully_Delivered"] = df_vp["Pending Qty Filled"] == 0
    df_vp["PO Delivery Date"] = pd.to_datetime(df_vp["PO Delivery Date"], errors="coerce")

    df_vp["Is_Late"] = (
        df_vp["PO Delivery Date"].dt.date.notna() &
        (df_vp["PO Delivery Date"].dt.date < today) &
        (df_vp["Pending Qty Filled"] > 0)
    )

    vendor_perf = (
        df_vp.groupby("PO Vendor", dropna=False)
        .agg(
            Total_PO_Count=("Purchase Doc", "nunique"),
            Fully_Delivered_PO_Count=("Is_Fully_Delivered", "sum"),
            Late_PO_Count=("Is_Late", "sum")
        )
        .reset_index()
    )
    vendor_perf["Pct_Fully_Delivered"] = (
        (vendor_perf["Fully_Delivered_PO_Count"] / vendor_perf["Total_PO_Count"] * 100).round(1)
    )
    vendor_perf["Pct_Late"] = (
        (vendor_perf["Late_PO_Count"] / vendor_perf["Total_PO_Count"] * 100).round(1)
    )

    # If we computed vendor_spend above, merge it for sorting
    if "vendor_spend" in locals():
        vendor_perf = vendor_perf.merge(
            vendor_spend[["PO Vendor", "Total_Spend_Cr"]],
            on="PO Vendor",
            how="left"
        )
        top10_vendor_perf = vendor_perf.sort_values("Total_Spend_Cr", ascending=False).head(10)
    else:
        top10_vendor_perf = vendor_perf.sort_values("Total_PO_Count", ascending=False).head(10)
        top10_vendor_perf["Total_Spend_Cr"] = None

    st.dataframe(
        top10_vendor_perf[
            ["PO Vendor","Total_PO_Count","Fully_Delivered_PO_Count","Late_PO_Count","Pct_Fully_Delivered","Pct_Late","Total_Spend_Cr"]
        ],
        use_container_width=True
    )

    melted_perf = top10_vendor_perf.melt(
        id_vars=["PO Vendor"],
        value_vars=["Pct_Fully_Delivered","Pct_Late"],
        var_name="Metric",
        value_name="Percentage"
    )
    fig_vendor_perf = px.bar(
        melted_perf,
        x="PO Vendor",
        y="Percentage",
        color="Metric",
        barmode="group",
        title="% Fully Delivered vs % Late (Top 10 Vendors by Spend)",
        labels={"Percentage":"% of POs","PO Vendor":"Vendor"}
    )
    st.plotly_chart(fig_vendor_perf, use_container_width=True)
else:
    st.info("ℹ️ Cannot compute Vendor Delivery Performance – required columns missing.")

# ------------------------------------
# 23) Monthly Unique PO Generation
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
    labels={"PO Month":"Month","Unique PO Count":"Number of Unique POs"},
    text="Unique PO Count"
)
fig_monthly_po.update_traces(textposition="outside")
st.plotly_chart(fig_monthly_po, use_container_width=True)

# ------------------------------------
# 24) Monthly Spend Trend by Entity
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

# Convert timestamp to string like "Apr-2023"
monthly_spend["Month_Str"] = monthly_spend["PO Month"].dt.strftime("%b-%Y")

fig_spend = px.line(
    monthly_spend,
    x="Month_Str",
    y="Spend (Cr ₹)",
    color="Entity",
    markers=True,
    title="Monthly Spend Trend by Entity",
    labels={"Month_Str":"Month","Spend (Cr ₹)":"Spend (Cr ₹)"}
)
fig_spend.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig_spend, use_container_width=True)

# ------------------------------------
# 25) Today’s Snapshot (KPIs)
# ------------------------------------
st.subheader("📅 Today’s Snapshot")
today = pd.Timestamp.today().normalize().date()

today_prs = filtered_df[
    pd.to_datetime(filtered_df["PR Date Submitted"]).dt.date == today
]
today_pos = filtered_df[
    pd.to_datetime(filtered_df["Po create Date"]).dt.date == today
]

pr_today_count = today_prs["PR Number"].nunique()
po_today_count = today_pos["Purchase Doc"].nunique()

# Count new Open PRs submitted today
if "open_df" in locals() and not open_df.empty:
    open_prs_today = open_df[
        pd.to_datetime(open_df["PR Date Submitted"]).dt.date == today
    ]["PR Number"].nunique()
else:
    open_prs_today = 0

# Count POs pending approval today
if "po_app_df" in locals():
    pending_approval_today = po_app_df[
        (pd.to_datetime(po_app_df["Po create Date"]).dt.date == today)
        & (po_app_df["PO Approved Date"].isna())
    ]["Purchase Doc"].nunique()
else:
    pending_approval_today = 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("PRs Submitted Today",       pr_today_count)
c2.metric("POs Created Today",          po_today_count)
c3.metric("New Open PRs Today",         open_prs_today)
c4.metric("POs Pending Approval Today", pending_approval_today)

# ------------------------------------
# 26) Top Buyers (This Month)
# ------------------------------------
st.subheader("🏆 Top Buyers (By # of PRs Closed This Month)")
this_month = pd.Timestamp.today().to_period("M")
this_month_prs = filtered_df[
    pd.to_datetime(filtered_df["PR Date Submitted"]).dt.to_period("M") == this_month
]

prs_per_buyer = (
    this_month_prs.groupby("PO.Creator")["PR Number"]
    .nunique()
    .reset_index(name="PR Count")
    .sort_values("PR Count", ascending=False)
)
st.dataframe(prs_per_buyer.head(5).reset_index(drop=True), use_container_width=True)

# ------------------------------------
# 27) Daily PR → PO Conversion Trend (%)
# ------------------------------------
st.subheader("📈 Daily PR → PO Conversion Trend (%)")
tmp_pr = filtered_df.copy()
tmp_pr["PR_Date"] = pd.to_datetime(tmp_pr["PR Date Submitted"])
daily_prs = (
    tmp_pr.groupby(tmp_pr["PR_Date"].dt.date)
    .agg(PRs=("PR Number", "nunique"))
    .reset_index()
    .rename(columns={"PR_Date": "Date"})
)

tmp_po = filtered_df.copy()
tmp_po["PO_Date"] = pd.to_datetime(tmp_po["Po create Date"])
daily_pos = (
    tmp_po.groupby(tmp_po["PO_Date"].dt.date)
    .agg(POs=("Purchase Doc", "nunique"))
    .reset_index()
    .rename(columns={"PO_Date": "Date"})
)

daily_merge = pd.merge(daily_prs, daily_pos, on="Date", how="outer").fillna(0)
daily_merge["Conversion %"] = (
    (daily_merge["POs"] / daily_merge["PRs"] * 100).round(1).fillna(0)
)

fig_conv = px.line(
    daily_merge.sort_values("Date"),
    x="Date",
    y="Conversion %",
    title="Daily PR → PO Conversion Rate (%)",
    markers=True,
    labels={"Conversion %": "Conversion %"}
)
st.plotly_chart(fig_conv, use_container_width=True)

# ------------------------------------
# 28) Work Assignments by Buyer
# ------------------------------------
st.subheader("📝 Work Assignments by Buyer")
if "open_df" in locals() and not open_df.empty:
    assignments = (
        open_df.groupby("PO.Creator")
        .agg(Open_PR_Count=("PR Number", "nunique"))
        .reset_index()
        .sort_values("Open_PR_Count", ascending=False)
    )
    st.dataframe(assignments, use_container_width=True)
else:
    st.info("ℹ️ No open PRs to display work assignments.")

# ------------------------------------
# 29) End of Dashboard
# ------------------------------------
