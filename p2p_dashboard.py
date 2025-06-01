import pandas as pd
import streamlit as st
import plotly.express as px
import io

# === Load and Combine Data ===
mepl_df = pd.read_excel("/Users/paurik/Downloads/MEPL.xlsx", skiprows=1)
mlpl_df = pd.read_excel("/Users/paurik/Downloads/MLPL.xlsx", skiprows=1)
mmw_df = pd.read_excel("/Users/paurik/Downloads/MMW.xlsx", skiprows=1)
mmpl_df = pd.read_excel("/Users/paurik/Downloads/MMPL.xlsx", skiprows=1)

# Add Entity label to each
mepl_df['Entity'] = "MEPL"
mlpl_df['Entity'] = "MLPL"
mmw_df['Entity'] = "MMW"
mmpl_df['Entity'] = "MMPL"

# Combine all data
df = pd.concat([mepl_df, mlpl_df, mmw_df, mmpl_df], ignore_index=True)
df.columns = df.columns.str.strip().str.replace('\xa0', ' ').str.replace(' +', ' ', regex=True)
df.rename(columns=lambda x: x.strip(), inplace=True)

# === Clean Date Columns ===
date_cols = ['PR Date Submitted', 'Po create Date']
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce').dt.date
    else:
        st.error(f"❌ '{col}' column not found in data.")

# === Buyer Group Classification ===
if 'Buyer Group' in df.columns:
    df['Buyer Group Code'] = df['Buyer Group'].str.extract(r'(\d+)').astype(float)
    def classify_buyer_group(row):
        if row['Buyer Group'] in ['ME_BG17', 'MLBG16']:
            return 'Direct'
        elif row['Buyer Group'] in ['Not Available'] or pd.isna(row['Buyer Group']):
            return 'Indirect'
        elif 1 <= row['Buyer Group Code'] <= 9:
            return 'Direct'
        elif 10 <= row['Buyer Group Code'] <= 18:
            return 'Indirect'
        else:
            return 'Other'
    df['Buyer.Type'] = df.apply(classify_buyer_group, axis=1)
else:
    df['Buyer.Type'] = 'Unknown'
    st.warning("⚠️ 'Buyer Group' column not found. Defaulted all to 'Unknown'.")

# === PO Orderer Mapping ===
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
df['PO Orderer'] = df['PO Orderer'].fillna("N/A").astype(str).str.strip()
df['PO.Creator'] = df['PO Orderer'].map(o_created_by_map).fillna(df['PO Orderer'])
df['PO.Creator'] = df['PO.Creator'].replace({'N/A': 'Dilip'})

# === PO Buyer Type based on known indirect list ===
indirect_buyers = ["Aatish", "Deepak", "Deepakex", "Dhruv", "Dilip", "Mukul", "Nayan", "Paurik", "Kamlesh", "Suresh"]
df['PO.BuyerType'] = df['PO.Creator'].apply(lambda x: 'Indirect' if x in indirect_buyers else 'Direct')

# === Sidebar Filters ===
st.sidebar.header("🔍 Filters")
pr_range = st.sidebar.date_input("PR Date Range", [pd.to_datetime(df['PR Date Submitted']).min(), pd.to_datetime(df['PR Date Submitted']).max()])
po_range = st.sidebar.date_input("PO Date Range", [pd.to_datetime(df['Po create Date']).min(), pd.to_datetime(df['Po create Date']).max()])
buyer_filter = st.sidebar.multiselect("Buyer Type", df['Buyer.Type'].unique(), default=df['Buyer.Type'].unique())
entity_filter = st.sidebar.multiselect("Entity", df['Entity'].unique(), default=df['Entity'].unique())
orderer_filter = st.sidebar.multiselect("PO Ordered By", df['PO.Creator'].unique(), default=df['PO.Creator'].unique())
po_buyer_type_filter = st.sidebar.multiselect("PO Buyer Type", df['PO.BuyerType'].unique(), default=df['PO.BuyerType'].unique())

# === Keyword Search ===
st.sidebar.header("🔎 Keyword Search")
search_term = st.sidebar.text_input("Search PR/PO/Product")

# === Filtered Data ===
filtered_df = df.copy()
filtered_df = filtered_df[filtered_df['PR Date Submitted'].between(pr_range[0], pr_range[1])]
po_filter_mask = filtered_df['Po create Date'].notna() & filtered_df['Po create Date'].between(po_range[0], po_range[1])
filtered_df = filtered_df[po_filter_mask | filtered_df['Po create Date'].isna()]
filtered_df = filtered_df[
    (filtered_df['Buyer.Type'].isin(buyer_filter)) &
    (filtered_df['Entity'].isin(entity_filter)) &
    (filtered_df['PO.Creator'].isin(orderer_filter)) &
    (filtered_df['PO.BuyerType'].isin(po_buyer_type_filter))
]

if search_term:
    mask = (
        filtered_df['PR Number'].astype(str).str.contains(search_term, case=False, na=False) |
        filtered_df['Purchase Doc'].astype(str).str.contains(search_term, case=False, na=False) |
        filtered_df['Product Name'].astype(str).str.contains(search_term, case=False, na=False)
    )
    search_results = filtered_df[mask]
    st.subheader(f"🔍 Search Results for '{search_term}'")
    st.dataframe(search_results)

# === Metrics ===
st.title("📊 Procure-to-Pay Dashboard")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total PRs", filtered_df['PR Number'].nunique())
col2.metric("Total POs", filtered_df['Purchase Doc'].nunique())
col3.metric("Line Items", len(filtered_df))
col4.metric("Entities", filtered_df['Entity'].nunique())
col5.metric("Spend (Cr ₹)", f"{filtered_df['Net Amount'].sum() / 1e7:,.2f}")

# === PR to PO Lead Time and Buyer-wise Cycle Time ===
st.subheader("⏱️ PR to PO Lead Time")
lead_df = filtered_df[filtered_df['Po create Date'].notna()].copy()
lead_df['Lead Time (Days)'] = (pd.to_datetime(lead_df['Po create Date']) - pd.to_datetime(lead_df['PR Date Submitted'])).dt.days
lead_avg = lead_df.groupby('Buyer.Type')['Lead Time (Days)'].mean().round(0).reset_index()
col1, col2 = st.columns(2)
col1.dataframe(lead_avg)
buyer_cycle = lead_df.groupby('PO.Creator')['Lead Time (Days)'].mean().round(0).reset_index()
col2.dataframe(buyer_cycle)

# === Monthly Analysis ===
st.subheader("📅 Monthly PR & PO Trends")
filtered_df['PR Month'] = pd.to_datetime(filtered_df['PR Date Submitted']).dt.to_period('M')
filtered_df['PO Month'] = pd.to_datetime(filtered_df['Po create Date']).dt.to_period('M')
monthly_summary = filtered_df.groupby('PR Month').agg({'PR Number': 'count', 'Purchase Doc': 'count'}).reset_index()
monthly_summary.columns = ['Month', 'PR Count', 'PO Count']
st.line_chart(monthly_summary.set_index('Month'))

# === Procurement Category Breakdown ===
st.subheader("📦 Procurement Category Spend")
if 'Procurement Category' in filtered_df.columns:
    category_spend = filtered_df.groupby('Procurement Category')['Net Amount'].sum().reset_index()
    category_spend['Spend (Cr ₹)'] = category_spend['Net Amount'] / 1e7
    fig_cat = px.bar(category_spend, x='Procurement Category', y='Spend (Cr ₹)', title='Spend by Category')
    st.plotly_chart(fig_cat, use_container_width=True)

# === Aging Buckets for PR to PO ===
st.subheader("🧮 PR to PO Aging Buckets")
aging_buckets = pd.cut(lead_df['Lead Time (Days)'], bins=[0, 7, 15, 30, 60, 90, 999], labels=['0-7','8-15','16-30','31-60','61-90','90+'])
age_summary = aging_buckets.value_counts(normalize=True).sort_index().reset_index()
age_summary.columns = ['Aging Bucket', 'Percentage']
age_summary['Percentage'] *= 100
fig_aging = px.bar(age_summary, x='Aging Bucket', y='Percentage', text='Percentage', title='PR to PO Aging Bucket Distribution (%)')
fig_aging.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
st.plotly_chart(fig_aging, use_container_width=True)

# === PRs and POs by Weekday ===
st.subheader("📆 PRs and POs by Weekday")
col1, col2 = st.columns(2)
col1.bar_chart(pd.to_datetime(filtered_df['PR Date Submitted']).dt.day_name().value_counts())
col2.bar_chart(pd.to_datetime(filtered_df['Po create Date']).dt.day_name().value_counts())

# === Open PRs ===
st.subheader("⚠️ Open PRs (Approved/InReview)")
open_df = filtered_df[filtered_df['PR Status'].isin(['Approved', 'InReview'])].copy()
if not open_df.empty:
    open_df['Pending Age (Days)'] = (
        pd.to_datetime(pd.Timestamp.today().date()) - pd.to_datetime(open_df['PR Date Submitted'])
    ).dt.days
    open_df_summary = open_df.groupby('PR Number').agg({
        'PR Date Submitted': 'first',
        'Pending Age (Days)': 'first',
        'Procurement Category': 'first',
        'Product Name': 'first',
        'Net Amount': 'sum',
        'PO Budget Code': 'first',
        'PR Status': 'first',
        'Buyer Group': 'first',
        'Buyer.Type': 'first',
        'Entity': 'first',
        'PO.Creator': 'first',
        'Purchase Doc': 'first'
    }).reset_index()
    st.metric("🔢 Open PRs", open_df_summary['PR Number'].nunique())
    st.bar_chart(pd.to_datetime(open_df_summary['PR Date Submitted']).dt.to_period('M').value_counts().sort_index())
    def highlight_age(val):
        return 'background-color: red' if val > 30 else ''
    st.dataframe(open_df_summary.style.applymap(highlight_age, subset=['Pending Age (Days)']))
    st.subheader("🏢 Open PRs by Entity")
    entity_counts = open_df_summary['Entity'].value_counts().reset_index()
    entity_counts.columns = ['Entity', 'Count']
    st.bar_chart(entity_counts.set_index('Entity'))
else:
    st.warning("⚠️ No open PRs match current filters or 'PR Status' criteria.")

# === Daily Trends ===
st.subheader("📅 Daily PR Trends")
daily_df = filtered_df.copy()
daily_df['PR Date'] = pd.to_datetime(daily_df['PR Date Submitted'])
daily_trend = daily_df.groupby('PR Date').size().reset_index(name='PR Count')
fig_daily = px.line(daily_trend, x='PR Date', y='PR Count', title='Daily PR Submissions')
st.plotly_chart(fig_daily, use_container_width=True)

# === Buyer-wise Spend ===
st.subheader("💰 Buyer-wise Spend (Cr ₹)")
buyer_spend = filtered_df.groupby('PO.Creator')['Net Amount'].sum().sort_values(ascending=False).reset_index()
buyer_spend['Net Amount (Cr)'] = buyer_spend['Net Amount'] / 1e7
fig_buyer = px.bar(buyer_spend, x='PO.Creator', y='Net Amount (Cr)', title='Spend by Buyer', labels={'Net Amount (Cr)': 'Spend (Cr ₹)'})
st.plotly_chart(fig_buyer, use_container_width=True)

## === PO Approval Summary and Detailed List ===
if 'PO Approved Date' in filtered_df.columns:
    st.subheader("📋 PO Approval Summary")
    
    po_approval_df = filtered_df[filtered_df['Po create Date'].notna()].copy()
    po_approval_df['PO Approved Date'] = pd.to_datetime(po_approval_df['PO Approved Date'], errors='coerce')
    
    total_pos = po_approval_df['Purchase Doc'].nunique()
    approved_pos = po_approval_df[po_approval_df['PO Approved Date'].notna()]['Purchase Doc'].nunique()
    pending_pos = total_pos - approved_pos
    
    po_approval_df['PO Approval Lead Time'] = (po_approval_df['PO Approved Date'] - pd.to_datetime(po_approval_df['Po create Date'])).dt.days
    
    avg_lead_time = po_approval_df['PO Approval Lead Time'].mean().round(1)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📦 Total POs", total_pos)
    col2.metric("✅ Approved POs", approved_pos)
    col3.metric("⏳ Pending Approval", pending_pos)
    col4.metric("⏱️ Avg Approval Lead Time (days)", avg_lead_time)

    # === Show Detailed Table Below ===
    st.subheader("📄 Detailed PO Approval Aging List")
    detail_df = po_approval_df[['PO.Creator', 'Purchase Doc', 'Po create Date', 'PO Approved Date', 'PO Approval Lead Time']].copy()
    detail_df = detail_df.sort_values(by='PO Approval Lead Time', ascending=False)
    st.dataframe(detail_df)

# === PO Status Count Summary ===
if 'PO Status' in filtered_df.columns:
    st.subheader("📊 PO Status Breakdown")
    
    po_status_summary = filtered_df['PO Status'].value_counts().reset_index()
    po_status_summary.columns = ['PO Status', 'Count']
    
    col1, col2 = st.columns([2, 3])
    with col1:
        st.dataframe(po_status_summary)
    with col2:
        fig_status = px.pie(po_status_summary, names='PO Status', values='Count', title='PO Status Distribution')
        fig_status.update_traces(textinfo='percent+label')
        st.plotly_chart(fig_status, use_container_width=True)
else:
    st.info("ℹ️ 'PO Status' column not found in the dataset.")

# === PO Delivery Status ===
st.subheader("🚚 PO Delivery Summary: Received vs Pending")

# Rename for clarity (optional)
delivery_df = filtered_df.copy()
delivery_df = delivery_df.rename(columns={
    'PO Quantity': 'PO Qty',
    'ReceivedQTY': 'Received Qty',
    'Pending QTY': 'Pending Qty'
})

# Add % received (if safe)
delivery_df['% Received'] = (delivery_df['Received Qty'] / delivery_df['PO Qty']) * 100
delivery_df['% Received'] = delivery_df['% Received'].fillna(0).round(1)

# 📦 Delivery Status Grid
po_delivery_summary = delivery_df.groupby(
    ['Purchase Doc', 'PO Vendor', 'Product Name', 'Item Description'], dropna=False
).agg({
    'PO Qty': 'sum',
    'Received Qty': 'sum',
    'Pending Qty': 'sum',
    '% Received': 'mean'
}).reset_index()

st.dataframe(po_delivery_summary.sort_values(by='Pending Qty', ascending=False))

# 📊 Pending Qty Chart
fig_pending = px.bar(
    po_delivery_summary.sort_values(by='Pending Qty', ascending=False).head(20),
    x='Purchase Doc',
    y='Pending Qty',
    color='PO Vendor',
    hover_data=['Product Name', 'Item Description'],
    title='Top 20 POs Awaiting Delivery (Pending Qty)',
    text='Pending Qty'
)
fig_pending.update_traces(textposition='outside')
st.plotly_chart(fig_pending, use_container_width=True)

# 📈 Delivery Performance Summary
total_po_lines = len(delivery_df)
fully_received = (delivery_df['Pending Qty'] == 0).sum()
partially_pending = (delivery_df['Pending Qty'] > 0).sum()
avg_receipt_pct = delivery_df['% Received'].mean().round(1)

st.markdown("### 📋 Delivery Performance Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("PO Lines", total_po_lines)
col2.metric("Fully Delivered", fully_received)
col3.metric("Pending Delivery", partially_pending)
col4.metric("Avg. Receipt %", f"{avg_receipt_pct}%")

# 📂 Download Button
st.download_button("📥 Download Delivery Status", po_delivery_summary.to_csv(index=False), file_name="PO_Delivery_Status.csv", mime="text/csv")
