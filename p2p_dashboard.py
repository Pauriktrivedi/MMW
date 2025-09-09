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

# --- 1) Load Data ---
@st.cache_data
def load_data():
    # Placeholder CSV (replace with your dataset path)
    df = pd.read_csv("p2p_data.csv")
    return df

# Load data
df = load_data()

# --- 2) Sidebar Filters ---
st.sidebar.header("Filters")

# Date Range Filter
date_col = "PO_Date"
df[date_col] = pd.to_datetime(df[date_col])
min_date, max_date = df[date_col].min(), df[date_col].max()
date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])

if len(date_range) == 2:
    df = df[(df[date_col] >= pd.to_datetime(date_range[0])) & (df[date_col] <= pd.to_datetime(date_range[1]))]

# PR Status Filter
pr_status = st.sidebar.multiselect("PR Status", options=df["PR_Status"].unique(), default=df["PR_Status"].unique())
df = df[df["PR_Status"].isin(pr_status)]

# Buyer Group Filter
buyer_group = st.sidebar.multiselect("Buyer Group", options=df["Buyer_Group"].unique(), default=df["Buyer_Group"].unique())
df = df[df["Buyer_Group"].isin(buyer_group)]

# --- 3) Dashboard Layout ---
st.title("📊 Procure-to-Pay Dashboard")

# KPI Section
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total PO Value", f"₹{df['PO_Value'].sum():,.0f}")
with col2:
    st.metric("Total PR Count", f"{df['PR_No'].nunique():,}")
with col3:
    st.metric("Total PO Count", f"{df['PO_No'].nunique():,}")

# --- 4) Charts ---

# PO Value Trend
fig_po_trend = px.line(
    df.groupby(df[date_col].dt.to_period("M"))["PO_Value"].sum().reset_index().rename(columns={date_col: "Month"}),
    x="Month", y="PO_Value", title="Monthly PO Value Trend"
)
st.plotly_chart(fig_po_trend, use_container_width=True)

# Entity-wise PR/PO count
entity_counts = df.groupby("Entity").agg({"PR_No": "nunique", "PO_No": "nunique"}).reset_index()
fig_entity = go.Figure(data=[
    go.Bar(name="PR Count", x=entity_counts["Entity"], y=entity_counts["PR_No"]),
    go.Bar(name="PO Count", x=entity_counts["Entity"], y=entity_counts["PO_No"])
])
fig_entity.update_layout(barmode="group", title="Entity-wise PR/PO Count")
st.plotly_chart(fig_entity, use_container_width=True)

# Spend by Buyer Group
fig_buyer = px.pie(df, names="Buyer_Group", values="PO_Value", title="Spend by Buyer Group")
st.plotly_chart(fig_buyer, use_container_width=True)

# PR Status Tracking
fig_status = px.bar(
    df.groupby("PR_Status")["PR_No"].count().reset_index().rename(columns={"PR_No": "Count"}),
    x="PR_Status", y="Count", title="PR Status Distribution"
)
st.plotly_chart(fig_status, use_container_width=True)

# --- 5) Data Table ---
st.subheader("📋 Detailed Data View")
st.dataframe(df)
