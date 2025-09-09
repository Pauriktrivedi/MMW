import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ====================================
#  Procure-to-Pay Dashboard (Streamlit)
# ====================================

# --- Page Configuration ---
st.set_page_config(
    page_title="Procure-to-Pay Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Procure-to-Pay Dashboard")

# --- File Upload ---
st.sidebar.header("Upload Excel Files")
mepl_file = st.sidebar.file_uploader("Upload MEPL.xlsx", type=["xlsx"])
mlpl_file = st.sidebar.file_uploader("Upload MLPL.xlsx", type=["xlsx"])
mmw_file  = st.sidebar.file_uploader("Upload mmw.xlsx", type=["xlsx"])
mmpl_file = st.sidebar.file_uploader("Upload mmpl.xlsx", type=["xlsx"])

# --- Load Data ---
@st.cache_data
def load_excel(file):
    return pd.read_excel(file, skiprows=1)

mepl_df = load_excel(mepl_file) if mepl_file else None
mlpl_df = load_excel(mlpl_file) if mlpl_file else None
mmw_df  = load_excel(mmw_file)  if mmw_file else None
mmpl_df = load_excel(mmpl_file) if mmpl_file else None

# --- Display Data Previews ---
if mepl_df is not None:
    st.subheader("MEPL Data Preview")
    st.dataframe(mepl_df.head())

if mlpl_df is not None:
    st.subheader("MLPL Data Preview")
    st.dataframe(mlpl_df.head())

if mmw_df is not None:
    st.subheader("MMW Data Preview")
    st.dataframe(mmw_df.head())

if mmpl_df is not None:
    st.subheader("MMPL Data Preview")
    st.dataframe(mmpl_df.head())

# --- Sample Charts ---
# Combine all uploaded data for demonstration
combined_df = pd.DataFrame()
for df in [mepl_df, mlpl_df, mmw_df, mmpl_df]:
    if df is not None:
        combined_df = pd.concat([combined_df, df], ignore_index=True)

if not combined_df.empty:
    st.subheader("PO Value Analysis")
    if 'PO Value' in combined_df.columns:
        fig = px.histogram(combined_df, x='PO Value', nbins=20, title='PO Value Distribution')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Entity-wise PR/PO Count")
    if 'Entity' in combined_df.columns:
        entity_count = combined_df.groupby('Entity').size().reset_index(name='Count')
        fig2 = px.bar(entity_count, x='Entity', y='Count', title='Entity-wise PR/PO Count')
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Spend Trends")
    if 'Date' in combined_df.columns and 'PO Value' in combined_df.columns:
        combined_df['Date'] = pd.to_datetime(combined_df['Date'], errors='coerce')
        spend_trend = combined_df.groupby('Date')['PO Value'].sum().reset_index()
        fig3 = px.line(spend_trend, x='Date', y='PO Value', title='Monthly Spend Trends')
        st.plotly_chart(fig3, use_container_width=True)
