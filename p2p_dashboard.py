import streamlit as st
import pandas as pd
import plotly.express as px

# ====================================
#  Procure-to-Pay Dashboard (Streamlit) with Canvas Upload
# ====================================

st.set_page_config(page_title="Procure-to-Pay Dashboard", layout="wide")

st.title("Procure-to-Pay Dashboard")

# --- Function to upload and load Excel files ---
def load_excel(file_label):
    uploaded_file = st.file_uploader(f"Upload {file_label}", type="xlsx")
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, skiprows=1)
        st.success(f"{file_label} loaded successfully!")
        return df
    else:
        st.warning(f"Please upload {file_label} to proceed.")
        return None

# --- Upload Excel Files ---
mepl_df = load_excel("MEPL.xlsx")
mlpl_df = load_excel("MLPL.xlsx")
mmw_df  = load_excel("mmw.xlsx")
mmpl_df = load_excel("mmpl.xlsx")

# --- Display Preview ---
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

# --- Example Chart (can customize later) ---
if mmw_df is not None:
    st.subheader("MMW PO Value Analysis")
    if 'PO Value' in mmw_df.columns and 'Date' in mmw_df.columns:
        fig = px.line(mmw_df, x='Date', y='PO Value', title='PO Value Over Time')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Columns 'Date' and 'PO Value' not found in MMW data.")
