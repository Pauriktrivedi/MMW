import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

df = pd.read_parquet("p2p_data.parquet")

st.sidebar.title("Filters")

date_cols = df.select_dtypes(include=["datetime"]).columns
if len(date_cols) > 0:
    date_col = date_cols[0]
    min_d, max_d = df[date_col].min(), df[date_col].max()
    start, end = st.sidebar.date_input("Date Range", [min_d, max_d])
    df = df[(df[date_col] >= pd.to_datetime(start)) & (df[date_col] <= pd.to_datetime(end))]

st.write("### P2P Dashboard")
st.write(f"Data covers till: **{df[date_col].max().date()}**")
st.dataframe(df)