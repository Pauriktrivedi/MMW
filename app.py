import subprocess
import streamlit as st
import pandas as pd
from pathlib import Path

# Auto rebuild parquet on app start
if Path("convert_to_parquet.py").exists():
    subprocess.run(["python", "convert_to_parquet.py"], check=False)

st.set_page_config(layout="wide")

df = pd.read_parquet("p2p_data.parquet")

st.sidebar.header("Filters")

min_date = df["Posting Date"].min()
max_date = df["Posting Date"].max()

start, end = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

filtered = df[(df["Posting Date"] >= pd.to_datetime(start)) & (df["Posting Date"] <= pd.to_datetime(end))]

st.caption(f"Data updated till: {max_date.strftime('%d %b %Y')}")

st.dataframe(filtered)