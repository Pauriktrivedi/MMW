# dept_category_spend_charts.py
# Drop-in Streamlit module to display Department-wise and Subcategory-wise spend bar charts.
# Reads the Spend_By_Department_Subcat.csv generated earlier and renders interactive Plotly charts.

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Dept & Category Spend Charts", layout="wide")
st.title("Department & Category Spend — Charts")

# Adjust this path if you store outputs elsewhere
DATA_PATH = Path("/mnt/data/outputs/Spend_By_Department_Subcat.csv")

if not DATA_PATH.exists():
    st.error(f"Spend summary not found at {DATA_PATH}. Run the mapping job first or update DATA_PATH.")
    st.stop()

# Load summary
df = pd.read_csv(DATA_PATH)
# Standardize column names
df.columns = [str(c).strip() for c in df.columns]

# Heuristics: find department, subcategory and amount columns
dept_col = next((c for c in df.columns if "depart" in c.lower()), None)
subcat_col = next((c for c in df.columns if "sub" in c.lower()), None)
amt_col = next((c for c in df.columns if any(k in c.lower() for k in ["net amount","amount","spend","total"])) , None)

if dept_col is None or subcat_col is None or amt_col is None:
    st.warning("Could not automatically detect Department, Subcategory or Amount column names in the summary file.")
    st.write("Detected columns:", df.columns.tolist())
    st.stop()

# Clean
df = df[[dept_col, subcat_col, amt_col]].dropna(subset=[amt_col])
# Convert amount to numeric
df[amt_col] = pd.to_numeric(df[amt_col], errors='coerce').fillna(0)

# Convert amount to crore for readability (if large Indian rupee amounts)
use_crore = st.checkbox("Show amounts in Crore (Cr ₹)", value=True)
if use_crore:
    df["Amount_Display"] = df[amt_col] / 1e7
    y_label = "Spend (Cr ₹)"
else:
    df["Amount_Display"] = df[amt_col]
    y_label = amt_col

# Department-wise bar chart (top 20)
st.subheader("Department-wise Spend")
df_dept = (
    df.groupby(dept_col, as_index=False)["Amount_Display"].sum()
    .sort_values("Amount_Display", ascending=False)
)

fig_dept = px.bar(
    df_dept.head(20),
    x=dept_col,
    y="Amount_Display",
    title="Top Departments by Spend",
    labels={"Amount_Display": y_label, dept_col: "Department"},
    text="Amount_Display",
)
fig_dept.update_traces(texttemplate="%{text:.2f}", textposition="outside")
fig_dept.update_layout(xaxis_tickangle=-45, margin=dict(b=160))

st.plotly_chart(fig_dept, use_container_width=True)

# Subcategory-wise bar chart (top 30)
st.subheader("Subcategory-wise Spend")
df_sub = (
    df.groupby(subcat_col, as_index=False)["Amount_Display"].sum()
    .sort_values("Amount_Display", ascending=False)
)

fig_sub = px.bar(
    df_sub.head(30),
    x=subcat_col,
    y="Amount_Display",
    title="Top Subcategories by Spend",
    labels={"Amount_Display": y_label, subcat_col: "Subcategory"},
    text="Amount_Display",
)
fig_sub.update_traces(texttemplate="%{text:.2f}", textposition="outside")
fig_sub.update_layout(xaxis_tickangle=-45, margin=dict(b=220))

st.plotly_chart(fig_sub, use_container_width=True)

# Download cleaned summary
st.markdown("---")
if st.button("Download cleaned summary CSV"):
    st.download_button("Download CSV", df.to_csv(index=False).encode('utf-8'), file_name="Spend_By_Department_Subcat_Cleaned.csv", mime='text/csv')

st.caption("Charts generated from Spend_By_Department_Subcat.csv — adjust DATA_PATH at top if file is stored elsewhere.")
