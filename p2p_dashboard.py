import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re

st.set_page_config(page_title="P2P Dashboard — Indirect (Final)", layout="wide", initial_sidebar_state="expanded")

# Header
st.markdown(
    """
    <div style="padding:6px 0 12px 0; margin-bottom:8px;">
      <h1 style="font-size:34px; line-height:1.05; margin:0; color:#0b1f3b;">P2P Dashboard — Indirect</h1>
      <div style="font-size:14px; color:#23395b; margin-top:6px; margin-bottom:6px;">Purchase-to-Pay overview (Indirect spend focus)</div>
      <hr style="border:0; height:1px; background:#e6eef6; margin-top:6px; margin-bottom:12px;" />
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------- Helpers -----------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = {}
    for c in df.columns:
        s = str(c).strip()
        s = s.replace("\xa0", " ")
        s = s.replace("\\", "_").replace("/", "_")
        s = "_".join(s.split())
        s = s.lower()
        s = ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in s)
        s = '_'.join([p for p in s.split('_') if p != ''])
        new_cols[c] = s
    return df.rename(columns=new_cols)

# ----------------- Load Data -----------------
@st.cache_data(show_spinner=False)
def load_all():
    fns = [("MEPL.xlsx", "MEPL"), ("MLPL.xlsx", "MLPL"), ("mmw.xlsx", "MMW"), ("mmpl.xlsx", "MMPL")]
    frames = []
    for fn, ent in fns:
        try:
            df = pd.read_excel(fn, skiprows=1)
            df['entity'] = ent
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    x = pd.concat(frames, ignore_index=True)
    x = normalize_columns(x)
    for c in ['pr_date_submitted', 'po_create_date', 'po_approved_date', 'po_delivery_date']:
        if c in x.columns:
            x[c] = pd.to_datetime(x[c], errors='coerce')
    return x

# load
df = load_all()
if df.empty:
    st.warning("No data loaded. Place MEPL.xlsx, MLPL.xlsx, mmw.xlsx, mmpl.xlsx next to this script or upload files in your environment.")

# ----------------- Column Setup -----------------
pr_col = 'pr_date_submitted' if 'pr_date_submitted' in df.columns else None
po_create_col = 'po_create_date' if 'po_create_date' in df.columns else None
net_amount_col = 'net_amount' if 'net_amount' in df.columns else None
purchase_doc_col = 'purchase_doc' if 'purchase_doc' in df.columns else None
po_vendor_col = 'po_vendor' if 'po_vendor' in df.columns else None
po_unit_rate_col = 'po_unit_rate' if 'po_unit_rate' in df.columns else None
po_delivery_col = 'po_delivery_date' if 'po_delivery_date' in df.columns else None
entity_col = 'entity'
TODAY = pd.Timestamp.now().normalize()

# initial working frame
fil = df.copy()
if pr_col and pr_col in fil.columns:
    # default FY range
    fil = fil.copy()

# ---------- Robust Sidebar Filters (place AFTER df & fil are defined, BEFORE tabs) ----------
# Debug info: helps identify why filters might be empty
st.sidebar.markdown("**DEBUG: data diagnostics**")
st.sidebar.write("Row count (df):", len(df))
st.sidebar.write("Row count (fil):", len(fil))
st.sidebar.write("Normalized columns:", df.columns.tolist()[:200])
st.sidebar.write("has po_vendor:", 'po_vendor' in df.columns)
st.sidebar.write("has product_name:", 'product_name' in df.columns)
st.sidebar.write("Sample vendors:", list(pd.Series(df.get('po_vendor', pd.Series(dtype=object))).dropna().astype(str).str.strip().unique()[:20]))
st.sidebar.write("Sample products:", list(pd.Series(df.get('product_name', pd.Series(dtype=object))).dropna().astype(str).str.strip().unique()[:20]))
st.sidebar.markdown("---")

# Financial year options
FY = {
    'All Years': (pd.Timestamp('2023-04-01'), pd.Timestamp('2026-03-31')),
    '2023': (pd.Timestamp('2023-04-01'), pd.Timestamp('2024-03-31')),
    '2024': (pd.Timestamp('2024-04-01'), pd.Timestamp('2025-03-31')),
    '2025': (pd.Timestamp('2025-04-01'), pd.Timestamp('2026-03-31'))
}

with st.sidebar:
    st.header("Filters")

    fy_key = st.selectbox("Financial Year", list(FY), index=0)
    pr_start, pr_end = FY[fy_key]

    # Build vendor/item choices from the full loaded df (safer than using fil which may be prefiltered)
    df_vendors = pd.Series(df.get('po_vendor', pd.Series(dtype=object))).astype(str).str.strip().replace({'nan':np.nan}).dropna()
    df_products = pd.Series(df.get('product_name', pd.Series(dtype=object))).astype(str).str.strip().replace({'nan':np.nan}).dropna()

    ALL_VENDORS = sorted(df_vendors.unique().tolist())[:1000]
    ALL_PRODUCTS = sorted(df_products.unique().tolist())[:1000]

    if not ALL_VENDORS:
        st.warning("No vendors found in source data (check normalized column names).")
    if not ALL_PRODUCTS:
        st.warning("No products found in source data (check normalized column names).")

    sel_v = st.multiselect("Vendor (pick one or more)", options=ALL_VENDORS, default=None, key="filter_vendor")
    sel_i = st.multiselect("Item / Product (pick one or more)", options=ALL_PRODUCTS, default=None, key="filter_item")

    if st.button("Reset Filters"):
        for k in ["filter_vendor","filter_item"]:
            if k in st.session_state:
                del st.session_state[k]
        st.experimental_rerun()

# Apply FY filter to fil
if pr_col and pr_col in fil.columns:
    fil = fil[(fil[pr_col] >= pr_start) & (fil[pr_col] <= pr_end)]

# Apply vendor/item filters from session_state
if 'filter_vendor' in st.session_state and st.session_state.get('filter_vendor'):
    fil = fil[fil['po_vendor'].astype(str).str.strip().isin(st.session_state.get('filter_vendor'))]
if 'filter_item' in st.session_state and st.session_state.get('filter_item'):
    fil = fil[fil['product_name'].astype(str).str.strip().isin(st.session_state.get('filter_item'))]

# ----------------- Tabs -----------------
T = st.tabs(['KPIs & Spend','PO/PR Timing','Delivery','Vendors','Dept & Services','Forecast','Search'])

# ----------------- KPIs & Spend -----------------
with T[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Total PRs', int(fil['pr_number'].nunique()) if 'pr_number' in fil.columns else 0)
    c2.metric('Total POs', int(fil['purchase_doc'].nunique()) if 'purchase_doc' in fil.columns else 0)
    c3.metric('Line Items', len(fil))
    spend_val = fil[net_amount_col].sum() if net_amount_col in fil.columns else 0
    c4.metric('Spend (Cr ₹)', f"{spend_val/1e7:,.2f}")

    st.markdown('---')
    if po_create_col and net_amount_col in fil.columns:
        t = fil.dropna(subset=[po_create_col]).copy()
        t['Month'] = t[po_create_col].dt.to_period('M').dt.to_timestamp()
        m = t.groupby('Month', as_index=False)[net_amount_col].sum().sort_values('Month')
        m['Cr'] = m[net_amount_col]/1e7
        m['Cum'] = m['Cr'].cumsum()
        fig = make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_bar(x=m['Month'], y=m['Cr'], name='Monthly Spend')
        fig.add_scatter(x=m['Month'], y=m['Cum'], name='Cumulative', mode='lines+markers', secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

# ----------------- PO/PR Timing -----------------
with T[1]:
    st.subheader('Lead Time (PR → PO)')
    if pr_col in fil.columns and po_create_col in fil.columns:
        ld = fil.dropna(subset=[po_create_col, pr_col]).copy()
        ld['lead_time_days'] = (ld[po_create_col] - ld[pr_col]).dt.days
        avg = ld['lead_time_days'].mean()
        st.metric('Avg Lead Time (days)', f"{avg:.1f}")

# ----------------- Delivery -----------------
with T[2]:
    st.subheader('Pending Deliveries')
    if 'pending_qty' in fil.columns:
        st.bar_chart(fil['pending_qty'])

# ----------------- Vendors -----------------
with T[3]:
    st.subheader('Top Vendors by Spend')
    if po_vendor_col in fil.columns and net_amount_col in fil.columns:
        vs = fil.groupby('po_vendor', dropna=False)[net_amount_col].sum().reset_index().sort_values(net_amount_col, ascending=False)
        vs['Cr'] = vs[net_amount_col]/1e7
        st.plotly_chart(px.bar(vs.head(10), x='po_vendor', y='Cr', text='Cr').update_traces(textposition='outside'), use_container_width=True)

# ----------------- Dept & Services (PR Department) -----------------
with T[4]:
    st.subheader('Dept & Services (PR Department)')
    pr_dept_candidates = [c for c in ['pr_department','pr_dept','pr_dept_name','pr_department_name','pr_department_code','dept_chart','department'] if c in fil.columns]
    if pr_dept_candidates:
        pr_dept_col = pr_dept_candidates[0]
        st.write(f"Using department column: **{pr_dept_col}** for Dept spend aggregation")
        d = fil.copy()
        d[pr_dept_col] = d[pr_dept_col].astype(str).str.strip().replace({'nan':pd.NA})
        if net_amount_col in d.columns:
            dep = d.groupby(pr_dept_col, dropna=False)[net_amount_col].sum().reset_index().sort_values(net_amount_col, ascending=False)
            dep['Cr'] = dep[net_amount_col]/1e7
            topn = min(30, len(dep))
            if topn > 0:
                st.plotly_chart(px.bar(dep.head(topn), x=pr_dept_col, y='Cr', title='Department-wise Spend (Top departments)').update_layout(xaxis_tickangle=-45, yaxis_title='Cr ₹'), use_container_width=True)
                st.dataframe(dep.head(200).assign(Spend_Cr=lambda df: (df[net_amount_col]/1e7).round(2)), use_container_width=True)
            else:
                st.info('No departments with spend found.')
        else:
            st.info('Net Amount column not present — cannot compute department spend.')
    else:
        st.info('No PR department-like column found in data. Expected columns: pr_department, pr_dept, pr_dept_name, dept_chart, etc.')

# ----------------- Forecast -----------------
with T[5]:
    st.subheader('Simple Moving Average Forecast')
    if po_create_col and net_amount_col in fil.columns:
        t = fil.dropna(subset=[po_create_col]).copy()
        t['Month'] = t[po_create_col].dt.to_period('M').dt.to_timestamp()
        m = t.groupby('Month')[net_amount_col].sum().sort_index()/1e7
        sma = m.rolling(3).mean()
        st.line_chart(pd.DataFrame({'Actual':m,'SMA':sma}))

# ----------------- Search -----------------
with T[6]:
    st.subheader('Keyword Search')
    q = st.text_input('Search Vendor/Product')
    if q:
        mask = fil['po_vendor'].str.contains(q, case=False, na=False) | fil['product_name'].str.contains(q, case=False, na=False)
        st.dataframe(fil[mask])

# EOF
