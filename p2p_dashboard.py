import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page
st.set_page_config(page_title="P2P Dashboard — Final", layout="wide", initial_sidebar_state="expanded")

# ---------------- Helpers ----------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = {}
    for c in df.columns:
        s = str(c).strip()
        s = s.replace('\xa0', ' ')
        s = s.replace('\\', '_').replace('/', '_')
        s = '_'.join(s.split())
        s = s.lower()
        s = ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in s)
        s = '_'.join([p for p in s.split('_') if p != ''])
        new_cols[c] = s
    return df.rename(columns=new_cols)

# ---------------- Load data ----------------
@st.cache_data
def load_all(file_list=None):
    if file_list is None:
        file_list = [("MEPL.xlsx","MEPL"),("MLPL.xlsx","MLPL"),("mmw.xlsx","MMW"),("mmpl.xlsx","MMPL")]
    frames = []
    for fn, ent in file_list:
        try:
            d = pd.read_excel(fn, skiprows=1)
            d['entity'] = ent
            frames.append(d)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = normalize_columns(df)
    for c in ['pr_date_submitted','po_create_date','po_approved_date','po_delivery_date']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')
    return df

# load
df = load_all()
if df.empty:
    st.warning('No data loaded. Place MEPL.xlsx / MLPL.xlsx / mmw.xlsx / mmpl.xlsx next to the app or upload via sidebar.')

# canonical column find helpers
def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

# column aliases (normalized names expected)
pr_col = find_col(df, ['pr_date_submitted','pr_date','pr date submitted'])
po_create_col = find_col(df, ['po_create_date','po create date'])
net_amount_col = find_col(df, ['net_amount','net amount','net_amount_inr'])
purchase_doc_col = find_col(df, ['purchase_doc','purchase doc','purchase_doc_number'])
pr_number_col = find_col(df, ['pr_number','pr number','pr_no'])
po_vendor_col = find_col(df, ['po_vendor','po vendor','vendor'])
po_unit_rate_col = find_col(df, ['po_unit_rate','po unit rate','po_unit_price'])

# ---------------- Sidebar filters (kept compact) ----------------
st.sidebar.header('Filters')
FY = {'All Years':(pd.Timestamp('2023-04-01'), pd.Timestamp('2026-03-31')),'2024':(pd.Timestamp('2024-04-01'), pd.Timestamp('2025-03-31')),'2025':(pd.Timestamp('2025-04-01'), pd.Timestamp('2026-03-31'))}
fy_key = st.sidebar.selectbox('Financial Year', list(FY))
pr_start, pr_end = FY[fy_key]

fil = df.copy()
if pr_col and pr_col in fil.columns:
    fil = fil[(fil[pr_col] >= pr_start) & (fil[pr_col] <= pr_end)]

# simple buyer/creator columns make-safe
for col in ['buyer_type','po_creator','po_buyer_type','entity']:
    if col not in fil.columns:
        fil[col] = ''
    fil[col] = fil[col].astype(str).str.strip()

# vendor / product filters
if 'po_vendor' not in fil.columns:
    fil['po_vendor'] = ''
if 'product_name' not in fil.columns:
    fil['product_name'] = ''
fil['po_vendor'] = fil['po_vendor'].astype(str).str.strip()
fil['product_name'] = fil['product_name'].astype(str).str.strip()

vendor_choices = sorted(fil['po_vendor'].dropna().unique().tolist())
item_choices = sorted(fil['product_name'].dropna().unique().tolist())
sel_v = st.sidebar.multiselect('Vendor', vendor_choices, default=vendor_choices)
sel_i = st.sidebar.multiselect('Item / Product', item_choices, default=item_choices)
if sel_v:
    fil = fil[fil['po_vendor'].isin(sel_v)]
if sel_i:
    fil = fil[fil['product_name'].isin(sel_i)]

# ---------------- Tabs ----------------
T = st.tabs(['KPIs & Spend','Timing','Delivery','Vendors','Dept & Services','Outliers','Forecast','Scorecards','Search'])

# KPIs & Spend
with T[0]:
    c1,c2,c3,c4,c5 = st.columns(5)
    total_prs = int(fil.get(pr_number_col, pd.Series(dtype=object)).nunique() if pr_number_col else 0)
    total_pos = int(fil.get(purchase_doc_col, pd.Series(dtype=object)).nunique() if purchase_doc_col else 0)
    c1.metric('Total PRs', total_prs)
    c2.metric('Total POs', total_pos)
    c3.metric('Line Items', len(fil))
    c4.metric('Entities', int(fil.get('entity', pd.Series(dtype=object)).nunique()))
    spend_val = fil.get(net_amount_col, pd.Series(0)).sum() if net_amount_col else 0
    c5.metric('Spend (Cr ₹)', f"{spend_val/1e7:,.2f}")

    st.markdown('---')
    dcol = po_create_col if po_create_col in fil.columns else pr_col
    if dcol and net_amount_col in fil.columns:
        t = fil.dropna(subset=[dcol]).copy()
        t['month'] = t[dcol].dt.to_period('M').dt.to_timestamp()
        m = t.groupby('month')[net_amount_col].sum().reset_index().sort_values('month')
        m['cr'] = m[net_amount_col]/1e7
        fig = make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_bar(x=m['month'].dt.strftime('%b-%Y'), y=m['cr'], name='Monthly Spend (Cr)')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

# Dept & Services tab: show PR Budget Description first, then PR Budget Code
with T[4]:
    st.header('Dept & Services — PR-driven (Updated Version)')

    # find PR/PO budget columns (original headers might be normalized by normalize_columns)
    pr_desc_col = find_col(fil, ['pr_budget_description','pr budget description','pr_budget_desc'])
    pr_code_col = find_col(fil, ['pr_budget_code','pr budget code','pr_budgetcode'])

    # Fallback: some sheets use different cases; try literal raw column names too
    if pr_desc_col is None:
        for raw in fil.columns:
            if 'pr budget' in raw or 'pr_budget' in raw and 'desc' in raw:
                pr_desc_col = raw
                break
    if pr_code_col is None:
        for raw in fil.columns:
            if 'pr budget' in raw or 'pr_budget' in raw and 'code' in raw:
                pr_code_col = raw
                break

    st.subheader('PR Budget Description Spend (Top 30)')
    if pr_desc_col and net_amount_col and pr_desc_col in fil.columns:
        dep_desc = fil.groupby(pr_desc_col, dropna=False)[net_amount_col].sum().reset_index().sort_values(net_amount_col, ascending=False)
        dep_desc['cr'] = dep_desc[net_amount_col]/1e7
        top_desc = dep_desc.head(30).copy()
        if not top_desc.empty:
            top_desc[pr_desc_col] = top_desc[pr_desc_col].astype(str)
            fig_desc = px.bar(top_desc, x=pr_desc_col, y='cr', title='PR Budget Description Spend (Top 30)', labels={pr_desc_col: 'PR Budget Description', 'cr': 'Cr'})
            fig_desc.update_layout(xaxis_tickangle=-45, yaxis_title='Cr')
            st.plotly_chart(fig_desc, use_container_width=True)

            # provide dropdown to drill into selected description
            desc_pick = st.selectbox('Drill PR Budget Description', ['-- All --'] + top_desc[pr_desc_col].tolist(), index=0, key='desc_pick')
            if desc_pick != '-- All --':
                sel_rows = fil[fil[pr_desc_col].astype(str) == desc_pick].copy()
            else:
                sel_rows = fil[fil[pr_desc_col].isin(top_desc[pr_desc_col].astype(str).tolist())].copy()
            st.dataframe(sel_rows.head(500), use_container_width=True)
        else:
            st.info('No PR Budget Description data available.')
    else:
        st.info('PR Budget Description column or Net Amount missing.')

    st.markdown('---')
    st.subheader('PR Budget Code Spend (Top 30)')
    if pr_code_col and net_amount_col and pr_code_col in fil.columns:
        dep_code = fil.groupby(pr_code_col, dropna=False)[net_amount_col].sum().reset_index().sort_values(net_amount_col, ascending=False)
        dep_code['cr'] = dep_code[net_amount_col]/1e7
        top_code = dep_code.head(30).copy()
        if not top_code.empty:
            top_code[pr_code_col] = top_code[pr_code_col].astype(str)
            fig_code = px.bar(top_code, x=pr_code_col, y='cr', title='PR Budget Code Spend (Top 30)', labels={pr_code_col: 'PR Budget Code', 'cr': 'Cr'})
            fig_code.update_layout(xaxis_tickangle=-45, yaxis_title='Cr')
            st.plotly_chart(fig_code, use_container_width=True)

            code_pick = st.selectbox('Drill PR Budget Code', ['-- All --'] + top_code[pr_code_col].tolist(), index=0, key='code_pick')
            if code_pick != '-- All --':
                sel_rows_c = fil[fil[pr_code_col].astype(str) == code_pick].copy()
            else:
                sel_rows_c = fil[fil[pr_code_col].astype(str).isin(top_code[pr_code_col].astype(str).tolist())].copy()
            st.dataframe(sel_rows_c.head(500), use_container_width=True)
        else:
            st.info('No PR Budget Code data available.')
    else:
        st.info('PR Budget Code column or Net Amount missing.')

# ----------------- Search (kept simple) -----------------
with T[8]:
    st.subheader('🔍 Keyword Search')
    valid_cols = [c for c in [pr_number_col, purchase_doc_col, 'product_name', po_vendor_col] if c in df.columns]
    query = st.text_input('Type vendor, product, PO, PR, etc.', '')
    if query and valid_cols:
        mask = pd.Series(False, index=df.index)
        q = query.lower()
        for c in valid_cols:
            mask = mask | df[c].astype(str).str.lower().str.contains(q, na=False)
        res = df[mask].copy()
        st.write(f'Found {len(res)} rows')
        st.dataframe(res, use_container_width=True)

# EOF
