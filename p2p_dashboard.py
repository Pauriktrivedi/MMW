import re
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="P2P Dashboard — Indirect", layout="wide", initial_sidebar_state="expanded")

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
    """Normalize column names to snake_case lowercase and remove NBSPs.

    Defensive about backslashes and punctuation to avoid unterminated string errors.
    """
    new_cols = {}
    for c in df.columns:
        s = str(c).strip()
        s = s.replace(" ", " ")
        s = s.replace("\", "_").replace("/", "_")
        s = "_".join(s.split())
        s = s.lower()
        s = ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in s)
        s = re.sub('_+', '_', s).strip('_')
        new_cols[c] = s
    return df.rename(columns=new_cols)

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

# Load data
df = load_all()
if df.empty:
    st.warning("No data loaded. Place MEPL.xlsx, MLPL.xlsx, mmw.xlsx, mmpl.xlsx next to this script.")

# Column shortcuts
pr_col = 'pr_date_submitted' if 'pr_date_submitted' in df.columns else None
po_create_col = 'po_create_date' if 'po_create_date' in df.columns else None
net_amount_col = 'net_amount' if 'net_amount' in df.columns else None
purchase_doc_col = 'purchase_doc' if 'purchase_doc' in df.columns else None
pr_number_col = 'pr_number' if 'pr_number' in df.columns else None
po_vendor_col = 'po_vendor' if 'po_vendor' in df.columns else None
entity_col = 'entity'
pr_dept_cols = [c for c in ['pr_department','pr_dept','department','pr_department_name'] if c in df.columns]
pr_dept_col = pr_dept_cols[0] if pr_dept_cols else None

TODAY = pd.Timestamp.now().normalize()

# Sidebar filters
st.sidebar.header('Filters')
FY = {
    'All Years': (pd.Timestamp('2023-04-01'), pd.Timestamp('2026-03-31')),
    '2023': (pd.Timestamp('2023-04-01'), pd.Timestamp('2024-03-31')),
    '2024': (pd.Timestamp('2024-04-01'), pd.Timestamp('2025-03-31')),
    '2025': (pd.Timestamp('2025-04-01'), pd.Timestamp('2026-03-31'))
}
fy_key = st.sidebar.selectbox('Financial Year', list(FY))
pr_start, pr_end = FY[fy_key]

fil = df.copy()
if pr_col and pr_col in fil.columns:
    fil = fil[(fil[pr_col] >= pr_start) & (fil[pr_col] <= pr_end)]

# Ensure vendor/item columns exist
if 'po_vendor' not in fil.columns:
    fil['po_vendor'] = ''
if 'product_name' not in fil.columns:
    fil['product_name'] = ''
fil['po_vendor'] = fil['po_vendor'].astype(str).str.strip()
fil['product_name'] = fil['product_name'].astype(str).str.strip()

# Vendor / Item multiselects — default to empty for performance
vendor_choices = sorted(fil['po_vendor'].dropna().unique().tolist())
item_choices = sorted(fil['product_name'].dropna().unique().tolist())
sel_v = st.sidebar.multiselect('Vendor (pick one or more)', vendor_choices, default=None)
sel_i = st.sidebar.multiselect('Item / Product (pick one or more)', item_choices, default=None)

if st.sidebar.button('Reset Filters'):
    for k in list(st.session_state.keys()):
        try:
            del st.session_state[k]
        except Exception:
            pass
    st.experimental_rerun()

if sel_v:
    fil = fil[fil['po_vendor'].isin(sel_v)]
if sel_i:
    fil = fil[fil['product_name'].isin(sel_i)]

# Tabs
T = st.tabs(['KPIs & Spend','PO/PR Timing','Delivery','Vendors','Dept & Services','Forecast','Search'])

# KPIs & Spend
with T[0]:
    c1, c2, c3, c4 = st.columns(4)
    total_prs = int(fil[pr_number_col].nunique()) if pr_number_col in fil.columns else 0
    total_pos = int(fil[purchase_doc_col].nunique()) if purchase_doc_col in fil.columns else 0
    c1.metric('Total PRs', total_prs)
    c2.metric('Total POs', total_pos)
    c3.metric('Line Items', len(fil))
    spend_val = fil[net_amount_col].sum() if net_amount_col in fil.columns else 0
    c4.metric('Spend (Cr ₹)', f"{spend_val/1e7:,.2f}")

    st.markdown('---')
    # Monthly spend + cumulative (limit to last 24 months for speed)
    if po_create_col and net_amount_col in fil.columns:
        t = fil.dropna(subset=[po_create_col]).copy()
        t['month'] = t[po_create_col].dt.to_period('M').dt.to_timestamp()
        m = t.groupby('month', as_index=False)[net_amount_col].sum().sort_values('month')
        # keep last 24 months
        if len(m) > 24:
            m = m.tail(24)
        m['cr'] = m[net_amount_col]/1e7
        m['cum'] = m['cr'].cumsum()
        fig = make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_bar(x=m['month'], y=m['cr'], name='Monthly Spend (Cr ₹)')
        fig.add_scatter(x=m['month'], y=m['cum'], name='Cumulative (Cr ₹)', mode='lines+markers', secondary_y=True)
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('Need date and Net Amount columns for spend charts.')

# PO/PR Timing
with T[1]:
    st.subheader('SLA (PR→PO ≤7d)')
    if pr_col in fil.columns and po_create_col in fil.columns:
        ld = fil.dropna(subset=[po_create_col, pr_col]).copy()
        ld['lead_time_days'] = (ld[po_create_col] - ld[pr_col]).dt.days
        avg = float(ld['lead_time_days'].mean()) if not ld.empty else 0.0
        st.metric('Avg Lead Time (days)', f"{avg:.1f}")

# Delivery
with T[2]:
    st.subheader('Delivery Summary')
    if 'pending_qty' in fil.columns:
        summary = fil.groupby('po_vendor', dropna=False)['pending_qty'].sum().reset_index().sort_values('pending_qty', ascending=False)
        st.dataframe(summary.head(20), use_container_width=True)

# Vendors
with T[3]:
    st.subheader('Top Vendors by Spend')
    if po_vendor_col in fil.columns and net_amount_col in fil.columns:
        vs = fil.groupby('po_vendor', dropna=False)[net_amount_col].sum().reset_index().sort_values(net_amount_col, ascending=False)
        vs['cr'] = vs[net_amount_col]/1e7
        st.plotly_chart(px.bar(vs.head(20), x='po_vendor', y='cr', text='cr').update_traces(textposition='outside'), use_container_width=True)

# Dept & Services
with T[4]:
    st.subheader('Department Spend (using PR department)')
    if pr_dept_col:
        dep = fil.groupby(pr_dept_col)[net_amount_col].sum().reset_index().sort_values(net_amount_col, ascending=False)
        dep['cr'] = dep[net_amount_col]/1e7
        st.plotly_chart(px.bar(dep.head(30), x=pr_dept_col, y='cr', title='Department-wise Spend (Top 30)').update_layout(xaxis_tickangle=-45), use_container_width=True)
    else:
        st.info('PR department column not found in data. Put a column named pr_department or pr_dept or department.')

# Forecast
with T[5]:
    st.subheader('SMA Forecast (next month)')
    if po_create_col and net_amount_col in fil.columns:
        t = fil.dropna(subset=[po_create_col]).copy()
        t['month'] = t[po_create_col].dt.to_period('M').dt.to_timestamp()
        m = t.groupby('month')[net_amount_col].sum().sort_index()/1e7
        if len(m) >= 3:
            sma = m.rolling(3).mean()
            st.line_chart(pd.DataFrame({'Actual':m, 'SMA':sma}))
        else:
            st.info('Not enough monthly points for SMA.')

# Search
with T[6]:
    st.subheader('Quick Search')
    q = st.text_input('Search Vendor/Product')
    if q:
        mask = fil['po_vendor'].str.contains(q, case=False, na=False) | fil['product_name'].str.contains(q, case=False, na=False)
        st.dataframe(fil[mask].head(200), use_container_width=True)

# EOF

