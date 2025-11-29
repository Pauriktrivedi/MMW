import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Stable, defensive P2P dashboard — minimal safe feature set
# Copy this full file into your app as p2p_dashboard.py and run with Streamlit.

st.set_page_config(page_title="P2P Dashboard — Stable Final", layout="wide", initial_sidebar_state="expanded")

# ---------------- Helpers ----------------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, snake-case-ish column names. Safe for many sources."""
    new = {}
    for c in df.columns:
        s = str(c).strip()
        s = s.replace(chr(160), ' ')
        s = s.replace('\\', '_').replace('/', '_')
        s = '_'.join(s.split())
        s = s.lower()
        s = ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in s)
        s = '_'.join([p for p in s.split('_') if p != ''])
        new[c] = s
    return df.rename(columns=new)


def first_existing(df, candidates, default=None):
    for c in candidates:
        if c in df.columns:
            return c
    return default


# ---------------- Load Data ----------------
@st.cache_data(show_spinner=False)
def load_all(files=None):
    if files is None:
        files = [("MEPL.xlsx","MEPL"),("MLPL.xlsx","MLPL"),("mmw.xlsx","MMW"),("mmpl.xlsx","MMPL")]
    frames = []
    for fn, tag in files:
        try:
            df = pd.read_excel(fn, skiprows=1)
            df['__source_file'] = tag
            frames.append(df)
        except FileNotFoundError:
            continue
        except Exception:
            # do not crash — skip problematic file
            continue
    if not frames:
        return pd.DataFrame()
    x = pd.concat(frames, ignore_index=True)
    x = normalize_columns(x)
    # coerce common date columns
    for c in ['pr_date_submitted', 'po_create_date', 'po_approved_date', 'po_delivery_date']:
        if c in x.columns:
            x[c] = pd.to_datetime(x[c], errors='coerce')
    return x


# load
df = load_all()
if df.empty:
    st.warning('No data loaded. Put MEPL.xlsx/MLPL.xlsx/mmw.xlsx/mmpl.xlsx next to the app or upload externally.')

# ---------------- Column discovery ----------------
pr_col = first_existing(df, ['pr_date_submitted','pr_date','pr_date_submitted_utc'])
po_create_col = first_existing(df, ['po_create_date','po_create_date_time','po_create'])
net_amount_col = first_existing(df, ['net_amount','net amount','amount','value'])
purchase_doc_col = first_existing(df, ['purchase_doc','purchase_doc_number','purchase_doc_no','purchase_doc_id'])
pr_number_col = first_existing(df, ['pr_number','pr_no','pr number'])
po_vendor_col = first_existing(df, ['po_vendor','vendor','supplier'])
pr_budget_code_col = first_existing(df, ['pr_budget_code','pr budget code','prbudgetcode'])
pr_budget_desc_col = first_existing(df, ['pr_budget_description','pr budget description','pr_budget_desc'])
entity_col = first_existing(df, ['entity','company','brand','entity_name'])

# create safe entity column
if entity_col and entity_col in df.columns:
    df['entity'] = df[entity_col].fillna('').astype(str).str.strip()
else:
    df['entity'] = df.get('__source_file','').fillna('').astype(str)

# ensure some budget columns exist to avoid KeyError later
for c in [pr_budget_code_col, pr_budget_desc_col]:
    if c and c not in df.columns:
        df[c] = ''

# ---------------- Sidebar filters ----------------
st.sidebar.header('Filters')
FY = {
    'All Years': (pd.Timestamp('2023-04-01'), pd.Timestamp('2026-03-31')),
    '2023': (pd.Timestamp('2023-04-01'), pd.Timestamp('2024-03-31')),
    '2024': (pd.Timestamp('2024-04-01'), pd.Timestamp('2025-03-31'))
}
fy_key = st.sidebar.selectbox('Financial Year', list(FY))
pr_start, pr_end = FY[fy_key]

fil = df.copy()
if pr_col and pr_col in fil.columns:
    fil = fil[(fil[pr_col] >= pr_start) & (fil[pr_col] <= pr_end)]

# additional date range (if available)
date_basis = pr_col if pr_col in fil.columns else (po_create_col if po_create_col in fil.columns else None)
if date_basis:
    min_d = fil[date_basis].dropna().min()
    max_d = fil[date_basis].dropna().max()
    if pd.notna(min_d) and pd.notna(max_d):
        dr = st.sidebar.date_input('Date range', (min_d.date(), max_d.date()))
        if isinstance(dr, tuple) and len(dr) == 2:
            sdt = pd.to_datetime(dr[0]); edt = pd.to_datetime(dr[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            fil = fil[(fil[date_basis] >= sdt) & (fil[date_basis] <= edt)]

# buyer_type fallback
if 'buyer_type' not in fil.columns:
    fil['buyer_type'] = fil.get('buyer_group','').astype(str).apply(lambda v: 'Indirect' if v.strip()=='' or v.strip().lower() in ['not available','na','n/a'] else 'Direct')

# entity filter default to all non-empty entities
entity_choices = sorted([e for e in fil['entity'].dropna().unique().tolist() if str(e).strip()!=''])
if not entity_choices:
    entity_choices = ['All']
sel_e = st.sidebar.multiselect('Entity', entity_choices, default=entity_choices)
if sel_e and 'All' not in sel_e:
    fil = fil[fil['entity'].isin(sel_e)]

# vendor filter
if po_vendor_col and po_vendor_col in fil.columns:
    fil['po_vendor'] = fil[po_vendor_col].fillna('').astype(str)
else:
    fil['po_vendor'] = fil.get('po_vendor','').astype(str)
vendor_choices = sorted([v for v in fil['po_vendor'].dropna().unique().tolist() if v.strip()!=''])
if vendor_choices:
    sel_v = st.sidebar.multiselect('Vendor', vendor_choices, default=vendor_choices)
    if sel_v:
        fil = fil[fil['po_vendor'].isin(sel_v)]

# ---------------- Tabs ----------------
T = st.tabs(['KPIs & Spend','PR/PO Timing','PO Approval','Delivery','Vendors','Dept & Services','Unit-rate Outliers','Forecast','Search','Full Data'])

# ---------------- KPIs & Spend ----------------
with T[0]:
    st.header('P2P Dashboard — Stable Final')
    c1,c2,c3,c4,c5 = st.columns(5)
    total_prs = int(fil[pr_number_col].nunique()) if pr_number_col and pr_number_col in fil.columns else 0
    total_pos = int(fil[purchase_doc_col].nunique()) if purchase_doc_col and purchase_doc_col in fil.columns else 0
    c1.metric('Total PRs', total_prs); c2.metric('Total POs', total_pos); c3.metric('Line Items', len(fil))
    c4.metric('Entities', int(fil['entity'].nunique()))
    spend_val = fil[net_amount_col].sum() if net_amount_col and net_amount_col in fil.columns else 0
    c5.metric('Spend (Cr ₹)', f"{spend_val/1e7:,.2f}")

    st.markdown('---')
    # monthly spend
    dcol = po_create_col if po_create_col and po_create_col in fil.columns else (pr_col if pr_col and pr_col in fil.columns else None)
    if dcol and net_amount_col and net_amount_col in fil.columns:
        t = fil.dropna(subset=[dcol]).copy()
        if not t.empty:
            t['month'] = t[dcol].dt.to_period('M').dt.to_timestamp()
            agg = t.groupby('month')[net_amount_col].sum().reset_index().sort_values('month')
            agg['cr'] = agg[net_amount_col]/1e7
            fig = make_subplots(specs=[[{"secondary_y":True}]])
            fig.add_bar(x=agg['month'].dt.strftime('%b-%Y'), y=agg['cr'], name='Monthly Spend (Cr)', text=agg['cr'])
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('No rows with a valid date for Monthly Spend')
    else:
        st.info('Need date and Net Amount columns for Monthly Spend chart')

    st.markdown('---')
    # entity trend - defensive
    if dcol and net_amount_col and net_amount_col in fil.columns and 'entity' in fil.columns:
        x = fil.dropna(subset=[dcol]).copy()
        if not x.empty:
            x['month'] = x[dcol].dt.to_period('M').dt.to_timestamp()
            x['entity'] = x['entity'].fillna('Unmapped')
            try:
                g = x.groupby(['month','entity'], dropna=False)[net_amount_col].sum().reset_index()
                if not g.empty:
                    fig_e = px.line(g, x=g['month'].dt.strftime('%b-%Y'), y=net_amount_col, color='entity', labels={net_amount_col:'Net Amount', 'x':'Month'})
                    fig_e.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_e, use_container_width=True)
            except KeyError as e:
                st.error(f'Grouping error: missing column for Entity Trend ({e})')
    else:
        st.info('Entity trend needs date + net amount + entity columns')

# ---------------- PR/PO Timing ----------------
with T[1]:
    st.subheader('PR/PO Timing (Lead times)')
    if pr_col and po_create_col and pr_col in fil.columns and po_create_col in fil.columns:
        tmp = fil.dropna(subset=[pr_col, po_create_col]).copy()
        if not tmp.empty:
            tmp['lead_days'] = (tmp[po_create_col] - tmp[pr_col]).dt.days
            avg = tmp['lead_days'].mean()
            st.metric('Average PR→PO lead time (days)', f"{avg:.1f}")
            st.plotly_chart(px.histogram(tmp, x='lead_days', nbins=30, title='Lead time distribution (days)'), use_container_width=True)
        else:
            st.info('No rows with both PR date and PO create date')
    else:
        st.info('PR/PO timing needs PR date and PO create date columns')

# ---------------- PO Approval ----------------nwith T[2]:
    pass

# ---------------- Delivery ----------------
with T[3]:
    st.subheader('Delivery — basic table')
    st.dataframe(fil.head(200), use_container_width=True)

# ---------------- Vendors ----------------
with T[4]:
    st.subheader('Vendors')
    if po_vendor_col and po_vendor_col in fil.columns and net_amount_col and net_amount_col in fil.columns:
        vs = fil.groupby(po_vendor_col, dropna=False)[net_amount_col].sum().reset_index().sort_values(net_amount_col, ascending=False)
        vs['cr'] = vs[net_amount_col]/1e7
        st.dataframe(vs.head(200), use_container_width=True)

# ---------------- Dept & Services ----------------
with T[5]:
    st.subheader('Dept & Services (PR Budget focus)')
    if pr_budget_desc_col and pr_budget_desc_col in fil.columns and net_amount_col and net_amount_col in fil.columns:
        agg = fil.groupby(pr_budget_desc_col)[net_amount_col].sum().reset_index().sort_values(net_amount_col, ascending=False)
        agg['cr'] = agg[net_amount_col]/1e7
        st.plotly_chart(px.bar(agg.head(30), x=pr_budget_desc_col, y='cr', title='PR Budget Description Spend (Top 30)', text='cr').update_traces(texttemplate='%{text:.2f}', textposition='outside'), use_container_width=True)
    else:
        st.info('PR Budget description or Net Amount missing')

# ---------------- Unit-rate Outliers ----------------nwith T[6]:
    pass

# ---------------- Forecast ----------------
with T[7]:
    st.subheader('Forecast (SMA) — basic')
    st.write('Forecast tab - see earlier stable script for fuller details')

# ---------------- Search ----------------
with T[8]:
    st.subheader('Search')
    st.dataframe(fil.head(200), use_container_width=True)

# ---------------- Full Data ----------------
with T[9]:
    st.subheader('Full Data')
    st.dataframe(fil.reset_index(drop=True), use_container_width=True)

# EOF
