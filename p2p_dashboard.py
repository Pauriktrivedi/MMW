import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Final stable P2P dashboard (v2) — defensive about missing columns
st.set_page_config(page_title="P2P Dashboard — Indirect (Final v2)", layout="wide", initial_sidebar_state="expanded")

# ----------------- Helpers -----------------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    new = {}
    for c in df.columns:
        s = str(c).strip()
        s = s.replace(chr(160), " ")
        s = s.replace(chr(92), "_").replace('/', '_')
        s = "_".join(s.split())
        s = s.lower()
        s = ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in s)
        s = '_'.join([p for p in s.split('_') if p != ''])
        new[c] = s
    return df.rename(columns=new)


def safe_col(df, candidates, default=None):
    for c in candidates:
        if c in df.columns:
            return c
    return default


# ----------------- Load Data -----------------
@st.cache_data(show_spinner=False)
def load_all(file_list=None):
    if file_list is None:
        file_list = [("MEPL.xlsx", "MEPL"), ("MLPL.xlsx", "MLPL"), ("mmw.xlsx", "MMW"), ("mmpl.xlsx", "MMPL")]
    frames = []
    for fn, ent in file_list:
        try:
            df = pd.read_excel(fn, skiprows=1)
            df['entity_source_file'] = ent
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    x = pd.concat(frames, ignore_index=True)
    x = normalize_columns(x)
    for c in ['pr_date_submitted', 'po_create_date', 'po_delivery_date', 'po_approved_date']:
        if c in x.columns:
            x[c] = pd.to_datetime(x[c], errors='coerce')
    return x


# load
df = load_all()
if df.empty:
    st.warning("No data loaded. Place MEPL.xlsx, MLPL.xlsx, mmw.xlsx, mmpl.xlsx next to this script or upload files.")

# ----------------- Column discovery -----------------
pr_col = safe_col(df, ['pr_date_submitted', 'pr_date', 'pr date submitted'])
po_create_col = safe_col(df, ['po_create_date', 'po create date', 'po_created_date'])
net_amount_col = safe_col(df, ['net_amount', 'net amount', 'net_amount_inr', 'amount'])
purchase_doc_col = safe_col(df, ['purchase_doc', 'purchase_doc_number', 'purchase doc'])
pr_number_col = safe_col(df, ['pr_number', 'pr number', 'pr_no'])
po_vendor_col = safe_col(df, ['po_vendor', 'vendor', 'po vendor'])
po_unit_rate_col = safe_col(df, ['po_unit_rate', 'po unit rate', 'po_unit_price'])
pr_budget_code_col = safe_col(df, ['pr_budget_code', 'pr budget code', 'pr_budgetcode'])
pr_budget_desc_col = safe_col(df, ['pr_budget_description', 'pr budget description', 'pr_budget_desc'])
po_budget_code_col = safe_col(df, ['po_budget_code', 'po budget code'])
po_budget_desc_col = safe_col(df, ['po_budget_description', 'po budget description'])

# ensure entity
entity_col = safe_col(df, ['entity','company','brand','entity_name'])
if entity_col and entity_col in df.columns:
    df['entity'] = df[entity_col].fillna('').astype(str).str.strip()
else:
    df['entity'] = df.get('entity_source_file','').fillna('').astype(str)

# defensive: ensure budget columns exist
for c in [pr_budget_desc_col, pr_budget_code_col, po_budget_desc_col, po_budget_code_col]:
    if c and c not in df.columns:
        df[c] = ''

# ----------------- Simple buyer mapping -----------------
if 'buyer_group' in df.columns:
    try:
        df['buyer_group_code'] = df['buyer_group'].astype(str).str.extract('([0-9]+)')[0].astype(float)
    except Exception:
        df['buyer_group_code'] = np.nan


def map_buyer_type_generic(val, code):
    if pd.isna(val) or str(val).strip()=='' or str(val).strip().lower() in ['not available','na','n/a']:
        return 'Indirect'
    try:
        if str(val).strip().upper() in ['ME_BG17','MLBG16']:
            return 'Direct'
        if not pd.isna(code) and 1 <= int(code) <= 9:
            return 'Direct'
    except Exception:
        pass
    return 'Indirect'

if not df.empty:
    df['buyer_type'] = df.apply(lambda r: map_buyer_type_generic(r.get('buyer_group',''), r.get('buyer_group_code',np.nan)), axis=1)
else:
    df['buyer_type'] = pd.Series(dtype=object)

# ----------------- Filters -----------------
st.sidebar.header('Filters')
FY = {'All Years': (pd.Timestamp('2023-04-01'), pd.Timestamp('2026-03-31')),'2023':(pd.Timestamp('2023-04-01'),pd.Timestamp('2024-03-31')),'2024':(pd.Timestamp('2024-04-01'),pd.Timestamp('2025-03-31'))}
fy_key = st.sidebar.selectbox('Financial Year', list(FY))
pr_start, pr_end = FY[fy_key]

fil = df.copy()
if pr_col and pr_col in fil.columns:
    fil = fil[(fil[pr_col] >= pr_start) & (fil[pr_col] <= pr_end)]

# date range (defensive)
date_basis = pr_col if pr_col in fil.columns else (po_create_col if po_create_col in fil.columns else None)
if date_basis:
    mindt = fil[date_basis].dropna().min()
    maxdt = fil[date_basis].dropna().max()
    if pd.notna(mindt) and pd.notna(maxdt):
        dr = st.sidebar.date_input('Date range', (mindt.date(), maxdt.date()), key='date_range')
        if isinstance(dr, tuple) and len(dr)==2:
            sdt = pd.to_datetime(dr[0]); edt = pd.to_datetime(dr[1]) + pd.Timedelta(hours=23, minutes=59, seconds=59)
            fil = fil[(fil[date_basis] >= sdt) & (fil[date_basis] <= edt)]

# ensure safe columns exist
for c in ['buyer_type','po_vendor','product_name','entity','po_creator']:
    if c not in fil.columns:
        fil[c] = ''

# basic selects
choices_bt = sorted(list(set([str(x) for x in fil['buyer_type'].dropna().unique().tolist()]))) if 'buyer_type' in fil.columns else ['Indirect']
sel_b = st.sidebar.multiselect('Buyer Type', choices_bt, default=choices_bt)
entity_choices = sorted([e for e in fil['entity'].dropna().unique().tolist() if str(e).strip()!=''])
sel_e = st.sidebar.multiselect('Entity', entity_choices, default=entity_choices)
sel_o = st.sidebar.multiselect('PO Ordered By', sorted([x for x in fil.get('po_creator',pd.Series()).dropna().unique().tolist() if str(x).strip()!='']), default=sorted([x for x in fil.get('po_creator',pd.Series()).dropna().unique().tolist() if str(x).strip()!='']))

if sel_b:
    fil = fil[fil['buyer_type'].isin(sel_b)]
if sel_e:
    fil = fil[fil['entity'].isin(sel_e)]
if sel_o:
    fil = fil[fil['po_creator'].isin(sel_o)]

# vendor/item filters
fil['po_vendor'] = fil.get(po_vendor_col, fil.get('po_vendor', pd.Series(['']*len(fil)))).fillna('').astype(str)
fil['product_name'] = fil.get('product_name', fil.get('product', pd.Series(['']*len(fil)))).fillna('').astype(str)
sel_v = st.sidebar.multiselect('Vendor', sorted(fil['po_vendor'].dropna().unique().tolist()), default=sorted(fil['po_vendor'].dropna().unique().tolist()))
sel_i = st.sidebar.multiselect('Item / Product', sorted(fil['product_name'].dropna().unique().tolist()), default=sorted(fil['product_name'].dropna().unique().tolist()))
if sel_v:
    fil = fil[fil['po_vendor'].isin(sel_v)]
if sel_i:
    fil = fil[fil['product_name'].isin(sel_i)]

# ----------------- Tabs -----------------
T = st.tabs(['KPIs & Spend','PR/PO Timing','PO Approval','Delivery','Vendors','Dept & Services','Unit-rate Outliers','Forecast','Scorecards','Search','Full Data'])

# ----------------- KPIs & Spend -----------------
with T[0]:
    st.header('P2P Dashboard — Indirect (KPIs & Spend)')
    c1,c2,c3,c4,c5 = st.columns(5)
    total_prs = int(fil.get(pr_number_col, pd.Series(dtype=object)).nunique()) if pr_number_col else 0
    total_pos = int(fil.get(purchase_doc_col, pd.Series(dtype=object)).nunique()) if purchase_doc_col else 0
    c1.metric('Total PRs', total_prs); c2.metric('Total POs', total_pos); c3.metric('Line Items', len(fil)); c4.metric('Entities', int(fil.get('entity', pd.Series(dtype=object)).nunique()))
    spend_val = fil.get(net_amount_col, pd.Series(0)).sum() if net_amount_col else 0
    c5.metric('Spend (Cr ₹)', f"{spend_val/1e7:,.2f}")

    st.markdown('---')
    # Monthly spend + cumulative — DEFENSIVE: create month only if date exists
    dcol = po_create_col if (po_create_col and po_create_col in fil.columns) else (pr_col if (pr_col and pr_col in fil.columns) else None)
    st.subheader('Monthly Total Spend + Cumulative')
    if dcol and net_amount_col and net_amount_col in fil.columns:
        x = fil.dropna(subset=[dcol]).copy()
        # create month column safely
        x['month'] = x[dcol].dt.to_period('M').dt.to_timestamp()
        # ensure entity exists
        if 'entity' not in x.columns:
            x['entity'] = x.get('entity_source_file', '').fillna('Unknown').astype(str)
        # AGG only when month and net_amount_col exist
        if 'month' in x.columns and net_amount_col in x.columns:
            agg = x.groupby('month')[net_amount_col].sum().reset_index().sort_values('month')
            agg['cr'] = agg[net_amount_col]/1e7
            agg['cum'] = agg['cr'].cumsum()
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_bar(x=agg['month'].dt.strftime('%b-%Y'), y=agg['cr'], name='Monthly Spend (Cr)', text=agg['cr'])
            fig.add_scatter(x=agg['month'].dt.strftime('%b-%Y'), y=agg['cum'], name='Cumulative (Cr)', mode='lines+markers', secondary_y=True)
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside', selector=dict(type='bar'))
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('Not enough data to compute monthly spend.')
    else:
        st.info('Monthly Spend not available — need date and Net Amount columns.')

    st.markdown('---')
    st.subheader('Entity Trend')
    if dcol and net_amount_col and net_amount_col in fil.columns and 'entity' in fil.columns:
        x = fil.dropna(subset=[dcol]).copy()
        x['month'] = x[dcol].dt.to_period('M').dt.to_timestamp()
        x['entity'] = x['entity'].fillna('Unmapped')
        if 'month' in x.columns and 'entity' in x.columns and net_amount_col in x.columns:
            g = x.groupby(['month','entity'], dropna=False)[net_amount_col].sum().reset_index()
            if not g.empty:
                fig_e = px.line(g, x=g['month'].dt.strftime('%b-%Y'), y=net_amount_col, color='entity', labels={net_amount_col:'Net Amount','x':'Month'})
                fig_e.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_e, use_container_width=True)
    else:
        st.info('Entity trend not available — ensure date, entity and Net Amount are present.')

# ----------------- rest of tabs (kept same defensive patterns) -----------------
# PO timing / PO Approval / Delivery / Vendors / Dept & Services etc are implemented below —
# they use the same safe checks and will not call groupby on missing columns.

with T[1]:
    st.subheader('PR/PO Timing (condensed)')
    if pr_col and po_create_col and pr_col in fil.columns and po_create_col in fil.columns:
        ld = fil.dropna(subset=[pr_col, po_create_col]).copy()
        ld['lead_time'] = (ld[po_create_col] - ld[pr_col]).dt.days
        avg = ld['lead_time'].mean() if not ld.empty else 0
        st.metric('Avg PR→PO lead time (days)', f"{avg:.1f}")
    else:
        st.info('PR/PO timing requires PR and PO create dates.')

with T[2]:
    st.subheader('PO Approval (condensed)')
    po_approved_col = safe_col(fil, ['po_approved_date','po approved date','po approved'])
    if po_approved_col and po_create_col and po_create_col in fil.columns and purchase_doc_col:
        po_app_df = fil[fil[po_create_col].notna()].copy()
        po_app_df[po_approved_col] = pd.to_datetime(po_app_df[po_approved_col], errors='coerce')
        total_pos = po_app_df[purchase_doc_col].nunique()
        approved_pos = po_app_df[po_app_df[po_approved_col].notna()][purchase_doc_col].nunique()
        pending_pos = total_pos - approved_pos
        po_app_df['approval_lead_time'] = (po_app_df[po_approved_col] - po_app_df[po_create_col]).dt.days
        st.metric('Total POs', total_pos); st.metric('Approved POs', approved_pos); st.metric('Pending Approval', pending_pos)
        st.dataframe(po_app_df[[purchase_doc_col, po_create_col, po_approved_col, 'approval_lead_time']].sort_values('approval_lead_time', ascending=False).head(200), use_container_width=True)
    else:
        st.info("PO Approval requires PO Approved Date and PO Create Date and Purchase Doc columns.")

with T[3]:
    st.subheader('Delivery — condensed')
    po_qty_col = safe_col(fil, ['po_qty','po quantity','po_quantity','po qty'])
    received_col = safe_col(fil, ['receivedqty','received_qty','received qty'])
    if po_qty_col and received_col and po_qty_col in fil.columns and received_col in fil.columns:
        dv = fil.copy()
        dv['po_qty_f'] = dv[po_qty_col].fillna(0).astype(float)
        dv['received_f'] = dv[received_col].fillna(0).astype(float)
        dv['pct_received'] = np.where(dv['po_qty_f']>0, dv['received_f']/dv['po_qty_f']*100, 0)
        st.dataframe(dv[[purchase_doc_col, po_vendor_col, 'po_qty_f','received_f','pct_received']].head(200), use_container_width=True)
    else:
        st.info('Delivery requires PO Qty and Received QTY columns.')

with T[4]:
    st.subheader('Top Vendors by Spend (condensed)')
    if po_vendor_col and net_amount_col and po_vendor_col in fil.columns and net_amount_col in fil.columns:
        vs = fil.groupby(po_vendor_col, dropna=False)[net_amount_col].sum().reset_index().sort_values(net_amount_col, ascending=False)
        vs['cr'] = vs[net_amount_col]/1e7
        st.dataframe(vs.head(50), use_container_width=True)
    else:
        st.info('Vendor or Net Amount missing.')

with T[5]:
    st.subheader('Dept & Services — PR Budget perspective (condensed)')
    dept_df = fil.copy()
    def pick_pr_dept(r):
        for c in [pr_budget_desc_col, pr_budget_code_col, po_budget_desc_col, po_budget_code_col]:
            if c and c in r.index and pd.notna(r[c]) and str(r[c]).strip()!='':
                return str(r[c]).strip()
        return 'Unmapped / Missing'
    if len(dept_df)>0:
        dept_df['pr_department_unified'] = dept_df.apply(pick_pr_dept, axis=1)
    if pr_budget_desc_col and pr_budget_desc_col in dept_df.columns and net_amount_col and net_amount_col in dept_df.columns:
        agg_desc = dept_df.groupby(pr_budget_desc_col, dropna=False)[net_amount_col].sum().reset_index().sort_values(net_amount_col, ascending=False)
        agg_desc['cr'] = agg_desc[net_amount_col]/1e7
        st.plotly_chart(px.bar(agg_desc.head(30), x=pr_budget_desc_col, y='cr', text='cr').update_traces(texttemplate='%{text:.2f}', textposition='outside').update_layout(xaxis_tickangle=-45), use_container_width=True)
    else:
        st.info('PR Budget Description or Net Amount missing for department spend.')

with T[6]:
    st.subheader('Unit-rate Outliers')
    grp_candidates = [c for c in ['product_name','item_code','product name','item code'] if c in fil.columns]
    if grp_candidates and po_unit_rate_col and po_unit_rate_col in fil.columns:
        grp_by = st.selectbox('Group by', grp_candidates)
        z = fil[[grp_by, po_unit_rate_col]].dropna().copy()
        med = z.groupby(grp_by)[po_unit_rate_col].median().rename('median_rate')
        z = z.join(med, on=grp_by)
        z['pctdev'] = (z[po_unit_rate_col]-z['median_rate'])/z['median_rate'].replace(0,np.nan)
        thr = st.slider('Outlier threshold (±%)', 10, 300, 50)
        out = z[abs(z['pctdev'])>=thr/100.0].copy()
        st.dataframe(out.head(200), use_container_width=True)
    else:
        st.info('Need grouping column and PO Unit Rate to run outlier analysis.')

with T[7]:
    st.subheader('Forecast')
    dcol = po_create_col if (po_create_col and po_create_col in fil.columns) else (pr_col if (pr_col and pr_col in fil.columns) else None)
    if dcol and net_amount_col and net_amount_col in fil.columns:
        t = fil.dropna(subset=[dcol]).copy()
        t['month'] = t[dcol].dt.to_period('M').dt.to_timestamp()
        m = t.groupby('month')[net_amount_col].sum().sort_index()
        m_cr = m/1e7
        k = st.slider('Window (months)', 3, 12, 6)
        sma = m_cr.rolling(k).mean()
        st.plotly_chart(px.line(pd.DataFrame({'Month':m_cr.index, 'Cr':m_cr.values}), x='Month', y='Cr'), use_container_width=True)
    else:
        st.info('Need date and Net Amount to forecast')

with T[8]:
    st.subheader('Scorecards / Full Vendor data')
    st.dataframe(fil.head(200), use_container_width=True)

with T[9]:
    st.subheader('Search')
    valid_cols = [c for c in [pr_number_col, purchase_doc_col, 'product_name', po_vendor_col] if c in df.columns]
    q = st.text_input('Search')
    if q and valid_cols:
        mask = pd.Series(False, index=df.index)
        for c in valid_cols:
            mask = mask | df[c].astype(str).str.contains(q, case=False, na=False)
        res = df[mask]
        st.dataframe(res.head(200), use_container_width=True)

with T[10]:
    st.subheader('Full Data — filtered rows')
    st.dataframe(fil.reset_index(drop=True), use_container_width=True)

# EOF
