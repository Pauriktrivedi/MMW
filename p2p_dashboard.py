import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Final corrected P2P dashboard — copy into p2p_dashboard.py and run with Streamlit
st.set_page_config(page_title="P2P Dashboard — Indirect (Final)", layout="wide", initial_sidebar_state="expanded")

# ----------------- Helpers -----------------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize arbitrary dataframe column names to snake_case."""
    new = {}
    for c in df.columns:
        s = str(c).strip()
        s = s.replace(chr(160), " ")  # NBSP
        s = s.replace(chr(92), "_").replace('/', '_')  # backslash and forward slash
        s = "_".join(s.split())
        s = s.lower()
        s = ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in s)
        s = '_'.join([p for p in s.split('_') if p != ''])
        new[c] = s
    return df.rename(columns=new)


def safe_col(df, candidates, default=None):
    """Return the first column name from candidates that exists in df, else default."""
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
        except FileNotFoundError:
            continue
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    x = pd.concat(frames, ignore_index=True)
    x = normalize_columns(x)
    # coerce likely date columns
    for c in ['pr_date_submitted', 'po_create_date', 'po_delivery_date', 'po_approved_date']:
        if c in x.columns:
            x[c] = pd.to_datetime(x[c], errors='coerce')
    return x


# load
df = load_all()
if df.empty:
    st.warning("No data loaded. Place MEPL.xlsx, MLPL.xlsx, mmw.xlsx, mmpl.xlsx next to this script or upload files.")

# ----------------- Column discovery / canonical names -----------------
pr_col = safe_col(df, ['pr_date_submitted', 'pr_date', 'pr date submitted'])
po_create_col = safe_col(df, ['po_create_date', 'po create date', 'po_created_date'])
net_amount_col = safe_col(df, ['net_amount', 'net amount', 'net_amount_inr', 'amount'])
purchase_doc_col = safe_col(df, ['purchase_doc', 'purchase_doc_number', 'purchase doc'])
pr_number_col = safe_col(df, ['pr_number', 'pr number', 'pr_no'])
po_vendor_col = safe_col(df, ['po_vendor', 'vendor', 'po vendor'])
po_unit_rate_col = safe_col(df, ['po_unit_rate', 'po unit rate', 'po_unit_price'])
pr_budget_code_col = safe_col(df, ['pr_budget_code', 'pr budget code', 'pr_budgetcode'])
pr_budget_desc_col = safe_col(df, ['pr_budget_description', 'pr budget description', 'pr_budget_desc'])
po_budget_code_col = safe_col(df, ['po_budget_code', 'po budget code', 'po_budgetcode'])
po_budget_desc_col = safe_col(df, ['po_budget_description', 'po budget description', 'po_budget_desc'])
pr_bu_col = safe_col(df, ['pr_bussiness_unit','pr_business_unit','pr business unit','pr_bu'])
po_bu_col = safe_col(df, ['po_bussiness_unit','po_business_unit','po business unit','po_bu'])

# Ensure budget-related columns exist to avoid KeyError in later code
for c in [pr_budget_desc_col, pr_budget_code_col, po_budget_desc_col, po_budget_code_col, pr_bu_col, po_bu_col]:
    if c and c not in df.columns:
        df[c] = ''

# ----------------- Buyer/PO creator mapping -----------------
if 'buyer_group' in df.columns:
    try:
        df['buyer_group_code'] = df['buyer_group'].astype(str).str.extract('([0-9]+)')[0].astype(float)
    except Exception:
        df['buyer_group_code'] = np.nan


def map_buyer_type_generic(val, code):
    if pd.isna(val) or str(val).strip() == '' or str(val).strip().lower() in ['not available','na','n/a']:
        return 'Indirect'
    try:
        if str(val).strip().upper() in ['ME_BG17','MLBG16']:
            return 'Direct'
        if not pd.isna(code) and 1 <= int(code) <= 9:
            return 'Direct'
        if not pd.isna(code) and 10 <= int(code) <= 18:
            return 'Indirect'
    except Exception:
        pass
    return 'Indirect'

if not df.empty:
    df['buyer_type'] = df.apply(lambda r: map_buyer_type_generic(r.get('buyer_group', ''), r.get('buyer_group_code', np.nan)), axis=1)
else:
    df['buyer_type'] = pd.Series(dtype=object)

# PO orderer -> creator mapping (example mapping)
map_orderer = {
    'mmw2324030': 'Dhruv', 'mmw2324062': 'Deepak', 'mmw2425154': 'Mukul', 'mmw2223104': 'Paurik',
    'mmw2021181': 'Nayan', 'mmw2223014': 'Aatish', 'mmw_ext_002': 'Deepakex', 'mmw2425024': 'Kamlesh',
}
if 'po_orderer' in df.columns:
    df['po_orderer_filled'] = df['po_orderer'].fillna('N/A').astype(str)
    df['po_creator'] = df['po_orderer_filled'].str.lower().map(map_orderer).fillna(df['po_orderer_filled'])
else:
    df['po_creator'] = pd.Series('', index=df.index)

# ----------------- Sidebar filters -----------------
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

# Date range filter (extra) - based on PR date if available else PO create
date_basis = pr_col if pr_col in fil.columns else (po_create_col if po_create_col in fil.columns else None)
if date_basis:
    mindt = fil[date_basis].dropna().min()
    maxdt = fil[date_basis].dropna().max()
    if pd.notna(mindt) and pd.notna(maxdt):
        dr = st.sidebar.date_input('Date range', (mindt.date(), maxdt.date()), key='date_range')
        if isinstance(dr, tuple) and len(dr) == 2:
            sdt = pd.to_datetime(dr[0]); edt = pd.to_datetime(dr[1]) + pd.Timedelta(hours=23, minutes=59, seconds=59)
            fil = fil[(fil[date_basis] >= sdt) & (fil[date_basis] <= edt)]

# defensive creation of filter columns
for c in ['buyer_type', 'po_creator', 'po_vendor', 'entity']:
    if c not in fil.columns:
        fil[c] = ''

# Buyer Type, Entity, PO Creator filters
choices_bt = sorted(fil['buyer_type'].dropna().unique().tolist()) if 'buyer_type' in fil.columns else ['Direct','Indirect']
choices_bt = [c for c in choices_bt if str(c).strip().lower() != 'other']
sel_b = st.sidebar.multiselect('Buyer Type', choices_bt, default=choices_bt)
entity_choices = sorted([e for e in fil['entity'].dropna().unique().tolist() if str(e).strip() != '']) if 'entity' in fil.columns else []
sel_e = st.sidebar.multiselect('Entity', entity_choices, default=entity_choices)
sel_o = st.sidebar.multiselect('PO Ordered By', sorted(fil['po_creator'].dropna().unique().tolist()), default=sorted(fil['po_creator'].dropna().unique().tolist()))

if sel_b:
    fil = fil[fil['buyer_type'].isin(sel_b)]
if sel_e and 'entity' in fil.columns:
    fil = fil[fil['entity'].isin(sel_e)]
if sel_o:
    fil = fil[fil['po_creator'].isin(sel_o)]

# Vendor + Item filters (multi-select)
if po_vendor_col and po_vendor_col in fil.columns:
    fil['po_vendor'] = fil[po_vendor_col].fillna('').astype(str)
else:
    fil['po_vendor'] = ''
if 'product_name' not in fil.columns:
    fil['product_name'] = ''

vendor_choices = sorted(fil['po_vendor'].dropna().unique().tolist())
item_choices = sorted(fil['product_name'].dropna().unique().tolist())
sel_v = st.sidebar.multiselect('Vendor (pick one or more)', vendor_choices, default=vendor_choices)
sel_i = st.sidebar.multiselect('Item / Product (pick one or more)', item_choices, default=item_choices)

if sel_v:
    fil = fil[fil['po_vendor'].isin(sel_v)]
if sel_i:
    fil = fil[fil['product_name'].isin(sel_i)]

# Reset filters
if st.sidebar.button('Reset Filters'):
    for k in list(st.session_state.keys()):
        try:
            del st.session_state[k]
        except Exception:
            pass
    st.experimental_rerun()

# ----------------- Tabs -----------------
T = st.tabs(['KPIs & Spend','PR/PO Timing','PO Approval','Delivery','Vendors','Dept & Services','Unit-rate Outliers','Forecast','Scorecards','Search','Full Data'])

# ----------------- KPIs & Spend -----------------
with T[0]:
    st.header('P2P Dashboard — Indirect (KPIs & Spend)')
    c1,c2,c3,c4,c5 = st.columns(5)
    total_prs = int(fil.get(pr_number_col, pd.Series(dtype=object)).nunique()) if pr_number_col else 0
    total_pos = int(fil.get(purchase_doc_col, pd.Series(dtype=object)).nunique()) if purchase_doc_col else 0
    c1.metric('Total PRs', total_prs)
    c2.metric('Total POs', total_pos)
    c3.metric('Line Items', len(fil))
    c4.metric('Entities', int(fil.get('entity', pd.Series(dtype=object)).nunique()))
    spend_val = fil.get(net_amount_col, pd.Series(0)).sum() if net_amount_col else 0
    c5.metric('Spend (Cr ₹)', f"{spend_val/1e7:,.2f}")

    st.markdown('---')
    # Monthly spend + cumulative
    dcol = po_create_col if (po_create_col and po_create_col in fil.columns) else (pr_col if (pr_col and pr_col in fil.columns) else None)
    st.subheader('Monthly Total Spend + Cumulative')
    if dcol and net_amount_col and net_amount_col in fil.columns:
        t = fil.dropna(subset=[dcol]).copy()
        t['month'] = t[dcol].dt.to_period('M').dt.to_timestamp()
        agg = t.groupby('month')[net_amount_col].sum().reset_index().sort_values('month')
        agg['cr'] = agg[net_amount_col]/1e7
        agg['cum'] = agg['cr'].cumsum()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_bar(x=agg['month'].dt.strftime('%b-%Y'), y=agg['cr'], name='Monthly Spend (Cr)', text=agg['cr'])
        fig.add_scatter(x=agg['month'].dt.strftime('%b-%Y'), y=agg['cum'], name='Cumulative (Cr)', mode='lines+markers', secondary_y=True)
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside', selector=dict(type='bar'))
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('Monthly Spend not available — need date and Net Amount columns.')

    st.markdown('---')
    st.subheader('Entity Trend')
    if dcol and net_amount_col and net_amount_col in fil.columns and 'entity' in fil.columns:
        x = fil.dropna(subset=[dcol]).copy()
        x['month'] = x[dcol].dt.to_period('M').dt.to_timestamp()
        if 'entity' in x.columns:
            g = x.groupby(['month','entity'], dropna=False)[net_amount_col].sum().reset_index()
            if not g.empty:
                fig_e = px.line(g, x=g['month'].dt.strftime('%b-%Y'), y=net_amount_col, color='entity', labels={net_amount_col:'Net Amount','x':'Month'})
                fig_e.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_e, use_container_width=True)

# ----------------- PR/PO Timing -----------------
with T[1]:
    st.subheader('SLA (PR→PO leadtime)')
    if pr_col and po_create_col and pr_col in fil.columns and po_create_col in fil.columns:
        ld = fil.dropna(subset=[pr_col, po_create_col]).copy()
        ld['lead_time'] = (ld[po_create_col] - ld[pr_col]).dt.days
        avg = ld['lead_time'].mean() if not ld.empty else 0
        fig = go.Figure(go.Indicator(mode='gauge+number', value=float(avg), number={'suffix':' d'}, gauge={'axis':{'range':[0,max(14,avg*1.2 if avg else 14)]}}))
        st.plotly_chart(fig, use_container_width=True)

# ----------------- PO Approval Summary -----------------
with T[2]:
    st.subheader('📋 PO Approval Summary')
    po_create = po_create_col
    po_approved = safe_col(fil, ['po_approved_date','po approved date','po_approved_date'])

    if po_approved and po_create and po_create in fil.columns and purchase_doc_col:
        po_app_df = fil[fil[po_create].notna()].copy()
        po_app_df[po_approved] = pd.to_datetime(po_app_df[po_approved], errors='coerce')

        total_pos = po_app_df[purchase_doc_col].nunique()
        approved_pos = po_app_df[po_app_df[po_approved].notna()][purchase_doc_col].nunique()
        pending_pos = total_pos - approved_pos

        po_app_df['approval_lead_time'] = (po_app_df[po_approved] - po_app_df[po_create]).dt.days
        avg_approval = po_app_df['approval_lead_time'].mean()

        c1,c2,c3,c4 = st.columns(4)
        c1.metric('Total POs', total_pos)
        c2.metric('Approved POs', approved_pos)
        c3.metric('Pending Approval', pending_pos)
        c4.metric('Avg Approval Lead Time (days)', f"{avg_approval:.1f}" if avg_approval==avg_approval else 'N/A')

        st.subheader('📄 PO Approval Details')
        show_cols = [col for col in [ 'po_creator', purchase_doc_col, po_create, po_approved, 'approval_lead_time'] if col]
        st.dataframe(po_app_df[show_cols].sort_values('approval_lead_time', ascending=False), use_container_width=True)
    else:
        st.info("ℹ️ 'PO Approved Date' column not found or Purchase Doc missing.")

# ----------------- Delivery -----------------
with T[3]:
    st.subheader('Delivery Summary')
    dv = fil.copy()
    po_qty_col = safe_col(dv, ['po_qty','po quantity','po_quantity','po qty'])
    received_col = safe_col(dv, ['receivedqty','received_qty','received qty','received_qty'])
    if po_qty_col and received_col and po_qty_col in dv.columns and received_col in dv.columns:
        dv['po_qty_f'] = dv[po_qty_col].fillna(0).astype(float)
        dv['received_f'] = dv[received_col].fillna(0).astype(float)
        dv['pct_received'] = np.where(dv['po_qty_f']>0, dv['received_f']/dv['po_qty_f']*100, 0)
        ag = dv.groupby([purchase_doc_col, po_vendor_col], dropna=False).agg({'po_qty_f':'sum','received_f':'sum','pct_received':'mean'}).reset_index()
        st.dataframe(ag.sort_values('po_qty_f', ascending=False).head(200), use_container_width=True)
    else:
        st.info('Delivery columns (PO Qty / Received QTY) not found.')

# ----------------- Vendors -----------------
with T[4]:
    st.subheader('Top Vendors by Spend')
    if po_vendor_col and net_amount_col and po_vendor_col in fil.columns and net_amount_col in fil.columns:
        vs = fil.groupby(po_vendor_col, dropna=False)[net_amount_col].sum().reset_index().sort_values(net_amount_col, ascending=False)
        vs['cr'] = vs[net_amount_col]/1e7
        st.dataframe(vs.head(50), use_container_width=True)
    else:
        st.info('Vendor / Net Amount columns not present.')

# ----------------- Dept & Services (PR Budget-focused) -----------------
with T[5]:
    st.subheader('Dept & Services — PR Budget perspective')
    dept_df = fil.copy()

    def pick_pr_dept(r):
        candidates = []
        for c in [pr_bu_col, pr_budget_desc_col, po_bu_col, po_budget_desc_col]:
            if c and c in r.index and pd.notna(r[c]) and str(r[c]).strip()!='':
                candidates.append(str(r[c]).strip())
        return candidates[0] if candidates else 'Unmapped / Missing'

    dept_df['pr_department_unified'] = dept_df.apply(pick_pr_dept, axis=1)

    # --- PR Budget Description (preferred) ---
    if pr_budget_desc_col and pr_budget_desc_col in dept_df.columns and net_amount_col and net_amount_col in dept_df.columns:
        agg_desc = dept_df.groupby(pr_budget_desc_col, dropna=False)[net_amount_col].sum().reset_index().sort_values(net_amount_col, ascending=False)
        agg_desc['cr'] = agg_desc[net_amount_col]/1e7
        top_desc = agg_desc.head(30)
        if not top_desc.empty:
            fig_desc = px.bar(top_desc, x=pr_budget_desc_col, y='cr', title='PR Budget Description Spend (Top 30)', labels={pr_budget_desc_col: 'PR Budget Description', 'cr':'Cr'}, text='cr')
            fig_desc.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig_desc.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_desc, use_container_width=True)

            pick_desc = st.selectbox('Drill into PR Budget Description', ['-- none --'] + top_desc[pr_budget_desc_col].astype(str).tolist())
            if pick_desc and pick_desc != '-- none --':
                sub = dept_df[dept_df[pr_budget_desc_col].astype(str) == pick_desc].copy()
                show_cols = [c for c in [pr_number_col, purchase_doc_col, pr_budget_code_col, pr_budget_desc_col, net_amount_col, po_vendor_col] if c in sub.columns]
                st.dataframe(sub[show_cols].sort_values(net_amount_col, ascending=False).head(500), use_container_width=True)
    else:
        st.info('PR Budget description or Net Amount column not found to show PR Budget Description spend.')

    st.markdown('---')
    # --- PR Budget Code ---
    if pr_budget_code_col and pr_budget_code_col in dept_df.columns and net_amount_col and net_amount_col in dept_df.columns:
        agg_code = dept_df.groupby(pr_budget_code_col, dropna=False)[net_amount_col].sum().reset_index().sort_values(net_amount_col, ascending=False)
        agg_code['cr'] = agg_code[net_amount_col]/1e7
        top_code = agg_code.head(30)
        if not top_code.empty:
            fig_code = px.bar(top_code, x=pr_budget_code_col, y='cr', title='PR Budget Code Spend (Top 30)', labels={pr_budget_code_col: 'PR Budget Code', 'cr':'Cr'}, text='cr')
            fig_code.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig_code.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_code, use_container_width=True)

            pick_code = st.selectbox('Drill into PR Budget Code', ['-- none --'] + top_code[pr_budget_code_col].astype(str).tolist())
            if pick_code and pick_code != '-- none --':
                sub2 = dept_df[dept_df[pr_budget_code_col].astype(str) == pick_code].copy()
                show_cols2 = [c for c in [pr_number_col, purchase_doc_col, pr_budget_code_col, pr_budget_desc_col, net_amount_col, po_vendor_col] if c in sub2.columns]
                st.dataframe(sub2[show_cols2].sort_values(net_amount_col, ascending=False).head(500), use_container_width=True)
    else:
        st.info('PR Budget code or Net Amount column not found to show PR Budget Code spend.')

# ----------------- Unit-rate Outliers -----------------
with T[6]:
    st.subheader('Unit-rate Outliers vs Historical Median')
    grp_candidates = [c for c in ['product_name','item_code','product name','item code'] if c in fil.columns]
    grp_by = st.selectbox('Group by', grp_candidates) if grp_candidates else None
    if grp_by and po_unit_rate_col and po_unit_rate_col in fil.columns:
        z = fil[[grp_by, po_unit_rate_col, purchase_doc_col, pr_number_col, po_vendor_col, 'item_description', po_create_col, net_amount_col]].dropna(subset=[grp_by, po_unit_rate_col]).copy()
        med = z.groupby(grp_by)[po_unit_rate_col].median().rename('median_rate')
        z = z.join(med, on=grp_by)
        z['pctdev'] = (z[po_unit_rate_col] - z['median_rate']) / z['median_rate'].replace(0, np.nan)
        thr = st.slider('Outlier threshold (±%)', 10, 300, 50, 5)
        out = z[abs(z['pctdev']) >= thr/100.0].copy()
        out['pctdev%'] = (out['pctdev']*100).round(1)
        st.dataframe(out.sort_values('pctdev%', ascending=False), use_container_width=True)

# ----------------- Forecast -----------------
with T[7]:
    st.subheader('Forecast Next Month Spend (SMA)')
    dcol = po_create_col if (po_create_col and po_create_col in fil.columns) else (pr_col if (pr_col and pr_col in fil.columns) else None)
    if dcol and net_amount_col and net_amount_col in fil.columns:
        t = fil.dropna(subset=[dcol]).copy()
        t['month'] = t[dcol].dt.to_period('M').dt.to_timestamp()
        m = t.groupby('month')[net_amount_col].sum().sort_index()
        m_cr = m/1e7
        k = st.slider('Window (months)', 3, 12, 6)
        sma = m_cr.rolling(k).mean()
        mu = m_cr.tail(k).mean() if len(m_cr) >= k else m_cr.mean()
        sd = m_cr.tail(k).std(ddof=1) if len(m_cr) >= k else m_cr.std(ddof=1)
        n = min(k, max(1, len(m_cr)))
        se = sd/np.sqrt(n) if sd==sd else 0
        lo, hi = float(mu-1.96*se), float(mu+1.96*se)
        nxt = (m_cr.index.max() + pd.offsets.MonthBegin(1)) if len(m_cr) > 0 else pd.Timestamp.now().to_period('M').to_timestamp()
        fdf = pd.DataFrame({'Month': list(m_cr.index) + [nxt], 'SpendCr': list(m_cr.values) + [np.nan], 'SMA': list(sma.values) + [mu]})
        fig = go.Figure()
        fig.add_bar(x=fdf['Month'], y=fdf['SpendCr'], name='Actual (Cr)')
        fig.add_scatter(x=fdf['Month'], y=fdf['SMA'], mode='lines+markers', name=f'SMA{k}')
        st.plotly_chart(fig, use_container_width=True)

# ----------------- Scorecards / Vendor details -----------------
with T[8]:
    st.subheader('Vendor Scorecard')
    if po_vendor_col and po_vendor_col in fil.columns:
        vendor = st.selectbox('Pick Vendor', sorted(fil[po_vendor_col].dropna().astype(str).unique().tolist()))
        vd = fil[fil[po_vendor_col].astype(str) == str(vendor)].copy()
        spend = vd.get(net_amount_col, pd.Series(0)).sum()/1e7 if net_amount_col else 0
        upos = int(vd.get(purchase_doc_col, pd.Series(dtype=object)).nunique()) if purchase_doc_col else 0
        k1,k2 = st.columns(2)
        k1.metric('Spend (Cr)', f"{spend:.2f}")
        k2.metric('Unique POs', upos)
        st.dataframe(vd.head(200), use_container_width=True)

# ----------------- Search -----------------
with T[9]:
    st.subheader('🔍 Keyword Search')
    valid_cols = [c for c in [pr_number_col, purchase_doc_col, 'product_name', po_vendor_col] if c in df.columns]
    query = st.text_input('Type vendor, product, PO, PR, etc.', '')
    cat_sel = st.multiselect('Filter by Procurement Category', sorted(df.get('procurement_category', pd.Series(dtype=object)).dropna().astype(str).unique().tolist())) if 'procurement_category' in df.columns else []
    vend_sel = st.multiselect('Filter by Vendor', sorted(df.get(po_vendor_col, pd.Series(dtype=object)).dropna().astype(str).unique().tolist())) if po_vendor_col in df.columns else []
    if query and valid_cols:
        mask = pd.Series(False, index=df.index)
        q = query.lower()
        for c in valid_cols:
            mask = mask | df[c].astype(str).str.lower().str.contains(q, na=False)
        res = df[mask].copy()
        if cat_sel:
            res = res[res['procurement_category'].astype(str).isin(cat_sel)]
        if vend_sel and po_vendor_col in df.columns:
            res = res[res[po_vendor_col].astype(str).isin(vend_sel)]
        st.write(f'Found {len(res)} rows')
        st.dataframe(res, use_container_width=True)
        st.download_button('⬇️ Download Search Results', res.to_csv(index=False), file_name='search_results.csv', mime='text/csv')
    else:
        st.caption('Start typing to search…')

# ----------------- Full Data (all rows) -----------------
with T[10]:
    st.subheader('Full Data — all filtered rows')
    try:
        st.dataframe(fil.reset_index(drop=True), use_container_width=True)
        csv = fil.to_csv(index=False)
        st.download_button('⬇️ Download full filtered data (CSV)', csv, file_name='p2p_full_filtered.csv', mime='text/csv')
    except Exception as e:
        st.error(f'Could not display full data: {e}')

# EOF
