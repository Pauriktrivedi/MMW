# Full P2P Dashboard with improved column auto-detection + diagnostics
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re

st.set_page_config(page_title="P2P Dashboard — Full (Auto-detect)", layout="wide", initial_sidebar_state="expanded")

# ---- Page header ----
st.title("P2P Dashboard — Indirect")
st.caption("Purchase-to-Pay overview (Indirect spend focus)")

# ----------------- Helpers -----------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names into safe snake_case keys (but keep originals in index)."""
    new_cols = {}
    for c in df.columns:
        s = str(c).strip()
        s = s.replace("\xa0", " ").replace("\\", "_").replace("/", "_")
        s = "_".join(s.split()).lower()
        s = re.sub(r'[^0-9a-z_]+', '_', s)
        s = re.sub(r'__+', '_', s).strip('_')
        new_cols[c] = s
    return df.rename(columns=new_cols)

def find_best_column(df, keywords):
    """Return first column name (original normalized) that contains any keyword substring.
       keywords: list of lowercase substrings to match.
       Returns None if not found.
    """
    if df is None or df.columns.empty:
        return None
    for col in df.columns:
        low = str(col).lower()
        for kw in keywords:
            if kw in low:
                return col
    return None

@st.cache_data(show_spinner=False)
def load_all_from_files(uploaded_files=None):
    """Load uploaded files or default files in working folder. Normalize columns."""
    frames = []
    if uploaded_files:
        for f in uploaded_files:
            try:
                df_temp = pd.read_excel(f, skiprows=1)
            except Exception:
                try:
                    df_temp = pd.read_excel(f)
                except Exception:
                    try:
                        df_temp = pd.read_csv(f)
                    except Exception:
                        continue
            df_temp['__source_entity'] = getattr(f, "name", "uploaded")
            frames.append(df_temp)
    else:
        defaults = [("MEPL.xlsx","MEPL"),("MLPL.xlsx","MLPL"),("mmw.xlsx","MMW"),("mmpl.xlsx","MMPL")]
        for fn, ent in defaults:
            try:
                df_temp = pd.read_excel(fn, skiprows=1)
            except Exception:
                try:
                    df_temp = pd.read_excel(fn)
                except Exception:
                    continue
            df_temp['__source_entity'] = ent
            frames.append(df_temp)

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True, sort=False)
    df = normalize_columns(df)
    return df

# ----------------- Sidebar Filters & Debug toggle -----------------
st.sidebar.header('Filters')

# Debug toggle (show diagnostics)
verbose = st.sidebar.checkbox('Verbose debug loader (show detected columns)', value=False)

# Financial Year choices
FY = {
    'All Years': (pd.Timestamp('2000-01-01'), pd.Timestamp('2099-12-31')),
    '2023': (pd.Timestamp('2023-04-01'), pd.Timestamp('2024-03-31')),
    '2024': (pd.Timestamp('2024-04-01'), pd.Timestamp('2025-03-31')),
    '2025': (pd.Timestamp('2025-04-01'), pd.Timestamp('2026-03-31')),
}
fy_key = st.sidebar.selectbox('Financial Year', list(FY.keys()), index=0)
fy_start, fy_end = FY[fy_key]

# Month dropdown (simple 12 months + All)
months = ['All Months','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
sel_month = st.sidebar.selectbox('Month (applies when Year selected)', months, index=0)

# Date range picker (optional) — overrides FY if chosen
date_range = st.sidebar.date_input('Select Date Range (optional)', value=[fy_start.date(), fy_end.date()], key='date_range_picker')
if isinstance(date_range, (list,tuple)) and len(date_range)==2:
    dr_start = pd.to_datetime(date_range[0])
    dr_end = pd.to_datetime(date_range[1])
else:
    dr_start, dr_end = fy_start, fy_end

st.sidebar.markdown('---')

# ----------------- Data Loading (bottom uploader preserved) -----------------
# Use any files already persisted in session; otherwise load defaults (if present)
uploaded_session = st.session_state.get('_bottom_uploaded', None)
df = load_all_from_files(uploaded_session)

# If no data loaded: show bottom uploader and stop early (so user can upload)
if df.empty:
    st.warning("No data loaded. Place default Excel files next to this script (MEPL.xlsx, MLPL.xlsx, mmw.xlsx, mmpl.xlsx) or use the bottom uploader at page bottom.")
    st.markdown('---')
    st.markdown('### Upload files (bottom of page)')
    new_files = st.file_uploader('Upload Excel/CSV files here', type=['xlsx','xls','csv'], accept_multiple_files=True, key='_bottom_uploader')
    if new_files:
        st.session_state['_bottom_uploaded'] = new_files
        st.experimental_rerun()
    st.stop()

# ----------------- Auto-detect important columns -----------------
# We'll search both normalized column names and original ones (already normalized by normalize_columns)
col_candidates = list(df.columns)

# detection heuristics (lowercase substrings)
PR_DATE_COL = find_best_column(df, ['pr_date','pr date','pr_date_submitted','pr_date_submitted','requestdate','prsubmitted','pr_submitted','prsubmitteddate','pr_sub'])
PO_CREATE_COL = find_best_column(df, ['po_create','po create','po_created','po_create_date','po_created_date','po_date','po created'])
PO_DELIVERY_COL = find_best_column(df, ['delivery','po_delivery','po delivery','delivered','delivery_date'])
NET_AMOUNT_COL = find_best_column(df, ['net amount','net_amount','netamount','amount','amount_in','value','net_amt'])
PURCHASE_DOC_COL = find_best_column(df, ['purchase_doc','purchase doc','po number','po_number','po no','po_no','po'])
PR_NUMBER_COL = find_best_column(df, ['pr number','pr_number','pr no','pr_no','pr'])
PO_VENDOR_COL = find_best_column(df, ['vendor','supplier','po_vendor','po vendor','supplier name'])
PRODUCT_NAME_COL = find_best_column(df, ['product','product name','item','item_name','product_name'])
PO_UNIT_RATE_COL = find_best_column(df, ['unit rate','unit_rate','rate','unitprice','unit price','po_unit_rate'])
PENDING_QTY_COL = find_best_column(df, ['pending_qty','pending qty','pending','pendingqty'])
RECEIVED_QTY_COL = find_best_column(df, ['receivedqty','received qty','received','qty_received'])
PO_DEPARTMENT_COL = find_best_column(df, ['po_department','po department','department','dept','pr_department','pr dept'])

# If verbose: show a diagnostics expandable
if verbose:
    with st.expander("Diagnostics: detected columns & sample headers", expanded=True):
        st.write("All normalized columns in dataframe (first 120):")
        st.write(list(df.columns)[:120])
        st.write("Auto-detected mapping (keys -> column):")
        mapping = {
            'PR date (pr_col)': PR_DATE_COL,
            'PO create (po_create_col)': PO_CREATE_COL,
            'PO delivery (po_delivery_col)': PO_DELIVERY_COL,
            'Net amount (net_amount_col)': NET_AMOUNT_COL,
            'Purchase doc (purchase_doc_col)': PURCHASE_DOC_COL,
            'PR number (pr_number_col)': PR_NUMBER_COL,
            'PO vendor (po_vendor_col)': PO_VENDOR_COL,
            'Product (product_name_col)': PRODUCT_NAME_COL,
            'PO unit rate (po_unit_rate_col)': PO_UNIT_RATE_COL,
            'Pending qty (pending_qty_col)': PENDING_QTY_COL,
            'Received qty (received_qty_col)': RECEIVED_QTY_COL,
            'PO department (po_department_col)': PO_DEPARTMENT_COL
        }
        st.json(mapping)
        st.markdown("**Preview (top 5 rows)**")
        st.dataframe(df.head(5))

# ----------------- Parse date columns safely -----------------
for date_col in [PR_DATE_COL, PO_CREATE_COL, PO_DELIVERY_COL]:
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

# Provide canonical names in code
pr_col = PR_DATE_COL
po_create_col = PO_CREATE_COL
po_delivery_col = PO_DELIVERY_COL
net_amount_col = NET_AMOUNT_COL
purchase_doc_col = PURCHASE_DOC_COL
pr_number_col = PR_NUMBER_COL
po_vendor_col = PO_VENDOR_COL
product_name_col = PRODUCT_NAME_COL
po_unit_rate_col = PO_UNIT_RATE_COL
pending_qty_col = PENDING_QTY_COL
received_qty_col = RECEIVED_QTY_COL
po_department_col = PO_DEPARTMENT_COL

# fallback: if pr_col not found, try typical normalized names explicitly
if pr_col is None and 'pr_date_submitted' in df.columns:
    pr_col = 'pr_date_submitted'
if po_vendor_col is None:
    # sometimes vendors stored under 'vendor_name' or 'vendor_name_lc' etc
    guess = find_best_column(df, ['vendor_name','supplier_name','supplier'])
    if guess:
        po_vendor_col = guess

# ----------------- Prepare derived date fields used for month/year filtering -----------------
if pr_col and pr_col in df.columns:
    df['pr_date_submitted__safe'] = pd.to_datetime(df[pr_col], errors='coerce')
else:
    df['pr_date_submitted__safe'] = pd.NaT

df['pr_year'] = df['pr_date_submitted__safe'].dt.year
df['pr_month_name'] = df['pr_date_submitted__safe'].dt.strftime('%b').fillna('Unknown')

# ----------------- Sidebar dynamic filter choices (populated from data) -----------------
buyer_choices = sorted(df.get('buyer_type', pd.Series('Indirect')).dropna().unique().astype(str).tolist())
vendor_choices = sorted(df[po_vendor_col].dropna().astype(str).unique().tolist()) if (po_vendor_col and po_vendor_col in df.columns) else []
item_choices = sorted(df[product_name_col].dropna().astype(str).unique().tolist()) if (product_name_col and product_name_col in df.columns) else []
dept_choices = sorted(df[po_department_col].dropna().astype(str).unique().tolist()) if (po_department_col and po_department_col in df.columns) else []

st.sidebar.markdown('### Additional Filters (data-driven)')
sel_buyer = st.sidebar.multiselect('Buyer Type', buyer_choices, default=buyer_choices)
sel_vendor = st.sidebar.multiselect('Vendor (pick one or more)', vendor_choices, default=vendor_choices if vendor_choices else [])
sel_item = st.sidebar.multiselect('Item / Product (pick one or more)', item_choices, default=item_choices if item_choices else [])
if dept_choices:
    sel_po_dept = st.sidebar.multiselect('PO Department', ['All Departments'] + dept_choices, default=['All Departments'])
else:
    sel_po_dept = ['All Departments']

# Reset Filters button
if st.sidebar.button('Reset Filters'):
    for k in list(st.session_state.keys()):
        if k.startswith('_') or k in ['filter_*']:
            del st.session_state[k]
    st.experimental_rerun()

# ----------------- Apply Filters -----------------
fil = df.copy()

# Date range filter (date picker overrides FY default)
fil = fil[(fil['pr_date_submitted__safe'] >= pd.to_datetime(dr_start)) & (fil['pr_date_submitted__safe'] <= pd.to_datetime(dr_end))]

# If user selected a specific month, apply it
if sel_month and sel_month != 'All Months':
    fil = fil[fil['pr_month_name'].str.startswith(sel_month)]

# Buyer filter
if sel_buyer:
    fil = fil[fil['buyer_type'].astype(str).isin(sel_buyer)]

# Vendor filter
if sel_vendor and po_vendor_col in fil.columns:
    fil = fil[fil[po_vendor_col].astype(str).isin(sel_vendor)]

# Item filter
if sel_item and product_name_col in fil.columns:
    fil = fil[fil[product_name_col].astype(str).isin(sel_item)]

# PO Department filter
if sel_po_dept and 'All Departments' not in sel_po_dept and po_department_col in fil.columns:
    fil = fil[fil[po_department_col].astype(str).isin(sel_po_dept)]

# ----------------- Small sanity / warnings if filters empty -----------------
if verbose:
    # Show a short summary of how many rows remain after filters
    st.sidebar.write(f"Rows after applying filters: {len(fil):,}")

if len(fil) == 0:
    st.warning("No rows match the current filters. Try selecting a wider date range or clearing month/buyer/vendor filters.")
    # continue — UI will show blank charts gracefully

# ----------------- Tabs (same layout as before) -----------------
T = st.tabs(['KPIs & Spend','PO/PR Timing','Delivery','Vendors','Dept & Services','Unit-rate Outliers','Forecast','Scorecards','Search'])

# ----------------- KPIs & Spend -----------------
with T[0]:
    c1,c2,c3,c4,c5 = st.columns(5)
    def nunique_safe(d, col):
        return int(d[col].nunique()) if (col and col in d.columns) else 0
    total_prs = nunique_safe(fil, pr_number_col)
    total_pos = nunique_safe(fil, purchase_doc_col)
    c1.metric('Total PRs', total_prs)
    c2.metric('Total POs', total_pos)
    c3.metric('Line Items', len(fil))
    c4.metric('Entities', int(fil.get('__source_entity', pd.Series(dtype=object)).nunique()))
    spend_val = fil[net_amount_col].sum() if (net_amount_col and net_amount_col in fil.columns) else 0
    c5.metric('Spend (Cr ₹)', f"{spend_val/1e7:,.2f}")

    st.markdown('---')
    dcol = po_create_col if (po_create_col and po_create_col in fil.columns) else (pr_col if (pr_col and pr_col in fil.columns) else None)
    st.subheader('Monthly Total Spend + Cumulative')
    if dcol and net_amount_col in fil.columns:
        t = fil.dropna(subset=[dcol]).copy()
        t['po_month'] = pd.to_datetime(t[dcol]).dt.to_period('M').dt.to_timestamp()
        t['month_str'] = t['po_month'].dt.strftime('%b-%Y')
        m = t.groupby(['po_month','month_str'], as_index=False)[net_amount_col].sum().sort_values('po_month')
        m['cr'] = m[net_amount_col]/1e7
        m['cumcr'] = m['cr'].cumsum()
        fig_spend = make_subplots(specs=[[{"secondary_y":True}]])
        fig_spend.add_bar(x=m['month_str'], y=m['cr'], name='Monthly Spend (Cr ₹)')
        fig_spend.add_scatter(x=m['month_str'], y=m['cumcr'], name='Cumulative (Cr ₹)', mode='lines+markers', secondary_y=True)
        fig_spend.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_spend, use_container_width=True)
    else:
        st.info('Monthly Spend chart not available — need a date and Net Amount column.')

    st.markdown('---')
    st.subheader('Entity Trend')
    if dcol and net_amount_col in fil.columns:
        x = fil.copy()
        x['po_month'] = pd.to_datetime(x[dcol]).dt.to_period('M').dt.to_timestamp()
        g = x.dropna(subset=['po_month']).groupby(['po_month', '__source_entity'], as_index=False)[net_amount_col].sum()
        if not g.empty:
            g['cr'] = g[net_amount_col]/1e7
            fig_entity = px.line(g, x=g['po_month'].dt.strftime('%b-%Y'), y='cr', color='__source_entity', markers=True, labels={'x':'Month','cr':'Cr ₹'})
            fig_entity.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_entity, use_container_width=True)
        else:
            st.info('Entity trend not available (no grouped data).')

# ----------------- PR/PO Timing -----------------
with T[1]:
    st.subheader('SLA (PR→PO ≤7d)')
    if pr_col and pr_col in fil.columns and po_create_col and po_create_col in fil.columns:
        ld = fil.dropna(subset=[po_create_col, pr_col]).copy()
        ld['lead_time_days'] = (pd.to_datetime(ld[po_create_col]) - pd.to_datetime(ld[pr_col])).dt.days
        avg = float(ld['lead_time_days'].mean()) if not ld.empty else 0.0
        max_range = max(14, avg*1.2 if avg else 14)
        fig = go.Figure(go.Indicator(mode='gauge+number', value=avg, number={'suffix':' d'},
                                     gauge={'axis':{'range':[0,max_range]}, 'bar':{'color':'darkblue'},
                                            'steps':[{'range':[0,7],'color':'lightgreen'},{'range':[7,max_range],'color':'lightcoral'}],
                                            'threshold':{'line':{'color':'red','width':4}, 'value':7}}))
        st.plotly_chart(fig, use_container_width=True)
        bins=[0,7,15,30,60,90,999]
        labels=['0-7','8-15','16-30','31-60','61-90','90+']
        ag = pd.cut(ld['lead_time_days'], bins=bins, labels=labels).value_counts(normalize=True).reindex(labels, fill_value=0).reset_index()
        ag.columns=['Bucket','Pct']
        ag['Pct'] = ag['Pct']*100
        st.plotly_chart(px.bar(ag, x='Bucket', y='Pct', text='Pct').update_traces(texttemplate='%{text:.1f}%', textposition='outside'), use_container_width=True)
    else:
        st.info('Need PR date and PO create date to compute SLA.')

    st.subheader('PR & PO per Month')
    tmp = fil.copy()
    if pr_col and pr_col in tmp.columns:
        tmp['pr_month'] = pd.to_datetime(tmp[pr_col]).dt.to_period('M')
    else:
        tmp['pr_month'] = pd.NaT
    if po_create_col and po_create_col in tmp.columns:
        tmp['po_month'] = pd.to_datetime(tmp[po_create_col]).dt.to_period('M')
    else:
        tmp['po_month'] = pd.NaT
    if pr_number_col and purchase_doc_col and pr_number_col in tmp.columns and purchase_doc_col in tmp.columns:
        ms = tmp.groupby('pr_month').agg({pr_number_col:'count', purchase_doc_col:'count'}).reset_index()
        if not ms.empty:
            ms.columns=['Month','PR Count','PO Count']
            ms['Month'] = ms['Month'].astype(str)
            st.line_chart(ms.set_index('Month'), use_container_width=True)
    st.subheader('Weekday Split')
    wd = fil.copy()
    if pr_col and pr_col in wd.columns:
        wd['pr_wk'] = pd.to_datetime(wd[pr_col]).dt.day_name()
    else:
        wd['pr_wk'] = ''
    if po_create_col and po_create_col in wd.columns:
        wd['po_wk'] = pd.to_datetime(wd[po_create_col]).dt.day_name()
    else:
        wd['po_wk'] = ''
    order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    prc = wd['pr_wk'].value_counts().reindex(order, fill_value=0)
    poc = wd['po_wk'].value_counts().reindex(order, fill_value=0)
    c1,c2 = st.columns(2)
    c1.bar_chart(prc)
    c2.bar_chart(poc)

# ----------------- Delivery -----------------
with T[2]:
    st.subheader('Delivery Summary')
    dv = fil.rename(columns={'po_quantity':'po_qty', 'receivedqty':'received_qty', 'pending_qty':'pending_qty'}).copy()
    if 'po_qty' in dv.columns and 'received_qty' in dv.columns:
        dv['%_received'] = np.where(dv['po_qty'].astype(float) > 0, (dv['received_qty'].astype(float) / dv['po_qty'].astype(float)) * 100, 0.0)
        group_cols = [purchase_doc_col, po_vendor_col, product_name_col, 'item_description']
        agcols = [c for c in group_cols if c in dv.columns]
        summ = dv.groupby(agcols, dropna=False).agg({'po_qty':'sum','received_qty':'sum','pending_qty':'sum','%_received':'mean'}).reset_index()
        if not summ.empty:
            st.dataframe(summ.sort_values('pending_qty', ascending=False), use_container_width=True)
            st.plotly_chart(px.bar(summ.sort_values('pending_qty', ascending=False).head(20), x=purchase_doc_col or 'purchase_doc', y='pending_qty', color=po_vendor_col if po_vendor_col in summ.columns else None, text='pending_qty', title='Top 20 Pending Qty').update_traces(textposition='outside'), use_container_width=True)
    else:
        st.info('Delivery columns (PO Qty / Received Qty) missing.')

# ----------------- Vendors -----------------
with T[3]:
    st.subheader('Top Vendors by Spend')
    if po_vendor_col and po_vendor_col in fil.columns and purchase_doc_col and purchase_doc_col in fil.columns and net_amount_col and net_amount_col in fil.columns:
        vs = fil.groupby(po_vendor_col, dropna=False).agg(Vendor_PO_Count=(purchase_doc_col,'nunique'), Total_Spend_Cr=(net_amount_col, lambda s: (s.sum()/1e7).round(2))).reset_index().sort_values('Total_Spend_Cr', ascending=False)
        st.dataframe(vs.head(10), use_container_width=True)
        st.plotly_chart(px.bar(vs.head(10), x=po_vendor_col, y='Total_Spend_Cr', text='Total_Spend_Cr', title='Top 10 Vendors (Cr ₹)').update_traces(textposition='outside'), use_container_width=True)
    else:
        st.info('Vendor / Spend columns not present to show Top Vendors.')

# ----------------- Dept & Services (simple view) -----------------
with T[4]:
    st.subheader('Dept & Services (simple view)')
    if net_amount_col in fil.columns:
        dept_col = po_department_col if (po_department_col and po_department_col in fil.columns) else None
        if dept_col:
            dep = fil.groupby(dept_col, dropna=False)[net_amount_col].sum().reset_index().sort_values(net_amount_col, ascending=False)
            dep['cr'] = dep[net_amount_col]/1e7
            st.plotly_chart(px.bar(dep.head(30), x=dept_col, y='cr', title='Department-wise Spend (Top 30)').update_layout(xaxis_tickangle=-45), use_container_width=True)
            st.dataframe(dep.head(50), use_container_width=True)
        else:
            st.info("No PO Department column detected; can't show Dept view.")
    else:
        st.info("No Net Amount column detected; can't show Dept spend.")

# ----------------- Unit-rate Outliers -----------------
with T[5]:
    st.subheader('Unit-rate Outliers vs Historical Median')
    grp_candidates = [c for c in [product_name_col, 'item_code'] if c and c in fil.columns]
    if grp_candidates and po_unit_rate_col and po_unit_rate_col in fil.columns:
        grp_by = st.selectbox('Group by', grp_candidates, index=0)
        z = fil[[grp_by, po_unit_rate_col, purchase_doc_col, pr_number_col, po_vendor_col, 'item_description', po_create_col, net_amount_col]].dropna(subset=[grp_by, po_unit_rate_col]).copy()
        med = z.groupby(grp_by)[po_unit_rate_col].median().rename('median_rate')
        z = z.join(med, on=grp_by)
        z['pctdev'] = (z[po_unit_rate_col] - z['median_rate']) / z['median_rate'].replace(0, np.nan)
        thr = st.slider('Outlier threshold (±%)', 10, 300, 50, 5)
        out = z[abs(z['pctdev']) >= thr/100.0].copy(); out['pctdev%'] = (out['pctdev']*100).round(1)
        st.dataframe(out.sort_values('pctdev%', ascending=False), use_container_width=True)
    else:
        st.info("Need 'PO Unit Rate' and a grouping column (product_name/item_code)")

# ----------------- Forecast -----------------
with T[6]:
    st.subheader('Forecast Next Month Spend (SMA)')
    dcol = po_create_col if (po_create_col and po_create_col in fil.columns) else (pr_col if (pr_col and pr_col in fil.columns) else None)
    if dcol and net_amount_col and net_amount_col in fil.columns:
        t = fil.dropna(subset=[dcol]).copy(); t['month'] = pd.to_datetime(t[dcol]).dt.to_period('M').dt.to_timestamp()
        m = t.groupby('month')[net_amount_col].sum().sort_index(); m_cr = (m/1e7)
        k = st.slider('Window (months)', 3, 12, 6, 1)
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
        fig.add_scatter(x=[nxt,nxt], y=[lo,hi], mode='lines', name='95% CI')
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Forecast for {nxt.strftime('%b-%Y')}: {mu:.2f} Cr (95% CI: {lo:.2f}–{hi:.2f})")
    else:
        st.info('Need date and Net Amount to forecast.')

# ----------------- Vendor Scorecards -----------------
with T[7]:
    st.subheader('Vendor Scorecard')
    if po_vendor_col and po_vendor_col in fil.columns:
        vendor = st.selectbox('Pick Vendor', sorted(fil[po_vendor_col].dropna().astype(str).unique().tolist()))
        vd = fil[fil[po_vendor_col].astype(str) == str(vendor)].copy()
        spend = vd.get(net_amount_col, pd.Series(0)).sum()/1e7 if net_amount_col else 0
        upos = int(vd.get(purchase_doc_col, pd.Series(dtype=object)).nunique()) if purchase_doc_col else 0
        k1,k2,k3,k4 = st.columns(4)
        k1.metric('Spend (Cr)', f"{spend:.2f}"); k2.metric('Unique POs', upos); k3.metric('Late PO count', 'N/A'); k4.metric('Pending Value (Cr)', 'N/A')
        if 'product_name' in vd.columns and po_unit_rate_col and po_unit_rate_col in vd.columns:
            med = vd.groupby('product_name')[po_unit_rate_col].median().rename('median_rate'); v2 = vd.join(med, on='product_name'); v2['var%'] = ((v2[po_unit_rate_col]-v2['median_rate'])/v2['median_rate'].replace(0,np.nan))*100
            st.plotly_chart(px.box(v2, x='product_name', y=po_unit_rate_col, points='outliers', title='Price variance by item'), use_container_width=True)
    else:
        st.info('No Vendor column present to show scorecards.')

# ----------------- Search (Keyword) -----------------
with T[8]:
    st.subheader('🔍 Keyword Search')
    valid_cols = [c for c in [pr_number_col, purchase_doc_col, product_name_col, po_vendor_col] if c and c in df.columns]
    query = st.text_input('Type vendor, product, PO, PR, etc.', '')
    cat_sel = st.multiselect('Filter by Procurement Category', sorted(df.get('procurement_category', pd.Series(dtype=object)).dropna().astype(str).unique().tolist())) if 'procurement_category' in df.columns else []
    vend_sel = st.multiselect('Filter by Vendor', vendor_choices) if (po_vendor_col and po_vendor_col in df.columns) else []
    if query and valid_cols:
        mask = pd.Series(False, index=df.index)
        q = query.lower()
        for c in valid_cols:
            mask = mask | df[c].astype(str).str.lower().str.contains(q, na=False)
        res = df[mask].copy()
        if cat_sel and 'procurement_category' in df.columns:
            res = res[res['procurement_category'].astype(str).isin(cat_sel)]
        if vend_sel and po_vendor_col in df.columns:
            res = res[res[po_vendor_col].astype(str).isin(vend_sel)]
        st.write(f'Found {len(res)} rows')
        st.dataframe(res, use_container_width=True)
        st.download_button('⬇️ Download Search Results', res.to_csv(index=False), file_name='search_results.csv', mime='text/csv', key='dl_search')
    else:
        st.caption('Start typing to search…')

# ----------------- Bottom uploader (persisted) -----------------
st.markdown('---')
st.markdown('### Upload & Debug (bottom of page)')
new_files = st.file_uploader('Upload Excel/CSV files here (bottom uploader) — select multiple if needed', type=['xlsx','xls','csv'], accept_multiple_files=True, key='_bottom_uploader')
if new_files:
    st.session_state['_bottom_uploaded'] = new_files
    st.experimental_rerun()
