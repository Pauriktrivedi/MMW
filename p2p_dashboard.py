# P2P Dashboard — Safe full app (copy/paste into app.py)
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re

st.set_page_config(page_title="P2P Dashboard — Full (Safe)", layout="wide", initial_sidebar_state="expanded")

# ---- Header ----
st.title("P2P Dashboard — Indirect")
st.caption("Purchase-to-Pay overview (Indirect spend focus)")

# ----------------- Helpers -----------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
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
    frames = []
    if uploaded_files:
        for f in uploaded_files:
            # try excel, fallback to csv
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

# ----------------- Sidebar filters -----------------
st.sidebar.header("Filters")

verbose = st.sidebar.checkbox("Verbose debug loader (show detected columns)", value=False)

FY = {
    'All Years': (pd.Timestamp('2000-01-01'), pd.Timestamp('2099-12-31')),
    '2023': (pd.Timestamp('2023-04-01'), pd.Timestamp('2024-03-31')),
    '2024': (pd.Timestamp('2024-04-01'), pd.Timestamp('2025-03-31')),
    '2025': (pd.Timestamp('2025-04-01'), pd.Timestamp('2026-03-31')),
}
fy_key = st.sidebar.selectbox("Financial Year", list(FY.keys()), index=0)
fy_start, fy_end = FY[fy_key]

months = ['All Months','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
sel_month = st.sidebar.selectbox("Month (applies when Year selected)", months, index=0)

date_range = st.sidebar.date_input("Select Date Range (optional)", value=[fy_start.date(), fy_end.date()], key='date_picker')
if isinstance(date_range, (list,tuple)) and len(date_range)==2:
    dr_start = pd.to_datetime(date_range[0])
    dr_end = pd.to_datetime(date_range[1])
else:
    dr_start, dr_end = fy_start, fy_end

st.sidebar.markdown("---")

# ----------------- Load data (supports bottom uploader session) -----------------
uploaded_session = st.session_state.get('_bottom_uploaded', None)
df = load_all_from_files(uploaded_session)

if df.empty:
    st.warning("No data loaded. Place defaults (MEPL.xlsx etc.) next to this script OR upload files using the bottom uploader (bottom of page).")
    st.markdown("---")
    st.markdown("### Upload files (bottom of page)")
    new_files = st.file_uploader("Upload Excel/CSV files here", type=['xlsx','xls','csv'], accept_multiple_files=True, key='_bottom_uploader')
    if new_files:
        st.session_state['_bottom_uploaded'] = new_files
        st.experimental_rerun()
    st.stop()

# ----------------- Auto-detect important columns (safe) -----------------
PR_DATE_COL = find_best_column(df, ['pr_date','pr date','pr_date_submitted','pr_date_submitted','prsubmitted','request_date','requested_date'])
PO_CREATE_COL = find_best_column(df, ['po_create','po create','po_created','po_create_date','po_created_date','po_date'])
PO_DELIVERY_COL = find_best_column(df, ['delivery','po_delivery','po delivery','delivered','delivery_date'])
NET_AMOUNT_COL = find_best_column(df, ['net amount','net_amount','netamount','amount','value'])
PURCHASE_DOC_COL = find_best_column(df, ['purchase_doc','purchase doc','po number','po_number','po no','po_no','po'])
PR_NUMBER_COL = find_best_column(df, ['pr number','pr_number','pr no','pr'])
PO_VENDOR_COL = find_best_column(df, ['vendor','supplier','po_vendor','supplier_name'])
PRODUCT_NAME_COL = find_best_column(df, ['product','product name','item','product_name'])
PO_UNIT_RATE_COL = find_best_column(df, ['unit rate','unit_rate','rate','unitprice'])
PENDING_QTY_COL = find_best_column(df, ['pending_qty','pending qty','pending'])
RECEIVED_QTY_COL = find_best_column(df, ['receivedqty','received qty','received'])
PO_DEPARTMENT_COL = find_best_column(df, ['po_department','po department','department','dept','pr_department'])

# If verbose show diagnostics
if verbose:
    with st.expander("Diagnostics: detected columns & preview", expanded=True):
        st.write("Detected columns (normalized):")
        st.write(list(df.columns)[:200])
        st.write("Auto-detected mapping:")
        st.json({
            "pr_col": PR_DATE_COL,
            "po_create_col": PO_CREATE_COL,
            "po_delivery_col": PO_DELIVERY_COL,
            "net_amount_col": NET_AMOUNT_COL,
            "purchase_doc_col": PURCHASE_DOC_COL,
            "pr_number_col": PR_NUMBER_COL,
            "po_vendor_col": PO_VENDOR_COL,
            "product_name_col": PRODUCT_NAME_COL,
            "po_unit_rate_col": PO_UNIT_RATE_COL,
            "pending_qty_col": PENDING_QTY_COL,
            "po_department_col": PO_DEPARTMENT_COL
        })
        st.markdown("**Data preview (top 5 rows)**")
        st.dataframe(df.head(5))

# ----------------- Parse date column(s) safely and derive month/year -----------------
if PR_DATE_COL and PR_DATE_COL in df.columns:
    df[PR_DATE_COL] = pd.to_datetime(df[PR_DATE_COL], errors='coerce')
else:
    # create a safe PR date column to avoid many checks later
    df['__pr_date_safe'] = pd.NaT
    PR_DATE_COL = '__pr_date_safe'

if PO_CREATE_COL and PO_CREATE_COL in df.columns:
    df[PO_CREATE_COL] = pd.to_datetime(df[PO_CREATE_COL], errors='coerce')
if PO_DELIVERY_COL and PO_DELIVERY_COL in df.columns:
    df[PO_DELIVERY_COL] = pd.to_datetime(df[PO_DELIVERY_COL], errors='coerce')

# derived fields used for filters (always present)
df['pr_date_safe'] = df[PR_DATE_COL] if PR_DATE_COL in df.columns else pd.NaT
df['pr_year'] = df['pr_date_safe'].dt.year
df['pr_month_name'] = df['pr_date_safe'].dt.strftime('%b').fillna('Unknown')

# ----------------- Ensure buyer_type exists (safe fallback) -----------------
# if buyer_type not present, try to create from other known columns, else default 'Indirect'
if 'buyer_type' not in df.columns:
    if PO_DEPARTMENT_COL and PO_DEPARTMENT_COL in df.columns:
        # nothing fancy — default to 'Indirect' (app previously used po_buyer_type fallback)
        df['buyer_type'] = 'Indirect'
    else:
        df['buyer_type'] = 'Indirect'

# ----------------- Build sidebar dynamic filters (only when column exists) -----------------
vendor_choices = sorted(df[PO_VENDOR_COL].dropna().astype(str).unique().tolist()) if (PO_VENDOR_COL and PO_VENDOR_COL in df.columns) else []
item_choices = sorted(df[PRODUCT_NAME_COL].dropna().astype(str).unique().tolist()) if (PRODUCT_NAME_COL and PRODUCT_NAME_COL in df.columns) else []
dept_choices = sorted(df[PO_DEPARTMENT_COL].dropna().astype(str).unique().tolist()) if (PO_DEPARTMENT_COL and PO_DEPARTMENT_COL in df.columns) else []
buyer_choices = sorted(df['buyer_type'].dropna().astype(str).unique().tolist())

st.sidebar.markdown("### Additional Filters (data-driven)")
sel_buyer = st.sidebar.multiselect('Buyer Type', buyer_choices, default=buyer_choices)
if vendor_choices:
    sel_vendor = st.sidebar.multiselect('Vendor (pick one or more)', vendor_choices, default=vendor_choices)
else:
    sel_vendor = []
if item_choices:
    sel_item = st.sidebar.multiselect('Item / Product (pick one or more)', item_choices, default=item_choices)
else:
    sel_item = []
if dept_choices:
    sel_po_dept = st.sidebar.multiselect('PO Department', ['All Departments'] + dept_choices, default=['All Departments'])
else:
    sel_po_dept = ['All Departments']

# Reset filters button
if st.sidebar.button('Reset Filters'):
    # only clear keys we created — avoid removing unrelated session state
    keys_to_clear = [k for k in st.session_state.keys() if k.startswith('_') and k != '_bottom_uploaded']
    for k in keys_to_clear:
        del st.session_state[k]
    st.experimental_rerun()

# ----------------- Apply filters safely -----------------
fil = df.copy()

# date range
fil = fil[(fil['pr_date_safe'] >= pd.to_datetime(dr_start)) & (fil['pr_date_safe'] <= pd.to_datetime(dr_end))]

# month
if sel_month and sel_month != 'All Months':
    # compare using abbreviated month names (Jan, Feb, ..)
    fil = fil[fil['pr_month_name'].str.startswith(sel_month)]

# buyer type
if sel_buyer and 'buyer_type' in fil.columns:
    fil = fil[fil['buyer_type'].astype(str).isin(sel_buyer)]

# vendor
if sel_vendor and (PO_VENDOR_COL and PO_VENDOR_COL in fil.columns):
    fil = fil[fil[PO_VENDOR_COL].astype(str).isin(sel_vendor)]

# item
if sel_item and (PRODUCT_NAME_COL and PRODUCT_NAME_COL in fil.columns):
    fil = fil[fil[PRODUCT_NAME_COL].astype(str).isin(sel_item)]

# po department
if sel_po_dept and 'All Departments' not in sel_po_dept and (PO_DEPARTMENT_COL and PO_DEPARTMENT_COL in fil.columns):
    fil = fil[fil[PO_DEPARTMENT_COL].astype(str).isin(sel_po_dept)]

# show small status when verbose
if verbose:
    st.sidebar.write(f"Rows after filters: {len(fil):,}")

if len(fil) == 0:
    st.warning("No rows match the current filters. Try a wider date-range, clear Month selection, or toggle verbose diagnostics to inspect column mapping.")

# ----------------- Tabs and charts (same layout, guarded where columns may not exist) -----------------
T = st.tabs(['KPIs & Spend','PO/PR Timing','Delivery','Vendors','Dept & Services','Unit-rate Outliers','Forecast','Scorecards','Search'])

with T[0]:
    c1,c2,c3,c4,c5 = st.columns(5)
    def nunique_safe(d, col):
        return int(d[col].nunique()) if (col and col in d.columns) else 0
    total_prs = nunique_safe(fil, PR_NUMBER_COL) if PR_NUMBER_COL else nunique_safe(fil, 'pr_number')
    total_pos = nunique_safe(fil, PURCHASE_DOC_COL) if PURCHASE_DOC_COL else nunique_safe(fil, 'purchase_doc')
    c1.metric('Total PRs', total_prs)
    c2.metric('Total POs', total_pos)
    c3.metric('Line Items', len(fil))
    c4.metric('Entities', int(fil.get('__source_entity', pd.Series(dtype=object)).nunique()))
    spend_val = fil[NET_AMOUNT_COL].sum() if (NET_AMOUNT_COL and NET_AMOUNT_COL in fil.columns) else 0
    c5.metric('Spend (Cr ₹)', f"{spend_val/1e7:,.2f}")

    st.markdown('---')
    dcol = PO_CREATE_COL if (PO_CREATE_COL and PO_CREATE_COL in fil.columns) else (PR_DATE_COL if (PR_DATE_COL and PR_DATE_COL in fil.columns) else None)
    st.subheader('Monthly Total Spend + Cumulative')
    if dcol and (NET_AMOUNT_COL and NET_AMOUNT_COL in fil.columns):
        t = fil.dropna(subset=[dcol]).copy()
        t['po_month'] = pd.to_datetime(t[dcol]).dt.to_period('M').dt.to_timestamp()
        t['month_str'] = t['po_month'].dt.strftime('%b-%Y')
        m = t.groupby(['po_month','month_str'], as_index=False)[NET_AMOUNT_COL].sum().sort_values('po_month')
        m['cr'] = m[NET_AMOUNT_COL]/1e7
        m['cumcr'] = m['cr'].cumsum()
        fig_spend = make_subplots(specs=[[{"secondary_y":True}]])
        fig_spend.add_bar(x=m['month_str'], y=m['cr'], name='Monthly Spend (Cr ₹)')
        fig_spend.add_scatter(x=m['month_str'], y=m['cumcr'], name='Cumulative (Cr ₹)', mode='lines+markers', secondary_y=True)
        fig_spend.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_spend, use_container_width=True)
    else:
        st.info('Monthly Spend chart not available — need a date column and a Net Amount column.')

with T[1]:
    st.subheader('SLA (PR→PO ≤7d)')
    if PR_DATE_COL and PR_DATE_COL in fil.columns and PO_CREATE_COL and PO_CREATE_COL in fil.columns:
        ld = fil.dropna(subset=[PO_CREATE_COL, PR_DATE_COL]).copy()
        ld['lead_time_days'] = (pd.to_datetime(ld[PO_CREATE_COL]) - pd.to_datetime(ld[PR_DATE_COL])).dt.days
        avg = float(ld['lead_time_days'].mean()) if not ld.empty else 0.0
        max_range = max(14, avg*1.2 if avg else 14)
        fig = go.Figure(go.Indicator(mode='gauge+number', value=avg, number={'suffix':' d'},
                                     gauge={'axis':{'range':[0,max_range]}, 'bar':{'color':'darkblue'},
                                            'steps':[{'range':[0,7],'color':'lightgreen'},{'range':[7,max_range],'color':'lightcoral'}],
                                            'threshold':{'line':{'color':'red','width':4}, 'value':7}}))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('Need PR and PO create dates to compute SLA.')

with T[2]:
    st.subheader('Delivery Summary')
    dv = fil.rename(columns={'po_quantity':'po_qty', 'receivedqty':'received_qty', 'pending_qty':'pending_qty'}).copy()
    group_cols = []
    if PURCHASE_DOC_COL and PURCHASE_DOC_COL in dv.columns:
        group_cols.append(PURCHASE_DOC_COL)
    if PO_VENDOR_COL and PO_VENDOR_COL in dv.columns:
        group_cols.append(PO_VENDOR_COL)
    if PRODUCT_NAME_COL and PRODUCT_NAME_COL in dv.columns:
        group_cols.append(PRODUCT_NAME_COL)
    if group_cols and (('po_qty' in dv.columns and 'received_qty' in dv.columns) or ('received_qty' in dv.columns and 'pending_qty' in dv.columns)):
        agcols = group_cols
        agg_map = {}
        if 'po_qty' in dv.columns: agg_map['po_qty'] = 'sum'
        if 'received_qty' in dv.columns: agg_map['received_qty'] = 'sum'
        if 'pending_qty' in dv.columns: agg_map['pending_qty'] = 'sum'
        summ = dv.groupby(agcols, dropna=False).agg(agg_map).reset_index()
        if not summ.empty:
            st.dataframe(summ.sort_values(by=summ.columns[-1], ascending=False), use_container_width=True)
    else:
        st.info('Delivery view needs PO Qty / Received Qty or Pending Qty and at least one grouping column.')

with T[3]:
    st.subheader('Top Vendors by Spend')
    if PO_VENDOR_COL and PO_VENDOR_COL in fil.columns and (PURCHASE_DOC_COL and PURCHASE_DOC_COL in fil.columns) and (NET_AMOUNT_COL and NET_AMOUNT_COL in fil.columns):
        vs = fil.groupby(PO_VENDOR_COL, dropna=False).agg(Vendor_PO_Count=(PURCHASE_DOC_COL, 'nunique'), Total_Spend_Cr=(NET_AMOUNT_COL, lambda s: (s.sum()/1e7).round(2))).reset_index().sort_values('Total_Spend_Cr', ascending=False)
        st.dataframe(vs.head(10), use_container_width=True)
        st.plotly_chart(px.bar(vs.head(10), x=PO_VENDOR_COL, y='Total_Spend_Cr', text='Total_Spend_Cr', title='Top 10 Vendors (Cr ₹)').update_traces(textposition='outside'), use_container_width=True)
    else:
        st.info('Vendor / Spend columns not available to show Top Vendors.')

with T[4]:
    st.subheader('Dept & Services (simple view)')
    if NET_AMOUNT_COL and NET_AMOUNT_COL in fil.columns and PO_DEPARTMENT_COL and PO_DEPARTMENT_COL in fil.columns:
        dep = fil.groupby(PO_DEPARTMENT_COL, dropna=False)[NET_AMOUNT_COL].sum().reset_index().sort_values(NET_AMOUNT_COL, ascending=False)
        dep['cr'] = dep[NET_AMOUNT_COL]/1e7
        if not dep.empty:
            st.plotly_chart(px.bar(dep.head(30), x=PO_DEPARTMENT_COL, y='cr', title='Department-wise Spend (Top 30)').update_layout(xaxis_tickangle=-45), use_container_width=True)
            st.dataframe(dep.head(50), use_container_width=True)
    else:
        st.info('Dept view requires a Net Amount column and a PO Department column.')

with T[5]:
    st.subheader('Unit-rate Outliers vs Historical Median')
    grp_candidates = [c for c in [PRODUCT_NAME_COL, 'item_code'] if c and c in fil.columns]
    if grp_candidates and PO_UNIT_RATE_COL and PO_UNIT_RATE_COL in fil.columns:
        grp_by = st.selectbox('Group by', grp_candidates, index=0)
        cols_needed = [grp_by, PO_UNIT_RATE_COL]
        z = fil.dropna(subset=cols_needed).copy()
        if not z.empty:
            med = z.groupby(grp_by)[PO_UNIT_RATE_COL].median().rename('median_rate')
            z = z.join(med, on=grp_by)
            z['pctdev'] = (z[PO_UNIT_RATE_COL] - z['median_rate']) / z['median_rate'].replace(0, np.nan)
            thr = st.slider('Outlier threshold (±%)', 10, 300, 50, 5)
            out = z[abs(z['pctdev']) >= thr/100.0].copy(); out['pctdev%'] = (out['pctdev']*100).round(1)
            st.dataframe(out.sort_values('pctdev%', ascending=False), use_container_width=True)
    else:
        st.info("Need 'PO Unit Rate' and a grouping column (product_name / item_code).")

with T[6]:
    st.subheader('Forecast Next Month Spend (SMA)')
    dcol = PO_CREATE_COL if (PO_CREATE_COL and PO_CREATE_COL in fil.columns) else (PR_DATE_COL if (PR_DATE_COL and PR_DATE_COL in fil.columns) else None)
    if dcol and NET_AMOUNT_COL and NET_AMOUNT_COL in fil.columns:
        t = fil.dropna(subset=[dcol]).copy(); t['month'] = pd.to_datetime(t[dcol]).dt.to_period('M').dt.to_timestamp()
        m = t.groupby('month')[NET_AMOUNT_COL].sum().sort_index(); m_cr = (m/1e7)
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

with T[7]:
    st.subheader('Vendor Scorecard')
    if PO_VENDOR_COL and PO_VENDOR_COL in fil.columns:
        vendor_list = sorted(fil[PO_VENDOR_COL].dropna().astype(str).unique().tolist())
        if vendor_list:
            vendor = st.selectbox('Pick Vendor', vendor_list)
            vd = fil[fil[PO_VENDOR_COL].astype(str) == str(vendor)].copy()
            spend = vd[NET_AMOUNT_COL].sum()/1e7 if NET_AMOUNT_COL and NET_AMOUNT_COL in vd.columns else 0
            upos = int(vd[PURCHASE_DOC_COL].nunique()) if PURCHASE_DOC_COL and PURCHASE_DOC_COL in vd.columns else 0
            k1,k2,k3,k4 = st.columns(4)
            k1.metric('Spend (Cr)', f"{spend:.2f}"); k2.metric('Unique POs', upos); k3.metric('Late PO count', 'N/A'); k4.metric('Pending Value (Cr)', 'N/A')
        else:
            st.info('No vendors available after filtering.')
    else:
        st.info('No Vendor column present to show vendor scorecards.')

with T[8]:
    st.subheader("🔍 Keyword Search")
    valid_cols = [c for c in [PR_NUMBER_COL, PURCHASE_DOC_COL, PRODUCT_NAME_COL, PO_VENDOR_COL] if c and c in df.columns]
    query = st.text_input('Type vendor, product, PO, PR, etc.', '')
    cat_sel = st.multiselect('Filter by Procurement Category', sorted(df.get('procurement_category', pd.Series(dtype=object)).dropna().astype(str).unique().tolist())) if 'procurement_category' in df.columns else []
    vend_sel = st.multiselect('Filter by Vendor', vendor_choices) if (PO_VENDOR_COL and PO_VENDOR_COL in df.columns) else []
    if query and valid_cols:
        mask = pd.Series(False, index=df.index)
        q = query.lower()
        for c in valid_cols:
            mask = mask | df[c].astype(str).str.lower().str.contains(q, na=False)
        res = df[mask].copy()
        if cat_sel and 'procurement_category' in df.columns:
            res = res[res['procurement_category'].astype(str).isin(cat_sel)]
        if vend_sel and PO_VENDOR_COL and PO_VENDOR_COL in df.columns:
            res = res[res[PO_VENDOR_COL].astype(str).isin(vend_sel)]
        st.write(f'Found {len(res)} rows')
        st.dataframe(res, use_container_width=True)
        st.download_button('⬇️ Download Search Results', res.to_csv(index=False), file_name='search_results.csv', mime='text/csv', key='dl_search')
    else:
        st.caption('Start typing to search…')

# ----------------- Bottom uploader (persisted to session) -----------------
st.markdown('---')
st.markdown('### Upload & Debug (bottom of page)')
new_files = st.file_uploader('Upload Excel/CSV files here (bottom uploader)', type=['xlsx','xls','csv'], accept_multiple_files=True, key='_bottom_uploader')
if new_files:
    st.session_state['_bottom_uploaded'] = new_files
    st.experimental_rerun()
