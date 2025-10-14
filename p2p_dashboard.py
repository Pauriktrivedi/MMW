# Full P2P Dashboard — expanded with requested filters and bottom uploader
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="P2P Dashboard — Full (Refactor)", layout="wide", initial_sidebar_state="expanded")

# ---- Page header ----
st.title("P2P Dashboard — Indirect")
st.caption("Purchase-to-Pay overview (Indirect spend focus)")

# ----------------- Helpers -----------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names into safe snake_case keys."""
    new_cols = {}
    for c in df.columns:
        s = str(c).strip().replace("\xa0", " ").replace("\\", "_").replace("/", "_")
        s = "_".join(s.split()).lower()
        s = ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in s)
        while '__' in s:
            s = s.replace('__', '_')
        s = s.strip('_')
        new_cols[c] = s
    return df.rename(columns=new_cols)


@st.cache_data(show_spinner=False)
def load_all_from_files(uploaded_files=None):
    """
    Load multiple uploaded files (UploadedFile objects) or default named files in repo.
    Try skiprows=1 first and fallback to read without skipping. Accept CSV fallback.
    """
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
            df_temp['entity'] = f.name.rsplit('.', 1)[0]
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
            df_temp['entity'] = ent
            frames.append(df_temp)

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True, sort=False)
    df = normalize_columns(df)
    return df

# ----------------- Sidebar Filters -----------------
st.sidebar.header('Filters')

# Financial Year choices (maps to actual date ranges)
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

# Date range picker (this will override FY if user picks custom range)
date_range = st.sidebar.date_input('Select Date Range (optional)', value=[fy_start.date(), fy_end.date()], key='date_range_picker')
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    try:
        dr_start = pd.to_datetime(date_range[0])
        dr_end = pd.to_datetime(date_range[1])
    except Exception:
        dr_start, dr_end = fy_start, fy_end
else:
    dr_start, dr_end = fy_start, fy_end

# small reset button
if st.sidebar.button('Reset Filters'):
    for k in list(st.session_state.keys()):
        if k.startswith('filter_') or k in ['_bottom_uploaded']:
            del st.session_state[k]
    st.experimental_rerun()

st.sidebar.markdown('---')
st.sidebar.info('Vendor / Item / PO Department filters will appear once data is loaded.')
st.sidebar.markdown('---')

# ----------------- Data Loading -----------------
# Look for bottom uploader saved in session state; if present use that, else defaults
uploaded_session = st.session_state.get('_bottom_uploaded', None)

df = load_all_from_files(uploaded_session)

# If no data loaded, show bottom uploader and stop
if df.empty:
    st.warning("No data loaded. Place default Excel files next to this script (MEPL.xlsx, MLPL.xlsx, mmw.xlsx, mmpl.xlsx) or use the bottom uploader.")
    st.markdown('---')
    st.markdown('### Upload files (bottom of page)')
    new_files = st.file_uploader('Upload Excel/CSV files here', type=['xlsx','xls','csv'], accept_multiple_files=True, key='_bottom_uploader')
    if new_files:
        st.session_state['_bottom_uploaded'] = new_files
        st.experimental_rerun()
    st.stop()

# ----------------- Column canonicalization & mapping -----------------
source_to_norm = {
    'pr number': 'pr_number', 'pr no': 'pr_number',
    'pr date submitted': 'pr_date_submitted',
    'purchase doc': 'purchase_doc', 'po create date': 'po_create_date', 'po delivery date': 'po_delivery_date',
    'po vendor': 'po_vendor', 'po quantity': 'po_quantity', 'po unit rate': 'po_unit_rate',
    'net amount': 'net_amount', 'pr status': 'pr_status', 'po status': 'po_status',
    'received qty': 'receivedqty', 'receivedqty': 'receivedqty', 'pending qty': 'pending_qty',
    'product name': 'product_name', 'product name friendly': 'product_name_friendly', 'item code': 'item_code',
    'item description': 'item_description', 'po budget code': 'po_budget_code', 'pr budget code': 'pr_budget_code',
    'po orderer': 'po_orderer', 'po department': 'po_department', 'pr department': 'pr_department'
}

col_map = {}
for c in list(df.columns):
    key = str(c).strip().lower()
    if key in source_to_norm:
        col_map[c] = source_to_norm[key]

if col_map:
    df = df.rename(columns=col_map)
df = normalize_columns(df)

# parse date fields safely
for dcol in ['pr_date_submitted','po_create_date','po_approved_date','po_delivery_date','last_po_date']:
    if dcol in df.columns:
        df[dcol] = pd.to_datetime(df[dcol], errors='coerce')

# prepare safe column references
pr_col = 'pr_date_submitted' if 'pr_date_submitted' in df.columns else None
po_create_col = 'po_create_date' if 'po_create_date' in df.columns else None
net_amount_col = 'net_amount' if 'net_amount' in df.columns else None
purchase_doc_col = 'purchase_doc' if 'purchase_doc' in df.columns else None
pr_number_col = 'pr_number' if 'pr_number' in df.columns else None
po_vendor_col = 'po_vendor' if 'po_vendor' in df.columns else None
po_unit_rate_col = 'po_unit_rate' if 'po_unit_rate' in df.columns else None
pending_qty_col = 'pending_qty' if 'pending_qty' in df.columns else None
received_qty_col = 'receivedqty' if 'receivedqty' in df.columns else None
po_department_col = 'po_department' if 'po_department' in df.columns else None
product_name_col = 'product_name' if 'product_name' in df.columns else None

# derived date fields
if pr_col:
    df['pr_date_submitted'] = pd.to_datetime(df[pr_col], errors='coerce')
else:
    df['pr_date_submitted'] = pd.NaT
df['pr_year'] = df['pr_date_submitted'].dt.year
df['pr_month_name'] = df['pr_date_submitted'].dt.strftime('%b')  # 'Jan' etc

# Buyer type fallback
if 'buyer_type' not in df.columns:
    df['buyer_type'] = 'Indirect'

# ----------------- Dynamic Filter choices -----------------
buyer_choices = sorted(df['buyer_type'].dropna().unique().astype(str).tolist())
vendor_choices = sorted(df[po_vendor_col].dropna().astype(str).unique().tolist()) if po_vendor_col in df.columns else []
item_choices = sorted(df[product_name_col].dropna().astype(str).unique().tolist()) if product_name_col in df.columns else []
dept_choices = sorted(df[po_department_col].dropna().astype(str).unique().tolist()) if po_department_col in df.columns else []

# display multi-selects in sidebar
st.sidebar.markdown('### Additional Filters (dynamic)')
sel_buyer = st.sidebar.multiselect('Buyer Type', buyer_choices, default=buyer_choices)
sel_vendor = st.sidebar.multiselect('Vendor (pick one or more)', vendor_choices, default=vendor_choices if vendor_choices else [])
sel_item = st.sidebar.multiselect('Item / Product (pick one or more)', item_choices, default=item_choices if item_choices else [])
if dept_choices:
    sel_po_dept = st.sidebar.multiselect('PO Department', ['All Departments'] + dept_choices, default=['All Departments'])
else:
    sel_po_dept = ['All Departments']

# ----------------- Apply Filters -----------------
fil = df.copy()

# Date range filter (date picker overrides FY default)
fil = fil[(fil['pr_date_submitted'] >= pd.to_datetime(dr_start)) & (fil['pr_date_submitted'] <= pd.to_datetime(dr_end))]

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

# ----------------- Tabs -----------------
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
    c4.metric('Entities', int(fil.get('entity', pd.Series(dtype=object)).nunique()))
    spend_val = fil[net_amount_col].sum() if (net_amount_col and net_amount_col in fil.columns) else 0
    c5.metric('Spend (Cr ₹)', f"{spend_val/1e7:,.2f}")

    st.markdown('---')
    dcol = po_create_col if po_create_col in fil.columns else (pr_col if pr_col in fil.columns else None)
    st.subheader('Monthly Total Spend + Cumulative')
    if dcol and net_amount_col in fil.columns:
        t = fil.dropna(subset=[dcol]).copy()
        t['po_month'] = t[dcol].dt.to_period('M').dt.to_timestamp()
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
        x['po_month'] = x[dcol].dt.to_period('M').dt.to_timestamp()
        g = x.dropna(subset=['po_month']).groupby(['po_month', 'entity'], as_index=False)[net_amount_col].sum()
        if not g.empty:
            g['cr'] = g[net_amount_col]/1e7
            fig_entity = px.line(g, x=g['po_month'].dt.strftime('%b-%Y'), y='cr', color='entity', markers=True, labels={'x':'Month','cr':'Cr ₹'})
            fig_entity.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_entity, use_container_width=True)
        else:
            st.info('Entity trend not available (no grouped data).')

# ----------------- PR/PO Timing -----------------
with T[1]:
    st.subheader('SLA (PR→PO ≤7d)')
    if pr_col in fil.columns and po_create_col in fil.columns:
        ld = fil.dropna(subset=[po_create_col, pr_col]).copy()
        ld['lead_time_days'] = (ld[po_create_col] - ld[pr_col]).dt.days
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
    if pr_col in tmp.columns:
        tmp['pr_month'] = tmp[pr_col].dt.to_period('M')
    else:
        tmp['pr_month'] = pd.NaT
    if po_create_col in tmp.columns:
        tmp['po_month'] = tmp[po_create_col].dt.to_period('M')
    else:
        tmp['po_month'] = pd.NaT
    if pr_number_col and purchase_doc_col:
        ms = tmp.groupby('pr_month').agg({pr_number_col:'count', purchase_doc_col:'count'}).reset_index()
        if not ms.empty:
            ms.columns=['Month','PR Count','PO Count']
            ms['Month'] = ms['Month'].astype(str)
            st.line_chart(ms.set_index('Month'), use_container_width=True)
    st.subheader('Weekday Split')
    wd = fil.copy()
    if pr_col in wd.columns:
        wd['pr_wk'] = wd[pr_col].dt.day_name()
    else:
        wd['pr_wk'] = ''
    if po_create_col in wd.columns:
        wd['po_wk'] = wd[po_create_col].dt.day_name()
    else:
        wd['po_wk'] = ''
    order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    prc = wd['pr_wk'].value_counts().reindex(order, fill_value=0)
    poc = wd['po_wk'].value_counts().reindex(order, fill_value=0)
    c1,c2 = st.columns(2)
    c1.bar_chart(prc)
    c2.bar_chart(poc)
    st.subheader('Open PRs')
    if 'pr_status' in fil.columns:
        op = fil[fil['pr_status'].isin(['Approved','InReview'])].copy()
        if not op.empty and pr_col in op.columns:
            TODAY = pd.Timestamp.now().normalize()
            op['pending_age_d'] = (TODAY - op[pr_col]).dt.days
            cols = [c for c in ['pr_number','pr_date_submitted','pending_age_d','procurement_category','product_name','net_amount','po_budget_code','pr_status','entity','po_creator',purchase_doc_col] if c in op.columns]
            st.dataframe(op[cols], use_container_width=True)

# ----------------- Delivery -----------------
with T[2]:
    st.subheader('Delivery Summary')
    dv = fil.rename(columns={'po_quantity':'po_qty', 'receivedqty':'received_qty', 'pending_qty':'pending_qty'}).copy()
    if 'po_qty' in dv.columns and 'received_qty' in dv.columns:
        dv['%_received'] = np.where(dv['po_qty'].astype(float) > 0, (dv['received_qty'].astype(float) / dv['po_qty'].astype(float)) * 100, 0.0)
        group_cols = [purchase_doc_col, 'po_vendor', 'product_name', 'item_description']
        agcols = [c for c in group_cols if c in dv.columns]
        summ = dv.groupby(agcols, dropna=False).agg({'po_qty':'sum','received_qty':'sum','pending_qty':'sum','%_received':'mean'}).reset_index()
        if not summ.empty:
            st.dataframe(summ.sort_values('pending_qty', ascending=False), use_container_width=True)
            st.plotly_chart(px.bar(summ.sort_values('pending_qty', ascending=False).head(20), x=purchase_doc_col or 'purchase_doc', y='pending_qty', color='po_vendor' if 'po_vendor' in summ.columns else None, text='pending_qty', title='Top 20 Pending Qty').update_traces(textposition='outside'), use_container_width=True)
        st.subheader('Top Pending Lines by Value')
        if 'pending_qty' in dv.columns and po_unit_rate_col in dv.columns:
            dv['pending_value'] = dv['pending_qty'].astype(float) * dv[po_unit_rate_col].astype(float)
            keep = [c for c in ['pr_number', purchase_doc_col, 'procurement_category', 'buying_legal_entity', 'pr_budget_description', 'product_name', 'item_description', 'pending_qty', po_unit_rate_col, 'pending_value'] if c in dv.columns]
            st.dataframe(dv.sort_values('pending_value', ascending=False).head(50)[keep], use_container_width=True)
    else:
        st.info('Delivery columns (PO Qty / Received Qty) missing.')

# ----------------- Vendors -----------------
with T[3]:
    st.subheader('Top Vendors by Spend')
    if po_vendor_col in fil.columns and purchase_doc_col in fil.columns and net_amount_col in fil.columns:
        vs = fil.groupby('po_vendor', dropna=False).agg(Vendor_PO_Count=(purchase_doc_col,'nunique'), Total_Spend_Cr=(net_amount_col, lambda s: (s.sum()/1e7).round(2))).reset_index().sort_values('Total_Spend_Cr', ascending=False)
        st.dataframe(vs.head(10), use_container_width=True)
        st.plotly_chart(px.bar(vs.head(10), x='po_vendor', y='Total_Spend_Cr', text='Total_Spend_Cr', title='Top 10 Vendors (Cr ₹)').update_traces(textposition='outside'), use_container_width=True)

    st.subheader('Vendor Delivery Performance (Top 10 by Spend)')
    if po_vendor_col in fil.columns and purchase_doc_col in fil.columns and po_delivery_col in fil.columns and pending_qty_col in fil.columns:
        vdf = fil.copy()
        vdf['pendingqtyfill'] = vdf[pending_qty_col].fillna(0).astype(float)
        vdf['is_fully_delivered'] = vdf['pendingqtyfill'] == 0
        vdf[po_delivery_col] = pd.to_datetime(vdf[po_delivery_col], errors='coerce')
        TODAY = pd.Timestamp.now().normalize()
        vdf['is_late'] = vdf[po_delivery_col].dt.date.notna() & (vdf[po_delivery_col].dt.date < TODAY.date()) & (vdf['pendingqtyfill'] > 0)
        perf = vdf.groupby('po_vendor', dropna=False).agg(Total_PO_Count=(purchase_doc_col,'nunique'), Fully_Delivered_PO_Count=('is_fully_delivered','sum'), Late_PO_Count=('is_late','sum')).reset_index()
        perf['Pct_Fully_Delivered'] = (perf['Fully_Delivered_PO_Count'] / perf['Total_PO_Count'] * 100).round(1)
        perf['Pct_Late'] = (perf['Late_PO_Count'] / perf['Total_PO_Count'] * 100).round(1)
        if po_vendor_col in fil.columns and net_amount_col in fil.columns:
            spend = fil.groupby('po_vendor', dropna=False)[net_amount_col].sum().rename('Spend').reset_index()
            perf = perf.merge(spend, left_on='po_vendor', right_on='po_vendor', how='left').fillna({'Spend':0})
        top10 = perf.sort_values('Spend', ascending=False).head(10)
        if not top10.empty:
            st.dataframe(top10[['po_vendor','Total_PO_Count','Fully_Delivered_PO_Count','Late_PO_Count','Pct_Fully_Delivered','Pct_Late']], use_container_width=True)
            melt = top10.melt(id_vars=['po_vendor'], value_vars=['Pct_Fully_Delivered','Pct_Late'], var_name='Metric', value_name='Percentage')
            st.plotly_chart(px.bar(melt, x='po_vendor', y='Percentage', color='Metric', barmode='group', title='% Fully Delivered vs % Late (Top 10 by Spend)'), use_container_width=True)

# ----------------- Smart Budget Mapper / Dept & Services -----------------
with T[4]:
    st.subheader('Dept & Services (Smart Mapper)')
    # Reuse earlier mapping logic (kept compact for brevity) — will show unmapped suggestions if mapping file present
    smart = fil.copy()
    smart['dept_chart'] = smart.get('dept_chart', pd.NA)
    smart['subcat_chart'] = smart.get('subcat_chart', pd.NA)
    # simple canonicalization using PO budget code if present
    if 'po_budget_code' in smart.columns:
        smart['po_budget_code'] = smart['po_budget_code'].astype(str)
    # show spend by dept if net_amount present
    if net_amount_col in smart.columns:
        dep = smart.groupby('dept_chart', dropna=False)[net_amount_col].sum().reset_index().sort_values(net_amount_col, ascending=False)
        if not dep.empty:
            dep['cr'] = dep[net_amount_col]/1e7
            st.plotly_chart(px.bar(dep.head(30), x='dept_chart', y='cr', title='Department-wise Spend (Top 30)').update_layout(xaxis_tickangle=-45), use_container_width=True)
            cA,cB = st.columns([2,1])
            dept_pick = cA.selectbox('Drill Department', dep['dept_chart'].astype(str).tolist(), key='dept_pick')
            topn = int(cB.number_input('Top N', 5, 100, 20, 5, key='dept_topn'))
            det = smart[smart['dept_chart'].astype(str) == str(dept_pick)].copy()
            k1,k2,k3 = st.columns(3)
            k1.metric('Lines', len(det)); k2.metric('PRs', int(det.get(pr_number_col, pd.Series(dtype=object)).nunique() if pr_number_col else 0)); k3.metric('Spend (Cr ₹)', f"{det.get(net_amount_col, pd.Series(0)).sum()/1e7:,.2f}")
            if 'subcat_chart' in det.columns:
                ss = det.groupby('subcat_chart', dropna=False)[net_amount_col].sum().reset_index().sort_values(net_amount_col, ascending=False)
                ss['cr'] = ss[net_amount_col]/1e7
                c1,c2 = st.columns(2)
                c1.plotly_chart(px.bar(ss.head(topn), x='subcat_chart', y='cr', title=f"{dept_pick} — Top Services").update_layout(xaxis_tickangle=-45), use_container_width=True)
                c2.plotly_chart(px.pie(ss.head(12), names='subcat_chart', values=net_amount_col, title=f"{dept_pick} — Service Share"), use_container_width=True)
                svc = st.selectbox('Drill Service', ss['subcat_chart'].astype(str).tolist(), key='svc_pick')
                sub = det[det['subcat_chart'].astype(str) == str(svc)].copy()
                cols = [c for c in ['po_budget_code','subcat_chart','dept_chart', purchase_doc_col, pr_number_col, 'procurement_category', 'product_name', 'item_description', 'po_vendor', net_amount_col] if c in sub.columns]
                if not cols:
                    cols = [c for c in ['dept_chart', purchase_doc_col, pr_number_col, net_amount_col] if c in sub.columns]
                st.dataframe(sub[cols], use_container_width=True)
        else:
            st.info('No dept-chart spend data found.')
    else:
        st.info('Net Amount missing — cannot show Dept spend.')

# ----------------- Unit-rate Outliers -----------------
with T[5]:
    st.subheader('Unit-rate Outliers vs Historical Median')
    grp_candidates = [c for c in ['product_name','item_code'] if c in fil.columns]
    if grp_candidates:
        grp_by = st.selectbox('Group by', grp_candidates, index=0)
    else:
        grp_by = None
    if grp_by and po_unit_rate_col in fil.columns:
        z = fil[[grp_by, po_unit_rate_col, purchase_doc_col, pr_number_col, 'po_vendor', 'item_description', po_create_col, net_amount_col]].dropna(subset=[grp_by, po_unit_rate_col]).copy()
        med = z.groupby(grp_by)[po_unit_rate_col].median().rename('median_rate')
        z = z.join(med, on=grp_by)
        z['pctdev'] = (z[po_unit_rate_col] - z['median_rate']) / z['median_rate'].replace(0, np.nan)
        thr = st.slider('Outlier threshold (±%)', 10, 300, 50, 5)
        out = z[abs(z['pctdev']) >= thr/100.0].copy(); out['pctdev%'] = (out['pctdev']*100).round(1)
        st.dataframe(out.sort_values('pctdev%', ascending=False), use_container_width=True)
        if not z.empty:
            st.plotly_chart(px.scatter(z, x=po_create_col, y=po_unit_rate_col, color=np.where(abs(z['pctdev'])>=thr/100.0,'Outlier','Normal'), hover_data=[grp_by, purchase_doc_col, 'po_vendor', 'median_rate']).update_layout(legend_title_text=''), use_container_width=True)
    else:
        st.info("Need 'PO Unit Rate' and a grouping column (product_name / item_code)")

# ----------------- Forecast -----------------
with T[6]:
    st.subheader('Forecast Next Month Spend (SMA)')
    dcol = po_create_col if po_create_col in fil.columns else (pr_col if pr_col in fil.columns else None)
    if dcol and net_amount_col in fil.columns:
        t = fil.dropna(subset=[dcol]).copy(); t['month'] = t[dcol].dt.to_period('M').dt.to_timestamp()
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
    if po_vendor_col in fil.columns:
        vendor = st.selectbox('Pick Vendor', sorted(fil[po_vendor_col].dropna().astype(str).unique().tolist()))
        vd = fil[fil[po_vendor_col].astype(str) == str(vendor)].copy()
        spend = vd.get(net_amount_col, pd.Series(0)).sum()/1e7 if net_amount_col else 0
        upos = int(vd.get(purchase_doc_col, pd.Series(dtype=object)).nunique()) if purchase_doc_col else 0
        if po_delivery_col in vd.columns and pending_qty_col in vd.columns:
            late = ((pd.to_datetime(vd[po_delivery_col], errors='coerce').dt.date < pd.Timestamp.now().date()) & (vd[pending_qty_col].fillna(0) > 0)).sum()
        else:
            late = np.nan
        if pending_qty_col in vd.columns and po_unit_rate_col in vd.columns:
            vd['pending_value'] = vd[pending_qty_col].fillna(0).astype(float) * vd[po_unit_rate_col].fillna(0).astype(float)
            pend_val = vd['pending_value'].sum()/1e7
        else:
            pend_val = np.nan
        k1,k2,k3,k4 = st.columns(4)
        k1.metric('Spend (Cr)', f"{spend:.2f}"); k2.metric('Unique POs', upos); k3.metric('Late PO count', None if pd.isna(late) else int(late)); k4.metric('Pending Value (Cr)', None if pd.isna(pend_val) else f"{pend_val:.2f}")
        if 'product_name' in vd.columns and po_unit_rate_col in vd.columns:
            med = vd.groupby('product_name')[po_unit_rate_col].median().rename('median_rate'); v2 = vd.join(med, on='product_name'); v2['var%'] = ((v2[po_unit_rate_col]-v2['median_rate'])/v2['median_rate'].replace(0,np.nan))*100
            st.plotly_chart(px.box(v2, x='product_name', y=po_unit_rate_col, points='outliers', title='Price variance by item'), use_container_width=True)
        if dcol := (po_create_col if po_create_col in vd.columns else (pr_col if pr_col in vd.columns else None)):
            vsp = vd.dropna(subset=[dcol]).groupby(pd.to_datetime(vd[dcol]).dt.to_period('M'))[net_amount_col].sum().to_timestamp()/1e7 if net_amount_col else pd.Series()
            if not vsp.empty:
                st.plotly_chart(px.line(vsp, labels={'value':'Spend (Cr)','index':'Month'}, title='Monthly Spend'), use_container_width=True)

# ----------------- Search (Keyword) -----------------
with T[8]:
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
        if cat_sel and 'procurement_category' in df.columns:
            res = res[res['procurement_category'].astype(str).isin(cat_sel)]
        if vend_sel and po_vendor_col in df.columns:
            res = res[res[po_vendor_col].astype(str).isin(vend_sel)]
        st.write(f'Found {len(res)} rows')
        st.dataframe(res, use_container_width=True)
        st.download_button('⬇️ Download Search Results', res.to_csv(index=False), file_name='search_results.csv', mime='text/csv', key='dl_search')
    elif not valid_cols:
        st.info('No searchable columns present.')
    else:
        st.caption('Start typing to search…')

# ----------------- Bottom uploader (persisted) -----------------
st.markdown('---')
st.markdown('### Upload & Debug (bottom of page)')
new_files = st.file_uploader('Upload Excel/CSV files here (bottom uploader) — select multiple if needed', type=['xlsx','xls','csv'], accept_multiple_files=True, key='_bottom_uploader')
if new_files:
    st.session_state['_bottom_uploaded'] = new_files
    st.experimental_rerun()
