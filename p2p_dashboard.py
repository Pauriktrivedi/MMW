import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="P2P Dashboard — Full (Refactor)", layout="wide", initial_sidebar_state="expanded")

# ---------- Helpers ----------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    new = {}
    for c in df.columns:
        s = str(c).strip()
        s = s.replace('\xa0',' ').replace('\\','_').replace('/','_')
        s = '_'.join(s.split())
        s = s.lower()
        s = ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in s)
        s = '_'.join([p for p in s.split('_') if p != ''])
        new[c] = s
    return df.rename(columns=new)


def safe_get(df, col, default=None):
    return df[col] if col in df.columns else pd.Series(default, index=df.index)


@st.cache_data(show_spinner=False)
def load_all_from_files(uploaded_files=None, try_skiprows=1):
    frames = []
    if uploaded_files:
        for f in uploaded_files:
            for skip in (try_skiprows, 0):
                try:
                    if str(f.name).lower().endswith('.csv'):
                        df_temp = pd.read_csv(f)
                    else:
                        df_temp = pd.read_excel(f, skiprows=skip)
                    df_temp['entity'] = f.name.rsplit('.',1)[0]
                    frames.append(df_temp)
                    break
                except Exception:
                    continue
    else:
        defaults = [("MEPL.xlsx","MEPL"),("MLPL.xlsx","MLPL"),("mmw.xlsx","MMW"),("mmpl.xlsx","MMPL")]
        for fn, ent in defaults:
            for skip in (try_skiprows, 0):
                try:
                    if fn.lower().endswith('.csv'):
                        df_temp = pd.read_csv(fn)
                    else:
                        df_temp = pd.read_excel(fn, skiprows=skip)
                    df_temp['entity'] = ent
                    frames.append(df_temp)
                    break
                except Exception:
                    continue
    if not frames:
        return pd.DataFrame()
    x = pd.concat(frames, ignore_index=True, sort=False)
    x = normalize_columns(x)
    # parse common date columns
    for c in ['pr_date_submitted','po_create_date','po_approved_date','po_delivery_date','last_po_date']:
        if c in x.columns:
            x[c] = pd.to_datetime(x[c], errors='coerce')
    return x


# ---------- UI Header ----------
st.markdown("""
<div style="padding:6px 0 12px 0;">
  <h1 style="font-size:34px; margin:0;">P2P Dashboard — Indirect</h1>
  <div style="font-size:14px; color:#333; margin-top:6px;">Purchase-to-Pay overview (Indirect spend focus)</div>
  <hr style="border:0; height:1px; background:#eee; margin-top:8px;" />
</div>
""", unsafe_allow_html=True)

# ---------- Sidebar Filters (top of file) ----------
st.sidebar.header('Filters')

FY = {
    'All Years': (pd.Timestamp('2023-04-01'), pd.Timestamp('2026-03-31')),
    '2023': (pd.Timestamp('2023-04-01'), pd.Timestamp('2024-03-31')),
    '2024': (pd.Timestamp('2024-04-01'), pd.Timestamp('2025-03-31')),
    '2025': (pd.Timestamp('2025-04-01'), pd.Timestamp('2026-03-31')),
}
fy_key = st.sidebar.selectbox('Financial Year', list(FY.keys()), index=0)
pr_start, pr_end = FY[fy_key]

# Load files (but keep uploader at bottom — show a small notice here)
st.sidebar.markdown('**Upload files** — available at bottom of page (use to replace data).')

# placeholder for additional toggles
st.sidebar.checkbox('Verbose debug', value=False, key='debug')

# We'll load data after creating uploader (but keep logic consistent):
uploaded_top = None  # nothing at top

# ---------- Load Data ----------
uploaded = None
# If user uploaded via bottom uploader in a prior run, preserve
if '_uploaded_files' in st.session_state:
    uploaded = st.session_state['_uploaded_files']
# show nothing here — actual upload control rendered at bottom

# load dataframe
try:
    df = load_all_from_files(uploaded)
except Exception as e:
    st.error(f"Data load failed: {e}")
    st.stop()

if df.empty:
    st.warning('No data loaded. Place default files next to app or upload files using the bottom uploader.')
    # still render uploader at bottom; stop here to avoid failing visualisations
    st.stop()

# ---------- Normalize & derive helper cols ----------
# Ensure required cols exist
df = df.copy()
for col in ['entity']:
    if col not in df.columns:
        df[col] = ''

# Common convenience names
pr_col = 'pr_date_submitted' if 'pr_date_submitted' in df.columns else None
po_create_col = 'po_create_date' if 'po_create_date' in df.columns else None
net_amount_col = 'net_amount' if 'net_amount' in df.columns else None
purchase_doc_col = 'purchase_doc' if 'purchase_doc' in df.columns else None
pr_number_col = 'pr_number' if 'pr_number' in df.columns else None
po_vendor_col = 'po_vendor' if 'po_vendor' in df.columns else None
po_dept_col = 'po_department' if 'po_department' in df.columns else ('po_dept' if 'po_dept' in df.columns else None)
product_col = 'product_name' if 'product_name' in df.columns else None

# Buyer type detection (robust)
buyer_group_col = 'buyer_group' if 'buyer_group' in df.columns else None
if buyer_group_col:
    try:
        df['buyer_group_code'] = df[buyer_group_col].astype(str).str.extract(r"(\d+)")[0].astype(float)
    except Exception:
        df['buyer_group_code'] = np.nan

def map_buyer_type(row):
    bg = str(row.get(buyer_group_col, '')).strip()
    code = row.get('buyer_group_code', np.nan)
    if bg in ['ME_BG17','MLBG16']:
        return 'Direct'
    if bg in ['Not Available'] or bg == '' or pd.isna(bg):
        return 'Indirect'
    try:
        if not pd.isna(code) and 1 <= int(code) <= 9:
            return 'Direct'
        if not pd.isna(code) and 10 <= int(code) <= 18:
            return 'Indirect'
    except Exception:
        pass
    return 'Other'

if buyer_group_col in df.columns:
    df['buyer_type'] = df.apply(map_buyer_type, axis=1)
else:
    df['buyer_type'] = 'Unknown'

# PO Orderer -> creator mapping
map_orderer = {
    'mmw2324030': 'Dhruv', 'mmw2324062': 'Deepak', 'mmw2425154': 'Mukul', 'mmw2223104': 'Paurik',
    'mmw2021181': 'Nayan', 'mmw2223014': 'Aatish', 'mmw_ext_002': 'Deepakex', 'mmw2425024': 'Kamlesh',
    'mmw2021184': 'Suresh', 'n/a': 'Dilip'
}
po_orderer_col = 'po_orderer' if 'po_orderer' in df.columns else None
if po_orderer_col:
    df['po_orderer'] = safe_get(df, po_orderer_col, pd.NA).fillna('N/A').astype(str).str.strip()
    df['po_orderer_lc'] = df['po_orderer'].str.lower()
    df['po_creator'] = df['po_orderer_lc'].map(map_orderer).fillna(df['po_orderer']).replace({'N/A':'Dilip'})
    indirect = set(['Aatish','Deepak','Deepakex','Dhruv','Dilip','Mukul','Nayan','Paurik','Kamlesh','Suresh'])
    df['po_buyer_type'] = np.where(df['po_creator'].isin(indirect), 'Indirect', 'Direct')
else:
    df['po_creator'] = ''
    df['po_buyer_type'] = ''

# ensure date columns parsed
for c in [pr_col, po_create_col]:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors='coerce')

# single today
TODAY = pd.Timestamp.now().normalize()

# ---------- Sidebar: Year / Month / Date range ----------
# Build months list based on underlying data and FY selection
base_date_col = pr_col if pr_col else po_create_col
if base_date_col and base_date_col in df.columns:
    # filter to FY range for month list
    months_df = df[(df[base_date_col] >= pr_start) & (df[base_date_col] <= pr_end)].dropna(subset=[base_date_col]).copy()
    months_periods = months_df[base_date_col].dt.to_period('M').sort_values().unique() if not months_df.empty else []
    month_labels = [p.strftime('%b-%Y') for p in months_periods]
else:
    months_periods = []
    month_labels = []

sel_month = st.sidebar.selectbox('Month (sub-filter of FY)', ['All Months'] + month_labels if month_labels else ['All Months'], index=0)
# apply FY and month filters to working 'fil'
fil = df.copy()
# apply FY
if base_date_col and base_date_col in fil.columns:
    fil = fil[(fil[base_date_col] >= pr_start) & (fil[base_date_col] <= pr_end)]
# apply month
if sel_month != 'All Months' and month_labels:
    target = pd.Period(months_periods[month_labels.index(sel_month)]) if month_labels else None
    if target is not None:
        fil = fil[fil[base_date_col].dt.to_period('M') == target]

# date range (optional override within current filtered set)
min_dt = fil[base_date_col].min() if base_date_col in fil.columns else None
max_dt = fil[base_date_col].max() if base_date_col in fil.columns else None
if pd.notna(min_dt) and pd.notna(max_dt):
    dr = st.sidebar.date_input('Date range (optional)', (min_dt.date(), max_dt.date()))
    if isinstance(dr, tuple) and len(dr) == 2:
        s,e = pd.to_datetime(dr[0]), pd.to_datetime(dr[1])
        fil = fil[(fil[base_date_col] >= s) & (fil[base_date_col] <= e)]

# ---------- Sidebar: PO Dept, Vendor, Item (single-select dropdowns w/ All) ----------
# PO Department
if po_dept_col and po_dept_col in fil.columns:
    dept_choices = ['All Departments'] + sorted(fil[po_dept_col].dropna().astype(str).unique().tolist())
    sel_dept = st.sidebar.selectbox('PO Dept', dept_choices, index=0)
    if sel_dept != 'All Departments':
        fil = fil[fil[po_dept_col].astype(str) == sel_dept]
else:
    sel_dept = None

# Vendor (single-select)
if po_vendor_col and po_vendor_col in fil.columns:
    vendor_choices = ['All Vendors'] + sorted(fil[po_vendor_col].dropna().astype(str).unique().tolist())
    sel_vendor = st.sidebar.selectbox('Vendor', vendor_choices, index=0)
    if sel_vendor != 'All Vendors':
        fil = fil[fil[po_vendor_col].astype(str) == sel_vendor]
else:
    sel_vendor = None

# Item / Product (single-select)
if product_col and product_col in fil.columns:
    item_choices = ['All Items'] + sorted(fil[product_col].dropna().astype(str).unique().tolist())
    sel_item = st.sidebar.selectbox('Item / Product', item_choices, index=0)
    if sel_item != 'All Items':
        fil = fil[fil[product_col].astype(str) == sel_item]
else:
    sel_item = None

# ---------- Sidebar: Buyer Type & Entity (multiselects) ----------
# Ensure columns exist
for c in ['buyer_type','entity','po_creator','po_buyer_type']:
    if c not in fil.columns:
        fil[c] = ''

sel_buyer = st.sidebar.multiselect('Buyer Type', sorted(fil['buyer_type'].dropna().unique().astype(str).tolist()), default=sorted(fil['buyer_type'].dropna().unique().astype(str).tolist()))
sel_entity = st.sidebar.multiselect('Entity', sorted(fil['entity'].dropna().unique().astype(str).tolist()), default=sorted(fil['entity'].dropna().unique().astype(str).tolist()))
sel_po_creator = st.sidebar.multiselect('PO Ordered By', sorted(fil['po_creator'].dropna().unique().astype(str).tolist()), default=sorted(fil['po_creator'].dropna().unique().astype(str).tolist()))
sel_po_buyer = st.sidebar.multiselect('PO Buyer Type', sorted(fil['po_buyer_type'].dropna().unique().astype(str).tolist()), default=sorted(fil['po_buyer_type'].dropna().unique().astype(str).tolist()))

# Apply the multiselect filters
if sel_buyer:
    fil = fil[fil['buyer_type'].astype(str).isin(sel_buyer)]
if sel_entity:
    fil = fil[fil['entity'].astype(str).isin(sel_entity)]
if sel_po_creator:
    fil = fil[fil['po_creator'].astype(str).isin(sel_po_creator)]
if sel_po_buyer:
    fil = fil[fil['po_buyer_type'].astype(str).isin(sel_po_buyer)]

# Reset filters
if st.sidebar.button('Reset Filters'):
    # clear upload state and filter keys then rerun
    for k in ['_uploaded_files']:
        if k in st.session_state:
            del st.session_state[k]
    st.experimental_rerun()

# ---------- Tabs ----------
T = st.tabs(['KPIs & Spend','PO/PR Timing','Delivery','Vendors','Dept & Services','Unit-rate Outliers','Forecast','Scorecards','Search','Full Data'])

# ---------- KPIs & Spend (combined) ----------
with T[0]:
    c1,c2,c3,c4,c5 = st.columns(5)
    total_prs = int(fil[pr_number_col].nunique()) if pr_number_col in fil.columns else 0
    total_pos = int(fil[purchase_doc_col].nunique()) if purchase_doc_col in fil.columns else 0
    c1.metric('Total PRs', total_prs)
    c2.metric('Total POs', total_pos)
    c3.metric('Line Items', len(fil))
    c4.metric('Entities', int(fil['entity'].nunique()))
    spend_val = fil[net_amount_col].sum() if net_amount_col in fil.columns else 0
    c5.metric('Spend (Cr \u20B9)', f"{spend_val/1e7:,.2f}")

    st.markdown('---')
    st.subheader('Monthly Total Spend + Cumulative')
    dcol = po_create_col if po_create_col in fil.columns else (pr_col if pr_col in fil.columns else None)
    if dcol and net_amount_col in fil.columns:
        t = fil.dropna(subset=[dcol]).copy()
        t['po_month'] = t[dcol].dt.to_period('M').dt.to_timestamp()
        t['month_str'] = t['po_month'].dt.strftime('%b-%Y')
        m = t.groupby(['po_month','month_str'], as_index=False)[net_amount_col].sum().sort_values('po_month')
        m['cr'] = m[net_amount_col]/1e7
        m['cumcr'] = m['cr'].cumsum()
        fig_spend = make_subplots(specs=[[{"secondary_y":True}]])
        fig_spend.add_bar(x=m['month_str'], y=m['cr'], name='Monthly Spend (Cr \u20B9)')
        fig_spend.add_scatter(x=m['month_str'], y=m['cumcr'], name='Cumulative (Cr \u20B9)', mode='lines+markers', secondary_y=True)
        fig_spend.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_spend, use_container_width=True)
    else:
        st.info('Monthly Spend chart not available — need a date and Net Amount column.')

    st.markdown('---')
    st.subheader('Entity Trend')
    if dcol and net_amount_col in fil.columns:
        x = fil.copy(); x['po_month'] = x[dcol].dt.to_period('M').dt.to_timestamp()
        g = x.dropna(subset=['po_month']).groupby(['po_month','entity'], as_index=False)[net_amount_col].sum()
        g['cr'] = g[net_amount_col]/1e7
        if not g.empty:
            fig_entity = px.line(g, x=g['po_month'].dt.strftime('%b-%Y'), y='cr', color='entity', markers=True, labels={'x':'Month','cr':'Cr \u20B9'})
            fig_entity.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_entity, use_container_width=True)
        else:
            st.info('No entity trend data available for the selected filters.')

# ---------- PO/PR Timing ----------
with T[1]:
    st.subheader('SLA (PR→PO ≤7d)')
    if pr_col and po_create_col and pr_col in fil.columns and po_create_col in fil.columns:
        ld = fil.dropna(subset=[po_create_col, pr_col]).copy()
        ld['lead_time_days'] = (ld[po_create_col] - ld[pr_col]).dt.days
        avg = float(ld['lead_time_days'].mean()) if not ld.empty else 0.0
        max_range = max(14, avg*1.2 if avg else 14)
        fig = go.Figure(go.Indicator(mode='gauge+number', value=avg, number={'suffix':' d'}, gauge={'axis':{'range':[0,max_range]}, 'bar':{'color':'darkblue'}, 'steps':[{'range':[0,7],'color':'lightgreen'},{'range':[7,max_range],'color':'lightcoral'}], 'threshold':{'line':{'color':'red','width':4}, 'value':7}}))
        st.plotly_chart(fig, use_container_width=True)
        bins=[0,7,15,30,60,90,999]
        labels=['0-7','8-15','16-30','31-60','61-90','90+']
        ag = pd.cut(ld['lead_time_days'], bins=bins, labels=labels).value_counts(normalize=True).reindex(labels, fill_value=0).reset_index()
        ag.columns=['Bucket','Pct']; ag['Pct']=ag['Pct']*100
        st.plotly_chart(px.bar(ag, x='Bucket', y='Pct', text='Pct').update_traces(texttemplate='%{text:.1f}%', textposition='outside'), use_container_width=True)
    st.subheader('PR & PO per Month')
    tmp = fil.copy()
    tmp['pr_month'] = tmp[pr_col].dt.to_period('M') if pr_col in tmp.columns else pd.NaT
    tmp['po_month'] = tmp[po_create_col].dt.to_period('M') if po_create_col in tmp.columns else pd.NaT
    if pr_number_col and purchase_doc_col and pr_number_col in tmp.columns and purchase_doc_col in tmp.columns:
        ms = tmp.groupby('pr_month').agg({pr_number_col:'count', purchase_doc_col:'count'}).reset_index()
        if not ms.empty:
            ms.columns=['Month','PR Count','PO Count']; ms['Month']=ms['Month'].astype(str)
            st.line_chart(ms.set_index('Month'), use_container_width=True)
    st.subheader('Weekday Split')
    wd = fil.copy(); wd['pr_wk'] = wd[pr_col].dt.day_name() if pr_col in wd.columns else ''
    wd['po_wk'] = wd[po_create_col].dt.day_name() if po_create_col in wd.columns else ''
    order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    prc = wd['pr_wk'].value_counts().reindex(order, fill_value=0)
    poc = wd['po_wk'].value_counts().reindex(order, fill_value=0)
    c1,c2 = st.columns(2); c1.bar_chart(prc); c2.bar_chart(poc)

# ---------- Delivery ----------
with T[2]:
    st.subheader('Delivery Summary')
    dv = fil.rename(columns={'po_quantity':'po_qty','receivedqty':'received_qty','pending_qty':'pending_qty'}).copy()
    if {'po_qty','received_qty'}.issubset(dv.columns):
        dv['%_received'] = np.where(dv['po_qty'].astype(float)>0, (dv['received_qty'].astype(float)/dv['po_qty'].astype(float))*100, 0.0)
        group_cols = [purchase_doc_col, 'po_vendor', product_col, 'item_description']
        agcols = [c for c in group_cols if c in dv.columns]
        summ = dv.groupby(agcols, dropna=False).agg({'po_qty':'sum','received_qty':'sum','pending_qty':'sum','%_received':'mean'}).reset_index()
        if not summ.empty:
            st.dataframe(summ.sort_values('pending_qty', ascending=False), use_container_width=True)

# ---------- Vendors ----------
with T[3]:
    st.subheader('Top Vendors by Spend')
    if po_vendor_col in fil.columns and purchase_doc_col in fil.columns and net_amount_col in fil.columns:
        vs = fil.groupby(po_vendor_col, dropna=False).agg(Vendor_PO_Count=(purchase_doc_col,'nunique'), Total_Spend_Cr=(net_amount_col, lambda s: (s.sum()/1e7).round(2))).reset_index().sort_values('Total_Spend_Cr', ascending=False)
        st.dataframe(vs.head(10), use_container_width=True)

# ---------- Dept & Services (simplified) ----------
with T[4]:
    st.subheader('Dept & Services (basic view)')
    if 'po_budget_code' in fil.columns and net_amount_col in fil.columns:
        dep = fil.groupby('po_budget_code')[net_amount_col].sum().reset_index().sort_values(net_amount_col, ascending=False)
        dep['cr'] = dep[net_amount_col]/1e7
        st.plotly_chart(px.bar(dep.head(30), x='po_budget_code', y='cr', title='Budget Code spend (Top 30)').update_layout(xaxis_tickangle=-45), use_container_width=True)

# ---------- Unit-rate Outliers ----------
with T[5]:
    st.subheader('Unit-rate Outliers vs Historical Median')
    grp_candidates = [c for c in [product_col, 'item_code'] if c in fil.columns]
    grp_by = st.selectbox('Group by', grp_candidates) if grp_candidates else None
    if grp_by and 'po_unit_rate' in fil.columns:
        z = fil[[grp_by,'po_unit_rate', purchase_doc_col, pr_number_col, 'po_vendor', 'item_description', po_create_col, net_amount_col]].dropna(subset=[grp_by,'po_unit_rate']).copy()
        med = z.groupby(grp_by)['po_unit_rate'].median().rename('median_rate')
        z = z.join(med, on=grp_by)
        z['pctdev'] = (z['po_unit_rate'] - z['median_rate'])/z['median_rate'].replace(0,np.nan)
        thr = st.slider('Outlier threshold (±%)', 10, 300, 50, 5)
        out = z[abs(z['pctdev'])>=thr/100.0].copy(); out['pctdev%'] = (out['pctdev']*100).round(1)
        st.dataframe(out.sort_values('pctdev%', ascending=False), use_container_width=True)

# ---------- Forecast ----------
with T[6]:
    st.subheader('Forecast Next Month Spend (SMA)')
    dcol = po_create_col if po_create_col in fil.columns else pr_col
    if dcol and net_amount_col in fil.columns:
        t = fil.dropna(subset=[dcol]).copy(); t['month'] = t[dcol].dt.to_period('M').dt.to_timestamp()
        m = t.groupby('month')[net_amount_col].sum().sort_index(); m_cr = m/1e7
        k = st.slider('Window (months)', 3, 12, 6, 1)
        sma = m_cr.rolling(k).mean()
        mu = m_cr.tail(k).mean() if len(m_cr)>=k else m_cr.mean()
        sd = m_cr.tail(k).std(ddof=1) if len(m_cr)>=k else m_cr.std(ddof=1)
        n = min(k, max(1,len(m_cr)))
        se = sd/np.sqrt(n) if sd==sd else 0
        lo, hi = float(mu-1.96*se), float(mu+1.96*se)
        nxt = (m_cr.index.max() + pd.offsets.MonthBegin(1)) if len(m_cr)>0 else pd.Timestamp.now().to_period('M').to_timestamp()
        fdf = pd.DataFrame({'Month': list(m_cr.index)+[nxt], 'SpendCr': list(m_cr.values)+[np.nan], 'SMA': list(sma.values)+[mu]})
        fig = go.Figure(); fig.add_bar(x=fdf['Month'], y=fdf['SpendCr'], name='Actual (Cr)'); fig.add_scatter(x=fdf['Month'], y=fdf['SMA'], mode='lines+markers', name=f'SMA{k}'); fig.add_scatter(x=[nxt,nxt], y=[lo,hi], mode='lines', name='95% CI')
        st.plotly_chart(fig, use_container_width=True)

# ---------- Scorecards / Vendor ----------
with T[7]:
    st.subheader('Vendor Scorecard')
    if po_vendor_col in fil.columns:
        vendor = st.selectbox('Pick Vendor', sorted(fil[po_vendor_col].dropna().astype(str).unique().tolist()))
        vd = fil[fil[po_vendor_col].astype(str)==str(vendor)].copy()
        spend = vd.get(net_amount_col, pd.Series(0)).sum()/1e7
        upos = int(vd.get(purchase_doc_col, pd.Series(dtype=object)).nunique())
        k1,k2,k3,k4 = st.columns(4)
        k1.metric('Spend (Cr)', f"{spend:.2f}"); k2.metric('Unique POs', upos)

# ---------- Search ----------
with T[8]:
    st.subheader('🔍 Keyword Search')
    valid_cols = [c for c in [pr_number_col, purchase_doc_col, product_col, po_vendor_col] if c in df.columns]
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
        st.download_button('⬇️ Download Search Results', res.to_csv(index=False), file_name='search_results.csv', mime='text/csv')
    else:
        st.caption('Start typing to search…')

# ---------- Full Data tab (filtered) ----------
with T[9]:
    st.subheader('Full Filtered Dataset')
    st.write(f'Rows: {len(fil)}')
    st.dataframe(fil.reset_index(drop=True), use_container_width=True)
    st.download_button('Download filtered data (CSV)', fil.to_csv(index=False), file_name='filtered_data.csv', mime='text/csv')

# ---------- Bottom uploader ----------
st.markdown('---')
st.markdown('### Upload Excel/CSV files (optional) — place files here to reload data')
new_files = st.file_uploader('Drag and drop files here', type=['xlsx','xls','csv'], accept_multiple_files=True, key='_bottom_uploader')
if new_files:
    st.session_state['_uploaded_files'] = new_files
    st.experimental_rerun()

# EOF
