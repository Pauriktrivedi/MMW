import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
        s = s.replace('\\', '_').replace('/', '_')
        s = "_".join(s.split())
        s = s.lower()
        s = ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in s)
        s = '_'.join([p for p in s.split('_') if p != ''])
        new_cols[c] = s
    return df.rename(columns=new_cols)


def safe_get(df, col, default=None):
    if col in df.columns:
        return df[col]
    return pd.Series(default, index=df.index)

# ----------------- Load Data -----------------
@st.cache_data(show_spinner=False)
def load_all(file_list=None):
    # loads default files from working directory if present
    if file_list is None:
        file_list = [("MEPL.xlsx","MEPL"),("MLPL.xlsx","MLPL"),("mmw.xlsx","MMW"),("mmpl.xlsx","MMPL")]
    frames = []
    for fn, ent in file_list:
        try:
            d = pd.read_excel(fn, skiprows=1)
            d['entity'] = ent
            frames.append(d)
        except Exception:
            # skip missing files quietly
            continue
    if not frames:
        return pd.DataFrame()
    x = pd.concat(frames, ignore_index=True)
    x = normalize_columns(x)
    for c in ['pr_date_submitted','po_create_date','po_approved_date','po_delivery_date']:
        if c in x.columns:
            x[c] = pd.to_datetime(x[c], errors='coerce')
    return x

# NOTE: Drag & drop / file upload controls intentionally removed as requested

df = load_all()
if df.empty:
    st.warning("No data loaded. Place MEPL.xlsx / MLPL.xlsx / mmw.xlsx / mmpl.xlsx in the app folder.")

# ----------------- Column setup -----------------
pr_col = 'pr_date_submitted' if 'pr_date_submitted' in df.columns else None
po_create_col = 'po_create_date' if 'po_create_date' in df.columns else None
po_delivery_col = 'po_delivery_date' if 'po_delivery_date' in df.columns else None
net_amount_col = 'net_amount' if 'net_amount' in df.columns else None
purchase_doc_col = 'purchase_doc' if 'purchase_doc' in df.columns else None
pr_number_col = 'pr_number' if 'pr_number' in df.columns else None
po_vendor_col = 'po_vendor' if 'po_vendor' in df.columns else None
po_unit_rate_col = 'po_unit_rate' if 'po_unit_rate' in df.columns else None
pending_qty_col = 'pending_qty' if 'pending_qty' in df.columns else None
entity_col = 'entity'

TODAY = pd.Timestamp.now().normalize()

# ----------------- Buyer/Creator logic -----------------
# keep original mapping logic but defensive
if 'buyer_group' in df.columns:
    try:
        df['buyer_group_code'] = df['buyer_group'].astype(str).str.extract(r"(\d+)")[0].astype(float)
    except Exception:
        df['buyer_group_code'] = np.nan


def map_buyer_type(row):
    bg = str(row.get('buyer_group','')).strip()
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

if not df.empty:
    df['buyer_type'] = df.apply(map_buyer_type, axis=1) if 'buyer_group' in df.columns else 'Unknown'

# PO orderer mapping
map_orderer = {
    'mmw2324030':'Dhruv','mmw2324062':'Deepak','mmw2425154':'Mukul','mmw2223104':'Paurik',
    'mmw2021181':'Nayan','mmw2223014':'Aatish','mmw_ext_002':'Deepakex','mmw2425024':'Kamlesh',
    'mmw2021184':'Suresh','n/a':'Dilip'
}
if not df.empty:
    df['po_orderer'] = safe_get(df, 'po_orderer', pd.NA).fillna('N/A').astype(str).str.strip()
    df['po_orderer_lc'] = df['po_orderer'].str.lower()
    df['po_creator'] = df['po_orderer_lc'].map(map_orderer).fillna(df['po_orderer']).replace({'N/A':'Dilip'})
    indirect = set(['Aatish','Deepak','Deepakex','Dhruv','Dilip','Mukul','Nayan','Paurik','Kamlesh','Suresh'])
    df['po_buyer_type'] = np.where(df['po_creator'].isin(indirect), 'Indirect', 'Direct')
else:
    df['po_creator'] = pd.Series(dtype=object)
    df['po_buyer_type'] = pd.Series(dtype=object)

# ----------------- Sidebar filters -----------------
st.sidebar.header('Filters')
FY = {
    'All Years': (pd.Timestamp('2023-04-01'), pd.Timestamp('2026-03-31')),
    '2023': (pd.Timestamp('2023-04-01'), pd.Timestamp('2024-03-31')),
    '2024': (pd.Timestamp('2024-04-01'), pd.Timestamp('2025-03-31')),
    '2025': (pd.Timestamp('2025-04-01'), pd.Timestamp('2026-03-31'))
}
fy_key = st.sidebar.selectbox('Financial Year', list(FY), index=0)
pr_start, pr_end = FY[fy_key]

fil = df.copy()
if pr_col and pr_col in fil.columns:
    fil = fil[(fil[pr_col] >= pr_start) & (fil[pr_col] <= pr_end)]

# Month dropdown under FY
month_basis = 'pr_date_submitted' if pr_col else ( 'po_create_date' if po_create_col else None )
if month_basis and month_basis in fil.columns:
    months = fil[month_basis].dropna().dt.to_period('M').astype(str).unique().tolist()
    months = sorted(months, key=lambda s: pd.Period(s))
    month_labels = [pd.Period(m).strftime('%b-%Y') for m in months]
    label_to_period = {pd.Period(m).strftime('%b-%Y'): m for m in months}
    if month_labels:
        sel_month = st.sidebar.selectbox('Month', ['All Months'] + month_labels, index=0)
        if sel_month != 'All Months':
            target_period = label_to_period[sel_month]
            fil = fil[fil[month_basis].dt.to_period('M').astype(str) == target_period]

# Date range picker (optional)
if month_basis and month_basis in fil.columns and fil[month_basis].dropna().any():
    min_dt = fil[month_basis].dropna().min(); max_dt = fil[month_basis].dropna().max()
    dr = st.sidebar.date_input('Date range', (pd.Timestamp(min_dt).date(), pd.Timestamp(max_dt).date()))
    if isinstance(dr, tuple) and len(dr) == 2:
        s,e = pd.to_datetime(dr[0]), pd.to_datetime(dr[1])
        fil = fil[(fil[month_basis] >= s) & (fil[month_basis] <= e)]

# Ensure filter columns exist
for col in ['buyer_type','entity','po_creator','po_buyer_type']:
    if col not in fil.columns:
        fil[col] = ''
    fil[col] = fil[col].astype(str).str.strip()

# Unified buyer type for clean filters
if 'buyer_type_unified' not in fil.columns:
    bt = safe_get(fil, 'buyer_type', pd.Series('', index=fil.index)).astype(str).str.strip()
    pbt = safe_get(fil, 'po_buyer_type', pd.Series('', index=fil.index)).astype(str).str.strip()
    fil['buyer_type_unified'] = np.where(bt != '', bt, pbt)
    fil['buyer_type_unified'] = fil['buyer_type_unified'].str.title().replace({'Other':'Indirect','Unknown':'Indirect','':'Indirect'})
    fil['buyer_type_unified'] = np.where(fil['buyer_type_unified'].str.lower() == 'direct', 'Direct', 'Indirect')

choices_bt = sorted(fil['buyer_type_unified'].dropna().unique().tolist()) if 'buyer_type_unified' in fil.columns else ['Direct','Indirect']
sel_b = st.sidebar.multiselect('Buyer Type', choices_bt, default=choices_bt)
sel_e = st.sidebar.multiselect('Entity', sorted(fil['entity'].dropna().unique().tolist()), default=sorted(fil['entity'].dropna().unique().tolist()))
sel_o = st.sidebar.multiselect('PO Ordered By', sorted(fil['po_creator'].dropna().unique().tolist()), default=sorted(fil['po_creator'].dropna().unique().tolist()))
sel_p = st.sidebar.multiselect('PO Buyer Type (raw)', sorted(fil['po_buyer_type'].dropna().unique().tolist()), default=sorted(fil['po_buyer_type'].dropna().unique().tolist()))

if sel_b:
    fil = fil[fil['buyer_type_unified'].isin(sel_b)]
if sel_e:
    fil = fil[fil['entity'].isin(sel_e)]
if sel_o:
    fil = fil[fil['po_creator'].isin(sel_o)]
if sel_p:
    fil = fil[fil['po_buyer_type'].isin(sel_p)]

# Vendor / Item filters
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

# Reset filters
if st.sidebar.button('Reset Filters'):
    # preserve page but clear session filters
    keys = [k for k in list(st.session_state.keys()) if k.startswith('filter_') or k in ['fy_key']]
    for k in keys:
        del st.session_state[k]
    st.experimental_rerun()

# ----------------- Tabs -----------------
T = st.tabs(['KPIs & Spend','PO/PR Timing','Delivery','Vendors','Dept & Services','Unit-rate Outliers','Forecast','Scorecards','Search'])

# ----------------- KPIs & Spend -----------------
with T[0]:
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric('Total PRs', int(fil[pr_number_col].nunique()) if pr_number_col in fil.columns else 0)
    c2.metric('Total POs', int(fil[purchase_doc_col].nunique()) if purchase_doc_col in fil.columns else 0)
    c3.metric('Line Items', len(fil))
    c4.metric('Entities', int(fil['entity'].nunique()))
    spend_val = fil[net_amount_col].sum() if net_amount_col in fil.columns else 0
    c5.metric('Spend (Cr ₹)', f"{spend_val/1e7:,.2f}")

    st.markdown('---')
    # Monthly spend + cumulative
    dcol = po_create_col if po_create_col in fil.columns else (pr_col if pr_col in fil.columns else None)
    if dcol and net_amount_col in fil.columns:
        t = fil.dropna(subset=[dcol]).copy()
        t['po_month'] = t[dcol].dt.to_period('M').dt.to_timestamp()
        t['month_str'] = t['po_month'].dt.strftime('%b-%Y')
        m = t.groupby(['po_month','month_str'], as_index=False)[net_amount_col].sum().sort_values('po_month')
        m['cr'] = m[net_amount_col]/1e7; m['cumcr'] = m['cr'].cumsum()
        fig = make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_bar(x=m['month_str'], y=m['cr'], name='Monthly Spend (Cr ₹)')
        fig.add_scatter(x=m['month_str'], y=m['cumcr'], name='Cumulative (Cr ₹)', mode='lines+markers', secondary_y=True)
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    # Entity trend
    st.markdown('---')
    if dcol and net_amount_col in fil.columns:
        x = fil.copy(); x['po_month'] = x[dcol].dt.to_period('M').dt.to_timestamp()
        g = x.dropna(subset=['po_month']).groupby(['po_month','entity'], as_index=False)[net_amount_col].sum(); g['cr']=g[net_amount_col]/1e7
        if not g.empty:
            fig2 = px.line(g, x=g['po_month'].dt.strftime('%b-%Y'), y='cr', color='entity', markers=True, labels={'x':'Month','cr':'Cr ₹'})
            fig2.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig2, use_container_width=True)

# ----------------- PO/PR Timing -----------------
with T[1]:
    st.subheader('SLA (PR→PO ≤7d)')
    if pr_col in fil.columns and po_create_col in fil.columns:
        ld = fil.dropna(subset=[po_create_col, pr_col]).copy(); ld['lead_time_days'] = (ld[po_create_col]-ld[pr_col]).dt.days
        avg = float(ld['lead_time_days'].mean()) if not ld.empty else 0.0
        max_range = max(14, avg*1.2 if avg else 14)
        fig = go.Figure(go.Indicator(mode='gauge+number', value=avg, number={'suffix':' d'}, gauge={'axis':{'range':[0,max_range]}, 'bar':{'color':'darkblue'}, 'steps':[{'range':[0,7],'color':'lightgreen'},{'range':[7,max_range],'color':'lightcoral'}], 'threshold':{'line':{'color':'red','width':4}, 'value':7}}))
        st.plotly_chart(fig, use_container_width=True)

    # PR & PO per month
    tmp = fil.copy()
    tmp['pr_month'] = tmp[pr_col].dt.to_period('M') if pr_col in tmp.columns else pd.NaT
    tmp['po_month'] = tmp[po_create_col].dt.to_period('M') if po_create_col in tmp.columns else pd.NaT
    if pr_number_col in tmp.columns and purchase_doc_col in tmp.columns:
        ms = tmp.groupby('pr_month').agg({pr_number_col:'count', purchase_doc_col:'count'}).reset_index()
        if not ms.empty:
            ms.columns=['Month','PR Count','PO Count']; ms['Month']=ms['Month'].astype(str)
            st.line_chart(ms.set_index('Month'), use_container_width=True)

# ----------------- Delivery -----------------
with T[2]:
    st.subheader('Delivery Summary')
    dv = fil.rename(columns={'po_quantity':'po_qty','receivedqty':'received_qty','pending_qty':'pending_qty'}).copy()
    if 'po_qty' in dv.columns and 'received_qty' in dv.columns:
        dv['pct_received'] = np.where(dv['po_qty'].astype(float)>0, (dv['received_qty'].astype(float)/dv['po_qty'].astype(float))*100, 0.0)
        agcols = [c for c in [purchase_doc_col,'po_vendor','product_name','item_description'] if c in dv.columns]
        summ = dv.groupby(agcols, dropna=False).agg({'po_qty':'sum','received_qty':'sum','pending_qty':'sum','pct_received':'mean'}).reset_index()
        if not summ.empty:
            st.dataframe(summ.sort_values('pending_qty', ascending=False), use_container_width=True)

# ----------------- Vendors -----------------
with T[3]:
    st.subheader('Top Vendors by Spend')
    if 'po_vendor' in fil.columns and net_amount_col in fil.columns:
        vs = fil.groupby('po_vendor', dropna=False)[net_amount_col].sum().reset_index().sort_values(net_amount_col, ascending=False)
        vs['cr'] = vs[net_amount_col]/1e7
        st.plotly_chart(px.bar(vs.head(10), x='po_vendor', y='cr', text='cr').update_traces(textposition='outside'), use_container_width=True)

# ----------------- Dept & Services -----------------
with T[4]:
    st.subheader('Dept & Services (PR Department)')
    # user requested: use PR department column for department spend bar chart
    dept_col_candidates = [c for c in ['pr_department','pr_dept','pr_dpt','pr department','pr dept'] if c in fil.columns]
    if dept_col_candidates:
        dept_col = dept_col_candidates[0]
        dep = fil.groupby(dept_col)[net_amount_col].sum().reset_index().sort_values(net_amount_col, ascending=False) if net_amount_col in fil.columns else pd.DataFrame()
        if not dep.empty:
            dep['cr'] = dep[net_amount_col]/1e7
            st.plotly_chart(px.bar(dep.head(30), x=dept_col, y='cr', title='PR Department Spend (Top 30)').update_layout(xaxis_tickangle=-45), use_container_width=True)
        else:
            st.info('PR department present but no spend values found.')
    else:
        st.info('No PR Department column found. Expected one of: ' + ','.join(["pr_department","pr_dept","pr_dpt","pr department","pr dept"]))

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

# ----------------- Forecast -----------------
with T[6]:
    st.subheader('Forecast Next Month Spend (SMA)')
    dcol = po_create_col if po_create_col in fil.columns else (pr_col if pr_col in fil.columns else None)
    if dcol and net_amount_col in fil.columns:
        t = fil.dropna(subset=[dcol]).copy(); t['month'] = t[dcol].dt.to_period('M').dt.to_timestamp()
        m = t.groupby('month')[net_amount_col].sum().sort_index(); m_cr=(m/1e7)
        k = st.slider('Window (months)', 3, 12, 6, 1)
        sma = m_cr.rolling(k).mean()
        st.line_chart(pd.DataFrame({'Actual':m_cr,'SMA':sma}))

# ----------------- Scorecards & Vendor -----------------
with T[7]:
    st.subheader('Vendor Scorecard')
    if 'po_vendor' in fil.columns:
        vendor = st.selectbox('Pick Vendor', sorted(fil['po_vendor'].dropna().unique().tolist()))
        vd = fil[fil['po_vendor'].astype(str) == str(vendor)].copy()
        spend = vd.get(net_amount_col, pd.Series(0)).sum()/1e7 if net_amount_col in vd.columns else 0
        upos = int(vd.get(purchase_doc_col, pd.Series(dtype=object)).nunique()) if purchase_doc_col in vd.columns else 0
        k1,k2,k3,k4 = st.columns(4)
        k1.metric('Spend (Cr)', f"{spend:.2f}"); k2.metric('Unique POs', upos); k3.metric('Late PO count', '')

# ----------------- Search -----------------
with T[8]:
    st.subheader('🔍 Keyword Search')
    valid_cols = [c for c in [pr_number_col, purchase_doc_col, 'product_name', 'po_vendor'] if c in df.columns]
    query = st.text_input('Type vendor, product, PO, PR, etc.', '')
    if query and valid_cols:
        mask = pd.Series(False, index=df.index)
        q = query.lower()
        for c in valid_cols:
            mask = mask | df[c].astype(str).str.lower().str.contains(q, na=False)
        res = df[mask].copy()
        st.write(f'Found {len(res)} rows')
        st.dataframe(res, use_container_width=True)
        st.download_button('⬇️ Download Search Results', res.to_csv(index=False), file_name='search_results.csv', mime='text/csv')

# EOF
