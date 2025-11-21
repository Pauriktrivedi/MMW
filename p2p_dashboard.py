import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(page_title="P2P Dashboard — Indirect", layout="wide", initial_sidebar_state="expanded")

# ----------------- Helpers -----------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to snake_case lowercase and remove NBSPs."""
    new_cols = {}
    for c in df.columns:
        s = str(c).strip()
        s = s.replace("\xa0", " ")
        # replace backslash and forward slash safely
        s = s.replace("\\", "_").replace("/", "_")
        s = "_".join(s.split())
        s = s.lower()
        s = ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in s)
        s = '_'.join([p for p in s.split('_') if p != ''])
        new_cols[c] = s
    return df.rename(columns=new_cols)

# ----------------- Load Data -----------------
@st.cache_data(show_spinner=False)
def load_all(file_list=None):
    """Load multiple Excel files (list of tuples (filename, Entity)).
    If file_list is None, try defaults and skip missing files.
    """
    if file_list is None:
        file_list = [("MEPL.xlsx","MEPL"),("MLPL.xlsx","MLPL"),("mmw.xlsx","MMW"),("mmpl.xlsx","MMPL")]
    frames = []
    for fn, ent in file_list:
        try:
            df = pd.read_excel(fn, skiprows=1)
            df['entity'] = ent
            frames.append(df)
        except FileNotFoundError:
            # skip missing file
            continue
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    x = pd.concat(frames, ignore_index=True)
    x = normalize_columns(x)
    # coerce common date columns (if present)
    for c in ['pr_date_submitted','po_create_date','po_approved_date','po_delivery_date']:
        if c in x.columns:
            x[c] = pd.to_datetime(x[c], errors='coerce')
    return x

# Load dataframe
df = load_all()
if df.empty:
    st.warning("No data loaded. Place MEPL.xlsx, MLPL.xlsx, mmw.xlsx, mmpl.xlsx next to this script.")

# ----------------- Column aliases -----------------
# We'll use normalized snake_case names; keep original fallback names too
pr_col = 'pr_date_submitted' if 'pr_date_submitted' in df.columns else ( 'pr_date' if 'pr_date' in df.columns else None)
po_create_col = 'po_create_date' if 'po_create_date' in df.columns else None
net_amount_col = 'net_amount' if 'net_amount' in df.columns else ( 'net_amount_inr' if 'net_amount_inr' in df.columns else None)
purchase_doc_col = 'purchase_doc' if 'purchase_doc' in df.columns else ( 'purchase_doc_number' if 'purchase_doc_number' in df.columns else None)
pr_number_col = 'pr_number' if 'pr_number' in df.columns else ( 'pr_no' if 'pr_no' in df.columns else None)
po_vendor_col = 'po_vendor' if 'po_vendor' in df.columns else ( 'vendor' if 'vendor' in df.columns else None)
po_unit_rate_col = 'po_unit_rate' if 'po_unit_rate' in df.columns else ( 'po_unit_price' if 'po_unit_price' in df.columns else None)
pending_qty_col = 'pending_qty' if 'pending_qty' in df.columns else ( 'pending_qty' if 'pending_qty' in df.columns else None)
received_qty_col = 'receivedqty' if 'receivedqty' in df.columns else None
po_orderer_col = 'po_orderer' if 'po_orderer' in df.columns else ( 'po_ordered_by' if 'po_ordered_by' in df.columns else None)
buyer_group_col = 'buyer_group' if 'buyer_group' in df.columns else None
procurement_cat_col = 'procurement_category' if 'procurement_category' in df.columns else None
entity_col = 'entity'

TODAY = pd.Timestamp.now().normalize()

# ----------------- Buyer mapping (lightweight) -----------------
if buyer_group_col in df.columns:
    try:
        df['buyer_group_code'] = df[buyer_group_col].astype(str).str.extract(r"(\d+)")[0].astype(float)
    except Exception:
        df['buyer_group_code'] = np.nan

def map_buyer_type(row):
    bg = str(row.get(buyer_group_col, '')).strip() if buyer_group_col in df.columns else ''
    code = row.get('buyer_group_code', np.nan)
    try:
        if bg in ['ME_BG17','MLBG16']:
            return 'Direct'
        if bg in ['Not Available'] or bg == '' or pd.isna(bg):
            return 'Indirect'
        if not pd.isna(code) and 1 <= int(code) <= 9:
            return 'Direct'
        if not pd.isna(code) and 10 <= int(code) <= 18:
            return 'Indirect'
    except Exception:
        pass
    return 'Other'

if not df.empty:
    df['buyer_type'] = df.apply(map_buyer_type, axis=1) if buyer_group_col in df.columns else 'Unknown'

# PO orderer mapping (light)
map_orderer = {
    'mmw2324030': 'Dhruv', 'mmw2324062': 'Deepak', 'mmw2425154': 'Mukul', 'mmw2223104': 'Paurik',
    'mmw2021181': 'Nayan', 'mmw2223014': 'Aatish', 'mmw_ext_002': 'Deepakex', 'mmw2425024': 'Kamlesh',
    'mmw2021184': 'Suresh', 'n/a': 'Dilip'
}

if not df.empty:
    df['po_orderer'] = df.get(po_orderer_col, pd.Series(['']*len(df))).fillna('N/A').astype(str).str.strip()
    df['po_orderer_lc'] = df['po_orderer'].str.lower()
    df['po_creator'] = df['po_orderer_lc'].map(map_orderer).fillna(df['po_orderer']).replace({'N/A': 'Dilip'})
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
fy_key = st.sidebar.selectbox('Financial Year', list(FY))
pr_start, pr_end = FY[fy_key]

fil = df.copy()
if pr_col and pr_col in fil.columns:
    fil = fil[(fil[pr_col] >= pr_start) & (fil[pr_col] <= pr_end)]

# Ensure columns exist to avoid KeyError
for col in ['buyer_type', entity_col, 'po_creator', 'po_buyer_type']:
    if col not in fil.columns:
        fil[col] = ''
    fil[col] = fil[col].astype(str).str.strip()

# Buyer / Entity / PO Ordered By / PO Buyer Type filters
choices_bt = sorted(fil['buyer_type'].dropna().unique().tolist()) if 'buyer_type' in fil.columns else ['Direct','Indirect']
sel_b = st.sidebar.multiselect('Buyer Type', choices_bt, default=choices_bt)
sel_e = st.sidebar.multiselect('Entity', sorted(fil[entity_col].dropna().unique().tolist()), default=sorted(fil[entity_col].dropna().unique().tolist()))
sel_o = st.sidebar.multiselect('PO Ordered By', sorted(fil['po_creator'].dropna().unique().tolist()), default=sorted(fil['po_creator'].dropna().unique().tolist()))
sel_p = st.sidebar.multiselect('PO Buyer Type (raw)', sorted(fil['po_buyer_type'].dropna().unique().tolist()), default=sorted(fil['po_buyer_type'].dropna().unique().tolist()))

if sel_b:
    fil = fil[fil['buyer_type'].isin(sel_b)]
if sel_e:
    fil = fil[fil[entity_col].isin(sel_e)]
if sel_o:
    fil = fil[fil['po_creator'].isin(sel_o)]
if sel_p:
    fil = fil[fil['po_buyer_type'].isin(sel_p)]

# ----------------- Vendor / Item filters (multi-select) -----------------
# Ensure vendor and product columns exist
if 'po_vendor' not in fil.columns:
    fil['po_vendor'] = ''
if 'product_name' not in fil.columns:
    fil['product_name'] = ''
fil['po_vendor'] = fil['po_vendor'].astype(str).str.strip()
fil['product_name'] = fil['product_name'].astype(str).str.strip()

vendor_choices = sorted(fil['po_vendor'].dropna().unique().tolist())
item_choices = sorted(fil['product_name'].dropna().unique().tolist())
sel_v = st.sidebar.multiselect('Vendor (pick one or more)', vendor_choices, default=vendor_choices, key='filter_vendor')
sel_i = st.sidebar.multiselect('Item / Product (pick one or more)', item_choices, default=item_choices, key='filter_item')

if st.sidebar.button('Reset Filters'):
    # clear a few known keys then rerun
    for k in ['filter_vendor','filter_item','buyer_type','entity','po_creator','po_buyer_type','fy_key']:
        if k in st.session_state:
            del st.session_state[k]
    st.experimental_rerun()

if sel_v:
    fil = fil[fil['po_vendor'].isin(sel_v)]
if sel_i:
    fil = fil[fil['product_name'].isin(sel_i)]

# ----------------- Tabs -----------------
T = st.tabs(['KPIs & Spend','PO/PR Timing','Delivery','Vendors','Dept & Services','Unit-rate Outliers','Forecast','Scorecards','Search'])

# ----------------- KPIs & Spend -----------------
with T[0]:
    c1,c2,c3,c4,c5 = st.columns(5)
    total_prs = int(fil.get(pr_number_col, pd.Series(dtype=object)).nunique() if pr_number_col else 0)
    total_pos = int(fil.get(purchase_doc_col, pd.Series(dtype=object)).nunique() if purchase_doc_col else 0)
    c1.metric('Total PRs', total_prs)
    c2.metric('Total POs', total_pos)
    c3.metric('Line Items', len(fil))
    c4.metric('Entities', int(fil.get(entity_col, pd.Series(dtype=object)).nunique()))
    spend_val = fil.get(net_amount_col, pd.Series(0)).sum() if net_amount_col else 0
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
        st.info('Monthly Spend chart not available — need date and Net Amount columns.')

    st.markdown('---')
    st.subheader('Entity Trend')
    if dcol and net_amount_col in fil.columns:
        x = fil.copy()
        x['po_month'] = x[dcol].dt.to_period('M').dt.to_timestamp()
        g = x.dropna(subset=['po_month']).groupby(['po_month', entity_col], as_index=False)[net_amount_col].sum()
        g['cr'] = g[net_amount_col]/1e7
        if not g.empty:
            fig_entity = px.line(g, x=g['po_month'].dt.strftime('%b-%Y'), y='cr', color=entity_col, markers=True, labels={'x':'Month','cr':'Cr ₹'})
            fig_entity.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_entity, use_container_width=True)
        else:
            st.info('Entity trend not available — need date and Net Amount columns.')

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
    ms = tmp.groupby('pr_month').agg({pr_number_col:'count', purchase_doc_col:'count'}).reset_index() if pr_number_col and purchase_doc_col else pd.DataFrame()
    if not ms.empty:
        ms.columns=['Month','PR Count','PO Count']
        ms['Month'] = ms['Month'].astype(str)
        st.line_chart(ms.set_index('Month'), use_container_width=True)

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

# ----------------- Vendors -----------------
with T[3]:
    st.subheader('Top Vendors by Spend')
    if po_vendor_col in fil.columns and purchase_doc_col in fil.columns and net_amount_col in fil.columns:
        vs = fil.groupby('po_vendor', dropna=False).agg(Vendor_PO_Count=(purchase_doc_col,'nunique'), Total_Spend_Cr=(net_amount_col, lambda s: (s.sum()/1e7).round(2))).reset_index().sort_values('Total_Spend_Cr', ascending=False)
        st.dataframe(vs.head(10), use_container_width=True)

# ----------------- Dept & Services (PR Dept logic) -----------------
with T[4]:
    st.subheader('PR Department Spend (Top 30)')

    dept_df = fil.copy()

    # Build unified PR department using available PR/PO budget & business unit fields
    # preference: PR Business Unit -> PR Budget description -> PO Business Unit -> PO Budget description
    for col in ['pr_budget_description','pr_bussiness_unit','pr_business_unit','pr_businessunit','pr_budget_desc']:
        if col not in dept_df.columns:
            dept_df[col] = ''
    for col in ['po_budget_description','po_bussiness_unit','po_business_unit','po_businessunit','po_budget_desc']:
        if col not in dept_df.columns:
            dept_df[col] = ''

    # Normalize and pick
    def pick_dept(row):
        candidates = [
            row.get('pr_bussiness_unit',''),
            row.get('pr_budget_description',''),
            row.get('po_bussiness_unit',''),
            row.get('po_budget_description','')
        ]
        for v in candidates:
            if pd.notna(v) and str(v).strip() != '':
                return str(v).strip()
        return 'Unmapped / Missing'

    dept_df['pr_department'] = dept_df.apply(pick_dept, axis=1)

    if net_amount_col in dept_df.columns:
        dep = dept_df.groupby('pr_department', dropna=False)[net_amount_col].sum().reset_index().sort_values(net_amount_col, ascending=False)
        if not dep.empty:
            dep['cr'] = dep[net_amount_col]/1e7
            st.subheader('PR Department Spend

# ---------- PO Budget Description Spend (Top 30) ----------
st.markdown('---')
st.subheader('PO Budget Description Spend (Top 30)')

po_budget_desc_col = po_budget_desc_col if 'po_budget_desc_col' in globals() else find_col(df, ['po_budget_description', 'po budget description', 'po_budget_desc', 'po budget'])

if po_budget_desc_col and net_amount_col and po_budget_desc_col in fil.columns and net_amount_col in fil.columns:
    dep_po = fil.groupby(po_budget_desc_col, dropna=False)[net_amount_col].sum().reset_index().sort_values(net_amount_col, ascending=False)
    dep_po['cr'] = dep_po[net_amount_col] / 1e7
    top_dep_po = dep_po.head(30)
    if not top_dep_po.empty:
        top_dep_po[po_budget_desc_col] = top_dep_po[po_budget_desc_col].astype(str)
        fig_po = px.bar(top_dep_po, x=po_budget_desc_col, y='cr',
                        title='PO Budget Description Spend (Top 30)',
                        labels={po_budget_desc_col: 'po_budget_description', 'cr': 'Cr'})
        fig_po.update_layout(xaxis_tickangle=-45, yaxis_title='Cr')
        st.plotly_chart(fig_po, use_container_width=True)
    else:
        st.info('No PO-budget spend rows found.')
else:
    missing = []
    if not po_budget_desc_col or po_budget_desc_col not in fil.columns:
        missing.append('PO Budget Description column')
    if not net_amount_col or net_amount_col not in fil.columns:
        missing.append('Net Amount column')
    st.info('Cannot show PO Budget Description spend — missing: ' + ', '.join(missing))
 (Top 30)')
            fig = px.bar(dep.head(30), x='pr_department', y='cr', title='PR Department Spend (Top 30)').update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(dep.head(100), use_container_width=True)
        else:
            st.info('No department spend data available (missing Net Amount).')
    else:
        st.info('Net Amount column not found; cannot compute departmental spend.')

# ----------------- Unit-rate Outliers -----------------
with T[5]:
    st.subheader('Unit-rate Outliers vs Historical Median')
    grp_candidates = [c for c in ['product_name','item_code'] if c in fil.columns]
    if grp_candidates:
        grp_by = st.selectbox('Group by', grp_candidates)
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

# ----------------- Scorecards / Vendor details -----------------
with T[7]:
    st.subheader('Vendor Scorecard')
    if po_vendor_col in fil.columns:
        vendor = st.selectbox('Pick Vendor', sorted(fil[po_vendor_col].dropna().astype(str).unique().tolist()))
        vd = fil[fil[po_vendor_col].astype(str) == str(vendor)].copy()
        spend = vd.get(net_amount_col, pd.Series(0)).sum()/1e7 if net_amount_col else 0
        upos = int(vd.get(purchase_doc_col, pd.Series(dtype=object)).nunique()) if purchase_doc_col else 0
        if po_delivery_col in vd.columns and pending_qty_col in vd.columns:
            late = ((pd.to_datetime(vd[po_delivery_col], errors='coerce').dt.date < TODAY.date()) & (vd[pending_qty_col].fillna(0) > 0)).sum()
        else:
            late = np.nan
        k1,k2,k3,k4 = st.columns(4)
        k1.metric('Spend (Cr)', f"{spend:.2f}"); k2.metric('Unique POs', upos); k3.metric('Late PO count', None if pd.isna(late) else int(late))

# ----------------- Search -----------------
with T[8]:
    st.subheader('🔍 Keyword Search')
    valid_cols = [c for c in [pr_number_col, purchase_doc_col, 'product_name', po_vendor_col] if c in df.columns]
    query = st.text_input('Type vendor, product, PO, PR, etc.', '')
    cat_sel = st.multiselect('Filter by Procurement Category', sorted(df.get(procurement_cat_col, pd.Series(dtype=object)).dropna().astype(str).unique().tolist())) if procurement_cat_col in df.columns else []
    vend_sel = st.multiselect('Filter by Vendor', sorted(df.get(po_vendor_col, pd.Series(dtype=object)).dropna().astype(str).unique().tolist())) if po_vendor_col in df.columns else []
    if query and valid_cols:
        mask = pd.Series(False, index=df.index)
        q = query.lower()
        for c in valid_cols:
            mask = mask | df[c].astype(str).str.lower().str.contains(q, na=False)
        res = df[mask].copy()
        if cat_sel and procurement_cat_col in df.columns:
            res = res[res[procurement_cat_col].astype(str).isin(cat_sel)]
        if vend_sel and po_vendor_col in df.columns:
            res = res[res[po_vendor_col].astype(str).isin(vend_sel)]
        st.write(f'Found {len(res)} rows')
        st.dataframe(res, use_container_width=True)
        st.download_button('⬇️ Download Search Results', res.to_csv(index=False), file_name='search_results.csv', mime='text/csv', key='dl_search')
    elif not valid_cols:
        st.info('No searchable columns present.')
    else:
        st.caption('Start typing to search…')

# EOF
