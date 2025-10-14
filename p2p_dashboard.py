# Complete app.py — P2P Dashboard (with Month + Date range + bottom uploader + PO Dept)
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="P2P Dashboard — Full (Refactor)", layout="wide", initial_sidebar_state="expanded")

# ---- Page header: guaranteed visible ----
import streamlit as _st
try:
    _st.session_state
except Exception:
    pass

_st.markdown(
    """
    <div style="background-color:transparent; padding:6px 0 12px 0; margin-bottom:6px;">
      <h1 style="font-size:34px; line-height:1.05; margin:0; color:#0b1f3b;">P2P Dashboard — Indirect</h1>
      <div style="font-size:14px; color:#23395b; margin-top:4px; margin-bottom:8px;">
         Purchase-to-Pay overview (Indirect spend focus)
      </div>
      <hr style="border:0; height:1px; background:#e6eef6; margin-top:8px; margin-bottom:12px;" />
    </div>
    """,
    unsafe_allow_html=True,
)

_st.write("## P2P Dashboard — Indirect")

# hide stray JSON debug widgets that sometimes show up in Streamlit
_st.markdown('''
<style>
[data-testid="stJson"], .stJson, pre.stCodeBlock, pre { display: none !important; }
</style>
''', unsafe_allow_html=True)

# ----------------- Helpers -----------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to snake_case lowercase and remove NBSPs."""
    new_cols = {}
    for c in df.columns:
        s = str(c).strip()
        s = s.replace("\xa0", " ")
        s = s.replace("\\", "_").replace("/", "_")
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
def load_all_from_files(uploaded_files=None):
    """
    Try reading uploaded files (or default filenames). Tries skiprows=1 first then fallback.
    Returns concatenated dataframe (normalized) or empty df.
    """
    frames = []
    if uploaded_files:
        for f in uploaded_files:
            try:
                if str(f.name).lower().endswith('.csv'):
                    df_temp = pd.read_csv(f)
                else:
                    df_temp = pd.read_excel(f, skiprows=1)
                df_temp['entity'] = f.name.rsplit('.',1)[0]
                frames.append(df_temp)
                continue
            except Exception:
                pass
            # fallback read
            try:
                if str(f.name).lower().endswith('.csv'):
                    df_temp = pd.read_csv(f)
                else:
                    df_temp = pd.read_excel(f, skiprows=0)
                df_temp['entity'] = f.name.rsplit('.',1)[0]
                frames.append(df_temp)
            except Exception:
                continue
    else:
        defaults = [("MEPL.xlsx","MEPL"),("MLPL.xlsx","MLPL"),("mmw.xlsx","MMW"),("mmpl.xlsx","MMPL")]
        for fn, ent in defaults:
            try:
                df_temp = pd.read_excel(fn, skiprows=1)
                df_temp['entity'] = ent
                frames.append(df_temp)
                continue
            except Exception:
                pass
            try:
                df_temp = pd.read_excel(fn, skiprows=0)
                df_temp['entity'] = ent
                frames.append(df_temp)
            except Exception:
                continue
    if not frames:
        return pd.DataFrame()
    x = pd.concat(frames, ignore_index=True, sort=False)
    x = normalize_columns(x)
    # safe date parse (some files may have different date columns)
    for c in ['pr_date_submitted','po_create_date','po_approved_date','po_delivery_date','last_po_date']:
        if c in x.columns:
            x[c] = pd.to_datetime(x[c], errors='coerce')
    return x

# Use uploaded file list from session_state (uploader is placed at bottom)
uploaded_in_session = st.session_state.get('_uploaded_files', None)

if uploaded_in_session:
    df = load_all_from_files(uploaded_in_session)
else:
    # initial load uses sidebar uploader (if used) or defaults
    # also check for any direct uploader at top (compat)
    top_uploaded = st.sidebar.file_uploader('Upload one or more Excel files (optional)', type=['xlsx','xls','csv'], accept_multiple_files=True)
    if top_uploaded:
        df = load_all_from_files(top_uploaded)
        # store top uploader result in session so bottom uploader shows same files
        st.session_state['_uploaded_files'] = top_uploaded
    else:
        df = load_all_from_files()

if df.empty:
    st.warning("No data loaded. You can upload Excel/CSV files using the uploader at the BOTTOM of this page (recommended) or place default Excel files next to this script (MEPL.xlsx, MLPL.xlsx, mmw.xlsx, mmpl.xlsx).")
    st.stop()

# ----------------- Column mapping -----------------
source_to_norm = {
    'pr number': 'pr_number','pr no':'pr_number','pr date submitted':'pr_date_submitted',
    'purchase doc':'purchase_doc','po create date':'po_create_date','po delivery date':'po_delivery_date',
    'po vendor':'po_vendor','po quantity':'po_quantity','po unit rate':'po_unit_rate','net amount':'net_amount',
    'po status':'po_status','po approved date':'po_approved_date','po orderer':'po_orderer','po budget code':'po_budget_code',
    'product name':'product_name','product name friendly':'product_name_friendly','item code':'item_code',
    'item description':'item_description','procurement category':'procurement_category',
    'pending qty':'pending_qty','received qty':'receivedqty','receivedqty':'receivedqty','pr quantity':'pr_quantity',
    'po department':'po_department','pr department':'pr_department'
}
col_map = {}
for c in df.columns:
    key = str(c).strip().lower()
    if key in source_to_norm:
        col_map[c] = source_to_norm[key]
if col_map:
    df = df.rename(columns=col_map)

# normalize and parse dates
df = normalize_columns(df)
for c in ['pr_date_submitted','po_create_date','po_approved_date','po_delivery_date','last_po_date']:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors='coerce')

# ----------------- Columns to use -----------------
pr_col = 'pr_date_submitted' if 'pr_date_submitted' in df.columns else None
po_create_col = 'po_create_date' if 'po_create_date' in df.columns else None
po_delivery_col = 'po_delivery_date' if 'po_delivery_date' in df.columns else None
net_amount_col = 'net_amount' if 'net_amount' in df.columns else None
purchase_doc_col = 'purchase_doc' if 'purchase_doc' in df.columns else None
pr_number_col = 'pr_number' if 'pr_number' in df.columns else None
po_vendor_col = 'po_vendor' if 'po_vendor' in df.columns else None
po_unit_rate_col = 'po_unit_rate' if 'po_unit_rate' in df.columns else None
pending_qty_col = 'pending_qty' if 'pending_qty' in df.columns else None
received_qty_col = 'receivedqty' if 'receivedqty' in df.columns else None
po_orderer_col = 'po_orderer' if 'po_orderer' in df.columns else None
buyer_group_col = 'buyer_group' if 'buyer_group' in df.columns else None
procurement_cat_col = 'procurement_category' if 'procurement_category' in df.columns else None
po_department_col = 'po_department' if 'po_department' in df.columns else None
entity_col = 'entity' if 'entity' in df.columns else None

TODAY = pd.Timestamp.now().normalize()

# ----------------- Buyer enrichment -----------------
if buyer_group_col in df.columns:
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

if not df.empty:
    if buyer_group_col in df.columns:
        df['buyer_type'] = df.apply(map_buyer_type, axis=1)
    else:
        df['buyer_type'] = 'Unknown'

# PO orderer mapping (same mapping as before)
map_orderer = {
    'mmw2324030': 'Dhruv', 'mmw2324062': 'Deepak', 'mmw2425154': 'Mukul', 'mmw2223104': 'Paurik',
    'mmw2021181': 'Nayan', 'mmw2223014': 'Aatish', 'mmw_ext_002': 'Deepakex', 'mmw2425024': 'Kamlesh',
    'mmw2021184': 'Suresh', 'n/a': 'Dilip'
}
if not df.empty:
    df['po_orderer'] = safe_get(df, po_orderer_col, pd.NA).fillna('N/A').astype(str).str.strip()
    df['po_orderer_lc'] = df['po_orderer'].str.lower()
    df['po_creator'] = df['po_orderer_lc'].map(map_orderer).fillna(df['po_orderer']).replace({'N/A':'Dilip'})
    indirect = set(['Aatish','Deepak','Deepakex','Dhruv','Dilip','Mukul','Nayan','Paurik','Kamlesh','Suresh'])
    df['po_buyer_type'] = np.where(df['po_creator'].isin(indirect), 'Indirect', 'Direct')
else:
    df['po_creator'] = pd.Series(dtype=object)
    df['po_buyer_type'] = pd.Series(dtype=object)

# ----------------- Sidebar Filters -----------------
st.sidebar.header('Filters')

# Financial year options (unchanged)
FY = {
    'All Years': (pd.Timestamp('2023-04-01'), pd.Timestamp('2026-03-31')),
    '2023': (pd.Timestamp('2023-04-01'), pd.Timestamp('2024-03-31')),
    '2024': (pd.Timestamp('2024-04-01'), pd.Timestamp('2025-03-31')),
    '2025': (pd.Timestamp('2025-04-01'), pd.Timestamp('2026-03-31')),
}
fy_key = st.sidebar.selectbox('Financial Year', list(FY), index=0)
pr_start_fy, pr_end_fy = FY[fy_key]

# Date range filter (overrides FY/month if set)
min_date = None
max_date = None
date_ref_col = None
# prefer PR date for date filters; fallback to PO create date
if pr_col in df.columns:
    date_ref_col = pr_col
elif po_create_col in df.columns:
    date_ref_col = po_create_col

if date_ref_col:
    min_date = df[date_ref_col].min()
    max_date = df[date_ref_col].max()

date_range = None
if min_date is not None and max_date is not None and not pd.isna(min_date) and not pd.isna(max_date):
    date_range = st.sidebar.date_input("Date range (overrides FY/month)", value=(min_date.date(), max_date.date()))
else:
    date_range = st.sidebar.date_input("Date range (overrides FY/month)", value=(pd.Timestamp.now().date(), pd.Timestamp.now().date()))

# Month dropdown (month-year) derived from selected date column
month_options = ['All Months']
if date_ref_col:
    months = df.dropna(subset=[date_ref_col]).copy()
    months['month'] = months[date_ref_col].dt.to_period('M').dt.to_timestamp()
    month_list = months['month'].dropna().dt.strftime('%b-%Y').drop_duplicates().tolist()
    month_options += month_list
selected_month = st.sidebar.selectbox('Month (applies when date range not set)', month_options, index=0)

# Buyer type, entity, PO ordered by, PO buyer type
# ensure columns exist in df (defensive)
for col in ['buyer_type','entity','po_creator','po_buyer_type','po_vendor','product_name','purchase_doc']:
    if col not in df.columns:
        df[col] = ''

fil = df.copy()

# Apply date filtering precedence: date_range (if not full-range) -> selected_month -> FY
# Helper: apply range only if user changed from defaults (we treat same-as-min/max as "not changed")
try:
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        dr_start, dr_end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        # apply only if user actually changed something (dr not equal to full min/max)
        if date_ref_col and not (dr_start == min_date and dr_end == max_date):
            fil = fil[(fil[date_ref_col] >= dr_start) & (fil[date_ref_col] <= dr_end)]
        else:
            # if user left defaults equal to file min/max, fall back to month/FY below
            pass
except Exception:
    # fall back to FY/month behavior
    pass

# If date range wasn't applied (or user left defaults), apply Month selection if not 'All Months'
if selected_month and selected_month != 'All Months' and date_ref_col:
    month_dt = pd.to_datetime(selected_month, format='%b-%Y', errors='coerce')
    if not pd.isna(month_dt):
        start_m = month_dt
        end_m = (month_dt + pd.offsets.MonthEnd(0))
        fil = fil[(fil[date_ref_col] >= start_m) & (fil[date_ref_col] <= end_m)]
else:
    # apply FY only if user didn't select a specific month and didn't apply a custom date-range
    if date_ref_col:
        fil = fil[(fil[date_ref_col] >= pr_start_fy) & (fil[date_ref_col] <= pr_end_fy)]

# unified buyer type
fil['buyer_type'] = fil['buyer_type'].astype(str).str.strip()
fil['entity'] = fil['entity'].astype(str).str.strip()
fil['po_creator'] = fil['po_creator'].astype(str).str.strip()
fil['po_buyer_type'] = fil['po_buyer_type'].astype(str).str.strip()
fil['po_vendor'] = fil['po_vendor'].astype(str).str.strip()
fil['product_name'] = fil['product_name'].astype(str).str.strip()

if 'buyer_type_unified' not in fil.columns:
    fil['buyer_type_unified'] = fil['buyer_type'].replace({'': np.nan}).fillna(fil['po_buyer_type']).fillna('Indirect')
    fil['buyer_type_unified'] = fil['buyer_type_unified'].str.title()

choices_bt = sorted(fil['buyer_type_unified'].dropna().unique().tolist())

# Sidebar selectors (multi-selects remain)
sel_b = st.sidebar.multiselect('Buyer Type', choices_bt, default=choices_bt)
sel_e = st.sidebar.multiselect('Entity', sorted(fil['entity'].dropna().unique().tolist()), default=sorted(fil['entity'].dropna().unique().tolist()))
sel_o = st.sidebar.multiselect('PO Ordered By', sorted(fil['po_creator'].dropna().unique().tolist()), default=sorted(fil['po_creator'].dropna().unique().tolist()))
sel_p = st.sidebar.multiselect('PO Buyer Type (raw)', sorted(fil['po_buyer_type'].dropna().unique().tolist()), default=sorted(fil['po_buyer_type'].dropna().unique().tolist()))

# Vendor & Item (multi-select)
vendor_choices = sorted(fil['po_vendor'].dropna().unique().tolist())
item_choices = sorted(fil['product_name'].dropna().unique().tolist())
sel_v = st.sidebar.multiselect('Vendor (pick one or more)', vendor_choices, default=vendor_choices)
sel_i = st.sidebar.multiselect('Item / Product (pick one or more)', item_choices, default=item_choices)

# PO Department selectbox (single-select with 'All Departments')
po_dept_list = sorted(fil[po_department_col].dropna().unique().tolist()) if po_department_col and po_department_col in fil.columns else []
po_dept_sel = st.sidebar.selectbox('PO Department', ['All Departments'] + po_dept_list, index=0)

# Reset filters
if st.sidebar.button('Reset Filters'):
    for k in ['filter_vendor','filter_item','filter_buyer','filter_entity','filter_po_creator','filter_po_buyer_type','fy_key','_uploaded_files']:
        if k in st.session_state:
            del st.session_state[k]
    # NOTE: we do a rerun so UI resets; uploaded files remain untouched
    st.experimental_rerun()

# Apply the rest of filters
if sel_b:
    fil = fil[fil['buyer_type_unified'].isin(sel_b)]
if sel_e:
    fil = fil[fil['entity'].isin(sel_e)]
if sel_o:
    fil = fil[fil['po_creator'].isin(sel_o)]
if sel_p:
    fil = fil[fil['po_buyer_type'].isin(sel_p)]
if sel_v:
    fil = fil[fil['po_vendor'].isin(sel_v)]
if sel_i:
    fil = fil[fil['product_name'].isin(sel_i)]
if po_dept_sel and po_dept_sel != 'All Departments' and po_department_col in fil.columns:
    fil = fil[fil[po_department_col] == po_dept_sel]

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
    c4.metric('Entities', int(fil.get(entity_col, pd.Series(dtype=object)).nunique()))
    spend_val = fil.get(net_amount_col, pd.Series(0)).sum() if net_amount_col in fil.columns else 0
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
            st.info('Entity trend not available — insufficient data.')
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
        bins=[0,7,15,30,60,90,999]
        labels=['0-7','8-15','16-30','31-60','61-90','90+']
        ag = pd.cut(ld['lead_time_days'], bins=bins, labels=labels).value_counts(normalize=True).reindex(labels, fill_value=0).reset_index()
        ag.columns=['Bucket','Pct']
        ag['Pct'] = ag['Pct']*100
        st.plotly_chart(px.bar(ag, x='Bucket', y='Pct', text='Pct').update_traces(texttemplate='%{text:.1f}%', textposition='outside'), use_container_width=True)

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
    else:
        ms = pd.DataFrame()
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
            op['pending_age_d'] = (TODAY - op[pr_col]).dt.days
            cols = [c for c in ['pr_number','pr_date_submitted','pending_age_d','procurement_category','product_name','net_amount','po_budget_code','pr_status',entity_col,'po_creator',purchase_doc_col] if c in op.columns]
            st.dataframe(op[cols], use_container_width=True)

    st.subheader('Lead Time by Buyer Type & Buyer')
    if pr_col in fil.columns and po_create_col in fil.columns:
        ld2 = fil.dropna(subset=[po_create_col, pr_col]).copy()
        ld2['lead_time_days'] = (ld2[po_create_col] - ld2[pr_col]).dt.days
        if 'buyer_type' in ld2.columns:
            st.dataframe(ld2.groupby('buyer_type')['lead_time_days'].mean().round(1).reset_index().sort_values('lead_time_days'), use_container_width=True)
        if 'po_creator' in ld2.columns:
            st.dataframe(ld2.groupby('po_creator')['lead_time_days'].mean().round(1).reset_index().sort_values('lead_time_days'), use_container_width=True)

    st.subheader('Daily PR Submissions')
    if pr_col in fil.columns:
        daily = fil.copy()
        daily['pr_date'] = pd.to_datetime(daily[pr_col], errors='coerce')
        dtrend = daily.groupby('pr_date').size().reset_index(name='PR Count')
        if not dtrend.empty:
            st.plotly_chart(px.line(dtrend, x='pr_date', y='PR Count', title='Daily PRs'), use_container_width=True)

    st.subheader('Monthly Unique PO Generation')
    if purchase_doc_col in fil.columns and po_create_col in fil.columns:
        pm = fil.dropna(subset=[po_create_col, purchase_doc_col]).copy()
        pm['po_month'] = pm[po_create_col].dt.to_period('M')
        mcount = pm.groupby('po_month')[purchase_doc_col].nunique().reset_index(name='Unique PO Count')
        mcount['po_month'] = mcount['po_month'].astype(str)
        if not mcount.empty:
            st.plotly_chart(px.bar(mcount, x='po_month', y='Unique PO Count', text='Unique PO Count', title='Unique POs per Month').update_traces(textposition='outside'), use_container_width=True)

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

# ----------------- Dept & Services (smart mapper) -----------------
with T[4]:
    st.subheader('Dept & Services (Smart Mapper)')
    st.info('Dept mapper view — uses mapping files if present; unmapped diagnostics shown when available.')
    # For brevity we reuse your existing logic for mapping if needed; show a simple summary here
    if net_amount_col in fil.columns:
        dsum = fil.groupby('po_department' if 'po_department' in fil.columns else 'pr_department', dropna=False)[net_amount_col].sum().reset_index()
        if not dsum.empty:
            dsum['cr'] = dsum[net_amount_col]/1e7
            st.dataframe(dsum.sort_values(net_amount_col, ascending=False).head(50), use_container_width=True)

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
            late = ((pd.to_datetime(vd[po_delivery_col], errors='coerce').dt.date < TODAY.date()) & (vd[pending_qty_col].fillna(0) > 0)).sum()
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

# ----------------- Bottom Uploader -----------------
st.markdown('---')
st.markdown('### Upload & Debug (bottom of page)')
new_files = st.file_uploader('Upload Excel/CSV files (optional) — drag here or Browse', type=['xlsx','xls','csv'], accept_multiple_files=True, key='_visible_bottom_uploader')
if new_files:
    # store in session_state and reload app so loader picks them up
    st.session_state['_uploaded_files'] = new_files
    st.experimental_rerun()
st.caption('Uploader placed at bottom of page. After uploading, the app will reload and show data from the uploaded files.')

# EOF
