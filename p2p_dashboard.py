# app.py - P2P Dashboard (uploader at bottom, year+month filter, date-range text / picker)
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime as dt

st.set_page_config(page_title="P2P Dashboard — Full (Refactor)", layout="wide", initial_sidebar_state="expanded")

# ---- Header ----
_st = st
try:
    _st.session_state
except Exception:
    pass

st.markdown(
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
st.write("## P2P Dashboard — Indirect")

# Hide stray JSON debug widgets
st.markdown('''<style>[data-testid="stJson"], .stJson, pre.stCodeBlock, pre { display: none !important; }</style>''', unsafe_allow_html=True)

# ---------------- Helpers ----------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
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

# ---------------- Load data ----------------
@st.cache_data(show_spinner=False)
def load_all_from_files(uploaded_files=None):
    """
    Try reading uploaded files (or default filenames). Try skiprows=1 then fallback to skiprows=0.
    Returns normalized DataFrame or empty DataFrame.
    """
    frames = []
    if uploaded_files:
        for f in uploaded_files:
            fname = str(f.name).lower()
            try:
                if fname.endswith('.csv'):
                    df_temp = pd.read_csv(f)
                else:
                    df_temp = pd.read_excel(f, skiprows=1)
                df_temp['entity'] = f.name.rsplit('.',1)[0]
                frames.append(df_temp)
                continue
            except Exception:
                pass
            # fallback
            try:
                if fname.endswith('.csv'):
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
    # parse dates where possible
    for c in ['pr_date_submitted','po_create_date','po_approved_date','po_delivery_date','last_po_date']:
        if c in x.columns:
            x[c] = pd.to_datetime(x[c], errors='coerce')
    return x

# We purposely do NOT put uploader in the sidebar.
# If user previously uploaded files (bottom uploader stored in session), use them
uploaded_files = st.session_state.get('_uploaded_files', None)

if uploaded_files:
    df = load_all_from_files(uploaded_files)
else:
    # initial load looks for defaults
    df = load_all_from_files()

if df.empty:
    st.warning("No data loaded. Use the uploader at the BOTTOM of the page (recommended) or place default Excel files next to the script (MEPL.xlsx, MLPL.xlsx, mmw.xlsx, mmpl.xlsx).")
    st.stop()

# ---------------- map incoming columns ----------------
source_to_norm = {
    # PR
    'pr number': 'pr_number','pr no': 'pr_number','pr date submitted':'pr_date_submitted','pr prepared by':'pr_prepared_by',
    'pr status':'pr_status','pr budget code':'pr_budget_code','pr budget description':'pr_budget_description',
    'pr business unit':'pr_business_unit','pr department':'pr_department',
    # PO
    'purchase doc':'purchase_doc','po create date':'po_create_date','po delivery date':'po_delivery_date',
    'po vendor':'po_vendor','po quantity':'po_quantity','po unit rate':'po_unit_rate','net amount':'net_amount',
    'po status':'po_status','po approved date':'po_approved_date','po orderer':'po_orderer',
    'last po number':'last_po_number','last po date':'last_po_date','last po vendor':'last_po_vendor',
    'po budget code':'po_budget_code','po budget description':'po_budget_description',
    'po business unit':'po_business_unit','po department':'po_department',
    # item
    'product name':'product_name','product name friendly':'product_name_friendly','item code':'item_code',
    'item description':'item_description','procurement category':'procurement_category','line':'line',
    # qtys
    'receivedqty':'receivedqty','received qty':'receivedqty','pending qty':'pending_qty','pending_qty':'pending_qty',
    'pr quantity':'pr_quantity','currency':'currency','unit rate':'unit_rate','pr value':'pr_value'
}

col_map = {}
for c in df.columns:
    key = str(c).strip().lower()
    if key in source_to_norm:
        col_map[c] = source_to_norm[key]
if col_map:
    df = df.rename(columns=col_map)

df = normalize_columns(df)
for c in ['pr_date_submitted','po_create_date','po_approved_date','po_delivery_date','last_po_date']:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors='coerce')

# ---------------- columns used ----------------
pr_col = 'pr_date_submitted' if 'pr_date_submitted' in df.columns else None
po_create_col = 'po_create_date' if 'po_create_date' in df.columns else None
net_amount_col = 'net_amount' if 'net_amount' in df.columns else None
purchase_doc_col = 'purchase_doc' if 'purchase_doc' in df.columns else None
pr_number_col = 'pr_number' if 'pr_number' in df.columns else None
po_vendor_col = 'po_vendor' if 'po_vendor' in df.columns else None
po_unit_rate_col = 'po_unit_rate' if 'po_unit_rate' in df.columns else None
pending_qty_col = 'pending_qty' if 'pending_qty' in df.columns else None
procurement_cat_col = 'procurement_category' if 'procurement_category' in df.columns else None
po_department_col = 'po_department' if 'po_department' in df.columns else None
entity_col = 'entity' if 'entity' in df.columns else None

TODAY = pd.Timestamp.now().normalize()

# ---------------- buyer & po_creator enrichment ----------------
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

if buyer_group_col:
    df['buyer_type'] = df.apply(map_buyer_type, axis=1)
else:
    df['buyer_type'] = 'Unknown'

map_orderer = {
    'mmw2324030': 'Dhruv', 'mmw2324062': 'Deepak', 'mmw2425154': 'Mukul', 'mmw2223104': 'Paurik',
    'mmw2021181': 'Nayan', 'mmw2223014': 'Aatish', 'mmw_ext_002': 'Deepakex', 'mmw2425024': 'Kamlesh',
    'mmw2021184': 'Suresh', 'n/a': 'Dilip'
}
df['po_orderer'] = safe_get(df, 'po_orderer', pd.NA).fillna('N/A').astype(str).str.strip()
df['po_orderer_lc'] = df['po_orderer'].str.lower()
df['po_creator'] = df['po_orderer_lc'].map(map_orderer).fillna(df['po_orderer']).replace({'N/A':'Dilip'})
indirect_set = set(['Aatish','Deepak','Deepakex','Dhruv','Dilip','Mukul','Nayan','Paurik','Kamlesh','Suresh'])
df['po_buyer_type'] = np.where(df['po_creator'].isin(indirect_set), 'Indirect', 'Direct')

# ---------------- Sidebar filters ----------------
st.sidebar.header('Filters')

# 1) Financial Year selector (keeps original FY options)
FY = {
    'All Years': (pd.Timestamp('2023-04-01'), pd.Timestamp('2026-03-31')),
    '2023': (pd.Timestamp('2023-04-01'), pd.Timestamp('2024-03-31')),
    '2024': (pd.Timestamp('2024-04-01'), pd.Timestamp('2025-03-31')),
    '2025': (pd.Timestamp('2025-04-01'), pd.Timestamp('2026-03-31')),
}
fy_key = st.sidebar.selectbox('Financial Year', list(FY), index=0)
fy_start, fy_end = FY[fy_key]

# date reference column preference (PR date preferred, else PO create date)
date_ref_col = pr_col if pr_col else (po_create_col if po_create_col else None)
min_date = df[date_ref_col].min() if date_ref_col else pd.Timestamp.now()
max_date = df[date_ref_col].max() if date_ref_col else pd.Timestamp.now()

# 2) Date range: provide picker (and also text fields for typed format DD-MMM-YYYY)
if date_ref_col:
    st.sidebar.markdown("**Date range (picker)**")
    try:
        dr_default = (min_date.date(), max_date.date())
    except Exception:
        dr_default = (pd.Timestamp.now().date(), pd.Timestamp.now().date())
    date_range_picker = st.sidebar.date_input("Select date range (picker)", value=dr_default)
    st.sidebar.markdown("**Or -- type date (DD-MMM-YYYY) to override picker**")
    start_text = st.sidebar.text_input("Start date (DD-MMM-YYYY)", value="", help="e.g. 01-Jan-2024")
    end_text = st.sidebar.text_input("End date (DD-MMM-YYYY)", value="", help="e.g. 31-Dec-2024")
else:
    date_range_picker = None
    start_text = end_text = ""

# 3) Year + Month filter: Year derived from data; user asked for simple 12 months based on year selection
# Build year list from date_ref_col (fallback to FY if date_ref_col missing)
years_available = []
if date_ref_col:
    years_available = sorted(df[date_ref_col].dropna().dt.year.unique().tolist(), reverse=True)
if not years_available:
    # fallback to FY keys if file has no dates
    years_available = [int(k) for k in ['2025','2024','2023'] if k.isdigit()]

year_sel = st.sidebar.selectbox('Year (for Month dropdown)', options=['All Years'] + [str(y) for y in years_available], index=0)

# Month dropdown: simple 12 months (All or Jan..Dec). Only applied when a specific year is selected.
month_options = ['All Months','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
month_sel = st.sidebar.selectbox('Month (applies when Year selected)', month_options, index=0)

# Other sidebar filters (Buyer Type, Entity, PO Ordered By, PO Buyer Type)
for col in ['buyer_type','entity','po_creator','po_buyer_type','po_vendor','product_name']:
    if col not in df.columns:
        df[col] = ''

fil = df.copy()

# Apply date filtering precedence:
# - if user filled text start/end (non-empty) and parsed OK -> use those
# - else if user used date picker AND changed from full file min/max -> use picker
# - else if user selected specific Year and Month -> filter by that
# - else apply FY

# try parse typed dates
def parse_typed_date(s):
    if not s or not str(s).strip():
        return None
    for fmt in ("%d-%b-%Y","%d-%m-%Y","%Y-%m-%d","%d/%m/%Y"):
        try:
            return pd.to_datetime(s.strip(), format=fmt)
        except Exception:
            pass
    try:
        return pd.to_datetime(s.strip(), dayfirst=True, errors='coerce')
    except Exception:
        return None

typed_start = parse_typed_date(start_text)
typed_end = parse_typed_date(end_text)

applied_date_filter = False
if date_ref_col:
    # 1) typed override
    if typed_start is not None and typed_end is not None and not pd.isna(typed_start) and not pd.isna(typed_end):
        fil = fil[(fil[date_ref_col] >= typed_start) & (fil[date_ref_col] <= typed_end)]
        applied_date_filter = True
    else:
        # 2) date picker -- treat as applied only if user changed from full extent
        try:
            if isinstance(date_range_picker, (list,tuple)) and len(date_range_picker) == 2:
                dp_start = pd.to_datetime(date_range_picker[0])
                dp_end = pd.to_datetime(date_range_picker[1])
                # if user changed from file min/max, apply
                if not (dp_start.date() == min_date.date() and dp_end.date() == max_date.date()):
                    fil = fil[(fil[date_ref_col] >= dp_start) & (fil[date_ref_col] <= dp_end)]
                    applied_date_filter = True
        except Exception:
            pass

# 3) If no date-range applied, apply Year+Month or FY
if not applied_date_filter:
    if year_sel != 'All Years':
        # apply Year filter
        y = int(year_sel)
        fil = fil[fil[date_ref_col].dt.year == y] if date_ref_col else fil
        if month_sel != 'All Months':
            # convert month short name to month number
            try:
                month_num = datetime_month_map = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}[month_sel]
                fil = fil[fil[date_ref_col].dt.month == month_num]
            except Exception:
                pass
    else:
        # apply FY as fallback (use PR date if present else no-op)
        if date_ref_col:
            fil = fil[(fil[date_ref_col] >= fy_start) & (fil[date_ref_col] <= fy_end)]

# Ensure strings & create unified buyer type
for col in ['buyer_type','entity','po_creator','po_buyer_type','po_vendor','product_name']:
    fil[col] = fil[col].astype(str).str.strip()

if 'buyer_type_unified' not in fil.columns:
    fil['buyer_type_unified'] = fil['buyer_type'].replace({'': np.nan}).fillna(fil['po_buyer_type']).fillna('Indirect').str.title()

# Sidebar multi-selects: Buyer Type, Entity, PO Ordered By, PO Buyer Type
choices_bt = sorted(fil['buyer_type_unified'].dropna().unique().tolist())
sel_b = st.sidebar.multiselect('Buyer Type', choices_bt, default=choices_bt)
sel_e = st.sidebar.multiselect('Entity', sorted(fil['entity'].dropna().unique().tolist()), default=sorted(fil['entity'].dropna().unique().tolist()))
sel_o = st.sidebar.multiselect('PO Ordered By', sorted(fil['po_creator'].dropna().unique().tolist()), default=sorted(fil['po_creator'].dropna().unique().tolist()))
sel_p = st.sidebar.multiselect('PO Buyer Type (raw)', sorted(fil['po_buyer_type'].dropna().unique().tolist()), default=sorted(fil['po_buyer_type'].dropna().unique().tolist()))

# Vendor & Item multi-selects (kept)
vendor_choices = sorted(fil['po_vendor'].dropna().unique().tolist())
item_choices = sorted(fil['product_name'].dropna().unique().tolist())
sel_v = st.sidebar.multiselect('Vendor (pick one or more)', vendor_choices, default=vendor_choices)
sel_i = st.sidebar.multiselect('Item / Product (pick one or more)', item_choices, default=item_choices)

# PO Department dropdown
po_dept_list = sorted(fil[po_department_col].dropna().unique().tolist()) if po_department_col in fil.columns else []
po_dept_sel = st.sidebar.selectbox('PO Department', ['All Departments'] + po_dept_list, index=0)

# Reset Filters
if st.sidebar.button('Reset Filters'):
    keys = ['_uploaded_files']
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]
    st.experimental_rerun()

# Apply sidebar selections to fil
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
if po_dept_sel != 'All Departments' and po_department_col in fil.columns:
    fil = fil[fil[po_department_col] == po_dept_sel]

# ---------------- Tabs & Views ----------------
T = st.tabs(['KPIs & Spend','PO/PR Timing','Delivery','Vendors','Dept & Services','Unit-rate Outliers','Forecast','Scorecards','Search'])

# KPIs & Spend
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

# PR/PO Timing (kept as before)
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
    wd['pr_wk'] = wd[pr_col].dt.day_name() if pr_col in wd.columns else ''
    wd['po_wk'] = wd[po_create_col].dt.day_name() if po_create_col in wd.columns else ''
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

# Delivery, Vendors, Dept & others (kept largely as earlier)
with T[2]:
    st.subheader('Delivery Summary')
    dv = fil.rename(columns={'po_quantity':'po_qty','receivedqty':'received_qty','pending_qty':'pending_qty'}).copy()
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

with T[3]:
    st.subheader('Top Vendors by Spend')
    if po_vendor_col in fil.columns and purchase_doc_col in fil.columns and net_amount_col in fil.columns:
        vs = fil.groupby('po_vendor', dropna=False).agg(Vendor_PO_Count=(purchase_doc_col,'nunique'), Total_Spend_Cr=(net_amount_col, lambda s: (s.sum()/1e7).round(2))).reset_index().sort_values('Total_Spend_Cr', ascending=False)
        st.dataframe(vs.head(10), use_container_width=True)
        st.plotly_chart(px.bar(vs.head(10), x='po_vendor', y='Total_Spend_Cr', text='Total_Spend_Cr', title='Top 10 Vendors (Cr ₹)').update_traces(textposition='outside'), use_container_width=True)

with T[4]:
    st.subheader('Dept & Services (Summary)')
    if net_amount_col in fil.columns:
        dept_col_choice = 'po_department' if 'po_department' in fil.columns else ('pr_department' if 'pr_department' in fil.columns else None)
        if dept_col_choice:
            dep = fil.groupby(dept_col_choice, dropna=False)[net_amount_col].sum().reset_index().sort_values(net_amount_col, ascending=False)
            dep['cr'] = dep[net_amount_col]/1e7
            st.dataframe(dep.head(50), use_container_width=True)
        else:
            st.info('No department columns found for Dept & Services view.')

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

with T[7]:
    st.subheader('Vendor Scorecard')
    if po_vendor_col in fil.columns:
        vendor = st.selectbox('Pick Vendor', sorted(fil[po_vendor_col].dropna().astype(str).unique().tolist()))
        vd = fil[fil[po_vendor_col].astype(str) == str(vendor)].copy()
        spend = vd.get(net_amount_col, pd.Series(0)).sum()/1e7 if net_amount_col else 0
        upos = int(vd.get(purchase_doc_col, pd.Series(dtype=object)).nunique()) if purchase_doc_col else 0
        if 'po_delivery_date' in vd.columns and pending_qty_col in vd.columns:
            late = ((pd.to_datetime(vd['po_delivery_date'], errors='coerce').dt.date < TODAY.date()) & (vd[pending_qty_col].fillna(0) > 0)).sum()
        else:
            late = np.nan
        if pending_qty_col in vd.columns and po_unit_rate_col in vd.columns:
            vd['pending_value'] = vd[pending_qty_col].fillna(0).astype(float) * vd[po_unit_rate_col].fillna(0).astype(float)
            pend_val = vd['pending_value'].sum()/1e7
        else:
            pend_val = np.nan
        k1,k2,
