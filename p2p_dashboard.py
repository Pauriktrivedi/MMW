import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="P2P Dashboard — Full (Refactor)", layout="wide", initial_sidebar_state="expanded")

# ---- Page header: guaranteed visible ----nimport streamlit as _st
# touch session_state to ensure session initialised (harmless)
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

# Extra plain-text fallback to ensure visibility
_st.write("## P2P Dashboard — Indirect")

# Quick CSS guard: hide any stray JSON/stJson widgets that came from debug prints
_st.markdown('''
<style>
/* hide Streamlit JSON widget and similar debug preformatted blocks */
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
        # keep alnum and underscore
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
    """Load files either from uploaded file buffers or from default filenames in working folder.
    Returns a single concatenated dataframe or empty df.
    """
    frames = []
    if uploaded_files:
        # uploaded_files are Streamlit UploadedFile objects
        for f in uploaded_files:
            try:
                df_temp = pd.read_excel(f, skiprows=1)
                df_temp['entity'] = f.name.rsplit('.',1)[0]
                frames.append(df_temp)
            except Exception:
                try:
                    # fallback: try without skiprows
                    df_temp = pd.read_excel(f)
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
            except Exception:
                try:
                    df_temp = pd.read_excel(fn)
                    df_temp['entity'] = ent
                    frames.append(df_temp)
                except Exception:
                    continue
    if not frames:
        return pd.DataFrame()
    x = pd.concat(frames, ignore_index=True, sort=False)
    x = normalize_columns(x)
    return x

# ----------------- Sidebar Filters (reordered per request) -----------------
st.sidebar.header('Filters')

# Financial Year picker (existing)
FY = {
    'All Years': (pd.Timestamp('2023-04-01'), pd.Timestamp('2026-03-31')),
    '2023': (pd.Timestamp('2023-04-01'), pd.Timestamp('2024-03-31')),
    '2024': (pd.Timestamp('2024-04-01'), pd.Timestamp('2025-03-31')),
    '2025': (pd.Timestamp('2025-04-01'), pd.Timestamp('2026-03-31')),
}
fy_key = st.sidebar.selectbox('Financial Year', list(FY), index=0)
pr_start, pr_end = FY[fy_key]

# Date range filter (user requested)
st.sidebar.markdown('**Date filters**')
# initial value will be adjusted after we load data
initial_date_range = (pr_start.date(), pr_end.date())
# placeholder — will re-create the widget after loading to use data-derived defaults
st.session_state.setdefault('placeholder_date_range', initial_date_range)

# Month filter placeholder — will be re-created after load
st.session_state.setdefault('placeholder_months', [])

# PO Department placeholder
st.session_state.setdefault('placeholder_po_depts', ['All Departments'])

# Vendor & Item placeholders
st.session_state.setdefault('placeholder_vendors', [])
st.session_state.setdefault('placeholder_items', [])

# Buyer type / entity / PO Ordered By placeholders
st.session_state.setdefault('placeholder_entities', [])
st.session_state.setdefault('placeholder_po_creators', [])
st.session_state.setdefault('placeholder_po_buyer_types', [])

# File uploader (moved to bottom later)
# We'll place a real uploader after the first pass of widgets; for now use a small spacer
st.sidebar.write('---')

# ----------------- Load dataframe -----------------
# Place uploader early so users can provide files before load
uploaded = st.sidebar.file_uploader('Upload one or more Excel/CSV files (optional)', type=['xlsx','xls','csv'], accept_multiple_files=True)

if uploaded:
    df = load_all_from_files(uploaded)
else:
    df = load_all_from_files()

# If no data loaded: show clear instruction and stop further computation
if df.empty:
    st.warning("No data loaded. Either upload Excel files using the sidebar or place default Excel files next to this script (MEPL.xlsx, MLPL.xlsx, mmw.xlsx, mmpl.xlsx).\n\nIf your file has a header row in row 1, it will be read; otherwise the app will attempt a second read without skipping rows.")
    st.stop()

# ----------------- Column mapping: align incoming sheet headers to app columns -----------------
source_to_norm = {
    # PR-level
    'pr number': 'pr_number', 'pr no': 'pr_number', 'pr date submitted': 'pr_date_submitted', 'pr prepared by': 'pr_prepared_by',
    'pr status': 'pr_status', 'pr budget code': 'pr_budget_code', 'pr budget description': 'pr_budget_description',
    'pr business unit': 'pr_business_unit', 'pr department': 'pr_department',
    # PO-level
    'purchase doc': 'purchase_doc', 'po create date': 'po_create_date', 'po delivery date': 'po_delivery_date', 'po vendor': 'po_vendor',
    'po quantity': 'po_quantity', 'po unit rate': 'po_unit_rate', 'net amount': 'net_amount', 'po status': 'po_status',
    'po approved date': 'po_approved_date', 'po orderer': 'po_orderer', 'last po number': 'last_po_number', 'last po date': 'last_po_date',
    'last po vendor': 'last_po_vendor', 'po budget code': 'po_budget_code', 'po budget description': 'po_budget_description',
    'po business unit': 'po_business_unit', 'po department': 'po_department',
    # Item / product
    'product name': 'product_name', 'product name friendly': 'product_name_friendly', 'item code': 'item_code', 'item description': 'item_description',
    'procurement category': 'procurement_category', 'line': 'line',
    # quantities / received / pending
    'receivedqty': 'receivedqty', 'received qty': 'receivedqty', 'pending qty': 'pending_qty', 'pending_qty': 'pending_qty',
    'pr quantity': 'pr_quantity', 'currency': 'currency', 'unit rate': 'unit_rate', 'pr value': 'pr_value',
}

# Apply mapping case-insensitively
col_map = {}
for c in df.columns:
    key = str(c).strip().lower()
    if key in source_to_norm:
        col_map[c] = source_to_norm[key]

if col_map:
    df = df.rename(columns=col_map)

# Ensure normalized column names
df = normalize_columns(df)

# Parse date columns safely
for c in ['pr_date_submitted','po_create_date','po_approved_date','po_delivery_date','last_po_date']:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors='coerce')

# ----------------- Prepare and normalize columns used in app -----------------
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
entity_col = 'entity'
po_department_col = 'po_department' if 'po_department' in df.columns else None

# single "today" value to avoid micro-inconsistencies
TODAY = pd.Timestamp.now().normalize()

# ----------------- Buyer/Creator logic -----------------
# Buyer group code extraction
if buyer_group_col in df.columns:
    try:
        df['buyer_group_code'] = df[buyer_group_col].astype(str).str.extract(r"(\d+)")[0].astype(float)
    except Exception:
        df['buyer_group_code'] = np.nan

# Buyer-type mapping function (keeps fallback safe)
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

# PO Orderer mapping
map_orderer = {
    'mmw2324030': 'Dhruv', 'mmw2324062': 'Deepak', 'mmw2425154': 'Mukul', 'mmw2223104': 'Paurik',
    'mmw2021181': 'Nayan', 'mmw2223014': 'Aatish', 'mmw_ext_002': 'Deepakex', 'mmw2425024': 'Kamlesh',
    'mmw2021184': 'Suresh', 'n/a': 'Dilip'
}

if not df.empty:
    df['po_orderer'] = safe_get(df, po_orderer_col, pd.NA).fillna('N/A').astype(str).str.strip()
    df['po_orderer_lc'] = df['po_orderer'].str.lower()
    df['po_creator'] = df['po_orderer_lc'].map(map_orderer).fillna(df['po_orderer']).replace({'N/A': 'Dilip'})
    indirect = set(['Aatish','Deepak','Deepakex','Dhruv','Dilip','Mukul','Nayan','Paurik','Kamlesh','Suresh'])
    df['po_buyer_type'] = np.where(df['po_creator'].isin(indirect), 'Indirect', 'Direct')
else:
    df['po_creator'] = pd.Series(dtype=object)
    df['po_buyer_type'] = pd.Series(dtype=object)

# ----------------- Populate dynamic sidebar choices now that df exists -----------------
# Months (Month-Year) list
if pr_col in df.columns:
    df['pr_month'] = df[pr_col].dt.to_period('M').dt.to_timestamp()
    month_options = df['pr_month'].dropna().dt.strftime('%b-%Y').sort_values().unique().tolist()
else:
    month_options = []

# Vendor and item options
vendor_options = sorted(df.get(po_vendor_col, pd.Series(dtype=object)).dropna().astype(str).unique().tolist())
item_options = sorted(df.get('product_name', pd.Series(dtype=object)).dropna().astype(str).unique().tolist())

# Entity, PO creators options
entity_options = sorted(df.get(entity_col, pd.Series(dtype=object)).dropna().astype(str).unique().tolist())
po_creator_options = sorted(df.get('po_creator', pd.Series(dtype=object)).dropna().astype(str).unique().tolist())
po_buyer_type_options = sorted(df.get('po_buyer_type', pd.Series(dtype=object)).dropna().astype(str).unique().tolist())

# PO Department options
po_dept_options = ['All Departments'] + sorted(df.get(po_department_col, pd.Series(dtype=object)).dropna().astype(str).unique().tolist())

# Now rebind sidebar widgets with populated choices (preserve previous selections where possible)
# Note: Streamlit requires widgets to be created in same order on every run — we re-create them here.
st.sidebar.write('---')
fy_key = st.sidebar.selectbox('Financial Year', list(FY), index=0, key='fy_key')

# Date range again (reflect data-derived defaults if present)
default_start = df[pr_col].min().date() if pr_col in df.columns and not df[pr_col].dropna().empty else pr_start.date()
default_end = df[pr_col].max().date() if pr_col in df.columns and not df[pr_col].dropna().empty else pr_end.date()
date_range = st.sidebar.date_input('PR Date range (start — end)', value=(default_start, default_end), key='date_range')

# Month multiselect
month_pick = st.sidebar.multiselect('Month(s) (select to filter)', options=month_options, default=month_options, key='month_pick')

# PO Department dropdown
po_department_sel = st.sidebar.selectbox('PO Department', po_dept_options, index=0, key='po_dept')

# Vendor & Item: allow both dropdown (single) and multi-select behavior
vendor_single = st.sidebar.selectbox('Vendor (single-select)', ['All Vendors'] + vendor_options, index=0, key='vendor_single')
vendor_sel_multi = st.sidebar.multiselect('Vendors (multi-select)', options=vendor_options, default=vendor_options, key='vendor_multi')

item_single = st.sidebar.selectbox('Item (single-select)', ['All Items'] + item_options, index=0, key='item_single')
item_sel_multi = st.sidebar.multiselect('Items / Products (multi-select)', options=item_options, default=item_options, key='item_multi')

# Buyer Type / Entity / PO Ordered By / PO Buyer Type
sel_b = st.sidebar.multiselect('Buyer Type', sorted(df.get('buyer_type', pd.Series(dtype=object)).dropna().unique().tolist()), default=sorted(df.get('buyer_type', pd.Series(dtype=object)).dropna().unique().tolist()), key='buyer_type')
sel_e = st.sidebar.multiselect('Entity', entity_options, default=entity_options, key='entity')
sel_o = st.sidebar.multiselect('PO Ordered By', po_creator_options, default=po_creator_options, key='po_creator')
sel_p = st.sidebar.multiselect('PO Buyer Type (raw)', po_buyer_type_options or ['Indirect','Direct'], default=po_buyer_type_options or ['Indirect','Direct'], key='po_buyer_type')

# Reset Filters button
if st.sidebar.button('Reset Filters'):
    for k in ['filter_vendor','filter_item','filter_buyer','filter_entity','filter_po_creator','filter_po_buyer_type','fy_key','vendor_multi','item_multi','month_pick','po_dept','date_range']:
        if k in st.session_state:
            del st.session_state[k]
    st.experimental_rerun()

# ----------------- Apply filters to dataframe -----------------
fil = df.copy()

# apply PR date range
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_d, end_d = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    if pr_col in fil.columns:
        fil = fil[(fil[pr_col] >= start_d) & (fil[pr_col] <= end_d)]

# apply month filter (if selected)
if month_pick:
    # convert fil pr_month to string and filter
    if pr_col in fil.columns:
        fil['pr_month_str'] = fil[pr_col].dt.to_period('M').dt.to_timestamp().dt.strftime('%b-%Y')
        fil = fil[fil['pr_month_str'].isin(month_pick)]

# apply PO department
if po_department_sel and po_department_sel != 'All Departments' and po_department_col in fil.columns:
    fil = fil[fil[po_department_col].astype(str) == str(po_department_sel)]

# apply buyer/entity/po_creator/po_buyer_type filters
if sel_b:
    fil = fil[fil['buyer_type'].isin(sel_b)]
if sel_e:
    fil = fil[fil[entity_col].isin(sel_e)]
if sel_o:
    fil = fil[fil['po_creator'].isin(sel_o)]
if sel_p:
    fil = fil[fil['po_buyer_type'].isin(sel_p)]

# apply vendor/item filters: prioritize multi-select if used; allow single-select 'All' to keep all
if vendor_sel_multi:
    fil = fil[fil.get(po_vendor_col, pd.Series('', index=fil.index)).astype(str).isin(vendor_sel_multi)]
elif vendor_single and vendor_single != 'All Vendors':
    fil = fil[fil.get(po_vendor_col, pd.Series('', index=fil.index)).astype(str) == vendor_single]

if item_sel_multi:
    fil = fil[fil.get('product_name', pd.Series('', index=fil.index)).astype(str).isin(item_sel_multi)]
elif item_single and item_single != 'All Items':
    fil = fil[fil.get('product_name', pd.Series('', index=fil.index)).astype(str) == item_single]

# ----------------- Tabs -----------------
T = st.tabs(['KPIs & Spend','PO/PR Timing','Delivery','Vendors','Dept & Services','Unit-rate Outliers','Forecast','Scorecards','Search'])

# ----------------- KPIs & Spend -----------------
with T[0]:
    # --- Top KPI metrics ---
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
    # --- Spend charts placed on same page as KPIs ---
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
    else:
        fig_spend = None

    st.subheader('Entity Trend')
    if dcol and net_amount_col in fil.columns:
        x = fil.copy()
        x['po_month'] = x[dcol].dt.to_period('M').dt.to_timestamp()
        g = x.dropna(subset=['po_month']).groupby(['po_month', entity_col], as_index=False)[net_amount_col].sum()
        g['cr'] = g[net_amount_col]/1e7
        if not g.empty:
            fig_entity = px.line(g, x=g['po_month'].dt.strftime('%b-%Y'), y='cr', color=entity_col, markers=True, labels={'x':'Month','cr':'Cr ₹'})
            fig_entity.update_layout(xaxis_tickangle=-45)
        else:
            fig_entity = None
    else:
        fig_entity = None

    if fig_spend is not None:
        st.plotly_chart(fig_spend, use_container_width=True)
    else:
        st.info('Monthly Spend chart not available — need date and Net Amount columns.')

    st.markdown('---')

    if fig_entity is not None:
        st.plotly_chart(fig_entity, use_container_width=True)
    else:
        st.info('Entity trend not available — need date and Net Amount columns.')

# EOF
