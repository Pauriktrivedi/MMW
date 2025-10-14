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
        # collapse multiple underscores
        while '__' in s:
            s = s.replace('__', '_')
        s = s.strip('_')
        new_cols[c] = s
    return df.rename(columns=new_cols)


@st.cache_data(show_spinner=False)
def load_all_from_files(uploaded_files=None):
    """
    Load multiple uploaded files (UploadedFile objects) or default named files in repo.
    Tries skiprows=1 first and falls back to skiprows=0 if that fails for each file.
    Returns a concatenated normalized DataFrame.
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
                    # if it's CSV try that as well
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

# ----------------- Sidebar Filters (top-left) -----------------
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

# Date range picker (alternative override)
date_range = st.sidebar.date_input('Select Date Range (optional)', value=[fy_start.date(), fy_end.date()], key='date_range_picker')
# normalize to timestamps
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    try:
        dr_start = pd.to_datetime(date_range[0])
        dr_end = pd.to_datetime(date_range[1])
    except Exception:
        dr_start, dr_end = fy_start, fy_end
else:
    dr_start, dr_end = fy_start, fy_end

# Buyer Type multiselect (we'll fill choices dynamically after data load)
# Vendor / Item - we want multi-selects (multiple selection allowed)
# PO Department - dropdown (we will use multi-select for flexibility)
st.sidebar.markdown('---')
st.sidebar.write('Vendor / Item / PO Department filters will appear once data is loaded.')
st.sidebar.markdown('---')

# A small 'Reset filters' button to clear session keys
if st.sidebar.button('Reset Filters'):
    for k in list(st.session_state.keys()):
        if k.startswith('filter_') or k in ['_bottom_uploaded']:
            del st.session_state[k]
    st.experimental_rerun()

# ----------------- Data Loading (main) -----------------
# First attempt to use files stored in session state (bottom uploader flow)
uploaded_session = st.session_state.get('_bottom_uploaded', None)
# Also allow immediate top uploader (optional quick load)
top_uploaded = st.sidebar.file_uploader('(Optional) Upload Excel/CSV files here (quick load)', type=['xlsx','xls','csv'], accept_multiple_files=True, key='top_uploader')

# Decide which files to use: top uploader, then session bottom uploader, then defaults
files_to_load = None
if top_uploaded:
    files_to_load = top_uploaded
elif uploaded_session:
    files_to_load = uploaded_session

df = load_all_from_files(files_to_load)

if df.empty:
    st.warning("No data loaded. Place default Excel files next to this script or use the bottom uploader to upload files.")
    # Still show the bottom uploader so user can upload
    st.markdown('---')
    st.markdown('### Upload files (bottom of page)')
    new_files = st.file_uploader('Upload Excel/CSV files here', type=['xlsx','xls','csv'], accept_multiple_files=True, key='_bottom_uploader')
    if new_files:
        st.session_state['_bottom_uploaded'] = new_files
        st.experimental_rerun()
    st.stop()

# ----------------- Column canonicalization & safe names -----------------
# Map common incoming names to expected normalized names (case-insensitive)
source_to_norm = {
    'pr number': 'pr_number', 'pr no': 'pr_number',
    'pr date submitted': 'pr_date_submitted',
    'purchase doc': 'purchase_doc', 'po create date': 'po_create_date', 'po delivery date': 'po_delivery_date',
    'po vendor': 'po_vendor', 'po quantity': 'po_quantity', 'po unit rate': 'po_unit_rate',
    'net amount': 'net_amount', 'pr status': 'pr_status', 'po status': 'po_status',
    'received qty': 'receivedqty', 'receivedqty': 'receivedqty', 'pending qty': 'pending_qty',
    'product name': 'product_name', 'item code': 'item_code', 'item description': 'item_description',
    'po budget code': 'po_budget_code', 'pr budget code': 'pr_budget_code',
    'po orderer': 'po_orderer', 'po department': 'po_department'
}

# Apply mapping against raw column names (before/after normalization)
col_map = {}
for c in list(df.columns):
    key = str(c).strip().lower()
    if key in source_to_norm:
        col_map[c] = source_to_norm[key]

if col_map:
    df = df.rename(columns=col_map)

# normalize column names finally (idempotent)
df = normalize_columns(df)

# parse date fields safely
for dcol in ['pr_date_submitted','po_create_date','po_approved_date','po_delivery_date','last_po_date']:
    if dcol in df.columns:
        df[dcol] = pd.to_datetime(df[dcol], errors='coerce')

# prepare safe column variables used across the app
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

# create derived date fields for filtering
if pr_col:
    df['pr_date_submitted'] = pd.to_datetime(df[pr_col], errors='coerce')
else:
    df['pr_date_submitted'] = pd.NaT

df['pr_year'] = df['pr_date_submitted'].dt.year
df['pr_month_name'] = df['pr_date_submitted'].dt.strftime('%b')  # e.g., 'Jan', 'Feb'

# ----------------- Build dynamic filter choices -----------------
# Buyer type unified (if buyer info present)
if 'buyer_type' not in df.columns:
    df['buyer_type'] = df.get('buyer_type', pd.Series(['Indirect'] * len(df)))

buyer_choices = sorted(df['buyer_type'].dropna().unique().astype(str).tolist())

vendor_choices = sorted(df[po_vendor_col].dropna().astype(str).unique().tolist()) if po_vendor_col in df.columns else []
item_choices = sorted(df[product_name_col].dropna().astype(str).unique().tolist()) if product_name_col in df.columns else []
dept_choices = sorted(df[po_department_col].dropna().astype(str).unique().tolist()) if po_department_col in df.columns else []

# Sidebar placement for these filters (multi-selects)
st.sidebar.markdown('### Additional Filters')
sel_buyer = st.sidebar.multiselect('Buyer Type', buyer_choices, default=buyer_choices)
sel_vendor = st.sidebar.multiselect('Vendor (pick one or more)', vendor_choices, default=vendor_choices if vendor_choices else [])
sel_item = st.sidebar.multiselect('Item / Product (pick one or more)', item_choices, default=item_choices if item_choices else [])
# PO Department as dropdown/multi-select (user asked dropdown; multi-select is more flexible)
if dept_choices:
    sel_po_dept = st.sidebar.multiselect('PO Department', ['All Departments'] + dept_choices, default=['All Departments'])
else:
    sel_po_dept = ['All Departments']

# ----------------- Apply Filters (year / month / date range / vendor / item / dept) -----------------
# Start with the whole frame and apply filters in order
fil = df.copy()

# Date range overriding logic:
# If user changed date_range in date_input, use that instead of FY default
# dr_start / dr_end created earlier from the date_input
fil = fil[(fil['pr_date_submitted'] >= pd.to_datetime(dr_start)) & (fil['pr_date_submitted'] <= pd.to_datetime(dr_end))]

# If user selected a specific FY (not 'All Years') AND no manual date range used,
# we also ensure within FY (we already set dr_start/dr_end default to FY, so this is safe)

# Month filtering
if sel_month and sel_month != 'All Months':
    # match by 3-letter month abbreviation
    fil = fil[fil['pr_month_name'].str.startswith(sel_month)]

# Buyer type filter
if sel_buyer:
    fil = fil[fil['buyer_type'].astype(str).isin(sel_buyer)]

# Vendor filter (multi-select)
if sel_vendor:
    fil = fil[fil[po_vendor_col].astype(str).isin(sel_vendor)] if po_vendor_col in fil.columns else fil

# Item filter (multi-select)
if sel_item:
    fil = fil[fil[product_name_col].astype(str).isin(sel_item)] if product_name_col in fil.columns else fil

# PO Department filter
if sel_po_dept and 'All Departments' not in sel_po_dept:
    if po_department_col in fil.columns:
        fil = fil[fil[po_department_col].astype(str).isin(sel_po_dept)]

# ----------------- Basic KPIs & Example outputs (you can integrate this into the full dashboard) -----------------
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
st.subheader('Preview (filtered rows)')
st.write(f"Filtered for FY `{fy_key}`  •  Month: `{sel_month}`  •  Range: `{pd.to_datetime(dr_start).date()}` → `{pd.to_datetime(dr_end).date()}`")
st.dataframe(fil.head(100), use_container_width=True)

# ----------------- Bottom uploader (persisted) -----------------
st.markdown('---')
st.markdown('### Upload & Debug (bottom of page)')
new_files = st.file_uploader('Upload Excel/CSV files here (bottom uploader) — select multiple if needed', type=['xlsx','xls','csv'], accept_multiple_files=True, key='_bottom_uploader')
if new_files:
    # save to session state and reload app (so all filters rebuild from new data)
    st.session_state['_bottom_uploaded'] = new_files
    st.experimental_rerun()
