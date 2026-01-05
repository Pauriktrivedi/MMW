import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from plotly.subplots import make_subplots
import time
import logging
import traceback
import re

# ---------- CONFIG ----------
# Set up a logger that works reliably with Streamlit
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler('app.log', mode='w')
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

DATA_DIR = Path(__file__).resolve().parent
LOGO_PATH = DATA_DIR / "matter_logo.png"
INDIRECT_BUYERS = {
    'Aatish', 'Deepak', 'Deepakex', 'Dhruv', 'Dilip', 'Mukul', 'Nayan', 'Paurik',
    'Kamlesh', 'Suresh', 'Priyam'
}

st.set_page_config(page_title="P2P Dashboard — Indirect (Final)", layout="wide", initial_sidebar_state="expanded")

# ---------- Helpers / Utilities (optimized) ----------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized, robust column normalizer."""
    if df is None or df.empty:
        return df
    cols = list(df.columns)
    new = []
    for c in cols:
        s = str(c).strip()
        s = s.replace(chr(160), " ")
        s = s.replace(chr(92), "_").replace('/', '_')
        s = '_'.join(s.split())
        s = s.lower()
        s = ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in s)
        s = '_'.join([p for p in s.split('_') if p != ''])
        new.append(s)
    df.columns = new
    return df

def safe_col(df, candidates, default=None):
    for c in candidates:
        if c in df.columns:
            return c
    return default

def memoized_compute(namespace: str, signature: tuple, compute_fn):
    """Session-state memoization keyed by the active filter tuple."""
    store = st.session_state.setdefault('_memo_cache', {})
    key = (namespace, signature)
    if key not in store:
        store[key] = compute_fn()
    return store[key]

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def _resolve_path(fn: str) -> Path:
    path = Path(fn)
    if path.exists():
        return path
    candidate = DATA_DIR / fn
    return candidate

def _read_excel(path: Path, entity: str) -> pd.DataFrame:
    # skiprows=1 was in original — preserve
    df = pd.read_excel(path, skiprows=1)
    df['entity_source_file'] = entity
    return df

def _finalize_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    x = pd.concat(frames, ignore_index=True)
    x = normalize_columns(x)
    # parse common date columns once
    for c in ['pr_date_submitted', 'po_create_date', 'po_delivery_date', 'po_approved_date']:
        if c in x.columns:
            x[c] = pd.to_datetime(x[c], errors='coerce')
    return x

@st.cache_data(show_spinner=False)
def load_all():
    """Loads and finalizes the dataset from the Parquet file."""
    parquet_path = DATA_DIR / "p2p_data.parquet"
    if not parquet_path.exists():
        st.warning("Data file (p2p_data.parquet) not found. Please run the conversion script first.")
        return pd.DataFrame()
    
    try:
        df = pd.read_parquet(parquet_path)
        # Ensure date columns are parsed correctly after loading from Parquet
        for c in ['pr_date_submitted', 'po_create_date', 'po_delivery_date', 'po_approved_date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Failed to load Parquet file: {e}")
        return pd.DataFrame()

# ---------- Vendor Master Parsing (New) ----------
@st.cache_data(show_spinner=False)
def load_vendor_master():
    """
    Parses 'meplvendor.xlsx', 'mlplvendor.xlsx', 'mmwvendor.xlsx', 'mmplvendor.xlsx' 
    if present in DATA_DIR. Returns a unified DataFrame of vendor details.
    Structure: Entity | VendorCode | VendorName | Address | Phone | Email | State | City
    """
    vendor_files = {
        'MEPL': 'meplvendor.xlsx',
        'MLPL': 'mlplvendor.xlsx',
        'MMW':  'mmwvendor.xlsx',
        'MMPL': 'mmplvendor.xlsx'
    }
    
    records = []
    
    for entity, fname in vendor_files.items():
        fpath = DATA_DIR / fname
        if not fpath.exists():
            continue
            
        try:
            # Read headerless to parse semi-structured content
            df = pd.read_excel(fpath, header=None)
            
            # Find all rows that start a vendor block (Col A == "Vendor account")
            start_indices = df.index[df.iloc[:, 0].astype(str).str.strip() == 'Vendor account'].tolist()
            
            for start_idx in start_indices:
                # Code
                code = df.iloc[start_idx, 2] # Col C
                
                # Name (Row + 6)
                name = None
                if start_idx + 6 < len(df):
                     val_g = str(df.iloc[start_idx + 6, 6]).strip() # Check label 'Vendor name'
                     if val_g == 'Vendor name':
                         name = df.iloc[start_idx + 6, 9] # Col I
                
                # Address (Row + 6)
                address = None
                if start_idx + 6 < len(df):
                    val_a = str(df.iloc[start_idx + 6, 0]).strip()
                    if val_a == 'Address':
                        address = df.iloc[start_idx + 6, 2]

                # Initialize other fields
                phone = None
                email = None
                state = None
                city = None
                
                # Scan next 40 rows for keywords in Col A
                limit = min(start_idx + 40, len(df))
                for r in range(start_idx, limit):
                    label = str(df.iloc[r, 0]).strip()
                    
                    if label == 'Telephone':
                        phone = df.iloc[r, 2]
                    elif label == 'Email':
                        email = df.iloc[r, 2]
                    elif label == 'State':
                        state = df.iloc[r, 2]
                    
                    # Stop if we hit next block
                    if r > start_idx and label == 'Vendor account':
                        break
                
                # Heuristic to parse City from Address if possible
                # Pattern: "City -Zip" (e.g., "Noida -201301")
                if address:
                    # Look for lines containing " -<digits>"
                    # or lines before state/country codes.
                    # Simple Regex approach:
                    # Matches "Word -123456" but ensures no newlines in the captured group
                    match = re.search(r'(?:^|\n)([^\n]+?)\s*-\s*\d{6}', str(address))
                    if match:
                        city = match.group(1).strip()
                    else:
                        # Fallback: sometimes it is "City-Zip" without space
                         match2 = re.search(r'(?:^|\n)([^\n]+?)-\d{6}', str(address))
                         if match2:
                             city = match2.group(1).strip()
                
                records.append({
                    'Entity': entity,
                    'VendorCode': str(code).strip(),
                    'VendorName': str(name).strip() if name else None,
                    'Address': str(address).strip() if address else None,
                    'Phone': str(phone).strip() if phone else None,
                    'Email': str(email).strip() if email else None,
                    'State': str(state).strip() if state else None,
                    'City': str(city).strip() if city else None
                })

        except Exception as e:
            logger.error(f"Error parsing vendor file {fname}: {e}")
            
    if records:
        v_df = pd.DataFrame(records)
        # Normalize name for matching
        v_df['VendorName_Norm'] = v_df['VendorName'].astype(str).str.lower().str.strip()
        return v_df
    
    return pd.DataFrame(columns=['Entity', 'VendorCode', 'VendorName', 'Address', 'Phone', 'Email', 'State', 'City', 'VendorName_Norm'])

# ---------- Fast type/coercion utilities ----------

def to_cat(df, col):
    if col in df.columns:
        df[col] = df[col].astype('category')

# ---------- Domain-specific vectorized helpers ----------

def compute_buyer_type_vectorized(df: pd.DataFrame) -> pd.Series:
    """Classify PRs into Direct/Indirect using Buyer Group + numeric code (vectorized)."""
    if df.empty:
        return pd.Series(dtype=object)
    group_col = safe_col(df, ['buyer_group', 'Buyer Group', 'buyer group'])
    if not group_col:  # default to Indirect if missing
        return pd.Series('Indirect', index=df.index, dtype=object)

    bg_raw = df[group_col].fillna('').astype(str).str.strip()
    # extract numeric code
    code_series = pd.to_numeric(bg_raw.str.extract(r'(\d+)')[0], errors='coerce')

    buyer_type = pd.Series('Direct', index=df.index, dtype=object)
    alias_direct = bg_raw.str.upper().isin({'ME_BG17', 'MLBG16'})
    buyer_type[alias_direct] = 'Direct'

    not_available = bg_raw.eq('') | bg_raw.str.lower().isin(['not available', 'na', 'n/a'])
    buyer_type[not_available] = 'Indirect'

    # vectorized ranges
    buyer_type[(code_series >= 1) & (code_series <= 9)] = 'Direct'
    buyer_type[(code_series >= 10) & (code_series <= 18)] = 'Indirect'
    buyer_type = buyer_type.fillna('Direct')
    return buyer_type

def compute_buyer_display(df: pd.DataFrame, purchase_doc_col: str | None, requester_col: str | None) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=object)
    po_creator = df.get('po_creator', pd.Series('', index=df.index)).fillna('').astype(str).str.strip()
    requester = df.get(requester_col, pd.Series('', index=df.index)).fillna('').astype(str).str.strip() if requester_col else pd.Series('', index=df.index)

    has_po = pd.Series(False, index=df.index)
    if purchase_doc_col and purchase_doc_col in df.columns:
        has_po = df[purchase_doc_col].fillna('').astype(str).str.strip() != ''

    # choose in vectorized manner with np.select
    conditions = [
        has_po & (po_creator != ''),
        (po_creator == '') & (requester != '')
    ]
    choices = [
        po_creator,
        requester
    ]
    buyer_display = np.select(conditions, choices, default='PR only - Unassigned')
    return pd.Series(buyer_display, index=df.index, dtype=object)

def compute_item_type_vectorized(df: pd.DataFrame) -> pd.Series:
    """Classifies items into 'Products' or 'Services' based on Category, Item Code, and Description."""
    if df.empty:
        return pd.Series(dtype=object)
    
    # 1. Explicit Service Categories
    service_cats = {
        'Service', 'Testing Services', 'IT Services', 'Recruitment', 'Repair & Maintenance', 
        'Plant Consultancy Services', 'Repairs and Maint.- Vehicle', 'Advertisement And Agency Cost', 
        'Customer Support Cost', 'Staff Welfare Cost', 'Electric Installation', 'Consulting Services', 
        'Office Maintenance', 'Insurance Expense', 'Legal and professional', 'Software License', 
        'SOFTWARE', 'Marketing', 'Network', 'Plant Maintenance', 'Transport'
    }
    
    # 2. Prepare columns
    cat_col = df.get('procurement_category', pd.Series('', index=df.index)).astype(str).fillna('')
    code_col = df.get('item_code', pd.Series('', index=df.index)).astype(str).fillna('').str.upper()
    prod_col = df.get('product_name', pd.Series('', index=df.index)).astype(str).fillna('').str.upper()
    desc_col = df.get('item_description', pd.Series('', index=df.index)).astype(str).fillna('').str.upper()
    
    # 3. Vectorized Masks
    mask_cat = cat_col.isin(service_cats)
    
    # Item Code Patterns: SER_, LBR_
    mask_code = code_col.str.startswith('SER') | code_col.str.startswith('LBR')
    
    # Keywords in Product/Description
    # Regex for distinct keywords to avoid partial matches like "Serviceable" (though likely safe)
    service_keywords = r'\b(AMC|ANNUAL MAINTENANCE|SERVICE|FEE|CHARGES|CONSULTANCY|LABOUR|INSTALLATION|FREIGHT|TRANSPORT|SUBSCRIPTION|WARRANTY)\b'
    mask_desc = prod_col.str.contains(service_keywords, regex=True) | desc_col.str.contains(service_keywords, regex=True)
    
    # 4. Final Logic: Any positive signal -> Service
    is_service = mask_cat | mask_code | mask_desc
    
    return np.where(is_service, 'Services', 'Products')


@st.cache_data(show_spinner=False)
def preprocess_data(_df: pd.DataFrame) -> pd.DataFrame:
    """Applies all expensive preprocessing steps to the raw dataframe."""
    if _df.empty:
        return _df
    df = _df.copy()

    # ensure entity
    entity_col = safe_col(df, ['entity','company','brand','entity_name'])
    if entity_col and entity_col in df.columns:
        df['entity'] = df[entity_col].fillna('').astype(str).str.strip()
    else:
        df['entity'] = df.get('entity_source_file', '').fillna('').astype(str)

    # defensive default columns
    pr_budget_desc_col = safe_col(df, ['pr_budget_description', 'pr budget description', 'pr_budget_desc', 'pr budget description'])
    pr_budget_code_col = safe_col(df, ['pr_budget_code', 'pr budget code', 'pr_budgetcode'])
    po_budget_desc_col = safe_col(df, ['po_budget_description', 'po budget description', 'po_budget_desc'])
    po_budget_code_col = safe_col(df, ['po_budget_code', 'po budget code', 'po_budgetcode'])
    pr_bu_col = safe_col(df, ['pr_bussiness_unit','pr_business_unit','pr business unit','pr_bu','pr bussiness unit','pr business unit'])
    po_bu_col = safe_col(df, ['po_bussiness_unit','po_business_unit','po business unit','po_bu','po bussiness unit','po business unit'])
    for c in [pr_budget_desc_col, pr_budget_code_col, po_budget_desc_col, po_budget_code_col, pr_bu_col, po_bu_col]:
        if c and c not in df.columns:
            df[c] = ''

    # buyer group code extraction (fast)
    if 'buyer_group' in df.columns:
        try:
            df['buyer_group_code'] = pd.to_numeric(df['buyer_group'].astype(str).str.extract('([0-9]+)')[0], errors='coerce')
        except Exception:
            df['buyer_group_code'] = np.nan

    # Buyer.Type
    if 'Buyer.Type' not in df.columns:
        df['Buyer.Type'] = compute_buyer_type_vectorized(df)
    df['Buyer.Type'] = df['Buyer.Type'].fillna('Direct').astype(str).str.strip().str.title()

    # normalize po_creator using mapping
    o_created_by_map = {
        'MMW2324030': 'Dhruv', 'MMW2324062': 'Deepak', 'MMW2425154': 'Mukul', 'MMW2223104': 'Paurik',
        'MMW2021181': 'Nayan', 'MMW2223014': 'Aatish', 'MMW_EXT_002': 'Deepakex', 'MMW2425024': 'Kamlesh',
        'MMW2021184': 'Suresh', 'N/A': 'Dilip', 'MMW2526019': 'Vraj', 'MMW2223240': 'Vatsal',
        'MMW2223219': '', 'MMW2021115': 'Priyam', 'MMW2425031': 'Preet', 'MMW222360IN': 'Ayush',
        'MMW2425132': 'Prateek.B', 'MMW2425025': 'Jaymin', 'MMW2425092': 'Suresh', 'MMW252617IN': 'Akaash',
        'MMW1920052': 'Nirmal', '2425036': '', 'MMW222355IN': 'Jaymin', 'MMW2324060': 'Chetan',
        'MMW222347IN': 'Vaibhav', 'MMW2425011': '', 'MMW1920036': 'Ankit', 'MMW2425143': 'Prateek.K',
        '2425027': '', 'MMW2223017': 'Umesh', 'MMW2021214': 'Raunak', 'Intechuser1': 'Intesh Data'
    }
    upper_map = {k.upper(): v for k, v in o_created_by_map.items()}
    po_orderer_col = safe_col(df, ['po_orderer', 'po orderer', 'po_orderer_code'])
    df['po_orderer'] = df[po_orderer_col].fillna('N/A').astype(str).str.strip() if po_orderer_col in df.columns else 'N/A'
    
    # Optimized po_creator mapping
    df['po_creator'] = df['po_orderer'].str.upper().map(upper_map).fillna(df['po_orderer'])
    # Robust Dilip mapping (handle nan/null strings case-insensitive)
    df['po_creator'] = df['po_creator'].fillna('Dilip').astype(str)
    mask_dilip = df['po_creator'].str.strip().str.lower().isin(['nan', 'n/a', 'na', '', 'none', 'null'])
    df.loc[mask_dilip, 'po_creator'] = 'Dilip'

    # po_buyer_type
    creator_clean = df['po_creator'].fillna('').astype(str).str.strip()
    df['po_buyer_type'] = np.where(creator_clean.isin(INDIRECT_BUYERS), 'Indirect', 'Direct')

    # Fix: Vraj should be considered Direct even if touching Indirect buyer groups
    df.loc[df['po_creator'] == 'Vraj', 'Buyer.Type'] = 'Direct'

    # pr_requester column detection and buyer_display
    purchase_doc_col = safe_col(df, ['purchase_doc', 'purchase_doc_number', 'purchase doc'])
    pr_requester_col = safe_col(df, ['pr_requester','requester','pr_requester_name','pr_requester_name','requester_name'])
    df['buyer_display'] = compute_buyer_display(df, purchase_doc_col, pr_requester_col)

    # Convert common columns to categorical to speed groupbys & joins
    po_vendor_col = safe_col(df, ['po_vendor', 'vendor', 'po vendor'])
    df['po_vendor'] = df[po_vendor_col].fillna('').astype(str) if po_vendor_col in df.columns else ''
    df['product_name'] = df['product_name'].fillna('').astype(str) if 'product_name' in df.columns else ''
    
    # Ensure purchase_doc is categorical to speed up groupby in Delivery tab
    if purchase_doc_col and purchase_doc_col in df.columns:
        to_cat(df, purchase_doc_col)

    # Compute Item.Type
    df['Item.Type'] = compute_item_type_vectorized(df)

    for c in ['entity', 'po_creator', 'buyer_display', po_vendor_col, 'Buyer.Type', 'procurement_category', 'product_name', 'Item.Type']:
        to_cat(df, c)
        
    return df

# ---------- Load & preprocess ----------
logger.info("Starting data loading...")
load_start_time = time.time()
df_raw = load_all()
vendor_master = load_vendor_master() # Load vendor details
load_end_time = time.time()
logger.info(f"Data loading took: {load_end_time - load_start_time:.2f} seconds")

logger.info("Starting data preprocessing...")
preprocess_start_time = time.time()
df = preprocess_data(df_raw)
preprocess_end_time = time.time()
logger.info(f"Data preprocessing took: {preprocess_end_time - preprocess_start_time:.2f} seconds")


if df.empty:
    st.warning("No data loaded. Place MEPL.xlsx, MLPL.xlsx, mmw.xlsx, mmpl.xlsx next to this script or upload files.")

# ---------- Canonical column detection ----------
# (Diagnostics section removed per user request)
# Ensure globals for overrides are set to None if not present (since diagnostics removed)
pr_qty_col = pr_unit_rate_col = pr_value_col = po_qty_col = po_unit_rate_col = net_col = None

# canonical column detection
pr_col = safe_col(df, ['pr_date_submitted', 'pr_date', 'pr date submitted'])
po_create_col = safe_col(df, ['po_create_date', 'po create date', 'po_created_date'])
net_amount_col = safe_col(df, ['net_amount', 'net amount', 'net_amount_inr', 'amount'])
purchase_doc_col = safe_col(df, ['purchase_doc', 'purchase_doc_number', 'purchase doc'])
pr_number_col = safe_col(df, ['pr_number', 'pr number', 'pr_no'])
po_vendor_col = safe_col(df, ['po_vendor', 'vendor', 'po vendor'])
po_unit_rate_col = safe_col(df, ['po_unit_rate', 'po unit rate', 'po_unit_price'])
pr_budget_code_col = safe_col(df, ['pr_budget_code', 'pr budget code', 'pr_budgetcode'])
pr_budget_desc_col = safe_col(df, ['pr_budget_description', 'pr budget description', 'pr_budget_desc', 'pr budget description'])
po_budget_code_col = safe_col(df, ['po_budget_code', 'po budget code', 'po_budgetcode'])
po_budget_desc_col = safe_col(df, ['po_budget_description', 'po budget description', 'po_budget_desc'])
pr_bu_col = safe_col(df, ['pr_bussiness_unit','pr_business_unit','pr business unit','pr_bu','pr bussiness unit','pr business unit'])
po_bu_col = safe_col(df, ['po_bussiness_unit','po_business_unit','po business unit','po_bu','po bussiness unit','po business unit'])
entity_col = safe_col(df, ['entity','company','brand','entity_name'])
pr_requester_col = safe_col(df, ['pr_requester','requester','pr_requester_name','pr_requester_name','requester_name'])

# ----------------- Sidebar filters -----------------
logger.info("Applying filters...")
filter_start_time = time.time()
if LOGO_PATH.exists():
    st.sidebar.image(str(LOGO_PATH), use_column_width=True)
st.sidebar.header('Filters')

FY = {
    'All Years': (pd.Timestamp('2023-04-01'), pd.Timestamp('2026-03-31')),
    '2023': (pd.Timestamp('2023-04-01'), pd.Timestamp('2024-03-31')),
    '2024': (pd.Timestamp('2024-04-01'), pd.Timestamp('2025-03-31')),
    '2025': (pd.Timestamp('2025-04-01'), pd.Timestamp('2026-03-31'))
}
fy_key = st.sidebar.selectbox('Financial Year', list(FY))
pr_start, pr_end = FY[fy_key]

# Work on a filtered view (avoid copies until necessary)
fil = df
# FY Filtering Logic: Use PR Date if available; fallback to PO Create Date for rows where PR Date is missing
if pr_col and pr_col in fil.columns:
    # Use PR date primarily, backfill with PO date for filtering check
    d_check = fil[pr_col]
    if po_create_col and po_create_col in fil.columns:
        d_check = d_check.fillna(fil[po_create_col])
    fil = fil[(d_check >= pr_start) & (d_check <= pr_end)]
elif po_create_col and po_create_col in fil.columns:
    # Fallback if PR column completely missing
    fil = fil[(fil[po_create_col] >= pr_start) & (fil[po_create_col] <= pr_end)]

# Date range filter
date_basis = pr_col if pr_col in fil.columns else (po_create_col if po_create_col in fil.columns else None)
dr = None
date_range_key = None
if date_basis:
    # compute min/max without copying
    mindt = fil[date_basis].dropna().min()
    maxdt = fil[date_basis].dropna().max()
    if pd.notna(mindt) and pd.notna(maxdt):
        dr = st.sidebar.date_input('Date range', (mindt.date(), maxdt.date()), key='date_range')
        if isinstance(dr, tuple) and len(dr) == 2:
            sdt = pd.to_datetime(dr[0]); edt = pd.to_datetime(dr[1]) + pd.Timedelta(hours=23, minutes=59, seconds=59)
            fil = fil[(fil[date_basis] >= sdt) & (fil[date_basis] <= edt)]
            date_range_key = (sdt.isoformat(), edt.isoformat())

# ensure defensive columns exist without expensive operations
for c in ['Buyer.Type', 'po_creator', 'po_vendor', 'entity', 'po_buyer_type']:
    if c not in fil.columns:
        fil[c] = ''

# Ensure PR-level Buyer.Type is tidy (reuse compute only if missing)
if 'Buyer.Type' not in fil.columns:
    fil['Buyer.Type'] = compute_buyer_type_vectorized(fil)
fil['Buyer.Type'] = fil['Buyer.Type'].fillna('Direct').astype(str).str.strip().str.title()
fil.loc[fil['Buyer.Type'].str.lower().isin(['direct', 'd']), 'Buyer.Type'] = 'Direct'
fil.loc[fil['Buyer.Type'].str.lower().isin(['indirect', 'i', 'in']), 'Buyer.Type'] = 'Indirect'
fil.loc[~fil['Buyer.Type'].isin(['Direct', 'Indirect']), 'Buyer.Type'] = 'Direct'



# Entity + PO ordered by filters (use categories for speed)
entity_choices = sorted([e for e in fil['entity'].cat.categories.tolist() if str(e).strip() != '']) if 'entity' in fil.columns and fil['entity'].dtype.name=='category' else sorted([e for e in fil['entity'].dropna().unique().tolist() if str(e).strip() != ''])
sel_e = st.sidebar.multiselect('Entity', entity_choices, default=entity_choices)

# Procurement Category filter
if 'procurement_category' in fil.columns:
    proc_cat_choices = sorted([str(x) for x in fil['procurement_category'].dropna().unique() if str(x).strip()])
    sel_pc = st.sidebar.multiselect('Procurement Category', proc_cat_choices, default=proc_cat_choices)
else:
    sel_pc = []
    proc_cat_choices = []

# PO Ordered By
creators = sorted([str(x) for x in fil.get('po_creator', pd.Series(dtype=object)).dropna().unique().tolist() if str(x).strip()!=''])
# Optimization: remove default=creators to speed up loading
sel_o = st.sidebar.multiselect('PO Ordered By', creators) 

# Buyer Type choices
choices_bt = sorted(fil['Buyer.Type'].dropna().unique().tolist())
sel_b = st.sidebar.multiselect('Buyer Type', choices_bt, default=choices_bt)

# Item Type Filter (Products vs Services)
item_type_opt = st.sidebar.radio("Item Type (Global)", ["All", "Products", "Services"], index=0)

# Vendor + Item filters
if po_vendor_col and po_vendor_col in fil.columns:
    if pd.api.types.is_categorical_dtype(fil[po_vendor_col]):
        vendor_choices = sorted(fil[po_vendor_col].cat.categories)
    else:
        vendor_choices = sorted(fil[po_vendor_col].unique())
    # Optimization: remove default=vendor_choices to speed up loading (no serialization of 1000 items)
    sel_v = st.sidebar.multiselect('Vendor (pick one or more)', vendor_choices) # default is None
else:
    sel_v = []
    vendor_choices = []

if 'product_name' in fil.columns:
    item_choices = sorted(fil['product_name'].unique())
    # Optimization: remove default=item_choices
    sel_i = st.sidebar.multiselect('Item / Product (pick one or more)', item_choices) # default is None
else:
    sel_i = []
    item_choices = []


# Apply filters using df.query() for potential performance improvement
query_parts = []
if sel_b and len(sel_b) < len(choices_bt):
    query_parts.append('`Buyer.Type` in @sel_b')
if sel_e and 'entity' in fil.columns and len(sel_e) < len(entity_choices):
    query_parts.append('entity in @sel_e')
if sel_pc and len(sel_pc) < len(proc_cat_choices):
    query_parts.append('procurement_category in @sel_pc')

if sel_o and len(sel_o) < len(creators):
    query_parts.append('po_creator in @sel_o')

if sel_v and len(sel_v) < len(vendor_choices):
    query_parts.append('po_vendor in @sel_v')

if sel_i and len(sel_i) < len(item_choices):
    query_parts.append('product_name in @sel_i')

if query_parts:
    fil = fil.query(' & '.join(query_parts))

# Apply Item Type Filter (Products vs Services)
if item_type_opt != "All" and 'Item.Type' in fil.columns:
    fil = fil[fil['Item.Type'] == item_type_opt]

filter_end_time = time.time()
logger.info(f"Filter application took: {filter_end_time - filter_start_time:.2f} seconds")


# Helper to create deterministic signature for caching
def _sel_key(values):
    return tuple(sorted(str(v) for v in values)) if values else ()
filter_signature = (
    fy_key, date_range_key, _sel_key(sel_b), _sel_key(sel_e), _sel_key(sel_pc),
    _sel_key(sel_o), _sel_key(sel_v), _sel_key(sel_i), item_type_opt
)

# Precompute month bucket once
trend_date_col = po_create_col if (po_create_col and po_create_col in fil.columns) else (pr_col if (pr_col and pr_col in fil.columns) else None)
if trend_date_col:
    fil['_month_bucket'] = fil[trend_date_col].dt.to_period('M').dt.to_timestamp()
else:
    fil['_month_bucket'] = pd.NaT

if st.sidebar.button('Reset Filters'):
    for k in list(st.session_state.keys()):
        try:
            del st.session_state[k]
        except Exception:
            pass
    st.rerun()


# ----------------- Tabs (structure preserved) -----------------
T = st.tabs(['KPIs & Spend','PR/PO Timing','PO Approval','Delivery','Vendors','Dept & Services','Unit-rate Outliers','Forecast','Savings','Scorecards','Search','Full Data', 'Geo Distribution'])

# ----------------- KPIs & Spend -----------------
with T[0]:
    st.header('P2P Dashboard — Indirect (KPIs & Spend)')
    c1,c2,c3,c4,c5 = st.columns(5)
    total_prs = int(fil.get(pr_number_col, pd.Series(dtype=object)).nunique()) if pr_number_col else 0
    total_pos = int(fil.get(purchase_doc_col, pd.Series(dtype=object)).nunique()) if purchase_doc_col else 0
    c1.metric('Total PRs', total_prs)
    c2.metric('Total POs', total_pos)
    c3.metric('Line Items', len(fil))
    c4.metric('Entities', int(fil.get('entity', pd.Series(dtype=object)).nunique()))
    spend_val = fil.get(net_amount_col, pd.Series(0)).sum() if net_amount_col else 0
    c5.metric('Spend (Cr ₹)', f"{spend_val/1e7:,.2f}")
    st.markdown('---')

    # Build monthly aggregated once
    def build_monthly():
        if not (trend_date_col and net_amount_col and net_amount_col in fil.columns):
            return pd.DataFrame()
        if 'entity' not in fil.columns:
            return pd.DataFrame()
        z = fil.loc[fil['_month_bucket'].notna(), ['_month_bucket', 'entity', net_amount_col]].copy()
        z['month'] = z['_month_bucket']
        
        # FIX: Strict FY filtering for the chart to avoid bleed into next FY
        z = z[(z['month'] >= pr_start) & (z['month'] <= pr_end)]
        
        # groupby using categorical 'entity' is fast
        return z.groupby(['month','entity'], dropna=False)[net_amount_col].sum().reset_index()

    st.subheader('Monthly Total Spend + Cumulative')
    if trend_date_col and net_amount_col and net_amount_col in fil.columns:
        me = memoized_compute('monthly_entity', filter_signature, build_monthly)
        if me.empty:
            st.info('No monthly/entity data to plot.')
        else:
            pivot = me.pivot(index='month', columns='entity', values=net_amount_col).fillna(0).sort_index()
            # ensure fixed entities first
            fixed_entities = ['MEPL','MLPL','MMW','MMPL']
            for ent in fixed_entities:
                if ent not in pivot.columns:
                    pivot[ent] = 0.0
            other_entities = [c for c in pivot.columns if c not in fixed_entities]
            ordered_entities = [e for e in fixed_entities if e in pivot.columns] + other_entities
            pivot = pivot[ordered_entities]

            pivot_cr = pivot / 1e7
            total_cr = pivot_cr.sum(axis=1)
            cum_cr = total_cr.cumsum()

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            xaxis_labels = pivot_cr.index.strftime('%b-%Y')
            colors = {'MEPL':'#1f77b4','MLPL':'#ff7f0e','MMW':'#2ca02c','MMPL':'#d62728'}
            for ent in ordered_entities:
                ent_vals = pivot_cr[ent].values
                text_vals = [f"{v:.2f}" if v > 0 else '' for v in ent_vals]
                fig.add_trace(go.Bar(x=xaxis_labels, y=ent_vals, name=ent, marker_color=colors.get(ent, None), text=text_vals, textposition='inside', hovertemplate='%{x}<br>'+ent+': %{y:.2f} Cr<extra></extra>'), secondary_y=False)
            highlight_color = '#FFD700'
            fig.add_trace(go.Scatter(x=xaxis_labels, y=cum_cr.values, mode='lines+markers+text',
                name='Cumulative (Cr)', line=dict(color=highlight_color, width=3),
                marker=dict(color=highlight_color, size=6), text=[f"{int(round(v, 0))}" for v in cum_cr.values],
                textposition='top center', textfont=dict(color=highlight_color, size=9),
                hovertemplate='%{x}<br>Cumulative: %{y:.2f} Cr<extra></extra>'),
                secondary_y=True)

            # Add total labels on top of each bar
            fig.add_trace(go.Scatter(
                x=xaxis_labels,
                y=total_cr,
                mode='text',
                text=[f'{v:.2f}' for v in total_cr],
                textposition='top center',
                showlegend=False,
                hovertemplate=None,
                hoverinfo='none'
            ), secondary_y=False)

            fig.update_layout(barmode='stack', xaxis_tickangle=-45, title='Monthly Spend (stacked by Entity) + Cumulative',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
            fig.update_yaxes(title_text='Monthly Spend (Cr)', secondary_y=False)
            fig.update_yaxes(title_text='Cumulative (Cr)', secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('Monthly Spend not available — need date and Net Amount columns.')

    st.markdown('---')
    st.subheader('Entity Trend')
    try:
        if trend_date_col and net_amount_col and net_amount_col in fil.columns and 'entity' in fil.columns:
            g = memoized_compute('monthly_entity', filter_signature, build_monthly)
            if not g.empty:
                fig_e = px.line(g, x=g['month'].dt.strftime('%b-%Y'), y=net_amount_col, color='entity', labels={net_amount_col:'Net Amount','x':'Month'})
                fig_e.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_e, use_container_width=True)
    except Exception as e:
        st.error(f'Could not render Entity Trend: {e}')

    # --- Procurement Category spend (new) ---
    st.markdown('---')
    st.subheader('Spend by Procurement Category')
    if 'procurement_category' in fil.columns and net_amount_col and net_amount_col in fil.columns:
        # build function for memoization
        def build_proc_cat_spend():
            pc = fil.groupby('procurement_category', dropna=False)[net_amount_col].sum().reset_index().sort_values(net_amount_col, ascending=False)
            pc['cr'] = pc[net_amount_col] / 1e7
            return pc
        pc_spend = memoized_compute('proc_cat_spend', filter_signature, build_proc_cat_spend)
        fig_pc = px.bar(pc_spend, x='procurement_category', y='cr', text='cr', title='Procurement Category Spend (Cr)')
        fig_pc.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_pc.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_pc, use_container_width=True)
    else:
        st.info('Procurement Category or Net Amount column not found — cannot show Procurement Category spend.')


    st.markdown('---')
    st.subheader('Buyer-wise Spend (Cr)')
    if 'buyer_display' in fil.columns and net_amount_col in fil.columns:
        def build_buyer_spend():
            grp = fil.groupby('buyer_display')[net_amount_col].sum().reset_index()
            grp['cr'] = grp[net_amount_col] / 1e7
            return grp.sort_values('cr', ascending=False)
        buyer_spend = memoized_compute('buyer_spend', filter_signature, build_buyer_spend)
        fig_buyer = px.bar(buyer_spend, x='buyer_display', y='cr', text='cr', title='Buyer-wise Spend (Cr)')
        fig_buyer.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_buyer.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_buyer, use_container_width=True)
        st.dataframe(buyer_spend, use_container_width=True)

        # Buyer trend (optimized grouping)
        try:
            if trend_date_col and net_amount_col and net_amount_col in fil.columns:
                def build_buyer_trend():
                    bt = fil.loc[fil['_month_bucket'].notna(), ['_month_bucket','buyer_display', net_amount_col]].copy()
                    bt['month'] = bt['_month_bucket']
                    return bt.groupby(['month','buyer_display'], dropna=False)[net_amount_col].sum().reset_index()
                bt_grouped = memoized_compute('buyer_trend', filter_signature, build_buyer_trend)
                if bt_grouped.empty:
                    st.info('No buyer trend data for the current filters.')
                else:
                    top_buyers = buyer_spend['buyer_display'].head(5).astype(str).tolist()
                    pick_mode = st.selectbox('Buyer trend: show', ['Top 5 by Spend', 'Choose buyers'], index=0)
                    if pick_mode == 'Choose buyers':
                        chosen = st.multiselect('Pick buyers to show on trend', sorted(buyer_spend['buyer_display'].astype(str).unique().tolist()), default=top_buyers)
                    else:
                        chosen = top_buyers
                    if chosen:
                        g_b = bt_grouped[bt_grouped['buyer_display'].isin(chosen)].copy()
                        if not g_b.empty:
                            # pivot by month faster than groupby in long loops
                            g_b['month'] = g_b['month'].dt.to_period('M').dt.to_timestamp()
                            full_range = pd.period_range(g_b['month'].min().to_period('M'), g_b['month'].max().to_period('M'), freq='M').to_timestamp()
                            pivot = (g_b.pivot_table(index='month', columns='buyer_display', values=net_amount_col, aggfunc='sum')
                                .reindex(full_range, fill_value=0)
                                .rename_axis('month')
                                .reset_index())
                            trend_long = pivot.melt(id_vars='month', var_name='buyer_display', value_name='value')

                            rolling_window = st.slider('Smooth buyer trend (months)', 1, 6, 1, key='buyer_trend_smooth')
                            if rolling_window > 1:
                                trend_long['value'] = (trend_long.sort_values(['buyer_display','month'])
                                    .groupby('buyer_display')['value']
                                    .transform(lambda s: s.rolling(rolling_window, min_periods=1).mean()))
                            trend_long = trend_long[trend_long['buyer_display'].isin(chosen)]

                            fig_b_trend = px.line(trend_long, x='month', y='value', color='buyer_display',
                                labels={'value':'Net Amount','month':'Month','buyer_display':'Buyer'}, title='Buyer-wise Monthly Trend')
                            fig_b_trend.update_layout(xaxis_tickformat='%b-%Y', hovermode='x unified', legend_title_text='Buyer')
                            fig_b_trend.update_traces(mode='lines+markers')
                            st.plotly_chart(fig_b_trend, use_container_width=True)
                        else:
                            st.info('No buyer trend rows for the selected buyers.')
        except Exception as e:
            st.error(f'Could not render Buyer Trend: {e}')
    else:
        st.info('Buyer display or Net Amount column missing — cannot compute buyer-wise spend.')

# ----------------- PR/PO Timing & Open PRs -----------------
with T[1]:
    st.subheader('PR/PO Timing')
    if pr_col and po_create_col and pr_col in fil.columns and po_create_col in fil.columns:
        def build_lead_df():
            lead = fil.loc[fil[pr_col].notna() & fil[po_create_col].notna(), [pr_col, po_create_col, 'Buyer.Type', 'po_creator']].copy()
            lead['Lead Time (Days)'] = (pd.to_datetime(lead[po_create_col]) - pd.to_datetime(lead[pr_col])).dt.days
            return lead
        lead_df = memoized_compute('lead_df', filter_signature, build_lead_df)

        SLA_DAYS = 7
        avg_lead = float(lead_df['Lead Time (Days)'].mean().round(1)) if not lead_df.empty else 0.0
        gauge_fig = go.Figure(go.Indicator(mode='gauge+number', value=avg_lead,
            number={'suffix':' days'},
            gauge={'axis':{'range':[0, max(14, avg_lead * 1.2 if avg_lead else 14)]},
                   'bar':{'color':'darkblue'},
                   'steps':[{'range':[0,SLA_DAYS],'color':'lightgreen'},{'range':[SLA_DAYS,max(14, avg_lead * 1.2 if avg_lead else 14)],'color':'lightcoral'}],
                   'threshold':{'line':{'color':'red','width':4}, 'value':SLA_DAYS}}))
        st.plotly_chart(gauge_fig, use_container_width=True)
        st.caption(f"Current Avg Lead Time: {avg_lead:.1f} days • Target ≤ {SLA_DAYS} days")

        st.subheader('⏱️ PR to PO Lead Time by Buyer Type & by Buyer')
        lead_avg_by_type = lead_df.groupby('Buyer.Type')['Lead Time (Days)'].mean().round(0).reset_index().rename(columns={'Buyer.Type':'Buyer Type'}) if 'Buyer.Type' in lead_df.columns else pd.DataFrame()
        lead_avg_by_buyer = lead_df.groupby('po_creator')['Lead Time (Days)'].mean().round(0).reset_index().rename(columns={'po_creator':'PO.Creator'}) if 'po_creator' in lead_df.columns else pd.DataFrame()
        c1,c2 = st.columns(2)
        c1.dataframe(lead_avg_by_type, use_container_width=True)
        c2.dataframe(lead_avg_by_buyer, use_container_width=True)

        st.subheader('📅 Monthly PR & PO Trends')
        tmp = fil
        tmp['PR Month'] = pd.to_datetime(tmp[pr_col], errors='coerce').dt.to_period('M') if pr_col in tmp.columns else pd.NaT
        tmp['PO Month'] = pd.to_datetime(tmp[po_create_col], errors='coerce').dt.to_period('M') if po_create_col in tmp.columns else pd.NaT

        pr_col_name = pr_number_col if pr_number_col else None
        po_col_name = purchase_doc_col if purchase_doc_col else None

        if pr_col_name and po_col_name and pr_col_name in tmp.columns:
            monthly_summary = tmp.groupby('PR Month').agg({pr_col_name: 'count', po_col_name: 'count'}).reset_index()
            monthly_summary.columns = ['Month', 'PR Count', 'PO Count']
            monthly_summary['Month'] = monthly_summary['Month'].astype(str)
            if not monthly_summary.empty:
                st.line_chart(monthly_summary.set_index('Month'), use_container_width=True)
        else:
            st.info('PR Number or Purchase Doc column missing — cannot show monthly PR/PO trend.')



        # Open PRs
        st.subheader("⚠️ Open PRs (Approved/InReview)")
        pr_status_col = safe_col(df, ['pr_status','pr status','status','prstatus','pr_status'])
        pr_date_col = pr_col if pr_col in df.columns else safe_col(df, ['pr_date_submitted','pr date submitted','pr_date','pr date'])
        pr_number_col_local = pr_number_col if pr_number_col in df.columns else safe_col(df, ['pr_number','pr number','pr_no','pr no'])

        if pr_status_col and pr_status_col in df.columns:
            def prepare_open_source(source_df: pd.DataFrame) -> pd.DataFrame:
                base = source_df.copy()
                if 'Buyer.Type' not in base.columns:
                    base['Buyer.Type'] = compute_buyer_type_vectorized(base)
                base['Buyer.Type'] = base['Buyer.Type'].fillna('Direct').astype(str).str.strip().str.title()
                base.loc[base['Buyer.Type'].str.lower().isin(['direct', 'd']), 'Buyer.Type'] = 'Direct'
                base.loc[base['Buyer.Type'].str.lower().isin(['indirect', 'i', 'in']), 'Buyer.Type'] = 'Indirect'
                base.loc[~base['Buyer.Type'].isin(['Direct','Indirect']), 'Buyer.Type'] = 'Direct'
                if sel_b:
                    base = base[base['Buyer.Type'].isin(sel_b)]
                return base

            using_global = False
            scoped_df = prepare_open_source(fil)
            open_df = scoped_df[scoped_df[pr_status_col].astype(str).isin(["Approved", "InReview"])].copy()
            if open_df.empty:
                global_df = prepare_open_source(df)
                open_df = global_df[global_df[pr_status_col].astype(str).isin(["Approved", "InReview"])].copy()
                if not open_df.empty:
                    using_global = True

            if open_df.empty:
                st.warning('⚠️ No open PRs match the current filters.')
            else:
                if using_global:
                    st.info('No filtered Open PRs were found — showing all Open PRs after applying only the Buyer Type selection.')
                # pending age
                if pr_date_col and pr_date_col in open_df.columns:
                    open_df["Pending Age (Days)"] = (pd.to_datetime(pd.Timestamp.today().date()) - pd.to_datetime(open_df[pr_date_col], errors='coerce')).dt.days
                else:
                    open_df["Pending Age (Days)"] = np.nan

                # aggregation
                agg_map = {}
                if pr_date_col and pr_date_col in open_df.columns:
                    agg_map[pr_date_col] = 'first'
                agg_map['Pending Age (Days)'] = 'first'
                pc_col = safe_col(open_df, ['procurement_category','procurement category','procurement_category'])
                if pc_col: agg_map[pc_col] = 'first'
                pn_col = safe_col(open_df, ['product_name','product name','productname'])
                if pn_col: agg_map[pn_col] = 'first'
                if net_amount_col and net_amount_col in open_df.columns:
                    agg_map[net_amount_col] = 'sum'
                pcode_col = safe_col(open_df, ['po_budget_code','po budget code','pr_budget_code','pr budget code'])
                if pcode_col: agg_map[pcode_col] = 'first'
                agg_map[pr_status_col] = 'first'
                bg_col = safe_col(open_df, ['buyer_group','buyer group','buyer_group'])
                if bg_col: agg_map[bg_col] = 'first'
                bt_col = safe_col(open_df, ['Buyer.Type','buyer_type','buyer.type'])
                if bt_col: agg_map[bt_col] = 'first'
                if 'entity' in open_df.columns: agg_map['entity'] = 'first'
                if 'po_creator' in open_df.columns: agg_map['po_creator'] = 'first'
                if purchase_doc_col and purchase_doc_col in open_df.columns:
                    agg_map[purchase_doc_col] = 'first'

                group_by_col = pr_number_col_local if pr_number_col_local and pr_number_col_local in open_df.columns else None

                if group_by_col:
                    open_summary = open_df.groupby(group_by_col).agg(agg_map).reset_index()
                else:
                    open_summary = open_df.reset_index().groupby('_row_id' if '_row_id' in open_df.columns else open_df.index.name or 'index').agg(agg_map).reset_index()

                open_summary = open_summary.drop(columns=['buyer_type','effective_buyer_type'], errors='ignore')
                st.metric("🔢 Open PRs", open_summary.shape[0])

                # highlight
                def highlight_age(val):
                    try:
                        return 'background-color: red' if float(val) > 30 else ''
                    except Exception:
                        return ''
                try:
                    styled = open_summary.copy()
                    rename_map = {}
                    if group_by_col: rename_map[group_by_col] = 'PR Number'
                    if pr_date_col and pr_date_col in styled.columns: rename_map[pr_date_col] = 'PR Date Submitted'
                    if net_amount_col and net_amount_col in styled.columns: rename_map[net_amount_col] = 'Net Amount'
                    styled = styled.rename(columns=rename_map)

                    highlight_cols = [c for c in styled.columns if 'Pending Age' in c or c == 'Pending Age (Days)']
                    if highlight_cols:
                        st.dataframe(styled.style.applymap(highlight_age, subset=highlight_cols), use_container_width=True)
                    else:
                        st.dataframe(styled, use_container_width=True)
                except Exception:
                    st.dataframe(open_summary, use_container_width=True)

                try:
                    csv_out = open_summary.to_csv(index=False)
                    st.download_button('⬇️ Download Open PRs (CSV)', csv_out, file_name='open_prs_summary.csv', mime='text/csv')
                except Exception:
                    pass

                st.subheader('🏢 Open PRs by Entity')
                if 'entity' in open_summary.columns:
                    ent_counts = open_summary['entity'].value_counts().reset_index()
                    ent_counts.columns = ['Entity','Count']
                    st.bar_chart(ent_counts.set_index('Entity'), use_container_width=True)
                else:
                    st.info('Entity column not found in Open PRs summary.')
        else:
            st.info("ℹ️ 'PR Status' column not found.")
    else:
        st.info('Need both PR Date and PO Create Date columns to compute SLA and lead times.')


# ---- Defensive PO Approval details (final stable version) ----
with T[2]:
    st.subheader("PO Approval Details")
    po_create = safe_col(df, ['po_create_date', 'po create date'])
    po_approved = safe_col(df, ['po_approved_date', 'po approved date'])

    def build_po_app_df():
        # Requires at least po_create; if po_approved missing, we can still show pending counts
        if not (po_create and po_create in fil.columns):
            return pd.DataFrame()
        
        cols = ['po_creator', purchase_doc_col, po_create]
        if po_approved and po_approved in fil.columns:
            cols.append(po_approved)
        if net_amount_col and net_amount_col in fil.columns:
            cols.append(net_amount_col)
        
        # Add Vendor and Product Name for display
        if po_vendor_col and po_vendor_col in fil.columns:
            cols.append(po_vendor_col)
        if 'product_name' in fil.columns:
            cols.append('product_name')
            
        p_df = fil[[c for c in cols if c in fil.columns]].copy()
        
        if po_approved and po_approved in p_df.columns:
            p_df['is_approved'] = p_df[po_approved].notna()
            p_df['approval_lead_time'] = (p_df[po_approved] - p_df[po_create]).dt.days
        else:
            p_df['is_approved'] = False
            p_df['approval_lead_time'] = np.nan
            
        return p_df

    po_app_df = memoized_compute('po_approval', filter_signature, build_po_app_df)

    if po_app_df.empty:
        st.info("PO Approval columns not found (need PO Create Date).")
    else:
        # Metrics: Count unique POs
        unique_pos_df = po_app_df.drop_duplicates(subset=[purchase_doc_col]) if purchase_doc_col in po_app_df.columns else po_app_df
        
        total_unique_pos = len(unique_pos_df)
        approved_count = unique_pos_df['is_approved'].sum()
        pending_count = total_unique_pos - approved_count
        
        # Pending Value (remains sum of line items for accuracy)
        pending_val = 0.0
        if net_amount_col and net_amount_col in po_app_df.columns:
            pending_val = po_app_df.loc[~po_app_df['is_approved'], net_amount_col].sum()

        avg_approval_time = unique_pos_df['approval_lead_time'].mean()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Avg Approval Time (Days)", f"{avg_approval_time:.1f}")
        m2.metric("Approved POs", int(approved_count))
        m3.metric("Pending POs", int(pending_count))
        m4.metric("Pending PO Value (Cr)", f"{pending_val/1e7:,.2f}")
        
        st.markdown('---')
        
        # New: Buyer-wise Pending Count & List
        st.subheader("Buyer-wise Pending Approvals")
        
        pending_df = po_app_df[~po_app_df['is_approved']].copy()
        
        if not pending_df.empty:
            # Aggregate to get unique POs with summed Net Amount and combined Product Names
            agg_dict = {}
            if 'po_creator' in pending_df.columns: agg_dict['po_creator'] = 'first'
            if po_create and po_create in pending_df.columns: agg_dict[po_create] = 'first'
            if po_vendor_col and po_vendor_col in pending_df.columns: agg_dict[po_vendor_col] = 'first'
            if net_amount_col and net_amount_col in pending_df.columns: agg_dict[net_amount_col] = 'sum'
            if 'product_name' in pending_df.columns: 
                agg_dict['product_name'] = lambda x: ', '.join(sorted(set(str(i) for i in x.dropna().unique() if str(i).strip() != '')))

            if purchase_doc_col and purchase_doc_col in pending_df.columns:
                unique_pending = pending_df.groupby(purchase_doc_col, as_index=False).agg(agg_dict)
            else:
                unique_pending = pending_df # Fallback
                
            # Group by Buyer
            buyer_pending_counts = unique_pending.groupby('po_creator').size().reset_index(name='Pending PO Count')
            buyer_pending_counts = buyer_pending_counts.sort_values('Pending PO Count', ascending=False)
            
            c_p1, c_p2 = st.columns(2)
            c_p1.write("**Pending Count by Buyer**")
            c_p1.dataframe(buyer_pending_counts, use_container_width=True)
            
            c_p2.write("**Pending PO List**")
            
            # Buyer Selection Dropdown
            buyers_list = sorted([str(x) for x in unique_pending['po_creator'].unique().tolist() if pd.notna(x) and str(x).strip() != ''])
            selected_buyer_pending = c_p2.selectbox("Filter by Buyer", ['All'] + buyers_list)
            
            # Filter Data
            if selected_buyer_pending != 'All':
                filtered_pending = unique_pending[unique_pending['po_creator'] == selected_buyer_pending].copy()
            else:
                filtered_pending = unique_pending.copy()
            
            # Calculate Age
            if po_create and po_create in filtered_pending.columns:
                filtered_pending['Age (Days)'] = (pd.Timestamp.now() - filtered_pending[po_create]).dt.days
            else:
                filtered_pending['Age (Days)'] = np.nan

            # Select relevant columns for the list
            # Requested: PO Number, Vendor Name, Product Name, Net Amount, Age
            display_cols = []
            rename_map = {}
            
            if purchase_doc_col and purchase_doc_col in filtered_pending.columns:
                display_cols.append(purchase_doc_col)
                rename_map[purchase_doc_col] = 'PO Number'
                
            if po_vendor_col and po_vendor_col in filtered_pending.columns:
                display_cols.append(po_vendor_col)
                rename_map[po_vendor_col] = 'Vendor Name'
            
            if 'product_name' in filtered_pending.columns:
                display_cols.append('product_name')
                rename_map['product_name'] = 'Product Name'
                
            if net_amount_col and net_amount_col in filtered_pending.columns:
                display_cols.append(net_amount_col)
                rename_map[net_amount_col] = 'Net Amount'
                
            display_cols.append('Age (Days)')
            
            # Ensure po_creator is available for sorting if not in display
            sort_col = 'po_creator' if 'po_creator' in filtered_pending.columns else display_cols[0]
            
            final_df = filtered_pending[display_cols].rename(columns=rename_map)
            
            c_p2.dataframe(final_df, use_container_width=True)
        else:
            st.success("No pending approvals found!")

        st.markdown('---')
        st.subheader("All Approval Details")
        
        desired = ['po_creator', purchase_doc_col, po_create, po_approved, 'approval_lead_time', 'is_approved', net_amount_col]
        show_cols = [c for c in desired if c and c in po_app_df.columns]
        if show_cols:
            po_detail = po_app_df[show_cols].copy()
            if 'approval_lead_time' in po_detail.columns:
                po_detail = po_detail.sort_values('approval_lead_time', ascending=False)
            if purchase_doc_col and purchase_doc_col in po_detail.columns:
                po_detail = po_detail.drop_duplicates(subset=[purchase_doc_col], keep='first')
            st.dataframe(po_detail, use_container_width=True)


# ----------------- Delivery -----------------
with T[3]:
    st.subheader('Delivery Summary')
    dv = fil
    po_qty_col = safe_col(dv, ['po_qty','po quantity','po_quantity','po qty'])
    received_col = safe_col(dv, ['receivedqty','received_qty','received qty','received_qty'])

    if po_qty_col and received_col and po_qty_col in dv.columns and received_col in dv.columns:
        def build_delivery():
            # Include Net Amount for Open Value calc
            cols = [po_qty_col, received_col, purchase_doc_col, po_vendor_col]
            if net_amount_col and net_amount_col in dv.columns:
                cols.append(net_amount_col)
                
            tmp = dv[[c for c in cols if c in dv.columns]].copy()
            tmp['po_qty_f'] = tmp[po_qty_col].fillna(0).astype(float)
            tmp['received_f'] = tmp[received_col].fillna(0).astype(float)
            
            if net_amount_col and net_amount_col in tmp.columns:
                tmp['net_val'] = tmp[net_amount_col].fillna(0).astype(float)
            else:
                tmp['net_val'] = 0.0

            # Group by PO
            # Aggregation: Sum Qty, Sum Received, Sum Net Amount (assuming net amount is line level)
            agg_rules = {'po_qty_f':'sum', 'received_f':'sum', 'net_val':'sum'}
            
            grp = tmp.groupby([purchase_doc_col, po_vendor_col], dropna=False).agg(agg_rules).reset_index()
            
            # Derived metrics
            grp['pct_received'] = np.where(grp['po_qty_f']>0, grp['received_f']/grp['po_qty_f']*100, 0)
            grp['is_open'] = grp['received_f'] < grp['po_qty_f']
            grp['is_partial'] = (grp['received_f'] > 0) & (grp['received_f'] < grp['po_qty_f'])
            
            # Open Value: Pro-rata? Or full line value if not fully received? 
            # Usually "Open PO Value" means the value of the goods NOT yet received.
            # Approximation: net_val * (1 - received/qty)
            ratio = np.where(grp['po_qty_f']>0, grp['received_f']/grp['po_qty_f'], 1.0)
            ratio = np.clip(ratio, 0, 1) # Ensure bounded
            grp['open_val'] = grp['net_val'] * (1 - ratio)
            
            return grp
            
        ag = memoized_compute('delivery_summary', filter_signature, build_delivery)
        
        # Metrics using unique POs
        open_pos = ag[ag['is_open']]
        closed_pos = ag[~ag['is_open']]
        
        cnt_open = open_pos[purchase_doc_col].nunique()
        cnt_closed = closed_pos[purchase_doc_col].nunique()
        val_open = open_pos['open_val'].sum()
        
        d1, d2, d3 = st.columns(3)
        d1.metric("Open POs", cnt_open)
        d2.metric("Closed POs", cnt_closed)
        d3.metric("Open PO Value (Cr)", f"{val_open/1e7:,.2f}")
        
        st.markdown("---")
        
        # Lists
        st.subheader("Open POs (Received < Ordered)")
        st.dataframe(open_pos.sort_values('open_val', ascending=False).head(500), use_container_width=True)
        
        st.subheader("Partial Delivery POs (0 < Received < Ordered)")
        partial_pos = ag[ag['is_partial']]
        st.dataframe(partial_pos.sort_values('open_val', ascending=False).head(500), use_container_width=True)
        
    else:
        st.info('Delivery columns (PO Qty / Received QTY) not found.')


# ----------------- Vendors -----------------
with T[4]:
    st.subheader('Vendor Insights & Service Buckets')
    
    if po_vendor_col and net_amount_col and po_vendor_col in fil.columns and net_amount_col in fil.columns:
        # Top level metrics
        total_spend = fil[net_amount_col].sum() / 1e7
        total_vendors = fil[po_vendor_col].nunique()
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Vendors", total_vendors)
        c2.metric("Total Spend (Cr)", f"{total_spend:.2f}")

        # ----------------- New: Buyer-wise Vendor Portfolio -----------------
        st.markdown("### 0. Buyer-wise Vendor Portfolio")
        
        # We need a context-aware dataframe that ignores Sidebar Buyer/Vendor filters 
        # but respects FY, Date Range, Entity, Category.
        # Construct df_context by applying base filters to df (raw processed data).
        
        df_context = df.copy()
        
        # Apply Time Filters
        if pr_col and pr_col in df_context.columns:
            d_check_ctx = df_context[pr_col]
            if po_create_col and po_create_col in df_context.columns:
                d_check_ctx = d_check_ctx.fillna(df_context[po_create_col])
            df_context = df_context[(d_check_ctx >= pr_start) & (d_check_ctx <= pr_end)]
        elif po_create_col and po_create_col in df_context.columns:
            df_context = df_context[(df_context[po_create_col] >= pr_start) & (df_context[po_create_col] <= pr_end)]
            
        if date_basis and date_range_key:
             # date_range_key is (sdt_iso, edt_iso)
             # we can re-use sdt, edt from global scope if available, or parse date_range_key
             # Actually, simpler to reuse the 'dr' logic if sdt/edt variables are available in scope.
             # They are defined in the sidebar section.
             try:
                 if 'sdt' in locals() and 'edt' in locals():
                     df_context = df_context[(df_context[date_basis] >= sdt) & (df_context[date_basis] <= edt)]
             except:
                 pass

        # Apply Entity & Category Filters (Sidebar logic replicated)
        if sel_e and 'entity' in df_context.columns and len(sel_e) < len(entity_choices):
            df_context = df_context[df_context['entity'].isin(sel_e)]
        if sel_pc and 'procurement_category' in df_context.columns and len(sel_pc) < len(proc_cat_choices):
            df_context = df_context[df_context['procurement_category'].isin(sel_pc)]
        if sel_b and 'Buyer.Type' in df_context.columns and len(sel_b) < len(choices_bt):
            df_context = df_context[df_context['Buyer.Type'].isin(sel_b)]

        # Get list of buyers from this context
        ctx_buyers = sorted([str(x) for x in df_context['po_creator'].dropna().unique().tolist() if str(x).strip() != ''])
        sel_buyer_portfolio = st.selectbox("Select Buyer to View Assigned Vendors", ['All'] + ctx_buyers)
        
        if sel_buyer_portfolio != 'All':
            # 1. Identify vendors this buyer has used
            buyer_specific_df = df_context[df_context['po_creator'] == sel_buyer_portfolio]
            my_vendor_list = buyer_specific_df[po_vendor_col].unique().tolist()
            
            # 2. Get ALL transactions for these vendors (to see other buyers)
            # This is the key requirement: "show multiple users for the vendors"
            portfolio_df = df_context[df_context[po_vendor_col].isin(my_vendor_list)].copy()
            
            if not portfolio_df.empty:
                # Aggregate
                port_agg = portfolio_df.groupby(po_vendor_col).agg(
                    Spend=(net_amount_col, 'sum'),
                    PO_Count=(purchase_doc_col, 'nunique') if purchase_doc_col in portfolio_df.columns else ('entity', 'count'),
                    Buyer_Count=('po_creator', 'nunique'),
                    Assigned_Buyers=('po_creator', lambda x: ', '.join(sorted(set(str(i) for i in x.dropna().unique() if str(i).strip() != ''))))
                ).reset_index()
                
                port_agg['Spend (Cr)'] = port_agg['Spend'] / 1e7
                port_agg = port_agg.sort_values('Spend (Cr)', ascending=False)
                
                # Merge with master data for contact info
                if not vendor_master.empty:
                    port_agg['VendorName_Norm'] = port_agg[po_vendor_col].astype(str).str.lower().str.strip()
                    vm_unique_p = vendor_master.sort_values('Entity').drop_duplicates(subset=['VendorName_Norm'], keep='first')
                    port_agg = pd.merge(port_agg, vm_unique_p[['VendorName_Norm', 'City', 'State', 'Phone']], 
                                      on='VendorName_Norm', how='left')
                    port_agg = port_agg.drop(columns=['VendorName_Norm'])
                
                st.write(f"Vendors associated with **{sel_buyer_portfolio}** (and other buyers interactions):")
                
                # Reorder columns
                cols_order = [po_vendor_col, 'Spend (Cr)', 'PO_Count', 'Buyer_Count', 'Assigned_Buyers']
                if 'City' in port_agg.columns: cols_order += ['City', 'State', 'Phone']
                
                st.dataframe(port_agg[cols_order], use_container_width=True)
            else:
                st.warning("No vendor transactions found for this buyer in the selected timeframe.")
        else:
            st.info("Select a buyer to see their vendor portfolio and cross-buyer collaboration.")

        st.markdown("---")
        
        # 1. Category / Service Bucket Drilldown
        st.markdown("### 1. Service Buckets (Categories) & Entity-wise Count")
        
        # Split layout for two charts
        col_cat, col_ent = st.columns(2)
        
        # Aggregate by Category
        if 'procurement_category' in fil.columns:
            cat_df = fil.groupby('procurement_category', dropna=False).agg(
                Spend=(net_amount_col, 'sum'),
                VendorCount=(po_vendor_col, 'nunique')
            ).reset_index().sort_values('Spend', ascending=False)
            cat_df['Spend (Cr)'] = cat_df['Spend'] / 1e7
            
            # Chart 1: Spend by Category
            fig_cat = px.bar(cat_df, x='procurement_category', y='Spend (Cr)', 
                             hover_data=['VendorCount'], text='Spend (Cr)',
                             title='Spend by Category')
            fig_cat.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            with col_cat:
                st.plotly_chart(fig_cat, use_container_width=True)
            
            # Selector
            cats = ['All'] + sorted(cat_df['procurement_category'].dropna().astype(str).unique().tolist())
            sel_cat = st.selectbox("Select Category to Filter Vendors", cats)
        else:
            sel_cat = 'All'
            with col_cat:
                st.info("Procurement Category not available.")

        # Chart 2: Vendor Count by Entity
        if 'entity' in fil.columns and po_vendor_col in fil.columns:
            ent_v_count = fil.groupby('entity')[po_vendor_col].nunique().reset_index()
            ent_v_count.columns = ['Entity', 'Vendor Count']
            if not ent_v_count.empty:
                fig_ent_count = px.pie(ent_v_count, values='Vendor Count', names='Entity', 
                                     title='Vendor Count by Entity', hole=0.4)
                with col_ent:
                    st.plotly_chart(fig_ent_count, use_container_width=True)

        # Filter for Vendor section
        v_df = fil.copy()
        if sel_cat != 'All' and 'procurement_category' in v_df.columns:
            v_df = v_df[v_df['procurement_category'].astype(str) == sel_cat]
            
        st.markdown("---")
        st.markdown(f"### 2. Vendors in '{sel_cat}'")
        
        # Vendor list in this context
        v_stats = v_df.groupby(po_vendor_col).agg(
            Spend=(net_amount_col, 'sum'),
            PO_Count=(purchase_doc_col, 'nunique') if purchase_doc_col in v_df.columns else ('entity', 'count'),
            Buyers=('po_creator', lambda x: ', '.join(sorted(set(str(i) for i in x.dropna().unique() if str(i).strip() != ''))))
        ).reset_index()
        v_stats['Spend (Cr)'] = v_stats['Spend'] / 1e7
        v_stats = v_stats.sort_values('Spend (Cr)', ascending=False)

        # Merge with master data to show enriched table
        if not vendor_master.empty:
            # We use normalized matching or left join on name if possible
            # But the vendor names in transaction data may vary slightly.
            # We'll try a naive merge on Vendor Name first.
            vm_unique = vendor_master.sort_values('Entity').drop_duplicates(subset=['VendorName_Norm'], keep='first')
            
            # Prepare join column
            v_stats['VendorName_Norm'] = v_stats[po_vendor_col].astype(str).str.lower().str.strip()
            
            # Join
            v_enriched = pd.merge(v_stats, vm_unique[['VendorName_Norm', 'Email', 'Phone', 'City', 'State']], 
                                  on='VendorName_Norm', how='left')
            
            # Display enriched table instead of just list
            st.markdown("#### Vendor List (Sorted by Spend)")
            display_cols = [po_vendor_col, 'Buyers', 'Spend (Cr)', 'PO_Count', 'City', 'State', 'Phone', 'Email']
            st.dataframe(v_enriched[display_cols], use_container_width=True)
            
            # Use enriched list for selection
            v_list = v_stats[po_vendor_col].astype(str).tolist()
        else:
            v_list = v_stats[po_vendor_col].astype(str).tolist()
            st.dataframe(v_stats[[po_vendor_col, 'Buyers', 'Spend (Cr)', 'PO_Count']], use_container_width=True)

        
        # Vendor Selector
        sel_vendor = st.selectbox("Select Vendor to View Details", v_list)
        
        if sel_vendor:
            # Vendor Detail View
            sub = v_df[v_df[po_vendor_col].astype(str) == sel_vendor]
            
            # Metrics for this vendor
            v_spend = sub[net_amount_col].sum() / 1e7
            v_pos = sub[purchase_doc_col].nunique() if purchase_doc_col in sub.columns else 0
            
            # Vendor Contact Details
            st.markdown("#### Vendor Contact & Info")
            
            found_contact = False
            if not vendor_master.empty:
                # Normalize selected vendor
                sel_norm = str(sel_vendor).lower().strip()
                match = vendor_master[vendor_master['VendorName_Norm'] == sel_norm]
                
                if not match.empty:
                    found_contact = True
                    # There might be multiple matches (across entities). Show unique ones.
                    
                    # Deduplicate by Code
                    u_matches = match.drop_duplicates(subset=['VendorCode'])
                    
                    for idx, row in u_matches.iterrows():
                        with st.expander(f"Contact Details ({row['Entity']}) - {row['VendorCode']}", expanded=True):
                            cd1, cd2 = st.columns(2)
                            cd1.write(f"**Phone:** {row['Phone'] if row['Phone'] else 'N/A'}")
                            cd1.write(f"**Email:** {row['Email'] if row['Email'] else 'N/A'}")
                            cd2.write(f"**Address:** {row['Address'] if row['Address'] else 'N/A'}")
                            cd2.write(f"**City/State:** {row['City'] if row['City'] else ''} {row['State'] if row['State'] else ''}")
            
            if not found_contact:
                st.caption("No contact details found in vendor master files.")
                
            st.markdown("---")

            vc1, vc2 = st.columns(2)
            vc1.metric(f"Spend: {sel_vendor}", f"{v_spend:.2f} Cr")
            vc2.metric("PO Count", v_pos)

            # Entity Breakdown Chart (New)
            if 'entity' in sub.columns:
                ent_breakdown = sub.groupby('entity')[net_amount_col].sum().reset_index()
                ent_breakdown['Spend (Cr)'] = ent_breakdown[net_amount_col] / 1e7
                if not ent_breakdown.empty:
                    fig_ent = px.pie(ent_breakdown, values='Spend (Cr)', names='entity', 
                                     title=f"Spend Breakdown by Entity: {sel_vendor}",
                                     hole=0.4)
                    st.plotly_chart(fig_ent, use_container_width=True)
            
            # Items / Services Table
            st.markdown("**Items / Services Provided:**")
            if 'product_name' in sub.columns:
                # Group by product name to see frequency or spend per item
                items = sub.groupby('product_name').agg(
                    Count=(purchase_doc_col, 'count'),
                    Total_Spend=(net_amount_col, 'sum')
                ).reset_index().sort_values('Total_Spend', ascending=False)
                items['Total_Spend'] = items['Total_Spend'].apply(lambda x: f"{x:,.2f}")
                st.dataframe(items, use_container_width=True)
            else:
                st.dataframe(sub.head(50))

            # --- START VPM SECTION ---
            st.markdown("---")
            st.markdown("### 4. Vendor Performance Management (VPM)")
            
            vpm1, vpm2 = st.columns(2)
            
            # A. Fulfillment Reliability (Fill Rate)
            # Detect local column names for safety
            qty_col_vpm = safe_col(sub, ['po_qty','po quantity','po_quantity','po qty'])
            rcv_col_vpm = safe_col(sub, ['receivedqty','received_qty','received qty','received_qty'])
            
            if qty_col_vpm and rcv_col_vpm and qty_col_vpm in sub.columns and rcv_col_vpm in sub.columns:
                sub_qty = pd.to_numeric(sub[qty_col_vpm], errors='coerce').fillna(0)
                sub_rcv = pd.to_numeric(sub[rcv_col_vpm], errors='coerce').fillna(0)
                
                total_ord = sub_qty.sum()
                total_rcv = sub_rcv.sum()
                
                fill_rate = (total_rcv / total_ord * 100) if total_ord > 0 else 0.0
                
                # Gauge chart
                fig_fill = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = fill_rate,
                    title = {'text': "Volume Fill Rate (%)"},
                    gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "green" if fill_rate >= 90 else "orange"}}
                ))
                fig_fill.update_layout(height=250, margin=dict(l=20,r=20,t=40,b=20))
                with vpm1:
                    st.plotly_chart(fig_fill, use_container_width=True)
            else:
                with vpm1:
                    st.info("PO Qty / Received Qty columns not found — cannot calculate Fill Rate.")

            # B. Promised Lead Time
            # We use po_delivery_date (Expected) - po_create_date
            del_date_col = safe_col(sub, ['po_delivery_date', 'po delivery date'])
            if po_create_col and del_date_col and del_date_col in sub.columns and po_create_col in sub.columns:
                sub_dates = sub[[po_create_col, del_date_col]].dropna().copy()
                if not sub_dates.empty:
                    sub_dates['lead_days'] = (sub_dates[del_date_col] - sub_dates[po_create_col]).dt.days
                    avg_lead_days = sub_dates['lead_days'].mean()
                    
                    with vpm2:
                        st.metric("Avg Promised Lead Time", f"{avg_lead_days:.1f} Days")
                        # Histogram of lead times
                        fig_lead = px.histogram(sub_dates, x='lead_days', nbins=20, title='Lead Time Distribution (Promised)', labels={'lead_days':'Days'})
                        fig_lead.update_layout(height=200, margin=dict(l=20,r=20,t=40,b=20))
                        st.plotly_chart(fig_lead, use_container_width=True)
                else:
                    with vpm2:
                        st.info("No valid dates for Lead Time calc.")
            else:
                 with vpm2:
                    st.info("PO Delivery Date / Create Date missing.")
            
            # C. Price Stability (Top Products)
            st.markdown("**Price Stability (Top 5 Products by Spend)**")
            if po_unit_rate_col and po_unit_rate_col in sub.columns and 'product_name' in sub.columns:
                 # Filter valid rates
                 sub_rate = sub[sub[po_unit_rate_col] > 0].copy()
                 if not sub_rate.empty:
                     stats = sub_rate.groupby('product_name').agg(
                         Total_Spend=(net_amount_col, 'sum') if net_amount_col else (po_unit_rate_col, 'count'),
                         Avg_Rate=(po_unit_rate_col, 'mean'),
                         Min_Rate=(po_unit_rate_col, 'min'),
                         Max_Rate=(po_unit_rate_col, 'max'),
                         Std_Dev=(po_unit_rate_col, 'std'),
                         Txn_Count=(po_unit_rate_col, 'count')
                     ).reset_index()
                     
                     # Filter for relevant products (at least 2 txns to calc stability)
                     stats = stats[stats['Txn_Count'] > 1].sort_values('Total_Spend', ascending=False).head(5)
                     
                     if not stats.empty:
                         stats['CV (%)'] = (stats['Std_Dev'] / stats['Avg_Rate'] * 100).fillna(0).round(1)
                         stats['Avg_Rate'] = stats['Avg_Rate'].map('{:,.2f}'.format)
                         stats['Min_Rate'] = stats['Min_Rate'].map('{:,.2f}'.format)
                         stats['Max_Rate'] = stats['Max_Rate'].map('{:,.2f}'.format)
                         
                         st.dataframe(stats[['product_name', 'Txn_Count', 'Avg_Rate', 'Min_Rate', 'Max_Rate', 'CV (%)']], use_container_width=True)
                         st.caption("*CV (Coefficient of Variation) indicates price volatility. Lower is better.*")
                     else:
                         st.caption("Not enough repeated transactions to calculate price stability.")
                 else:
                     st.caption("No positive unit rates found.")
            else:
                 st.info("Unit Rate or Product Name column missing.")

        st.markdown("---")
        st.markdown("### 3. Reverse Lookup (Service -> Vendors)")
        # Simple text search for products to find vendors
        srv_query = st.text_input("Enter Service/Item keyword (e.g. 'Laptop', 'Housekeeping')", "")
        if srv_query and 'product_name' in fil.columns:
            mask = fil['product_name'].astype(str).str.lower().str.contains(srv_query.lower())
            found = fil[mask]
            if not found.empty:
                # Group by Vendor
                v_found = found.groupby(po_vendor_col).agg(
                    Spend=(net_amount_col, 'sum'),
                    Matches=('product_name', 'count'),
                    Buyers=('po_creator', lambda x: ', '.join(sorted(set(str(i) for i in x.dropna().unique() if str(i).strip() != ''))))
                ).reset_index().sort_values('Spend', ascending=False)
                v_found['Spend'] = v_found['Spend'].apply(lambda x: f"{x/1e7:.4f} Cr")
                
                # Merge contact details if available
                if not vendor_master.empty:
                    # Prepare join column
                    v_found['VendorName_Norm'] = v_found[po_vendor_col].astype(str).str.lower().str.strip()
                    
                    # Deduplicate master data
                    vm_unique = vendor_master.sort_values('Entity').drop_duplicates(subset=['VendorName_Norm'], keep='first')
                    
                    # Join
                    v_found = pd.merge(v_found, vm_unique[['VendorName_Norm', 'Email', 'Phone', 'City', 'State']], 
                                      on='VendorName_Norm', how='left')
                    
                    # Cleanup for display
                    v_found = v_found.drop(columns=['VendorName_Norm'])
                
                st.write(f"Vendors supplying '{srv_query}':")
                st.dataframe(v_found, use_container_width=True)
            else:
                st.warning("No matches found.")
                
    else:
        st.info('Vendor / Net Amount columns not present.')


# ----------------- Dept & Services -----------------
with T[5]:
    st.subheader('Dept & Services — PR Budget perspective')
    dept_df = fil
    # replace expensive apply with column-wise bfill
    dept_cols = [c for c in [pr_bu_col, pr_budget_desc_col, po_bu_col, po_budget_desc_col, pr_budget_code_col] if c]
    if dept_cols:
        # prepare as strings, replace empty with NaN then backfill
        dept_df_local = dept_df[dept_cols].astype(str).replace({'': np.nan}).bfill(axis=1).iloc[:, 0].fillna('Unmapped / Missing')
        dept_df = dept_df.copy()
        dept_df['pr_department_unified'] = dept_df_local
    else:
        dept_df = dept_df.copy()
        dept_df['pr_department_unified'] = 'Unmapped / Missing'

    def build_dept_df():
        return dept_df
    dept_df = memoized_compute('dept_df', filter_signature, build_dept_df)

    if pr_budget_desc_col and pr_budget_desc_col in dept_df.columns and net_amount_col and net_amount_col in dept_df.columns:
        def build_desc():
            df_ = dept_df.groupby(pr_budget_desc_col, dropna=False)[net_amount_col].sum().reset_index().sort_values(net_amount_col, ascending=False)
            df_['cr'] = df_[net_amount_col]/1e7
            return df_
        agg_desc = memoized_compute('dept_desc', filter_signature, build_desc)
        top_desc = agg_desc.head(30)
        if not top_desc.empty:
            fig_desc = px.bar(top_desc, x=pr_budget_desc_col, y='cr', title='PR Budget Description Spend (Top 30)', labels={pr_budget_desc_col: 'PR Budget Description', 'cr':'Cr'}, text='cr')
            fig_desc.update_traces(texttemplate='%{text:.2f}', textposition='outside'); fig_desc.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_desc, use_container_width=True)

            pick_desc = st.selectbox('Drill into PR Budget Description', ['-- none --'] + top_desc[pr_budget_desc_col].astype(str).tolist())
            if pick_desc and pick_desc != '-- none --':
                sub = dept_df[dept_df[pr_budget_desc_col].astype(str) == pick_desc].copy()
                show_cols = [c for c in [pr_number_col, purchase_doc_col, pr_budget_code_col, pr_budget_desc_col, net_amount_col, po_vendor_col] if c in sub.columns]
                st.dataframe(sub[show_cols].sort_values(net_amount_col, ascending=False).head(500), use_container_width=True)
    else:
        st.info('PR Budget description or Net Amount column not found to show PR Budget Description spend.')

    st.markdown('---')
    if pr_budget_code_col and pr_budget_code_col in dept_df.columns and net_amount_col and net_amount_col in dept_df.columns:
        def build_code():
            df_ = dept_df.groupby(pr_budget_code_col, dropna=False)[net_amount_col].sum().reset_index().sort_values(net_amount_col, ascending=False)
            df_['cr'] = df_[net_amount_col]/1e7
            return df_
        agg_code = memoized_compute('dept_code', filter_signature, build_code)
        top_code = agg_code.head(30)
        if not top_code.empty:
            fig_code = px.bar(top_code, x=pr_budget_code_col, y='cr', title='PR Budget Code Spend (Top 30)', labels={pr_budget_code_col: 'PR Budget Code', 'cr':'Cr'}, text='cr')
            fig_code.update_traces(texttemplate='%{text:.2f}', textposition='outside'); fig_code.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_code, use_container_width=True)

            pick_code = st.selectbox('Drill into PR Budget Code', ['-- none --'] + top_code[pr_budget_code_col].astype(str).tolist())
            if pick_code and pick_code != '-- none --':
                sub2 = dept_df[dept_df[pr_budget_code_col].astype(str) == pick_code].copy()
                show_cols2 = [c for c in [pr_number_col, purchase_doc_col, pr_budget_code_col, pr_budget_desc_col, net_amount_col, po_vendor_col] if c in sub2.columns]
                st.dataframe(sub2[show_cols2].sort_values(net_amount_col, ascending=False).head(500), use_container_width=True)
    else:
        st.info('PR Budget code or Net Amount column not found to show PR Budget Code spend.')


# ----------------- Unit-rate Outliers -----------------
with T[6]:
    st.subheader('Unit-rate Outliers vs Historical Median')
    grp_candidates = [c for c in ['product_name','item_code','product name','item code'] if c in fil.columns]
    grp_by = st.selectbox('Group by', grp_candidates) if grp_candidates else None
    if grp_by and po_unit_rate_col and po_unit_rate_col in fil.columns:
        cols_needed = [grp_by, po_unit_rate_col, purchase_doc_col, pr_number_col, po_vendor_col, 'item_description', po_create_col, net_amount_col]
        available_cols = [c for c in cols_needed if c in fil.columns]
        def build_unit_base():
            z = fil[available_cols].dropna(subset=[grp_by, po_unit_rate_col]).copy()
            med = z.groupby(grp_by)[po_unit_rate_col].median().rename('median_rate')
            z = z.join(med, on=grp_by)
            z['pctdev'] = (z[po_unit_rate_col] - z['median_rate']) / z['median_rate'].replace(0, np.nan)
            return z
        z = memoized_compute('unit_outlier', filter_signature + (grp_by,), build_unit_base)
        thr = st.slider('Outlier threshold (±%)', 10, 300, 50, 5)
        out = z[abs(z['pctdev']) >= thr/100.0].copy()
        out['pctdev%'] = (out['pctdev']*100).round(1)
        st.dataframe(out.sort_values('pctdev%', ascending=False), use_container_width=True)

# ----------------- Forecast -----------------
with T[7]:
    st.subheader('Forecast Next Month Spend (SMA)')
    if trend_date_col and net_amount_col and net_amount_col in fil.columns:
        def build_monthly_total():
            t = fil.loc[fil['_month_bucket'].notna(), ['_month_bucket', net_amount_col]].copy()
            t['month'] = t['_month_bucket']
            return t.groupby('month')[net_amount_col].sum().sort_index()
        m = memoized_compute('monthly_total', filter_signature, build_monthly_total)
        m_cr = m/1e7
        k = st.slider('Window (months)', 3, 12, 6)
        sma = m_cr.rolling(k).mean()
        mu = m_cr.tail(k).mean() if len(m_cr) >= k else m_cr.mean()
        sd = m_cr.tail(k).std(ddof=1) if len(m_cr) >= k else m_cr.std(ddof=1)
        n = min(k, max(1, len(m_cr)))
        se = sd/np.sqrt(n) if sd==sd else 0
        lo, hi = float(mu-1.96*se), float(mu+1.96*se)
        nxt = (m_cr.index.max() + pd.offsets.MonthBegin(1)) if len(m_cr) > 0 else pd.Timestamp.now().to_period('M').to_timestamp()
        fdf = pd.DataFrame({
            'Month': list(m_cr.index) + [nxt],
            'SpendCr': list(m_cr.values) + [np.nan],
            'SMA': list(sma.values) + [mu]
        })
        fig = go.Figure();
        fig.add_bar(x=fdf['Month'], y=fdf['SpendCr'], name='Actual (Cr)');
        fig.add_scatter(x=fdf['Month'], y=fdf['SMA'], mode='lines+markers', name=f'SMA{k}')
        st.plotly_chart(fig, use_container_width=True)

# ----------------- Savings -----------------
with T[8]:
    st.subheader('Savings — PR → PO')
    # detect PR/PO rate/value/quantity columns
    pr_qty_col = safe_col(fil, ['pr_quantity','pr qty','pr_quantity','pr quantity','pr quantity','pr_quantity'])
    pr_unit_rate_col = safe_col(fil, ['unit_rate','pr_unit_rate','pr unit rate','pr_unit_rate'])
    pr_value_col = safe_col(fil, ['pr_value','pr value','pr_value'])
    po_qty_col = safe_col(fil, ['po_quantity','po qty','po_quantity','po quantity'])
    po_unit_rate_col = safe_col(fil, ['po_unit_rate','po unit rate','po_unit_rate'])
    net_col = safe_col(fil, ['net_amount','net amount','net_amount_inr','net_amount'])

    if (pr_qty_col or pr_unit_rate_col or pr_value_col) and (po_qty_col or po_unit_rate_col or net_col):
        try:
            def build_savings():
                z = fil.copy()
                # ensure any categorical columns used are converted to safe types first
                for col in [pr_qty_col, pr_unit_rate_col, pr_value_col, po_qty_col, po_unit_rate_col, net_col]:
                    if col and col in z.columns and pd.api.types.is_categorical_dtype(z[col]):
                        z[col] = z[col].astype(object)

                # compute PR line value: prefer PR Value if present else PR Qty * PR Unit Rate
                if pr_value_col and pr_value_col in z.columns:
                    z['pr_line_value'] = pd.to_numeric(z[pr_value_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0.0)
                else: # safe multiplication with defaults
                    pr_q = pd.to_numeric(z.get(pr_qty_col, 0).astype(str).str.replace(',', ''), errors='coerce').fillna(0.0)
                    pr_r = pd.to_numeric(z.get(pr_unit_rate_col, 0).astype(str).str.replace(',', ''), errors='coerce').fillna(0.0)
                    z['pr_line_value'] = pr_q * pr_r

                # compute PO line net: prefer Net Amount if present else PO Qty * PO Unit Rate
                if net_col and net_col in z.columns:
                    z['po_line_value'] = pd.to_numeric(z[net_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0.0)
                else:
                    po_q = pd.to_numeric(z.get(po_qty_col, 0).astype(str).str.replace(',', ''), errors='coerce').fillna(0.0)
                    po_r = pd.to_numeric(z.get(po_unit_rate_col, 0).astype(str).str.replace(',', ''), errors='coerce').fillna(0.0)
                    z['po_line_value'] = po_q * po_r

                # unit rates numeric where available
                z['pr_unit_rate_f'] = pd.to_numeric(z.get(pr_unit_rate_col, np.nan), errors='coerce')
                z['po_unit_rate_f'] = pd.to_numeric(z.get(po_unit_rate_col, np.nan), errors='coerce')

                # savings absolute and percent (use pr_line_value as denominator when >0)
                z['savings_abs'] = z['pr_line_value'] - z['po_line_value']
                z['savings_pct'] = np.where(z['pr_line_value'] > 0, (z['savings_abs'] / z['pr_line_value']) * 100.0, np.nan)
                # per-unit pct if unit rates present
                z['unit_rate_pct_saved'] = np.where(
                    z['pr_unit_rate_f'] > 0,
                    (z['pr_unit_rate_f'] - z['po_unit_rate_f']) / z['pr_unit_rate_f'] * 100.0,
                    np.nan
                )
                # select display columns (only those that exist)
                disp_cols = [
                    pr_number_col, purchase_doc_col, pr_qty_col, pr_unit_rate_col, pr_value_col,
                    po_qty_col, po_unit_rate_col, net_col, 'pr_line_value', 'po_line_value', 'savings_abs',
                    'savings_pct', 'unit_rate_pct_saved', 'po_vendor', 'buyer_display', 'entity', 'procurement_category'
                ]
                disp_cols = [c for c in disp_cols if c and c in z.columns]
                return z[disp_cols].copy()

            # compute and memoize
            savings_df = memoized_compute('savings', filter_signature, build_savings)
            if savings_df.empty:
                st.info('No matching PR/PO rows found to compute savings.')
            else:
                # KPIs
                total_pr_value = float(savings_df['pr_line_value'].sum())
                total_po_value = float(savings_df['po_line_value'].sum())
                total_savings = total_pr_value - total_po_value
                pct_saved_overall = (total_savings / total_pr_value * 100.0) if total_pr_value > 0 else np.nan

                k1,k2,k3,k4 = st.columns(4)
                k1.metric('Total PR Value (Cr)', f"{total_pr_value/1e7:,.2f}")
                k2.metric('Total PO Value (Cr)', f"{total_po_value/1e7:,.2f}")
                k3.metric('Total Savings (Cr)', f"{total_savings/1e7:,.2f}")
                k4.metric('% Saved vs PR', f"{pct_saved_overall:.2f}%" if pct_saved_overall==pct_saved_overall else 'N/A')
                st.markdown('---')

                # Histogram % saved
                st.subheader('Distribution of % Saved (per line)')
                fig_hist = px.histogram(savings_df, x='savings_pct', nbins=50, title='% Saved per Line (PR→PO)', labels={'savings_pct':'% Saved'})
                st.plotly_chart(fig_hist, use_container_width=True)

                # Top savings by absolute value
                st.subheader('Top Savings — Absolute (Cr)')
                top_abs = savings_df.sort_values('savings_abs', ascending=False).head(20).copy()
                top_abs['savings_cr'] = top_abs['savings_abs']/1e7
                x_axis_for_top = 'po_vendor' if 'po_vendor' in top_abs.columns else purchase_doc_col
                fig_top_abs = px.bar(top_abs, x=x_axis_for_top, y='savings_cr', hover_data=['pr_line_value','po_line_value'], title='Top 20 Savings by Absolute Value (Cr)')
                st.plotly_chart(fig_top_abs, use_container_width=True)

                # Category level
                st.subheader('Savings by Procurement Category')
                if 'procurement_category' in savings_df.columns:
                    pc = savings_df.groupby('procurement_category', dropna=False)[['pr_line_value','po_line_value','savings_abs']].sum().reset_index()
                    pc['savings_cr'] = pc['savings_abs']/1e7
                    pc['pct_saved'] = np.where(pc['pr_line_value']>0, pc['savings_abs']/pc['pr_line_value']*100.0, np.nan)
                    fig_pc = px.bar(pc.sort_values('savings_cr', ascending=False), x='procurement_category', y='savings_cr', text='pct_saved', title='Procurement Category — Savings (Cr)')
                    fig_pc.update_traces(texttemplate='%{text:.2f}%')
                    fig_pc.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_pc, use_container_width=True)
                else:
                    st.info('Procurement Category not available for category breakdown.')

                # PR unit vs PO unit scatter
                if 'pr_unit_rate_f' in savings_df.columns and 'po_unit_rate_f' in savings_df.columns:
                    st.subheader('PR Unit Rate vs PO Unit Rate (scatter)')
                    sc = savings_df.dropna(subset=['pr_unit_rate_f','po_unit_rate_f']).copy()
                    fig_sc = px.scatter(sc, x='pr_unit_rate_f', y='po_unit_rate_f', size='pr_line_value',
                        hover_data=[pr_number_col, purchase_doc_col, 'po_vendor'],
                        title='PR Unit Rate vs PO Unit Rate')
                    st.plotly_chart(fig_sc, use_container_width=True)

                st.markdown('---')
                st.subheader('Detailed Savings List')
                st.dataframe(savings_df.sort_values('savings_abs', ascending=False).reset_index(drop=True), use_container_width=True)
                try:
                    st.download_button('⬇️ Download Savings CSV', savings_df.to_csv(index=False), file_name='savings_detail.csv', mime='text/csv')
                except Exception:
                    pass
        except Exception as ex:
            st.error('Error while computing Savings — full traceback below:')
            st.text(traceback.format_exc())
    else:
        st.info('Required PR/PO quantity/unit/value columns or Net Amount not present — cannot compute savings.')


# ----------------- Vendor Scorecard -----------------
with T[9]:
    st.subheader('Vendor Scorecard')
    if po_vendor_col and po_vendor_col in fil.columns:
        vendor = st.selectbox('Pick Vendor', sorted(fil[po_vendor_col].dropna().astype(str).unique().tolist()))
        vd = fil[fil[po_vendor_col].astype(str) == str(vendor)].copy()
        spend = vd.get(net_amount_col, pd.Series(0)).sum()/1e7 if net_amount_col else 0
        upos = int(vd.get(purchase_doc_col, pd.Series(dtype=object)).nunique()) if purchase_doc_col else 0
        k1,k2 = st.columns(2); k1.metric('Spend (Cr)', f"{spend:.2f}"); k2.metric('Unique POs', upos)
        st.dataframe(vd.head(200), use_container_width=True)

# ----------------- Search -----------------
with T[10]:
    st.subheader('🔍 Keyword Search')
    search_df = df # search on processed data
    valid_cols = [c for c in [pr_number_col, purchase_doc_col, 'product_name', po_vendor_col] if c in search_df.columns]
    query = st.text_input('Type vendor, product, PO, PR, etc.', '')
    cat_sel = st.multiselect('Filter by Procurement Category',
        sorted(search_df.get('procurement_category', pd.Series(dtype=object)).dropna().astype(str).unique().tolist())) if 'procurement_category' in search_df.columns else []
    vend_sel = st.multiselect('Filter by Vendor',
        sorted(search_df.get(po_vendor_col, pd.Series(dtype=object)).dropna().astype(str).unique().tolist())) if po_vendor_col in search_df.columns else []

    if query and valid_cols:
        q = query.lower()
        masks = []
        for c in valid_cols:
            masks.append(search_df[c].astype(str).str.lower().str.contains(q, na=False))
        mask_any = np.logical_or.reduce(masks) if masks else pd.Series(False, index=search_df.index)
        res = search_df[mask_any].copy()
        if cat_sel:
            res = res[res['procurement_category'].astype(str).isin(cat_sel)]
        if vend_sel and po_vendor_col in df.columns:
            res = res[res[po_vendor_col].astype(str).isin(vend_sel)]
        st.write(f'Found {len(res)} rows')
        st.dataframe(res, use_container_width=True)
        try:
            st.download_button('⬇️ Download Search Results', res.to_csv(index=False), file_name='search_results.csv', mime='text/csv')
        except Exception:
            pass
    else:
        st.caption('Start typing to search…')


# ----------------- Full Data -----------------
with T[11]:
    st.subheader('Full Data — all filtered rows')
    try:
        st.dataframe(fil.reset_index(drop=True), use_container_width=True)
        csv = convert_df_to_csv(fil)
        st.download_button('⬇️ Download full filtered data (CSV)', csv, file_name='p2p_full_filtered.csv', mime='text/csv')
    except Exception as e:
        st.error(f'Could not display full data: {e}')

# ----------------- Geo Distribution -----------------
with T[12]:
    st.subheader('Geo Distribution of Processed Orders (India)')
    
    # Needs vendor master and filtered data
    if po_vendor_col and po_vendor_col in fil.columns and not vendor_master.empty:
        # 1. Prepare Transaction Data
        # We need unique POs per vendor + spend info for drilldown
        
        # Select columns: Vendor, PO Number, Net Amount, Product
        cols_to_keep_geo = [po_vendor_col, purchase_doc_col]
        if net_amount_col and net_amount_col in fil.columns:
            cols_to_keep_geo.append(net_amount_col)
        if 'product_name' in fil.columns:
            cols_to_keep_geo.append('product_name')
            
        df_geo_base = fil[cols_to_keep_geo].copy()
        
        if purchase_doc_col not in df_geo_base.columns:
             # Fallback if no PO col, use row count (less accurate for "Orders Processed" but safe)
             df_geo_base['dummy_po'] = df_geo_base.index
             po_col_geo = 'dummy_po'
        else:
             po_col_geo = purchase_doc_col
             
        df_geo_base['VendorName_Norm'] = df_geo_base[po_vendor_col].astype(str).str.lower().str.strip()
        
        # 2. Prepare Vendor Master (Deduplicate to get one State per vendor)
        # We prioritize the master entries. If duplicates, pick first.
        # Ensure State is present
        if 'State' in vendor_master.columns:
            vm_geo = vendor_master[vendor_master['State'].notna()].copy()
            vm_geo['VendorName_Norm'] = vm_geo['VendorName_Norm'].astype(str).str.lower().str.strip()
            # Deduplicate
            vm_geo = vm_geo.drop_duplicates(subset=['VendorName_Norm'], keep='first')
            
            # 3. Merge (Keep City if available)
            cols_vm_geo = ['VendorName_Norm', 'State']
            if 'City' in vm_geo.columns:
                cols_vm_geo.append('City')
                
            merged_geo = pd.merge(df_geo_base, vm_geo[cols_vm_geo], on='VendorName_Norm', how='inner')
            
            if not merged_geo.empty:
                # 4. Aggregation by State
                # Clean State Names for GeoJSON matching
                # Common issue: "Maharashtra" vs "MAHARASHTRA" -> Title Case
                merged_geo['State_Clean'] = merged_geo['State'].astype(str).str.strip().str.title()
                
                # Fix common mismatches for India GeoJSON
                state_corrections = {
                    'Delhi': 'NCT of Delhi',
                    'New Delhi': 'NCT of Delhi',
                    'Telengana': 'Telangana',
                    'Orissa': 'Odisha',
                    'Andaman And Nicobar Islands': 'Andaman & Nicobar Island',
                    'J&K': 'Jammu & Kashmir',
                    # Add more as discovered
                }
                merged_geo['State_Clean'] = merged_geo['State_Clean'].replace(state_corrections)

                geo_stats = merged_geo.groupby('State_Clean')[po_col_geo].nunique().reset_index()
                geo_stats.columns = ['State', 'PO_Count']
                
                total_pos_geo = geo_stats['PO_Count'].sum()
                geo_stats['Percentage'] = (geo_stats['PO_Count'] / total_pos_geo * 100)
                
                # 5. Plot
                st.markdown(f"**Total Mapped POs:** {total_pos_geo}")
                
                # Public GeoJSON for India
                geojson_url = "https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson"
                
                # Coordinates for State Labels
                STATE_COORDS = {
                    "Andhra Pradesh": (15.91, 79.74), "Arunachal Pradesh": (28.21, 94.72), "Assam": (26.20, 92.93),
                    "Bihar": (25.09, 85.31), "Chhattisgarh": (21.27, 81.86), "Goa": (15.29, 74.12),
                    "Gujarat": (22.25, 71.19), "Haryana": (29.05, 76.08), "Himachal Pradesh": (31.10, 77.17),
                    "Jharkhand": (23.61, 85.27), "Karnataka": (15.31, 75.71), "Kerala": (10.85, 76.27),
                    "Madhya Pradesh": (22.97, 78.65), "Maharashtra": (19.75, 75.71), "Manipur": (24.66, 93.90),
                    "Meghalaya": (25.46, 91.36), "Mizoram": (23.16, 92.93), "Nagaland": (26.15, 94.56),
                    "Odisha": (20.95, 85.09), "Punjab": (31.14, 75.34), "Rajasthan": (27.02, 74.21),
                    "Sikkim": (27.53, 88.51), "Tamil Nadu": (11.12, 78.65), "Telangana": (18.11, 79.01),
                    "Tripura": (23.94, 91.98), "Uttar Pradesh": (26.84, 80.94), "Uttarakhand": (30.06, 79.01),
                    "West Bengal": (22.98, 87.85), "Andaman & Nicobar Island": (11.74, 92.65),
                    "Chandigarh": (30.73, 76.77), "Dadra and Nagar Haveli and Daman and Diu": (20.18, 73.01),
                    "NCT of Delhi": (28.70, 77.10), "Jammu & Kashmir": (33.77, 76.57),
                    "Ladakh": (34.15, 77.57), "Lakshadweep": (10.56, 72.64), "Puducherry": (11.94, 79.80)
                }

                # Prepare labels for map
                geo_stats['lat'] = geo_stats['State'].map(lambda x: STATE_COORDS.get(x, (None, None))[0])
                geo_stats['lon'] = geo_stats['State'].map(lambda x: STATE_COORDS.get(x, (None, None))[1])
                geo_stats['label'] = geo_stats.apply(lambda row: f"{row['PO_Count']}\n({row['Percentage']:.1f}%)", axis=1)

                # Filter labels to avoid clutter (e.g. only > 1% or significant count)
                # Keep all but use dynamic text color/style or simple threshold
                
                try:
                    # Base Choropleth
                    fig_map = px.choropleth(
                        geo_stats,
                        geojson=geojson_url,
                        featureidkey='properties.ST_NM',
                        locations='State',
                        color='PO_Count',
                        color_continuous_scale='Reds',
                        hover_data=['Percentage', 'PO_Count'],
                        title='PO Count by Vendor State (Heatmap)'
                    )
                    
                    # Add Text Labels - Improved for Visibility
                    # Only show labels with some significance to reduce overlapping
                    df_labels = geo_stats.dropna(subset=['lat', 'lon']).copy()
                    
                    # Heuristic: Filter overlap for very small percentages if clustered?
                    # For now, just render them with a clearer font/background
                    if not df_labels.empty:
                        fig_map.add_trace(go.Scattergeo(
                            lon=df_labels['lon'],
                            lat=df_labels['lat'],
                            text=df_labels['label'],
                            mode='text',
                            textfont=dict(color='black', size=12, family='Arial'),
                            textposition='middle center',
                            showlegend=False
                        ))

                    fig_map.update_geos(fitbounds="locations", visible=False)
                    fig_map.update_traces(hovertemplate='<b>%{location}</b><br>PO Count: %{z}<br>Percentage: %{customdata[0]:.2f}%<extra></extra>')
                    fig_map.update_layout(height=600, margin={"r":0,"t":30,"l":0,"b":0})
                    
                    st.plotly_chart(fig_map, use_container_width=True)
                    
                    # --- DRILL DOWN SECTION ---
                    st.markdown("---")
                    st.subheader("State-wise Detailed Insights")
                    
                    # Selector
                    sorted_states = geo_stats.sort_values('PO_Count', ascending=False)['State'].tolist()
                    sel_state = st.selectbox("Select State for Details", sorted_states)
                    
                    if sel_state:
                        # Filter merged data for this state
                        state_df = merged_geo[merged_geo['State_Clean'] == sel_state].copy()
                        
                        if not state_df.empty:
                            # State Metrics
                            s_pos = state_df[po_col_geo].nunique()
                            s_vendors = state_df[po_vendor_col].nunique()
                            s_spend = state_df[net_amount_col].sum()/1e7 if net_amount_col and net_amount_col in state_df.columns else 0.0
                            
                            m1, m2, m3 = st.columns(3)
                            m1.metric("PO Count", s_pos)
                            m2.metric("Active Vendors", s_vendors)
                            m3.metric("Total Spend (Cr)", f"{s_spend:.2f}")
                            
                            st.markdown(f"#### Deep Dive: {sel_state}")
                            
                            col_v, col_p, col_c = st.columns(3)
                            
                            # 1. Top Vendors
                            v_grp = state_df.groupby(po_vendor_col).agg(
                                POs=(po_col_geo, 'nunique'),
                                Spend=(net_amount_col, 'sum') if net_amount_col and net_amount_col in state_df.columns else (po_col_geo, 'count')
                            ).reset_index().sort_values('Spend', ascending=False).head(20)
                            
                            if net_amount_col and net_amount_col in state_df.columns:
                                v_grp['Spend (Cr)'] = (v_grp['Spend']/1e7).map('{:,.2f}'.format)
                                v_show = v_grp[[po_vendor_col, 'POs', 'Spend (Cr)']]
                            else:
                                v_show = v_grp[[po_vendor_col, 'POs']]
                                
                            col_v.write("**Top Vendors**")
                            col_v.dataframe(v_show, use_container_width=True, hide_index=True)
                            
                            # 2. Top Products
                            if 'product_name' in state_df.columns:
                                p_grp = state_df.groupby('product_name').agg(
                                    POs=(po_col_geo, 'nunique'),
                                    Spend=(net_amount_col, 'sum') if net_amount_col and net_amount_col in state_df.columns else (po_col_geo, 'count')
                                ).reset_index().sort_values('Spend', ascending=False).head(20)
                                
                                if net_amount_col and net_amount_col in state_df.columns:
                                    p_grp['Spend (Cr)'] = (p_grp['Spend']/1e7).map('{:,.2f}'.format)
                                    p_show = p_grp[['product_name', 'POs', 'Spend (Cr)']]
                                else:
                                    p_show = p_grp[['product_name', 'POs']]
                                    
                                col_p.write("**Top Products**")
                                col_p.dataframe(p_show, use_container_width=True, hide_index=True)
                            else:
                                col_p.info("Product Name column missing.")
                                
                            # 3. Top Cities (if available)
                            if 'City' in state_df.columns:
                                c_grp = state_df.groupby('City').agg(
                                    POs=(po_col_geo, 'nunique'),
                                    Vendors=(po_vendor_col, 'nunique'),
                                    Spend=(net_amount_col, 'sum') if net_amount_col and net_amount_col in state_df.columns else (po_col_geo, 'count')
                                ).reset_index().sort_values('Spend', ascending=False).head(20)
                                
                                if net_amount_col and net_amount_col in state_df.columns:
                                    c_grp['Spend (Cr)'] = (c_grp['Spend']/1e7).map('{:,.2f}'.format)
                                    c_show = c_grp[['City', 'Vendors', 'Spend (Cr)']]
                                else:
                                    c_show = c_grp[['City', 'Vendors', 'POs']]
                                
                                col_c.write("**Top Cities**")
                                col_c.dataframe(c_show, use_container_width=True, hide_index=True)
                            else:
                                col_c.info("City info not available.")
                        else:
                            st.info("No data for selected state.")

                except Exception as e:
                    st.error(f"Error rendering map: {e}")
            else:
                st.warning("No matched vendor locations found. Ensure Vendor Master is loaded and has 'State' info.")
        else:
            st.warning("Vendor Master file does not have a 'State' column.")
    else:
        st.info("Vendor Master data missing or no transactions available.")

# EOF
