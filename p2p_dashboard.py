# p2p_dashboard_indirect_final.py
# Restored stable + optimized filtering using index maps for fast multi-filter intersections.

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from plotly.subplots import make_subplots

DATA_DIR = Path(__file__).resolve().parent
RAW_FILES = [("MEPL.xlsx", "MEPL"), ("MLPL.xlsx", "MLPL"), ("mmw.xlsx", "MMW"), ("mmpl.xlsx", "MMPL")]
LOGO_PATH = DATA_DIR / "matter_logo.png"
INDIRECT_BUYERS = {
    'Aatish', 'Deepak', 'Deepakex', 'Dhruv', 'Dilip', 'Mukul', 'Nayan', 'Paurik', 'Kamlesh', 'Suresh', 'Priyam'
}

st.set_page_config(page_title="P2P Dashboard — Indirect (Final)", layout="wide", initial_sidebar_state="expanded")

# ----------------- Helpers -----------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    new = {}
    for c in df.columns:
        s = str(c).strip()
        s = s.replace(chr(160), " ")
        s = s.replace(chr(92), "_").replace('/', '_')
        s = "_".join(s.split())
        s = s.lower()
        s = ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in s)
        s = '_'.join([p for p in s.split('_') if p != ''])
        new[c] = s
    return df.rename(columns=new)

def safe_col(df, candidates, default=None):
    for c in candidates:
        if c in df.columns:
            return c
    return default

def memoized_compute(namespace: str, signature: tuple, compute_fn):
    """Simple session-state memoization keyed by the active filter tuple."""
    store = st.session_state.setdefault('_memo_cache', {})
    key = (namespace, signature)
    if key not in store:
        store[key] = compute_fn()
    return store[key]

def compute_buyer_type_vectorized(df: pd.DataFrame) -> pd.Series:
    """Classify PRs into Direct/Indirect using Buyer Group + numeric code."""
    if df.empty:
        return pd.Series(dtype=object)
    group_col = safe_col(df, ['buyer_group', 'Buyer Group', 'buyer group'])
    if not group_col:
        return pd.Series('Indirect', index=df.index, dtype=object)
    bg_raw = df[group_col].fillna('').astype(str).str.strip()
    code_series = pd.to_numeric(bg_raw.str.extract(r'(\d+)')[0], errors='coerce')
    buyer_type = pd.Series('Direct', index=df.index, dtype=object)
    alias_direct = bg_raw.str.upper().isin({'ME_BG17', 'MLBG16'})
    buyer_type[alias_direct] = 'Direct'
    not_available = bg_raw.eq('') | bg_raw.str.lower().isin(['not available', 'na', 'n/a'])
    buyer_type[not_available] = 'Indirect'
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
    buyer_display = np.where(has_po & (po_creator != ''), po_creator, '')
    buyer_display = np.where((buyer_display == '') & (requester != ''), requester, buyer_display)
    buyer_display = np.where(buyer_display == '', 'PR only - Unassigned', buyer_display)
    return pd.Series(buyer_display, index=df.index, dtype=object)

def _resolve_path(fn: str) -> Path:
    path = Path(fn)
    if path.exists():
        return path
    candidate = DATA_DIR / fn
    return candidate

def _read_excel(path: Path, entity: str) -> pd.DataFrame:
    df = pd.read_excel(path, skiprows=1)
    df['entity_source_file'] = entity
    return df

def _finalize_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    x = pd.concat(frames, ignore_index=True)
    x = normalize_columns(x)
    for c in ['pr_date_submitted', 'po_create_date', 'po_delivery_date', 'po_approved_date']:
        if c in x.columns:
            x[c] = pd.to_datetime(x[c], errors='coerce')
    return x

@st.cache_data(show_spinner=False)
def load_all(file_list=None):
    if file_list is None:
        file_list = RAW_FILES
    frames: list[pd.DataFrame] = []
    for fn, ent in file_list:
        path = _resolve_path(fn)
        if not path.exists():
            continue
        try:
            frames.append(_read_excel(path, ent))
        except Exception as exc:
            st.warning(f"Failed to read {path.name}: {exc}")
    return _finalize_frames(frames)

# ----------------- Load Data -----------------
df = load_all()
if df.empty:
    st.warning("No data loaded. Place MEPL.xlsx, MLPL.xlsx, mmw.xlsx, mmpl.xlsx next to this script or upload files.")

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
if entity_col and entity_col in df.columns:
    df['entity'] = df[entity_col].fillna('').astype(str).str.strip()
else:
    df['entity'] = df.get('entity_source_file', '').fillna('').astype(str)

for c in [pr_budget_desc_col, pr_budget_code_col, po_budget_desc_col, po_budget_code_col, pr_bu_col, po_bu_col]:
    if c and c not in df.columns:
        df[c] = ''

# buyer group code extraction if present
if 'buyer_group' in df.columns:
    try:
        df['buyer_group_code'] = df['buyer_group'].astype(str).str.extract('([0-9]+)')[0].astype(float)
    except Exception:
        df['buyer_group_code'] = np.nan
else:
    df['buyer_group_code'] = np.nan

# compute Buyer.Type
if 'Buyer.Type' not in df.columns:
    df['Buyer.Type'] = compute_buyer_type_vectorized(df)

# normalize po_creator
po_orderer_col = safe_col(df, ['po_orderer', 'po orderer', 'po_orderer_code']) or 'po_orderer'
if po_orderer_col not in df.columns:
    df['po_orderer'] = 'N/A'
else:
    df[po_orderer_col] = df[po_orderer_col].fillna('N/A').astype(str).str.strip()

o_created_by_map = {
    'MMW2324030': 'Dhruv', 'MMW2324062': 'Deepak', 'MMW2425154': 'Mukul', 'MMW2223104': 'Paurik',
    'MMW2021181': 'Nayan', 'MMW2223014': 'Aatish', 'MMW_EXT_002': 'Deepakex', 'MMW2425024': 'Kamlesh',
    'MMW2021184': 'Suresh', 'N/A': 'Dilip', 'MMW2526019': 'Vraj', 'MMW2223240': 'Vatsal', 'MMW2223219': '',
    'MMW2021115': 'Priyam', 'MMW2425031': 'Preet', 'MMW222360IN': 'Ayush', 'MMW2425132': 'Prateek.B',
    'MMW2425025': 'Jaymin', 'MMW2425092': 'Suresh', 'MMW252617IN': 'Akaash', 'MMW1920052': 'Nirmal',
    '2425036': '', 'MMW222355IN': 'Jaymin', 'MMW2324060': 'Chetan', 'MMW222347IN': 'Vaibhav',
    'MMW2425011': '', 'MMW1920036': 'Ankit', 'MMW2425143': 'Prateek.K', '2425027': '', 'MMW2223017': 'Umesh',
    'MMW2021214': 'Raunak', 'Intechuser1': 'Intesh Data'
}
upper_map = {k.upper(): v for k, v in o_created_by_map.items()}

df['po_creator'] = df.get(po_orderer_col, df.get('po_orderer', pd.Series('N/A', index=df.index))).astype(str).str.upper().map(upper_map).fillna(df.get(po_orderer_col, df.get('po_orderer', pd.Series('N/A', index=df.index))).astype(str))
df['po_creator'] = df['po_creator'].replace({'N/A': 'Dilip', '': 'Dilip'}).astype(str).str_strip()

creator_clean = df['po_creator'].fillna('').astype(str).str.strip()
df['po_buyer_type'] = np.where(creator_clean.isin(INDIRECT_BUYERS), 'Indirect', 'Direct')

pr_requester_col = safe_col(df, ['pr_requester','requester','pr_requester_name','requester_name'])
df['buyer_display'] = compute_buyer_display(df, purchase_doc_col, pr_requester_col)

# ----------------- Sidebar filters -----------------
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

# Start with df, then apply date FY filter (no copy)
fil = df
if pr_col and pr_col in fil.columns:
    fil = fil[(fil[pr_col] >= pr_start) & (fil[pr_col] <= pr_end)]

# Date range filter using pre-parsed datetimes
date_basis = pr_col if pr_col in fil.columns else (po_create_col if po_create_col in fil.columns else None)
date_range_key = None
if date_basis:
    mindt = fil[date_basis].dropna().min()
    maxdt = fil[date_basis].dropna().max()
    if pd.notna(mindt) and pd.notna(maxdt):
        dr = st.sidebar.date_input('Date range', (mindt.date(), maxdt.date()), key='date_range')
        if isinstance(dr, tuple) and len(dr) == 2:
            sdt = pd.to_datetime(dr[0]); edt = pd.to_datetime(dr[1]) + pd.Timedelta(hours=23, minutes=59, seconds=59)
            fil = fil[(fil[date_basis] >= sdt) & (fil[date_basis] <= edt)]
            date_range_key = (sdt.isoformat(), edt.isoformat())

# ----------------- ULTRA-FAST INDEX-BASED FILTERS -----------------
@st.cache_data
def build_index_map(series: pd.Series):
    """Return dict: value_str -> ndarray of integer row positions (np.int64)."""
    s = series.fillna('').astype(str)
    groups = s.groupby(s).groups  # dict(value -> index Int64Index)
    index_map = {str(k): np.fromiter(v.values, dtype=int) for k, v in groups.items()}
    return index_map

# Convert relevant filter columns to string (do not convert to category here: we'll use index map)
for col in ['entity','po_creator','po_vendor','product_name','buyer_display','Buyer.Type']:
    if col in fil.columns:
        fil[col] = fil[col].fillna('').astype(str)

# Sidebar choices (from current filtered frame)
entity_choices = sorted(fil['entity'].unique().tolist()) if 'entity' in fil.columns else []
sel_e = st.sidebar.multiselect('Entity', entity_choices, default=entity_choices)

creator_choices = sorted(fil['po_creator'].unique().tolist()) if 'po_creator' in fil.columns else []
sel_o = st.sidebar.multiselect('PO Ordered By', creator_choices, default=creator_choices)

bt_choices = sorted(fil['Buyer.Type'].unique().tolist()) if 'Buyer.Type' in fil.columns else []
sel_b = st.sidebar.multiselect('Buyer Type', bt_choices, default=bt_choices)

vendor_choices = sorted(fil['po_vendor'].unique().tolist()) if 'po_vendor' in fil.columns else []
sel_v = st.sidebar.multiselect('Vendor (pick one or more)', vendor_choices, default=vendor_choices)

item_choices = sorted(fil['product_name'].unique().tolist()) if 'product_name' in fil.columns else []
sel_i = st.sidebar.multiselect('Item / Product (pick one or more)', item_choices, default=item_choices)

# Build maps (cached)
idx_entity = build_index_map(fil['entity']) if 'entity' in fil.columns else {}
idx_creator = build_index_map(fil['po_creator']) if 'po_creator' in fil.columns else {}
idx_bt = build_index_map(fil['Buyer.Type']) if 'Buyer.Type' in fil.columns else {}
idx_vendor = build_index_map(fil['po_vendor']) if 'po_vendor' in fil.columns else {}
idx_item = build_index_map(fil['product_name']) if 'product_name' in fil.columns else {}

def intersect_arrays(list_of_arrays):
    """Robust intersection of integer index arrays.
    Coerce inputs to 1-D integer numpy arrays, ignore non-convertible entries,
    and operate on unique values to avoid strange dtypes causing TypeError.
    """
    if not list_of_arrays:
        return np.array([], dtype=int)

    arrays = []
    for a in list_of_arrays:
        try:
            arr = np.asarray(a)
            if arr.size == 0:
                continue
            # Coerce to integers where possible
            arr = arr.astype(int)
            arrays.append(np.unique(arr))
        except Exception:
            # Skip non-convertible arrays
            continue

    if not arrays:
        return np.array([], dtype=int)

    # Start with first array
    res = arrays[0].copy()
    for arr in arrays[1:]:
        # intersect unique sorted arrays
        res = np.intersect1d(res, arr, assume_sorted=False)
        if res.size == 0:
            break
    return res

# Collect arrays for selected filters
arrays = []
if sel_e:
    arrays += [idx_entity.get(x, np.array([], dtype=int)) for x in sel_e]
if sel_o:
    arrays += [idx_creator.get(x, np.array([], dtype=int)) for x in sel_o]
if sel_b:
    arrays += [idx_bt.get(x, np.array([], dtype=int)) for x in sel_b]
if sel_v:
    arrays += [idx_vendor.get(x, np.array([], dtype=int)) for x in sel_v]
if sel_i:
    arrays += [idx_item.get(x, np.array([], dtype=int)) for x in sel_i]

# Compute final filtered index
arrays = [a for a in arrays if a.size > 0]
if not arrays:
    # No effective selection -> show all rows by default
    final_idx = np.arange(len(fil), dtype=int)
else:
    final_idx = intersect_arrays(arrays)

# Apply final selection
if final_idx.size == 0:
    fil = fil.iloc[0:0]
else:
    fil = fil.iloc[final_idx]

# Ensure trend_date_col is available (avoid NameError)
trend_date_col = None
if po_create_col and po_create_col in fil.columns:
    trend_date_col = po_create_col
elif pr_col and pr_col in fil.columns:
    trend_date_col = pr_col

# Preview limit
MAX_PREVIEW = 5000
if len(fil) > MAX_PREVIEW:
    st.warning(f"Filtered result has {len(fil)} rows — showing top {MAX_PREVIEW} rows.")
    st.dataframe(fil.head(MAX_PREVIEW), use_container_width=True)
else:
    st.dataframe(fil, use_container_width=True)

if st.sidebar.button('Reset Filters'):
    for k in list(st.session_state.keys()):
        try:
            del st.session_state[k]
        except Exception:
            pass
    st.experimental_rerun()

# signature for memoization
def _sel_key(values):
    return tuple(sorted(str(v) for v in values)) if values else ()

filter_signature = (
    fy_key, date_range_key, _sel_key(sel_b), _sel_key(sel_e), _sel_key(sel_o), _sel_key(sel_v), _sel_key(sel_i),
)

# Precompute month bucket once to avoid repeated conversions downstream
if trend_date_col:
    fil['_month_bucket'] = fil[trend_date_col].dt.to_period('M').dt.to_timestamp()
else:
    fil['_month_bucket'] = pd.NaT

# ----------------- Tabs -----------------
T = st.tabs(['KPIs & Spend','PR/PO Timing','PO Approval','Delivery','Vendors','Dept & Services','Unit-rate Outliers','Forecast','Scorecards','Search','Full Data'])

# ----------------- KPIs & Spend -----------------
with T[0]:
    st.header('P2P Dashboard — Indirect (KPIs & Spend)')
    c1,c2,c3,c4,c5 = st.columns(5)
    total_prs = int(fil.get(pr_number_col, pd.Series(dtype=object)).nunique()) if pr_number_col else 0
    total_pos = int(fil.get(purchase_doc_col, pd.Series(dtype=object)).nunique()) if purchase_doc_col else 0
    c1.metric('Total PRs', total_prs); c2.metric('Total POs', total_pos)
    c3.metric('Line Items', len(fil))
    c4.metric('Entities', int(fil.get('entity', pd.Series(dtype=object)).nunique()))
    spend_val = fil.get(net_amount_col, pd.Series(0)).sum() if net_amount_col else 0
    c5.metric('Spend (Cr ₹)', f"{spend_val/1e7:,.2f}")
    st.markdown('---')

    def build_monthly():
        if not (trend_date_col and net_amount_col and net_amount_col in fil.columns):
            return pd.DataFrame()
        if 'entity' not in fil.columns:
            return pd.DataFrame()
        z = fil.dropna(subset=['_month_bucket']).copy()
        z['month'] = z['_month_bucket']
        z['entity'] = z['entity'].fillna('Unmapped')
        return z.groupby(['month','entity'], observed=True)[net_amount_col].sum().reset_index()

    st.subheader('Monthly Total Spend + Cumulative')
    if trend_date_col and net_amount_col and net_amount_col in fil.columns:
        me = memoized_compute('monthly_entity', filter_signature, build_monthly)
        if me.empty:
            st.info('No monthly/entity data to plot.')
        else:
            pivot = me.pivot(index='month', columns='entity', values=net_amount_col).fillna(0).sort_index()
            fixed_entities = ['MEPL','MLPL','MMW','MMPL']
            colors = {'MEPL':'#1f77b4','MLPL':'#ff7f0e','MMW':'#2ca02c','MMPL':'#d62728'}
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
            for ent in ordered_entities:
                ent_vals = pivot_cr[ent].values
                text_vals = [f"{v:.2f}" if v > 0 else '' for v in ent_vals]
                fig.add_trace(go.Bar(x=xaxis_labels, y=ent_vals, name=ent, marker_color=colors.get(ent, None), text=text_vals, textposition='inside', hovertemplate='%{x}<br>'+ent+': %{y:.2f} Cr<extra></extra>'), secondary_y=False)
            highlight_color = '#FFD700'
            fig.add_trace(go.Scatter(x=xaxis_labels, y=cum_cr.values, mode='lines+markers+text', name='Cumulative (Cr)', line=dict(color=highlight_color, width=3), marker=dict(color=highlight_color, size=6), text=[f"{int(round(v, 0))}" for v in cum_cr.values], textposition='top center', textfont=dict(color=highlight_color, size=9), hovertemplate='%{x}<br>Cumulative: %{y:.2f} Cr<extra></extra>'), secondary_y=True)
            fig.update_layout(barmode='stack', xaxis_tickangle=-45, title='Monthly Spend (stacked by Entity) + Cumulative', legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
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

    st.markdown('---')
    st.subheader('Buyer-wise Spend (Cr)')
    if 'buyer_display' in fil.columns and net_amount_col in fil.columns:
        def build_buyer_spend():
            grp = fil.groupby('buyer_display', observed=True)[net_amount_col].sum().reset_index()
            grp['cr'] = grp[net_amount_col] / 1e7
            return grp.sort_values('cr', ascending=False)
        buyer_spend = memoized_compute('buyer_spend', filter_signature, build_buyer_spend)
        fig_buyer = px.bar(buyer_spend, x='buyer_display', y='cr', text='cr', title='Buyer-wise Spend (Cr)')
        fig_buyer.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_buyer.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_buyer, use_container_width=True)
        st.dataframe(buyer_spend, use_container_width=True)

# end of file
