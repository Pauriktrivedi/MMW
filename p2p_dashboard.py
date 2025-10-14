import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="P2P Dashboard — Full (Refactor)", layout="wide", initial_sidebar_state="expanded")

# ---- Page header: guaranteed visible ----
import streamlit as _st
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
    for c in ['pr_date_submitted','po_create_date','po_approved_date','po_delivery_date','last_po_date']:
        if c in x.columns:
            x[c] = pd.to_datetime(x[c], errors='coerce')
    return x

# Use uploaded files stored in session_state if present (uploader is at bottom of page)
uploaded_in_session = st.session_state.get('_uploaded_files', None)
if uploaded_in_session:
    df = load_all_from_files(uploaded_in_session)
else:
    # initial load falls back to defaults
    df = load_all_from_files()

if df.empty:
    st.warning("No data loaded. You can upload Excel/CSV files using the uploader at the BOTTOM of this page, or place default Excel files next to this script (MEPL.xlsx, MLPL.xlsx, mmw.xlsx, mmpl.xlsx).")
    st.stop()

# ----------------- Column mapping -----------------
source_to_norm = {
    'pr number': 'pr_number','pr no':'pr_number','pr date submitted':'pr_date_submitted',
    'purchase doc':'purchase_doc','po create date':'po_create_date','po delivery date':'po_delivery_date',
    'po vendor':'po_vendor','po quantity':'po_quantity','po unit rate':'po_unit_rate','net amount':'net_amount',
    'po status':'po_status','po approved date':'po_approved_date','po orderer':'po_orderer','po budget code':'po_budget_code',
    'product name':'product_name','item code':'item_code','item description':'item_description','procurement category':'procurement_category',
    'pending qty':'pending_qty','received qty':'receivedqty','receivedqty':'receivedqty','pr quantity':'pr_quantity','po department':'po_department'
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

# ----------------- Columns used -----------------
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

map_orderer = {'mmw2324030':'Dhruv','mmw2324062':'Deepak','mmw2425154':'Mukul','mmw2223104':'Paurik','mmw2021181':'Nayan','mmw2223014':'Aatish','mmw_ext_002':'Deepakex','mmw2425024':'Kamlesh','mmw2021184':'Suresh','n/a':'Dilip'}
if not df.empty:
    df['po_orderer'] = safe_get(df, po_orderer_col, pd.NA).fillna('N/A').astype(str).str.strip()
    df['po_orderer_lc'] = df['po_orderer'].str.lower()
    df['po_creator'] = df['po_orderer_lc'].map(map_orderer).fillna(df['po_orderer']).replace({'N/A':'Dilip'})
    indirect = set(['Aatish','Deepak','Deepakex','Dhruv','Dilip','Mukul','Nayan','Paurik','Kamlesh','Suresh'])
    df['po_buyer_type'] = np.where(df['po_creator'].isin(indirect), 'Indirect', 'Direct')
else:
    df['po_creator'] = pd.Series(dtype=object)
    df['po_buyer_type'] = pd.Series(dtype=object)

# ----------------- Sidebar Filters (compact) -----------------
st.sidebar.header('Filters')
FY = {'All Years': (pd.Timestamp('2023-04-01'), pd.Timestamp('2026-03-31')),'2023': (pd.Timestamp('2023-04-01'), pd.Timestamp('2024-03-31')),'2024': (pd.Timestamp('2024-04-01'), pd.Timestamp('2025-03-31')),'2025': (pd.Timestamp('2025-04-01'), pd.Timestamp('2026-03-31'))}
fy_key = st.sidebar.selectbox('Financial Year', list(FY), index=0)
pr_start, pr_end = FY[fy_key]

fil = df.copy()
if pr_col and pr_col in fil.columns:
    fil = fil[(fil[pr_col] >= pr_start) & (fil[pr_col] <= pr_end)]

for col in ['buyer_type', entity_col, 'po_creator', 'po_buyer_type']:
    if col not in fil.columns:
        fil[col] = ''
    fil[col] = fil[col].astype(str).str.strip()

if 'buyer_type_unified' not in fil.columns:
    bt = safe_get(fil, 'buyer_type', pd.Series('', index=fil.index)).astype(str).str.strip()
    pbt = safe_get(fil, 'po_buyer_type', pd.Series('', index=fil.index)).astype(str).str.strip()
    fil['buyer_type_unified'] = np.where(bt != '', bt, pbt)
    fil['buyer_type_unified'] = fil['buyer_type_unified'].str.title().replace({'Other':'Indirect','Unknown':'Indirect','':'Indirect','Na':'Indirect','N/A':'Indirect'})
    fil['buyer_type_unified'] = np.where(fil['buyer_type_unified'].str.lower() == 'direct', 'Direct', 'Indirect')

choices_bt = sorted(fil['buyer_type_unified'].dropna().unique().tolist()) if 'buyer_type_unified' in fil.columns else ['Direct','Indirect']
sel_b = st.sidebar.multiselect('Buyer Type', choices_bt, default=choices_bt)
sel_e = st.sidebar.multiselect('Entity', sorted(fil[entity_col].dropna().unique().tolist()), default=sorted(fil[entity_col].dropna().unique().tolist()))
sel_o = st.sidebar.multiselect('PO Ordered By', sorted(fil['po_creator'].dropna().unique().tolist()), default=sorted(fil['po_creator'].dropna().unique().tolist()))
sel_p = st.sidebar.multiselect('PO Buyer Type (raw)', sorted(fil['po_buyer_type'].dropna().unique().tolist()), default=sorted(fil['po_buyer_type'].dropna().unique().tolist()))

# Vendor & Item multi-selects
vendor_choices = sorted(fil['po_vendor'].dropna().unique().tolist())
item_choices = sorted(fil['product_name'].dropna().unique().tolist())
sel_v = st.sidebar.multiselect('Vendor (pick one or more)', vendor_choices, default=vendor_choices)
sel_i = st.sidebar.multiselect('Item / Product (pick one or more)', item_choices, default=item_choices)

# PO Department dropdown
po_dept_choices = ['All Departments'] + (sorted(fil['po_department'].dropna().unique().tolist()) if 'po_department' in fil.columns else [])
po_dept_sel = st.sidebar.selectbox('PO Department', po_dept_choices, index=0)

if st.sidebar.button('Reset Filters'):
    keys_to_clear = ['filter_vendor','filter_item','filter_buyer','filter_entity','filter_po_creator','filter_po_buyer_type','fy_key']
    for k in keys_to_clear:
        if k in st.session_state:
            del st.session_state[k]
    st.experimental_rerun()

# apply filters
if sel_b:
    fil = fil[fil['buyer_type_unified'].isin(sel_b)]
if sel_e:
    fil = fil[fil[entity_col].isin(sel_e)]
if sel_o:
    fil = fil[fil['po_creator'].isin(sel_o)]
if sel_p:
    fil = fil[fil['po_buyer_type'].isin(sel_p)]
if sel_v:
    fil = fil[fil['po_vendor'].isin(sel_v)]
if sel_i:
    fil = fil[fil['product_name'].isin(sel_i)]
if po_dept_sel and po_dept_sel != 'All Departments' and 'po_department' in fil.columns:
    fil = fil[fil['po_department'] == po_dept_sel]

# ----------------- Tabs & content (unchanged) -----------------
T = st.tabs(['KPIs & Spend','PO/PR Timing','Delivery','Vendors','Dept & Services','Unit-rate Outliers','Forecast','Scorecards','Search'])

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

# ... rest of tabs unchanged (kept for brevity in canvas) ...

# ----------------- BOTTOM-UPLOADER (visible at bottom of main page) -----------------
st.markdown('---')
st.markdown('### Upload & Debug (bottom of page)')
new_files = st.file_uploader('Upload Excel/CSV files (optional) — drag here or Browse', type=['xlsx','xls','csv'], accept_multiple_files=True, key='_visible_bottom_uploader')
if new_files:
    # store in session_state and reload app so loader picks them up
    st.session_state['_uploaded_files'] = new_files
    st.experimental_rerun()

st.caption('Uploader placed at bottom of page as requested. After uploading, the app will reload and show data from the uploaded files.')
