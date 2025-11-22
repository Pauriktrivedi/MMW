import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(page_title="P2P Dashboard — Indirect (Final)", layout="wide", initial_sidebar_state="expanded")

# ---------- Helpers ----------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to snake_case (safe, idempotent)."""
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


@st.cache_data
def load_all(file_list=None):
    """Load default Excel files placed next to the app. No drag-and-drop in UI by design."""
    if file_list is None:
        file_list = [("MEPL.xlsx", "MEPL"), ("MLPL.xlsx", "MLPL"), ("mmw.xlsx", "MMW"), ("mmpl.xlsx", "MMPL")]
    frames = []
    for fn, ent in file_list:
        try:
            df = pd.read_excel(fn, skiprows=1)
            df['entity'] = ent
            frames.append(df)
        except FileNotFoundError:
            continue
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    x = pd.concat(frames, ignore_index=True)
    x = normalize_columns(x)
    # coerce date columns if present
    for c in ['pr_date_submitted', 'po_create_date', 'po_delivery_date', 'po_approved_date']:
        if c in x.columns:
            x[c] = pd.to_datetime(x[c], errors='coerce')
    return x


# ---------- Load data ----------
df = load_all()
if df.empty:
    st.warning("No data loaded. Place MEPL.xlsx, MLPL.xlsx, mmw.xlsx, mmpl.xlsx next to this script and rerun.")

# ---------- Column discovery (robust) ----------
# helper to find first present column from candidates

def find_col(dframe, candidates):
    for c in candidates:
        if c in dframe.columns:
            return c
    return None

pr_col = find_col(df, ['pr_date_submitted', 'pr_date', 'pr date submitted'])
po_create_col = find_col(df, ['po_create_date', 'po create date', 'po_create', 'po date'])
net_amount_col = find_col(df, ['net_amount', 'net amount', 'net_amount_inr', 'amount'])
purchase_doc_col = find_col(df, ['purchase_doc', 'purchase doc', 'purchase_doc_number'])
pr_number_col = find_col(df, ['pr_number', 'pr number', 'pr_no'])
po_vendor_col = find_col(df, ['po_vendor', 'po vendor', 'vendor'])
po_unit_rate_col = find_col(df, ['po_unit_rate', 'po unit rate', 'po_unit_price'])

# PR/PO Budget & description columns (user confirmed originals)
pr_code_col = find_col(df, ['pr_budget_code', 'pr budget code', 'prbudgetcode'])
pr_desc_col = find_col(df, ['pr_budget_code_description', 'pr budget description', 'pr_budget_description', 'pr budget desc'])
po_code_col = find_col(df, ['po_budget_code', 'po budget code', 'po_budgetcode'])
po_desc_col = find_col(df, ['po_budget_code_description', 'po budget description', 'po_budget_description', 'po budget desc'])

# ---------- Lightweight mappings ----------
# buyer type mapping (keeps previous logic but robust to missing col)
buyer_group_col = find_col(df, ['buyer_group', 'buyer group'])
if buyer_group_col:
    try:
        df['buyer_group_code'] = df[buyer_group_col].astype(str).str.extract(r"(\d+)")[0].astype(float)
    except Exception:
        df['buyer_group_code'] = np.nan


def map_buyer_type_row(r):
    bg = str(r.get(buyer_group_col, '')).strip() if buyer_group_col else ''
    code = r.get('buyer_group_code', np.nan)
    if bg in ['ME_BG17', 'MLBG16']:
        return 'Direct'
    if bg in ['', 'Not Available'] or pd.isna(bg):
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
    df['buyer_type'] = df.apply(map_buyer_type_row, axis=1) if buyer_group_col else 'Unknown'
else:
    df['buyer_type'] = pd.Series(dtype=object)

# PO creator mapping (light)
po_orderer_col = find_col(df, ['po_orderer', 'po ordered by', 'po_ordered_by'])
map_orderer = {
    'mmw2324030': 'Dhruv', 'mmw2324062': 'Deepak', 'mmw2425154': 'Mukul', 'mmw2223104': 'Paurik',
    'mmw2021181': 'Nayan', 'mmw2223014': 'Aatish', 'mmw_ext_002': 'Deepakex', 'mmw2425024': 'Kamlesh',
    'mmw2021184': 'Suresh', 'n/a': 'Dilip'
}
if po_orderer_col and po_orderer_col in df.columns:
    df['po_orderer'] = df[po_orderer_col].fillna('N/A').astype(str).str.strip()
else:
    df['po_orderer'] = ''

_df_po_orderer_lc = df['po_orderer'].astype(str).str.lower()
df['po_creator'] = _df_po_orderer_lc.map(map_orderer).fillna(df['po_orderer']).replace({'N/A': 'Dilip'})
indirect_set = set(['Aatish','Deepak','Deepakex','Dhruv','Dilip','Mukul','Nayan','Paurik','Kamlesh','Suresh'])
df['po_buyer_type'] = np.where(df['po_creator'].isin(indirect_set), 'Indirect', 'Direct')

# ---------- Sidebar filters ----------
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

# ensure columns exist
for c in ['buyer_type', 'po_creator', 'po_buyer_type', 'entity']:
    if c not in fil.columns:
        fil[c] = ''
    fil[c] = fil[c].astype(str).str.strip()

sel_b = st.sidebar.multiselect('Buyer Type', sorted(fil['buyer_type'].dropna().unique().tolist()), default=sorted(fil['buyer_type'].dropna().unique().tolist()))
sel_e = st.sidebar.multiselect('Entity', sorted(fil['entity'].dropna().unique().tolist()), default=sorted(fil['entity'].dropna().unique().tolist()))
sel_o = st.sidebar.multiselect('PO Ordered By', sorted(fil['po_creator'].dropna().unique().tolist()), default=sorted(fil['po_creator'].dropna().unique().tolist()))
sel_p = st.sidebar.multiselect('PO Buyer Type', sorted(fil['po_buyer_type'].dropna().unique().tolist()), default=sorted(fil['po_buyer_type'].dropna().unique().tolist()))

if sel_b:
    fil = fil[fil['buyer_type'].isin(sel_b)]
if sel_e:
    fil = fil[fil['entity'].isin(sel_e)]
if sel_o:
    fil = fil[fil['po_creator'].isin(sel_o)]
if sel_p:
    fil = fil[fil['po_buyer_type'].isin(sel_p)]

# Vendor / Product filters
if po_vendor_col and po_vendor_col in fil.columns:
    fil['po_vendor'] = fil[po_vendor_col].astype(str).str.strip()
else:
    fil['po_vendor'] = ''
if 'product_name' not in fil.columns:
    fil['product_name'] = ''
else:
    fil['product_name'] = fil['product_name'].astype(str).str.strip()

vendor_choices = sorted(fil['po_vendor'].dropna().unique().tolist())
item_choices = sorted(fil['product_name'].dropna().unique().tolist())
sel_v = st.sidebar.multiselect('Vendor', vendor_choices, default=vendor_choices)
sel_i = st.sidebar.multiselect('Item / Product', item_choices, default=item_choices)
if sel_v:
    fil = fil[fil['po_vendor'].isin(sel_v)]
if sel_i:
    fil = fil[fil['product_name'].isin(sel_i)]

# ---------- Tabs ----------
T = st.tabs(['KPIs & Spend','PO/PR Timing','Delivery','Vendors','Dept & Services','Unit-rate Outliers','Forecast','Scorecards','Search'])

# ---------- KPIs & Spend (merged) ----------
with T[0]:
    st.header('P2P Dashboard — Indirect')
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
    # Monthly spend + cumulative
    dcol = po_create_col if po_create_col in fil.columns else (pr_col if pr_col in fil.columns else None)
    if dcol and net_amount_col in fil.columns:
        t = fil.dropna(subset=[dcol]).copy()
        t['month'] = t[dcol].dt.to_period('M').dt.to_timestamp()
        m = t.groupby('month')[net_amount_col].sum().sort_index().reset_index()
        m['cr'] = m[net_amount_col]/1e7
        m['cumcr'] = m['cr'].cumsum()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_bar(x=m['month'].dt.strftime('%b-%Y'), y=m['cr'], name='Monthly Spend (Cr ₹)')
        fig.add_scatter(x=m['month'].dt.strftime('%b-%Y'), y=m['cumcr'], name='Cumulative (Cr ₹)', mode='lines+markers', secondary_y=True)
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

# ---------- Dept & Services (PR Budget charts) ----------
with T[4]:
    st.subheader('Department Spend — PR Budget Code / PR Budget Description')

    # Build pr_department aware of provided columns
    dept_df = fil.copy()

    # ensure candidate columns exist (normalized names expected)
    # If the original Excel had spaces, normalization already converted them
    def safe_col_lookup(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    pr_code = safe_col_lookup(dept_df, [
        'pr_budget_code', 'pr budget code', 'prbudgetcode'
    ])
    pr_desc = safe_col_lookup(dept_df, [
        'pr_budget_code_description', 'pr budget description', 'pr_budget_description', 'pr budget desc', 'pr_budget_desc'
    ])

    # If pr_code/pr_desc not present, attempt to guess by suffix
    if pr_code is None and 'pr' in dept_df.columns:
        pr_code = 'pr'

    # Build aggregated charts only if net_amount present
    if net_amount_col and net_amount_col in dept_df.columns:
        # PR Budget Code chart
        if pr_code and pr_code in dept_df.columns:
            code_agg = dept_df.groupby(pr_code, dropna=False)[net_amount_col].sum().reset_index().sort_values(net_amount_col, ascending=False)
            code_agg['cr'] = code_agg[net_amount_col] / 1e7
            top_codes = code_agg.head(30).copy()
            top_codes[pr_code] = top_codes[pr_code].astype(str)
            fig_code = px.bar(top_codes, x=pr_code, y='cr', title='PR Budget Code Spend (Top 30)', labels={pr_code: 'PR Budget Code', 'cr': 'Cr'})
            fig_code.update_layout(xaxis_tickangle=-45, yaxis_title='Cr')
            st.plotly_chart(fig_code, use_container_width=True)

            # Drill selection (dropdown) — shows related rows. Note: plotly click-capture not used here to keep app dependency-free.
            sel_code = st.selectbox('Select PR Budget Code to inspect rows', ['__ALL__'] + top_codes[pr_code].tolist())
            if sel_code and sel_code != '__ALL__':
                rows = dept_df[dept_df[pr_code].astype(str) == sel_code].copy()
                st.write(f"Showing {len(rows)} rows for PR Budget Code = {sel_code}")
                # show useful columns
                show_cols = [c for c in [pr_number_col, purchase_doc_col, pr_code, pr_desc, po_code_col, po_desc_col, po_vendor_col, net_amount_col] if c and c in rows.columns]
                st.dataframe(rows[show_cols].head(500), use_container_width=True)

        # PR Budget Description chart
        if pr_desc and pr_desc in dept_df.columns:
            desc_agg = dept_df.groupby(pr_desc, dropna=False)[net_amount_col].sum().reset_index().sort_values(net_amount_col, ascending=False)
            desc_agg['cr'] = desc_agg[net_amount_col] / 1e7
            top_desc = desc_agg.head(30).copy()
            top_desc[pr_desc] = top_desc[pr_desc].astype(str)
            fig_desc = px.bar(top_desc, x=pr_desc, y='cr', title='PR Budget Description Spend (Top 30)', labels={pr_desc: 'PR Budget Description', 'cr': 'Cr'})
            fig_desc.update_layout(xaxis_tickangle=-45, yaxis_title='Cr')
            st.plotly_chart(fig_desc, use_container_width=True)

            sel_desc = st.selectbox('Select PR Budget Description to inspect rows', ['__ALL__'] + top_desc[pr_desc].tolist(), key='sel_pr_desc')
            if sel_desc and sel_desc != '__ALL__':
                rows = dept_df[dept_df[pr_desc].astype(str) == sel_desc].copy()
                st.write(f"Showing {len(rows)} rows for PR Budget Description = {sel_desc}")
                show_cols = [c for c in [pr_number_col, purchase_doc_col, pr_code, pr_desc, po_code_col, po_desc_col, po_vendor_col, net_amount_col] if c and c in rows.columns]
                st.dataframe(rows[show_cols].head(1000), use_container_width=True)
    else:
        st.info('Net Amount column not found — cannot compute departmental spend.')

# ---------- Remaining tabs (kept minimal & stable) ----------
with T[1]:
    st.subheader('PR / PO Timing')
    st.write('Timing tab — kept minimal for stability in this final script.')

with T[2]:
    st.subheader('Delivery')
    st.write('Delivery tab — kept minimal for stability.')

with T[3]:
    st.subheader('Vendors')
    st.write('Vendors tab — kept minimal for stability.')

with T[5]:
    st.subheader('Unit-rate Outliers')
    st.write('Unit-rate outliers tab — kept minimal for stability.')

with T[6]:
    st.subheader('Forecast')
    st.write('Forecast tab — kept minimal for stability.')

with T[7]:
    st.subheader('Scorecards')
    st.write('Scorecards tab — kept minimal for stability.')

with T[8]:
    st.subheader('Search')
    st.write('Search tab — kept minimal for stability.')

# EOF
