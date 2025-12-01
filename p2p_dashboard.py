# paste from line 1 into p2p_dashboard_indirect_final.py
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
    'Aatish', 'Deepak', 'Deepakex', 'Dhruv', 'Dilip',
    'Mukul', 'Nayan', 'Paurik', 'Kamlesh', 'Suresh', 'Priyam'
}

# Run with: streamlit run p2p_dashboard_indirect_final.py
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
        st.warning("⚠️ 'Buyer Group' column not found. All Buyer.Type set to 'Unknown'.")
        return pd.Series('Unknown', index=df.index, dtype=object)

    bg_raw = df[group_col].fillna('').astype(str).str.strip()
    code_series = pd.to_numeric(bg_raw.str.extract(r'(\d+)')[0], errors='coerce')

    # Keep both normalized and user-facing code columns for downstream use.
    df['buyer_group_code'] = code_series
    df['Buyer Group Code'] = code_series

    buyer_type = pd.Series('Other', index=df.index, dtype=object)

    alias_direct = bg_raw.str.upper().isin({'ME_BG17', 'MLBG16'})
    buyer_type[alias_direct] = 'Direct'

    not_available = bg_raw.eq('') | bg_raw.str.lower().isin(['not available', 'na', 'n/a'])
    buyer_type[not_available] = 'Indirect'

    buyer_type[(code_series >= 1) & (code_series <= 9)] = 'Direct'
    buyer_type[(code_series >= 10) & (code_series <= 18)] = 'Indirect'

    buyer_type = buyer_type.fillna('Other')
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

# buyer group code extraction
if 'buyer_group' in df.columns:
    try:
        df['buyer_group_code'] = df['buyer_group'].astype(str).str.extract('([0-9]+)')[0].astype(float)
    except Exception:
        df['buyer_group_code'] = np.nan

df['Buyer.Type'] = compute_buyer_type_vectorized(df)

# normalize po_creator
po_orderer_col = safe_col(df, ['po_orderer', 'po orderer', 'po_orderer_code'])
if po_orderer_col and po_orderer_col in df.columns:
    df[po_orderer_col] = df[po_orderer_col].fillna('N/A').astype(str).str.strip()
else:
    df['po_orderer'] = 'N/A'
    po_orderer_col = 'po_orderer'

# mapping dictionary for creators
o_created_by_map = {
    'MMW2324030': 'Dhruv', 'MMW2324062': 'Deepak', 'MMW2425154': 'Mukul', 'MMW2223104': 'Paurik',
    'MMW2021181': 'Nayan', 'MMW2223014': 'Aatish', 'MMW_EXT_002': 'Deepakex', 'MMW2425024': 'Kamlesh',
    'MMW2021184': 'Suresh', 'N/A': 'Dilip',
    'MMW2526019': 'Vraj', 'MMW2223240': 'Vatsal', 'MMW2223219': '', 'MMW2021115': 'Priyam',
    'MMW2425031': 'Preet', 'MMW222360IN': 'Ayush', 'MMW2425132': 'Prateek.B', 'MMW2425025': 'Jaymin',
    'MMW2425092': 'Suresh', 'MMW252617IN': 'Akaash', 'MMW1920052': 'Nirmal', '2425036': '',
    'MMW222355IN': 'Jaymin', 'MMW2324060': 'Chetan', 'MMW222347IN': 'Vaibhav', 'MMW2425011': '',
    'MMW1920036': 'Ankit', 'MMW2425143': 'Prateek.K', '2425027': '', 'MMW2223017': 'Umesh',
    'MMW2021214': 'Raunak', 'Intechuser1': 'Intesh Data'
}
upper_map = {k.upper(): v for k, v in o_created_by_map.items()}
df['po_creator'] = df[po_orderer_col].astype(str).str.upper().map(upper_map).fillna(df[po_orderer_col].astype(str))
df['po_creator'] = df['po_creator'].replace({'N/A': 'Dilip', '': 'Dilip'})

creator_clean = df['po_creator'].fillna('').astype(str).str.strip()
df['po_buyer_type'] = np.where(creator_clean.isin(INDIRECT_BUYERS), 'Indirect', 'Direct')

pr_requester_col = safe_col(df, ['pr_requester','requester','pr_requester_name','pr_requester_name','requester_name'])
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

fil = df.copy()
if pr_col and pr_col in fil.columns:
    fil = fil[(fil[pr_col] >= pr_start) & (fil[pr_col] <= pr_end)]

# Date range filter
date_basis = pr_col if pr_col in fil.columns else (po_create_col if po_create_col in fil.columns else None)
dr = None
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

# defensive columns
for c in ['Buyer.Type', 'po_creator', 'po_vendor', 'entity', 'po_buyer_type']:
    if c not in fil.columns:
        fil[c] = ''

# Ensure PR-level Buyer.Type is tidy
if 'Buyer.Type' not in fil.columns:
    fil['Buyer.Type'] = compute_buyer_type_vectorized(fil)
fil['Buyer.Type'] = fil['Buyer.Type'].fillna('Unknown').astype(str).str.strip().str.title()
fil.loc[fil['Buyer.Type'].str.lower().isin(['direct', 'd']), 'Buyer.Type'] = 'Direct'
fil.loc[fil['Buyer.Type'].str.lower().isin(['indirect', 'i', 'in']), 'Buyer.Type'] = 'Indirect'
fil.loc[~fil['Buyer.Type'].isin(['Direct', 'Indirect']), 'Buyer.Type'] = 'Other'

# Entity + PO ordered by filters
entity_choices = sorted([e for e in fil['entity'].dropna().unique().tolist() if str(e).strip() != '']) if 'entity' in fil.columns else []
sel_e = st.sidebar.multiselect('Entity', entity_choices, default=entity_choices)
sel_o = st.sidebar.multiselect('PO Ordered By', sorted([str(x) for x in fil.get('po_creator', pd.Series(dtype=object)).dropna().unique().tolist() if str(x).strip()!='']), default=sorted([str(x) for x in fil.get('po_creator', pd.Series(dtype=object)).dropna().unique().tolist() if str(x).strip()!='']))

# Buyer Type choices from effective buyer type
choices_bt = sorted(fil['Buyer.Type'].dropna().unique().tolist())
sel_b = st.sidebar.multiselect('Buyer Type', choices_bt, default=choices_bt)

# Apply filters: effective buyer type -> entity -> po_creator
if sel_b:
    fil = fil[fil['Buyer.Type'].isin(sel_b)]
if sel_e and 'entity' in fil.columns:
    fil = fil[fil['entity'].isin(sel_e)]
if sel_o:
    fil = fil[fil['po_creator'].isin(sel_o)]

# Vendor + Item filters
if po_vendor_col and po_vendor_col in fil.columns:
    fil['po_vendor'] = fil[po_vendor_col].fillna('').astype(str)
else:
    fil['po_vendor'] = ''
if 'product_name' not in fil.columns:
    fil['product_name'] = ''

vendor_choices = sorted([v for v in fil['po_vendor'].dropna().unique().tolist() if str(v).strip()!=''])
item_choices = sorted([v for v in fil['product_name'].dropna().unique().tolist() if str(v).strip()!=''])
sel_v = st.sidebar.multiselect('Vendor (pick one or more)', vendor_choices, default=vendor_choices)
sel_i = st.sidebar.multiselect('Item / Product (pick one or more)', item_choices, default=item_choices)

if sel_v:
    fil = fil[fil['po_vendor'].isin(sel_v)]
if sel_i:
    fil = fil[fil['product_name'].isin(sel_i)]

def _sel_key(values):
    return tuple(sorted(str(v) for v in values)) if values else ()

filter_signature = (
    fy_key,
    date_range_key,
    _sel_key(sel_b),
    _sel_key(sel_e),
    _sel_key(sel_o),
    _sel_key(sel_v),
    _sel_key(sel_i),
)

# Precompute month bucket once to avoid repeated conversions downstream
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
    st.experimental_rerun()

# ----------------- Tabs -----------------
T = st.tabs(['KPIs & Spend','PR/PO Timing','PO Approval','Delivery','Vendors','Dept & Services','Unit-rate Outliers','Forecast','Scorecards','Search','Full Data'])

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

    def build_monthly():
        if not (trend_date_col and net_amount_col and net_amount_col in fil.columns):
            return pd.DataFrame()
        if 'entity' not in fil.columns:
            return pd.DataFrame()
        z = fil.dropna(subset=['_month_bucket']).copy()
        z['month'] = z['_month_bucket']
        z['entity'] = z['entity'].fillna('Unmapped')
        return z.groupby(['month','entity'], dropna=False)[net_amount_col].sum().reset_index()

    # ---------------- Monthly stacked entity spend + cumulative ----------------
    st.subheader('Monthly Total Spend + Cumulative')
    if trend_date_col and net_amount_col and net_amount_col in fil.columns:
        me = memoized_compute('monthly_entity', filter_signature, build_monthly)
        if me.empty:
            st.info('No monthly/entity data to plot.')
        else:
            pivot = me.pivot(index='month', columns='entity', values=net_amount_col).fillna(0).sort_index()

            # Fixed order + colors (Option B)
            fixed_entities = ['MEPL','MLPL','MMW','MMPL']
            colors = {'MEPL':'#1f77b4','MLPL':'#ff7f0e','MMW':'#2ca02c','MMPL':'#d62728'}

            # Ensure fixed columns exist (zero if missing)
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
                fig.add_trace(
                    go.Bar(
                        x=xaxis_labels,
                        y=ent_vals,
                        name=ent,
                        marker_color=colors.get(ent, None),
                        text=text_vals,
                        textposition='inside',
                        hovertemplate='%{x}<br>'+ent+': %{y:.2f} Cr<extra></extra>'
                    ),
                    secondary_y=False
                )

            # cumulative line — highlighted color + per-point labels in same color
            highlight_color = '#FFD700'  # gold highlight — change if you prefer a different color
            fig.add_trace(
                go.Scatter(
                    x=xaxis_labels,
                    y=cum_cr.values,
                    mode='lines+markers+text',
                    name='Cumulative (Cr)',
                    line=dict(color=highlight_color, width=3),
                    marker=dict(color=highlight_color, size=6),
                    # round to 0 decimals and show as integers
                    text=[f"{int(round(v, 0))}" for v in cum_cr.values],
                    # position and slightly smaller font to avoid overlap
                    textposition='top center',
                    textfont=dict(color=highlight_color, size=9),
                    hovertemplate='%{x}<br>Cumulative: %{y:.2f} Cr<extra></extra>'
                ),
                secondary_y=True
            )

            fig.update_layout(
                barmode='stack',
                xaxis_tickangle=-45,
                title='Monthly Spend (stacked by Entity) + Cumulative',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            fig.update_yaxes(title_text='Monthly Spend (Cr)', secondary_y=False)
            fig.update_yaxes(title_text='Cumulative (Cr)', secondary_y=True)

            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('Monthly Spend not available — need date and Net Amount columns.')

    st.markdown('---')

    # ---------------- Entity Trend (line) ----------------
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

    # ---------------- Buyer-wise Spend (bar) + Buyer Trend (line) ----------------
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

        # Buyer trend similar to Entity trend
        try:
            if trend_date_col and net_amount_col and net_amount_col in fil.columns:
                def build_buyer_trend():
                    bt = fil.dropna(subset=['_month_bucket']).copy()
                    bt['month'] = bt['_month_bucket']
                    return bt.groupby(['month','buyer_display'], dropna=False)[net_amount_col].sum().reset_index()
                bt_grouped = memoized_compute('buyer_trend', filter_signature, build_buyer_trend)
                top_buyers = buyer_spend['buyer_display'].head(5).astype(str).tolist()
                pick_mode = st.selectbox('Buyer trend: show', ['Top 5 by Spend', 'Choose buyers'], index=0)
                if pick_mode == 'Choose buyers':
                    chosen = st.multiselect('Pick buyers to show on trend', sorted(buyer_spend['buyer_display'].astype(str).unique().tolist()), default=top_buyers)
                else:
                    chosen = top_buyers
                if chosen:
                    g_b = bt_grouped[bt_grouped['buyer_display'].isin(chosen)].copy()
                    if not g_b.empty:
                        fig_b_trend = px.line(g_b, x=g_b['month'].dt.strftime('%b-%Y'), y=net_amount_col, color='buyer_display', labels={net_amount_col:'Net Amount','x':'Month'}, title='Buyer-wise Monthly Trend')
                        fig_b_trend.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig_b_trend, use_container_width=True)
        except Exception as e:
            st.error(f'Could not render Buyer Trend: {e}')
    else:
        st.info('Buyer display or Net Amount column missing — cannot compute buyer-wise spend.')

# ----------------- PR/PO Timing & Open PRs -----------------
with T[1]:
    st.subheader('PR/PO Timing')
    if pr_col and po_create_col and pr_col in fil.columns and po_create_col in fil.columns:
        def build_lead_df():
            lead = fil.dropna(subset=[pr_col, po_create_col]).copy()
            lead['Lead Time (Days)'] = (pd.to_datetime(lead[po_create_col]) - pd.to_datetime(lead[pr_col])).dt.days
            return lead
        lead_df = memoized_compute('lead_df', filter_signature, build_lead_df)
        SLA_DAYS = 7
        avg_lead = float(lead_df['Lead Time (Days)'].mean().round(1)) if not lead_df.empty else 0.0
        gauge_fig = go.Figure(go.Indicator(mode='gauge+number', value=avg_lead, number={'suffix':' days'}, gauge={'axis':{'range':[0, max(14, avg_lead * 1.2 if avg_lead else 14)]}, 'bar':{'color':'darkblue'}, 'steps':[{'range':[0,SLA_DAYS],'color':'lightgreen'},{'range':[SLA_DAYS,max(14, avg_lead * 1.2 if avg_lead else 14)],'color':'lightcoral'}], 'threshold':{'line':{'color':'red','width':4}, 'value':SLA_DAYS}}))
        st.plotly_chart(gauge_fig, use_container_width=True)
        st.caption(f"Current Avg Lead Time: {avg_lead:.1f} days   •   Target ≤ {SLA_DAYS} days")

        st.subheader('⏱️ PR to PO Lead Time by Buyer Type & by Buyer')
        if 'Buyer.Type' in lead_df.columns:
            lead_avg_by_type = lead_df.groupby('Buyer.Type')['Lead Time (Days)'].mean().round(0).reset_index().rename(columns={'Buyer.Type':'Buyer Type'})
        else:
            lead_avg_by_type = pd.DataFrame()
        if 'po_creator' in lead_df.columns:
            lead_avg_by_buyer = lead_df.groupby('po_creator')['Lead Time (Days)'].mean().round(0).reset_index().rename(columns={'po_creator':'PO.Creator'})
        else:
            lead_avg_by_buyer = pd.DataFrame()
        c1,c2 = st.columns(2)
        c1.dataframe(lead_avg_by_type, use_container_width=True)
        c2.dataframe(lead_avg_by_buyer, use_container_width=True)

        st.subheader('📅 Monthly PR & PO Trends')
        tmp = fil.copy()
        if pr_col in tmp.columns:
            tmp['PR Month'] = pd.to_datetime(tmp[pr_col], errors='coerce').dt.to_period('M')
        else:
            tmp['PR Month'] = pd.NaT
        if po_create_col in tmp.columns:
            tmp['PO Month'] = pd.to_datetime(tmp[po_create_col], errors='coerce').dt.to_period('M')
        else:
            tmp['PO Month'] = pd.NaT
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

        # ------------------------------------
        # Open PRs (Approved / InReview) — hybrid behavior with Buyer Type respected
        # ------------------------------------
        st.subheader("⚠️ Open PRs (Approved/InReview)")
        pr_status_col = safe_col(df, ['pr_status','pr status','status','prstatus','pr_status'])
        pr_date_col = pr_col if pr_col in df.columns else safe_col(df, ['pr_date_submitted','pr date submitted','pr_date','pr date'])
        pr_number_col_local = pr_number_col if pr_number_col in df.columns else safe_col(df, ['pr_number','pr number','pr_no','pr no'])

        if pr_status_col and pr_status_col in df.columns:
            def prepare_open_source(source_df: pd.DataFrame) -> pd.DataFrame:
                """Ensure Buyer.Type exists and re-apply Buyer Type selection locally."""
                base = source_df.copy()
                if 'Buyer.Type' not in base.columns:
                    base['Buyer.Type'] = compute_buyer_type_vectorized(base)
                base['Buyer.Type'] = base['Buyer.Type'].fillna('Other').astype(str).str.title()
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

                # compute pending age if PR date exists
                if pr_date_col and pr_date_col in open_df.columns:
                    open_df["Pending Age (Days)"] = (pd.to_datetime(pd.Timestamp.today().date()) - pd.to_datetime(open_df[pr_date_col], errors='coerce')).dt.days
                else:
                    open_df["Pending Age (Days)"] = np.nan

                # aggregation map
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
                if purchase_doc_col and purchase_doc_col in open_df.columns: agg_map[purchase_doc_col] = 'first'

                group_by_col = pr_number_col_local if pr_number_col_local and pr_number_col_local in open_df.columns else None
                if group_by_col:
                    open_summary = open_df.groupby(group_by_col).agg(agg_map).reset_index()
                else:
                    open_summary = open_df.reset_index().groupby('_row_id' if '_row_id' in open_df.columns else open_df.index.name or 'index').agg(agg_map).reset_index()

                open_summary = open_summary.drop(columns=['buyer_type','effective_buyer_type'], errors='ignore')

                # show count and charts
                st.metric("🔢 Open PRs", open_summary.shape[0])

                # highlight overdue
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

                # download
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

# ----------------- Remaining tabs (Approval / Delivery / Vendors / etc.) -----------------
with T[2]:
    st.subheader('📋 PO Approval Summary')
    po_create = po_create_col
    po_approved = safe_col(fil, ['po_approved_date','po approved date','po_approved_date'])
    if po_approved and po_create and po_create in fil.columns and purchase_doc_col:
        def build_po_approval():
            df_ = fil[fil[po_create].notna()].copy()
            df_[po_approved] = pd.to_datetime(df_[po_approved], errors='coerce')
            df_['approval_lead_time'] = (df_[po_approved] - df_[po_create]).dt.days
            return df_
        po_app_df = memoized_compute('po_approval', filter_signature, build_po_approval)
        total_pos = po_app_df[purchase_doc_col].nunique()
        approved_pos = po_app_df[po_app_df[po_approved].notna()][purchase_doc_col].nunique()
        pending_pos = total_pos - approved_pos
        avg_approval = po_app_df['approval_lead_time'].mean()
        c1,c2,c3,c4 = st.columns(4)
        c1.metric('Total POs', total_pos); c2.metric('Approved POs', approved_pos); c3.metric('Pending Approval', pending_pos); c4.metric('Avg Approval Lead Time (days)', f"{avg_approval:.1f}" if avg_approval==avg_approval else 'N/A')
        st.subheader('📄 PO Approval Details')
        show_cols = [col for col in [ 'po_creator', purchase_doc_col, po_create, po_approved, 'approval_lead_time'] if col]
        po_detail = po_app_df[show_cols].sort_values('approval_lead_time', ascending=False)
        if purchase_doc_col and purchase_doc_col in po_detail.columns:
            po_detail = po_detail.drop_duplicates(subset=[purchase_doc_col], keep='first')
        st.dataframe(po_detail, use_container_width=True)
    else:
        st.info("ℹ️ 'PO Approved Date' column not found or Purchase Doc missing.")

with T[3]:
    st.subheader('Delivery Summary')
    dv = fil.copy()
    po_qty_col = safe_col(dv, ['po_qty','po quantity','po_quantity','po qty'])
    received_col = safe_col(dv, ['receivedqty','received_qty','received qty','received_qty'])
    if po_qty_col and received_col and po_qty_col in dv.columns and received_col in dv.columns:
        def build_delivery():
            tmp = dv[[po_qty_col, received_col, purchase_doc_col, po_vendor_col]].copy()
            tmp['po_qty_f'] = tmp[po_qty_col].fillna(0).astype(float)
            tmp['received_f'] = tmp[received_col].fillna(0).astype(float)
            tmp['pct_received'] = np.where(tmp['po_qty_f']>0, tmp['received_f']/tmp['po_qty_f']*100, 0)
            return tmp.groupby([purchase_doc_col, po_vendor_col], dropna=False).agg({'po_qty_f':'sum','received_f':'sum','pct_received':'mean'}).reset_index()
        ag = memoized_compute('delivery_summary', filter_signature, build_delivery)
        st.dataframe(ag.sort_values('po_qty_f', ascending=False).head(200), use_container_width=True)
    else:
        st.info('Delivery columns (PO Qty / Received QTY) not found.')

with T[4]:
    st.subheader('Top Vendors by Spend')
    if po_vendor_col and net_amount_col and po_vendor_col in fil.columns and net_amount_col in fil.columns:
        def build_vendor_spend():
            df_ = fil.groupby(po_vendor_col, dropna=False)[net_amount_col].sum().reset_index().sort_values(net_amount_col, ascending=False)
            df_['cr'] = df_[net_amount_col]/1e7
            return df_
        vs = memoized_compute('vendor_spend', filter_signature, build_vendor_spend)
        st.dataframe(vs.head(50), use_container_width=True)
    else:
        st.info('Vendor / Net Amount columns not present.')

with T[5]:
    st.subheader('Dept & Services — PR Budget perspective')
    dept_df = fil.copy()
    def pick_pr_dept(r):
        candidates = []
        for c in [pr_bu_col, pr_budget_desc_col, po_bu_col, po_budget_desc_col, pr_budget_code_col]:
            if c and c in r.index and pd.notna(r[c]) and str(r[c]).strip()!='':
                candidates.append(str(r[c]).strip())
        return candidates[0] if candidates else 'Unmapped / Missing'
    def build_dept_df():
        dept = fil.copy()
        dept['pr_department_unified'] = dept.apply(pick_pr_dept, axis=1)
        return dept
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

with T[7]:
    st.subheader('Forecast Next Month Spend (SMA)')
    if trend_date_col and net_amount_col and net_amount_col in fil.columns:
        def build_monthly_total():
            t = fil.dropna(subset=['_month_bucket']).copy()
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
        fdf = pd.DataFrame({'Month': list(m_cr.index) + [nxt], 'SpendCr': list(m_cr.values) + [np.nan], 'SMA': list(sma.values) + [mu]})
        fig = go.Figure(); fig.add_bar(x=fdf['Month'], y=fdf['SpendCr'], name='Actual (Cr)'); fig.add_scatter(x=fdf['Month'], y=fdf['SMA'], mode='lines+markers', name=f'SMA{k}')
        st.plotly_chart(fig, use_container_width=True)

with T[8]:
    st.subheader('Vendor Scorecard')
    if po_vendor_col and po_vendor_col in fil.columns:
        vendor = st.selectbox('Pick Vendor', sorted(fil[po_vendor_col].dropna().astype(str).unique().tolist()))
        vd = fil[fil[po_vendor_col].astype(str) == str(vendor)].copy()
        spend = vd.get(net_amount_col, pd.Series(0)).sum()/1e7 if net_amount_col else 0
        upos = int(vd.get(purchase_doc_col, pd.Series(dtype=object)).nunique()) if purchase_doc_col else 0
        k1,k2 = st.columns(2); k1.metric('Spend (Cr)', f"{spend:.2f}"); k2.metric('Unique POs', upos)
        st.dataframe(vd.head(200), use_container_width=True)

with T[9]:
    st.subheader('🔍 Keyword Search')
    valid_cols = [c for c in [pr_number_col, purchase_doc_col, 'product_name', po_vendor_col] if c in df.columns]
    query = st.text_input('Type vendor, product, PO, PR, etc.', '')
    cat_sel = st.multiselect('Filter by Procurement Category', sorted(df.get('procurement_category', pd.Series(dtype=object)).dropna().astype(str).unique().tolist())) if 'procurement_category' in df.columns else []
    vend_sel = st.multiselect('Filter by Vendor', sorted(df.get(po_vendor_col, pd.Series(dtype=object)).dropna().astype(str).unique().tolist())) if po_vendor_col in df.columns else []
    if query and valid_cols:
        mask = pd.Series(False, index=df.index)
        q = query.lower()
        for c in valid_cols:
            mask = mask | df[c].astype(str).str.lower().str.contains(q, na=False)
        res = df[mask].copy()
        if cat_sel:
            res = res[res['procurement_category'].astype(str).isin(cat_sel)]
        if vend_sel and po_vendor_col in df.columns:
            res = res[res[po_vendor_col].astype(str).isin(vend_sel)]
        st.write(f'Found {len(res)} rows')
        st.dataframe(res, use_container_width=True)
        st.download_button('⬇️ Download Search Results', res.to_csv(index=False), file_name='search_results.csv', mime='text/csv')
    else:
        st.caption('Start typing to search…')

with T[10]:
    st.subheader('Full Data — all filtered rows')
    try:
        st.dataframe(fil.reset_index(drop=True), use_container_width=True)
        csv = fil.to_csv(index=False)
        st.download_button('⬇️ Download full filtered data (CSV)', csv, file_name='p2p_full_filtered.csv', mime='text/csv')
    except Exception as e:
        st.error(f'Could not display full data: {e}')

# EOF
