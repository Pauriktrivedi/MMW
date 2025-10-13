import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="P2P Dashboard — Indirect (Fast)", layout="wide", initial_sidebar_state="expanded")

# ===== Lightweight header =====
st.markdown("""
<div style="padding:8px 0 12px 0; margin-bottom:8px;">
  <h1 style="font-size:32px; margin:0; color:#0b1f3b;">P2P Dashboard — Indirect</h1>
  <div style="font-size:13px; color:#23395b; margin-top:6px;">Purchase-to-Pay overview (Indirect spend focus)</div>
  <hr style="border:0; height:1px; background:#e6eef6; margin-top:10px; margin-bottom:12px;" />
</div>
""", unsafe_allow_html=True)

# ===== Helpers =====

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    new = {}
    for c in df.columns:
        s = str(c).strip()
        s = s.replace(" "," ")
        s = s.replace('\','_').replace('/', '_')
        s = '_'.join(s.split()).lower()
        s = ''.join(ch if (ch.isalnum() or ch=='_') else '_' for ch in s)
        s = '_'.join([p for p in s.split('_') if p!=''])
        new[c] = s
    return df.rename(columns=new)

@st.cache_data(ttl=3600, show_spinner=False)
def load_all(files=None, usecols=None):
    """Load the available Excel files. If none found, return empty df.
    The `usecols` argument (list) lets pandas read only necessary columns to speed things up.
    """
    if files is None:
        files = [("MEPL.xlsx","MEPL"),("MLPL.xlsx","MLPL"),("mmw.xlsx","MMW"),("mmpl.xlsx","MMPL")]
    frames = []
    for fn, ent in files:
        try:
            if usecols:
                tmp = pd.read_excel(fn, skiprows=1, usecols=usecols)
            else:
                tmp = pd.read_excel(fn, skiprows=1)
            tmp['entity'] = ent
            frames.append(tmp)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    x = pd.concat(frames, ignore_index=True)
    x = normalize_columns(x)
    # coerce common dates
    for c in ['pr_date_submitted','po_create_date','po_approved_date','po_delivery_date']:
        if c in x.columns:
            x[c] = pd.to_datetime(x[c], errors='coerce')
    return x

# === Load minimal columns first to speed startup ===
minimal_cols = None  # keep None to read full sheet if files are small; set to list of cols if needed
df = load_all(usecols=minimal_cols)

if df.empty:
    st.warning("No data loaded. Put MEPL.xlsx/MLPL.xlsx/mmw.xlsx/mmpl.xlsx next to the app or update load_all().")

# canonical column names (safely detect what's available)
cols = set(df.columns)
pr_col = 'pr_date_submitted' if 'pr_date_submitted' in cols else None
po_create_col = 'po_create_date' if 'po_create_date' in cols else None
net_amount_col = 'net_amount' if 'net_amount' in cols else None
purchase_doc_col = 'purchase_doc' if 'purchase_doc' in cols else None
pr_number_col = 'pr_number' if 'pr_number' in cols else None
po_vendor_col = 'po_vendor' if 'po_vendor' in cols else None
product_col = 'product_name' if 'product_name' in cols else None
entity_col = 'entity'

TODAY = pd.Timestamp.now().normalize()

# ----- Sidebar filters (fast defaults) -----
st.sidebar.header('Filters')
FY = {
    'All Years': (pd.Timestamp('2023-04-01'), pd.Timestamp('2026-03-31')),
    '2023': (pd.Timestamp('2023-04-01'), pd.Timestamp('2024-03-31')),
    '2024': (pd.Timestamp('2024-04-01'), pd.Timestamp('2025-03-31')),
    '2025': (pd.Timestamp('2025-04-01'), pd.Timestamp('2026-03-31'))
}
fy_key = st.sidebar.selectbox('Financial Year', list(FY), index=0)
pr_start, pr_end = FY[fy_key]

# create working filtered frame 'fil' and avoid many copies
fil = df.copy()
if pr_col:
    fil = fil[(fil[pr_col] >= pr_start) & (fil[pr_col] <= pr_end)]

# ensure vendor/product exist
if po_vendor_col not in fil.columns:
    fil['po_vendor'] = ''
    po_vendor_col = 'po_vendor'
if product_col not in fil.columns:
    fil['product_name'] = ''
    product_col = 'product_name'

# Build choices lazily but limit size of widgets to keep UI snappy
vendor_choices = sorted(fil[po_vendor_col].dropna().unique().tolist())
product_choices = sorted(fil[product_col].dropna().unique().tolist())

# Use empty default for multiselect to avoid rendering thousands of tags by default
sel_v = st.sidebar.multiselect('Vendor (pick one or more)', vendor_choices, default=None, key='sv')
sel_p = st.sidebar.multiselect('Item / Product (pick one or more)', product_choices, default=None, key='sp')

# Reset filters (clears only our keys)
if st.sidebar.button('Reset Filters'):
    for k in ['sv','sp','fy_key']:
        if k in st.session_state:
            del st.session_state[k]
    st.experimental_rerun()

# Apply filters (fast boolean masks)
if sel_v:
    mask_v = fil[po_vendor_col].isin(sel_v)
    fil = fil[mask_v]
if sel_p:
    mask_p = fil[product_col].isin(sel_p)
    fil = fil[mask_p]

# ----- Tabs -----
T = st.tabs(['KPIs & Spend','PO/PR Timing','Delivery','Vendors','Dept & Services','Unit-rate Outliers','Forecast','Search'])

# ===== KPIs & Spend (combined) =====
with T[0]:
    c1,c2,c3,c4,c5 = st.columns(5)
    total_prs = int(fil[pr_number_col].nunique()) if pr_number_col in fil.columns else 0
    total_pos = int(fil[purchase_doc_col].nunique()) if purchase_doc_col in fil.columns else 0
    line_items = len(fil)
    entities = int(fil[entity_col].nunique()) if entity_col in fil.columns else 0
    spend_val = fil[net_amount_col].sum() if net_amount_col in fil.columns else 0
    c1.metric('Total PRs', total_prs)
    c2.metric('Total POs', total_pos)
    c3.metric('Line Items', line_items)
    c4.metric('Entities', entities)
    c5.metric('Spend (Cr ₹)', f"{spend_val/1e7:,.2f}")

    st.markdown('---')
    # Monthly spend (limit to last 24 months for speed)
    if po_create_col and net_amount_col in fil.columns:
        t = fil.dropna(subset=[po_create_col])[[po_create_col, net_amount_col]].copy()
        t['month'] = t[po_create_col].dt.to_period('M').dt.to_timestamp()
        m = t.groupby('month')[net_amount_col].sum().sort_index()
        m = m.tail(24)
        dfm = m.reset_index().rename(columns={net_amount_col:'amount'})
        dfm['cr'] = dfm['amount']/1e7
        dfm['cumcr'] = dfm['cr'].cumsum()
        fig = make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_bar(x=dfm['month'], y=dfm['cr'], name='Monthly Spend (Cr ₹)')
        fig.add_scatter(x=dfm['month'], y=dfm['cumcr'], name='Cumulative (Cr ₹)', mode='lines+markers', secondary_y=True)
        fig.update_layout(xaxis_tickangle=-45, height=380)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('Monthly spend requires Po create Date and Net Amount columns.')

    st.markdown('---')
    # Entity trend (top 4 entities)
    if po_create_col and net_amount_col in fil.columns and entity_col in fil.columns:
        x = fil.dropna(subset=[po_create_col, net_amount_col])[[po_create_col, entity_col, net_amount_col]].copy()
        x['month'] = x[po_create_col].dt.to_period('M').dt.to_timestamp()
        g = x.groupby(['month', entity_col])[net_amount_col].sum().reset_index()
        g['cr'] = g[net_amount_col]/1e7
        top_entities = g.groupby(entity_col)['cr'].sum().nlargest(4).index.tolist()
        g = g[g[entity_col].isin(top_entities)]
        fig2 = px.line(g, x=g['month'].dt.strftime('%b-%Y'), y='cr', color=entity_col, markers=True, labels={'cr':'Cr ₹','x':'Month'})
        fig2.update_layout(xaxis_tickangle=-45, height=320)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info('Entity trend requires Po create Date, Net Amount and Entity.')

# ===== Dept & Services: use PR department column if present =====
with T[4]:
    st.subheader('Dept & Services (PR Department)')
    # check a few likely normalized names for PR dept
    candidates = [c for c in ['pr_department','pr_dept','pr_dept_name','pr_department_name','dept_chart','department'] if c in fil.columns]
    if candidates:
        pr_dept_col = candidates[0]
        st.write(f"Using department column: **{pr_dept_col}** for aggregation")
        d = fil.copy()
        d[pr_dept_col] = d[pr_dept_col].astype(str).str.strip().replace({'nan':pd.NA})
        if net_amount_col in d.columns:
            dep = d.groupby(pr_dept_col, dropna=False)[net_amount_col].sum().reset_index().sort_values(net_amount_col, ascending=False)
            dep['Cr'] = dep[net_amount_col]/1e7
            topn = min(30, len(dep))
            if topn>0:
                st.plotly_chart(px.bar(dep.head(topn), x=pr_dept_col, y='Cr').update_layout(xaxis_tickangle=-45, yaxis_title='Cr ₹'), use_container_width=True)
                st.dataframe(dep.head(200).assign(Spend_Cr=lambda df: (df[net_amount_col]/1e7).round(2)), use_container_width=True)
            else:
                st.info('No departments with spend found.')
        else:
            st.info('Net Amount missing — cannot compute department spend.')
    else:
        st.info('No PR department-like column found. Consider mapping or add a column named pr_department.')

# ===== Light-weight Search tab =====
with T[7]:
    st.subheader('🔍 Keyword Search (fast)')
    valid_cols = [c for c in [pr_number_col, purchase_doc_col, product_col, po_vendor_col] if c in fil.columns]
    q = st.text_input('Search Vendor/Product/PO/PR')
    if q and valid_cols:
        mask = pd.Series(False, index=fil.index)
        for c in valid_cols:
            mask = mask | fil[c].astype(str).str.contains(q, case=False, na=False)
        res = fil.loc[mask]
        st.write(f'Found {len(res)} rows — displaying up to 500')
        st.dataframe(res.head(500), use_container_width=True)

# EOF
