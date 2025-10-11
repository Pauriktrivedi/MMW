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

# Extra plain-text fallback to ensure visibility
_st.write("## P2P Dashboard — Indirect")

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
def load_all(file_list=None):
    """Load multiple Excel files (list of tuples (filename, Entity)).
    If file_list is None, try defaults and skip missing files.
    """
    if file_list is None:
        file_list = [("MEPL.xlsx","MEPL"),("MLPL.xlsx","MLPL"),("mmw.xlsx","MMW"),("mmpl.xlsx","MMPL")]
    frames = []
    for fn, ent in file_list:
        try:
            df = pd.read_excel(fn, skiprows=1)
            df['entity'] = ent
            frames.append(df)
        except FileNotFoundError:
            # skip missing file but inform user later via returned empty
            continue
    if not frames:
        return pd.DataFrame()
    x = pd.concat(frames, ignore_index=True)
    x = normalize_columns(x)
    # coerce common date columns (if present)
    for c in ['pr_date_submitted','po_create_date','po_approved_date','po_delivery_date']:
        if c in x.columns:
            x[c] = pd.to_datetime(x[c], errors='coerce')
    return x

# Allow optional file upload for local testing
with st.sidebar.expander("Upload data files (optional)"):
    uploaded = st.file_uploader("Upload one or more Excel files", type=["xlsx","xls"], accept_multiple_files=True, key="files")

file_list = None
if uploaded:
    # map to (buffer, filename-without-extension) pairs
    file_list = []
    for f in uploaded:
        try:
            df_temp = pd.read_excel(f, skiprows=1)
            # use the uploaded file name (without extension) as entity tag
            ent = f.name.rsplit('.',1)[0]
            df_temp['entity'] = ent
            df_temp = normalize_columns(df_temp)
            file_list.append((f.name, ent))
        except Exception:
            # fallback: still push file path placeholder so load_all will try reading
            file_list.append((f.name, f.name.rsplit('.',1)[0]))

# Load base dataframe
if file_list:
    # When user uploaded files, we'll read them via pandas directly to construct a DataFrame
    frames = []
    for f in uploaded:
        try:
            df2 = pd.read_excel(f, skiprows=1)
            df2['entity'] = f.name.rsplit('.',1)[0]
            frames.append(df2)
        except Exception:
            continue
    if frames:
        df = pd.concat(frames, ignore_index=True)
        df = normalize_columns(df)
        for c in ['pr_date_submitted','po_create_date','po_approved_date','po_delivery_date']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors='coerce')
    else:
        df = load_all()
else:
    df = load_all()

if df.empty:
    st.warning("No data loaded. Either place default Excel files next to this script (MEPL.xlsx, MLPL.xlsx, mmw.xlsx, mmpl.xlsx) or upload files using the sidebar.")

# ----------------- Prepare and normalize columns used in app -----------------
# unify common column names used later to expected snake_case
# Create helper series for frequently used columns to avoid KeyError
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
    df['buyer_type'] = df.apply(map_buyer_type, axis=1) if buyer_group_col in df.columns else 'Unknown'

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

# ----------------- Sidebar Filters -----------------
st.sidebar.header('Filters')
FY = {
    'All Years': (pd.Timestamp('2023-04-01'), pd.Timestamp('2026-03-31')),
    '2023': (pd.Timestamp('2023-04-01'), pd.Timestamp('2024-03-31')),
    '2024': (pd.Timestamp('2024-04-01'), pd.Timestamp('2025-03-31')),
    '2025': (pd.Timestamp('2025-04-01'), pd.Timestamp('2026-03-31')),
    '2026': (pd.Timestamp('2026-04-01'), pd.Timestamp('2027-03-31'))
}
fy_key = st.sidebar.selectbox('Financial Year', list(FY))
pr_start, pr_end = FY[fy_key]

fil = df.copy()
if pr_col and pr_col in fil.columns:
    fil = fil[(fil[pr_col] >= pr_start) & (fil[pr_col] <= pr_end)]

# ensure filter columns exist
for col in ['buyer_type', entity_col, 'po_creator', 'po_buyer_type']:
    if col not in fil.columns:
        fil[col] = ''
    fil[col] = fil[col].astype(str).str.strip()

# Ensure buyer_type_unified exists in the filtered frame — defensive creation to avoid KeyError
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
if sel_b:
    fil = fil[fil['buyer_type_unified'].isin(sel_b)]
if sel_e:
    fil = fil[fil[entity_col].isin(sel_e)]
if sel_o:
    fil = fil[fil['po_creator'].isin(sel_o)]
if sel_p:
    fil = fil[fil['po_buyer_type'].isin(sel_p)]

# ----------------- Page header -----------------
with st.container():
    st.markdown("<h1 style='margin:0.2rem 0 0.0rem 0'>P2P Dashboard — Indirect</h1>", unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom:0.6rem'><strong>Purchase-to-Pay overview (Indirect spend focus)</strong></div>", unsafe_allow_html=True)
    st.markdown('---')

# ----------------- Tabs -----------------
T = st.tabs(['KPIs & Spend','PO/PR Timing','Delivery','Vendors','Dept & Services','Unit-rate Outliers','Forecast','Scorecards','Search'])

# ----------------- KPIs & Spend -----------------
with T[0]:
    # --- Top KPI metrics ---
    c1,c2,c3,c4,c5 = st.columns(5)
    total_prs = int(fil.get(pr_number_col, pd.Series(dtype=object)).nunique() if pr_number_col else 0)
    total_pos = int(fil.get(purchase_doc_col, pd.Series(dtype=object)).nunique() if purchase_doc_col else 0)
    c1.metric('Total PRs', total_prs)
    c2.metric('Total POs', total_pos)
    c3.metric('Line Items', len(fil))
    c4.metric('Entities', int(fil.get(entity_col, pd.Series(dtype=object)).nunique()))
    spend_val = fil.get(net_amount_col, pd.Series(0)).sum() if net_amount_col else 0
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

    # layout: metrics on top, then spend charts stacked vertically
    if fig_spend is not None:
        st.plotly_chart(fig_spend, use_container_width=True)
    else:
        st.info('Monthly Spend chart not available — need date and Net Amount columns.')

    st.markdown('---')

    if fig_entity is not None:
        st.plotly_chart(fig_entity, use_container_width=True)
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
    ms = tmp.groupby('pr_month').agg({pr_number_col:'count', purchase_doc_col:'count'}).reset_index() if pr_number_col and purchase_doc_col else pd.DataFrame()
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

    # Lead time by Buyer Type & Buyer
    st.subheader('Lead Time by Buyer Type & Buyer')
    if pr_col in fil.columns and po_create_col in fil.columns:
        ld2 = fil.dropna(subset=[po_create_col, pr_col]).copy()
        ld2['lead_time_days'] = (ld2[po_create_col] - ld2[pr_col]).dt.days
        if 'buyer_type' in ld2.columns:
            st.dataframe(ld2.groupby('buyer_type')['lead_time_days'].mean().round(1).reset_index().sort_values('lead_time_days'), use_container_width=True)
        if 'po_creator' in ld2.columns:
            st.dataframe(ld2.groupby('po_creator')['lead_time_days'].mean().round(1).reset_index().sort_values('lead_time_days'), use_container_width=True)

    # Daily PR Trends
    st.subheader('Daily PR Submissions')
    if pr_col in fil.columns:
        daily = fil.copy()
        daily['pr_date'] = pd.to_datetime(daily[pr_col], errors='coerce')
        dtrend = daily.groupby('pr_date').size().reset_index(name='PR Count')
        if not dtrend.empty:
            st.plotly_chart(px.line(dtrend, x='pr_date', y='PR Count', title='Daily PRs'), use_container_width=True)

    # Monthly Unique PO Generation
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
    dv = fil.rename(columns={
        'po_quantity':'po_qty', 'receivedqty':'received_qty', 'pending_qty':'pending_qty'
    }).copy()
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

# ----------------- Smart Budget Mapper / Dept & Services -----------------
with T[4]:
    st.subheader('Dept & Services (Smart Mapper)')

    def _norm_series_local(s):
        s = s.astype(str).str.upper().str.strip()
        s = s.str.replace('\xa0',' ', regex=False).str.replace('&','AND', regex=False).str.replace('R&D','RANDD', regex=False)
        s = s.str.replace(r'[/\\_\-]+','.', regex=True).str.replace(r'\s+','', regex=True).str.replace(r'\.{2,}','.', regex=True)
        s = s.str.replace(r'^\.+|\.+$','', regex=True).str.replace(r'[^A-Z0-9\.]','', regex=True)
        return s

    def _norm_one_local(x):
        return _norm_series_local(pd.Series([x])).iloc[0] if pd.notna(x) else ''

    # small built-in mapping that'll be safe if mapping files aren't present
    P3F = {}
    P3 = {}
    P3s = {}
    EM = {}
    EMs = {}
    EP = {}
    EPs = {}

    smart = fil.copy()
    smart['dept_chart'], smart['subcat_chart'], smart['__src'] = pd.NA, pd.NA, 'UNMAPPED'

    # try to load mapping files if present in current directory
    exp = None
    for name in ['Expanded_Budget_Code_Mapping.xlsx', 'Final_Budget_Mapping_Completed_Verified.xlsx']:
        try:
            exp = pd.read_excel(name)
            break
        except Exception:
            exp = None

    if exp is not None and not exp.empty:
        m = exp.copy(); m.columns = m.columns.astype(str).str.strip()
        code_col = next((c for c in m.columns if c.lower().strip() in ['budget code','code','budget_code']), None)
        dept_col = next((c for c in m.columns if 'department' in c.lower()), None)
        subc_col = next((c for c in m.columns if ('subcat' in c.lower()) or ('sub category' in c.lower()) or ('subcategory' in c.lower())), None)
        p3_col = next((c for c in m.columns if 'prefix_3' in c.lower()), None)
        ent_col = next((c for c in m.columns if c.lower().strip() in ['entity','domain','company','prefix_1','brand']), None)
        if code_col and dept_col:
            m['__code'] = _norm_series_local(m[code_col])
            tmp = m.dropna(subset=['__code']).drop_duplicates('__code')
            EM = dict(zip(tmp['__code'], tmp[dept_col].astype(str).str.strip()))
            if subc_col:
                EMs = dict(zip(tmp['__code'], tmp[subc_col].astype(str).str.strip()))
        if p3_col:
            m['__p3'] = _norm_series_local(m[p3_col])
        if ent_col:
            m['__ent'] = _norm_series_local(m[ent_col])
        if p3_col and dept_col and ent_col and not m.empty:
            g = m.dropna(subset=['__p3','__ent']).copy()
            if not g.empty:
                EP = g.groupby(['__ent','__p3'])[dept_col].agg(lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0]).to_dict()
                if subc_col: EPs = g.groupby(['__ent','__p3'])[subc_col].agg(lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0]).to_dict()
        if p3_col and dept_col and not m.empty:
            g2 = m.dropna(subset=['__p3']).copy()
            if not g2.empty:
                P3 = g2.groupby('__p3')[dept_col].agg(lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0]).to_dict()
                if subc_col: P3s = g2.groupby('__p3')[subc_col].agg(lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0]).to_dict()

    def pick_p3_local(code):
        if not isinstance(code, str) or not code:
            return None
        segs = code.split('.')
        for j in range(len(segs)-1, -1, -1):
            if (segs[j] in P3) or (segs[j] in P3F):
                return segs[j]
        return None

    def map_one_local(code_raw, ent_raw):
        code = _norm_one_local(code_raw); ent = _norm_one_local(ent_raw)
        if not code:
            return (pd.NA, pd.NA, 'UNMAPPED')
        if code in EM:
            return (EM.get(code), EMs.get(code, pd.NA), 'EXACT')
        parts = code.split('.')
        if len(parts) > 1:
            for i in range(1, len(parts)):
                suf = '.'.join(parts[i:])
                if suf in EM:
                    return (EM.get(suf), EMs.get(suf, pd.NA), 'HIER')
        p3 = pick_p3_local(code)
        if p3 and ent and (ent, p3) in EP:
            return (EP.get((ent,p3)), EPs.get((ent,p3), pd.NA), 'ENTITY_PFX')
        if p3 and (p3 in P3):
            return (P3.get(p3), P3s.get(p3, pd.NA), 'PFX3')
        if p3 and (p3 in P3F):
            return (P3F.get(p3), pd.NA, 'KEYWORD')
        return (pd.NA, pd.NA, 'UNMAPPED')

    # apply mapping using first available budget code column
    bcol_candidates = [c for c in ['po_budget_code','pr_budget_code','po_budget_code_1'] if c in smart.columns]
    if bcol_candidates:
        base = smart[bcol_candidates[0]]
        ent = smart.get(entity_col, pd.Series([pd.NA]*len(smart)))
        mp = pd.DataFrame([map_one_local(c,e) for c,e in zip(base.tolist(), ent.tolist())], columns=['dept_chart','subcat_chart','__src'], index=smart.index)
        for c in ['dept_chart','subcat_chart','__src']:
            smart[c] = smart[c].combine_first(mp[c]) if c in smart.columns else mp[c]
    smart['dept_chart'].fillna('Unmapped / Missing', inplace=True)

    # ---- Debug: show top unmapped budget codes and suggestions ----
    import difflib

    # determine which budget-code column we used (safe fallback)
    bcol_candidates = [c for c in ['po_budget_code','pr_budget_code','po_budget_code_1'] if c in smart.columns]
    base_col = bcol_candidates[0] if bcol_candidates else None

    # identify unmapped rows (use __src==UNMAPPED OR dept_chart == 'Unmapped / Missing')
    is_unmapped = (smart.get('__src', pd.Series('', index=smart.index)) == 'UNMAPPED') | (smart.get('dept_chart','').astype(str).str.contains('Unmapped', na=False))
    unmapped = smart[is_unmapped].copy()

    st.markdown('### Unmapped budget codes — quick diagnostics')
    if unmapped.empty:
        st.info('No unmapped lines detected (good!).')
    else:
        # Top N budget codes by count and spend
        topn = 30
        if base_col:
            spend_col = net_amount_col if net_amount_col in unmapped.columns else (unmapped.columns[0] if len(unmapped.columns)>0 else None)
            agg = unmapped.groupby(base_col, dropna=False).agg(
                Lines=('procurement_category' if 'procurement_category' in unmapped.columns else base_col, 'count'),
                Spend=(spend_col, 'sum')
            ).reset_index().sort_values('Lines', ascending=False)
            if 'Spend' in agg.columns:
                agg['SpendCr'] = agg['Spend'] / 1e7
            agg = agg.rename(columns={base_col: 'BudgetCode', 'Lines': 'Count'})
            cols_to_show = ['BudgetCode','Count'] + (['SpendCr'] if 'SpendCr' in agg.columns else [])
            st.dataframe(agg.head(topn)[cols_to_show], use_container_width=True)
        else:
            st.write(f'No budget-code column found among {bcol_candidates} — showing sample unmapped rows.')
            st.dataframe(unmapped.head(50), use_container_width=True)

        # Download full list of unmapped rows for offline review
        csv_buf = unmapped.to_csv(index=False)
        st.download_button('Download unmapped rows (CSV)', csv_buf, file_name='unmapped_rows.csv', mime='text/csv')

        # Fuzzy suggestions against mapping keys (if mapping dicts exist)
        mapping_candidates = []
        try:
            if isinstance(EM, dict) and EM:
                mapping_candidates += list(EM.keys())
        except Exception:
            pass
        try:
            if isinstance(P3, dict) and P3:
                mapping_candidates += list(P3.keys())
        except Exception:
            pass
        try:
            if isinstance(P3F, dict) and P3F:
                mapping_candidates += list(P3F.keys())
        except Exception:
            pass
        mapping_candidates = list(dict.fromkeys(mapping_candidates))  # unique preserve order

        if base_col and mapping_candidates:
            # normalize unmapped budget-code values similar to mapping logic but without regex
            def norm(v):
                try:
                    s = str(v).upper().strip()
                    s = s.replace(chr(160),' ').replace('&','AND').replace('R&D','RANDD')
                    s = s.replace('/','.')
                    s = s.replace('_','.')
                    s = s.replace('-','.')
                    s = s.replace(chr(92), '.')  # backslash -> dot
                    # remove any non-alnum or dot
                    s = ''.join(ch for ch in s if (ch.isalnum() or ch == '.'))
                    while '..' in s:
                        s = s.replace('..','.')
                    s = s.strip('.')
                    s = s.replace(' ','')
                    return s
                except Exception:
                    return str(v)

            # build list of unique budget codes to suggest for
            code_list = unmapped[base_col].dropna().astype(str).str.strip().unique().tolist()
            suggestions = []
            for code in code_list[:200]:  # limit work to first 200 for performance
                ncode = norm(code)
                matches = difflib.get_close_matches(ncode, mapping_candidates, n=5, cutoff=0.55)
                suggestions.append({'BudgetCode': code, 'Normalized': ncode, 'TopMatches': ';'.join(matches) if matches else ''})
            sug_df = pd.DataFrame(suggestions)
            if not sug_df.empty:
                st.markdown('**Suggested close matches (auto-suggest)** — review and copy into alias/override file as needed')
                st.dataframe(sug_df.head(200), use_container_width=True)
                st.download_button('Download suggested aliases (CSV)', sug_df.to_csv(index=False), file_name='alias_suggestions.csv', mime='text/csv')
            else:
                st.info('No mapping keys found to suggest fuzzy matches.')
        else:
            st.info('No mapping candidates found (EM/P3/P3F empty) — consider uploading or placing mapping Excel files in working folder.')

    # Canonical labels + optional overrides
    DEPT_ALIASES = {
        'HR & ADMIN':'HR & Admin','HUMAN RESOURCES':'HR & Admin','LEGAL':'Legal & IP','LEGAL & IP':'Legal & IP',
        'PROGRAM':'Program','R AND D':'R&D','R&D':'R&D','RANDD':'R&D','RESEARCH & DEVELOPMENT':'R&D','INFRASTRUCTURE':'Infra',
        'INFRA':'Infra','CUSTOMER SUCCESS':'Customer Success','MFG':'Manufacturing','MANUFACTURING':'Manufacturing','DESIGN':'Design',
        'MARKETING':'Marketing','SALES':'Sales','SS & SCM':'SS & SCM','SUPPLY CHAIN':'SS & SCM','FINANCE':'Finance','RENTAL OFFICES':'Rental Offices'
    }
    SUBCAT_ALIASES = {'HOUSEKEEPING':'Admin, Housekeeping and Security','ADMIN, HOUSEKEEPING AND SECURITY':'Admin, Housekeeping and Security','ELECTRICITY EXPENSES':'Electricity','PANTY':'Pantry and Canteen','PANTRY':'Pantry and Canteen','TRAVEL':'Travel & Other'}

    def canonize(s, mapping):
        return s.apply(lambda v: mapping.get(str(v).strip().upper(), str(v).strip()) if pd.notna(v) else v)

    with st.sidebar.expander('Alias overrides (optional)'):
        up = st.file_uploader('Upload alias overrides CSV/XLSX', type=['csv','xlsx'], key='alias_up')
        if up is not None:
            try:
                al = pd.read_csv(up) if up.name.lower().endswith('.csv') else pd.read_excel(up)
                al.columns = al.columns.astype(str).str.strip()
                if {'Department','Dept.Alias'}.issubset(al.columns):
                    for k,v in al[['Department','Dept.Alias']].dropna().values:
                        DEPT_ALIASES[str(k).upper().strip()] = str(v).strip()
                if {'Subcategory','Subcat.Alias'}.issubset(al.columns):
                    for k,v in al[['Subcategory','Subcat.Alias']].dropna().values:
                        SUBCAT_ALIASES[str(k).upper().strip()] = str(v).strip()
                st.success('Aliases loaded')
            except Exception as e:
                st.warning(f'Alias file error: {e}')

    smart['dept_chart'] = canonize(smart['dept_chart'].astype(str), DEPT_ALIASES)
    if 'subcat_chart' in smart.columns:
        smart['subcat_chart'] = canonize(smart['subcat_chart'].astype(str), SUBCAT_ALIASES)

    st.caption({'map_src_counts': smart.get('__src', pd.Series()).value_counts(dropna=False).to_dict() if '__src' in smart.columns else {}})

    if net_amount_col in smart.columns:
        dep = smart.groupby('dept_chart', dropna=False)[net_amount_col].sum().reset_index().sort_values(net_amount_col, ascending=False)
        if not dep.empty:
            dep['cr'] = dep[net_amount_col]/1e7
            st.plotly_chart(px.bar(dep.head(30), x='dept_chart', y='cr', title='Department-wise Spend (Top 30)').update_layout(xaxis_tickangle=-45), use_container_width=True)
            cA,cB = st.columns([2,1])
            dept_pick = cA.selectbox('Drill Department', dep['dept_chart'].astype(str).tolist(), key='dept_pick')
            topn = int(cB.number_input('Top N', 5, 100, 20, 5, key='dept_topn'))
            det = smart[smart['dept_chart'].astype(str) == str(dept_pick)].copy()
            k1,k2,k3 = st.columns(3)
            k1.metric('Lines', len(det)); k2.metric('PRs', int(det.get(pr_number_col, pd.Series(dtype=object)).nunique() if pr_number_col else 0)); k3.metric('Spend (Cr ₹)', f"{det.get(net_amount_col, pd.Series(0)).sum()/1e7:,.2f}")
            if 'subcat_chart' in det.columns:
                ss = det.groupby('subcat_chart', dropna=False)[net_amount_col].sum().reset_index().sort_values(net_amount_col, ascending=False)
                ss['cr'] = ss[net_amount_col]/1e7
                c1,c2 = st.columns(2)
                c1.plotly_chart(px.bar(ss.head(topn), x='subcat_chart', y='cr', title=f"{dept_pick} — Top Services").update_layout(xaxis_tickangle=-45), use_container_width=True)
                c2.plotly_chart(px.pie(ss.head(12), names='subcat_chart', values=net_amount_col, title=f"{dept_pick} — Service Share"), use_container_width=True)
                svc = st.selectbox('Drill Service', ss['subcat_chart'].astype(str).tolist(), key='svc_pick')
                sub = det[det['subcat_chart'].astype(str) == str(svc)].copy()
                cols = [c for c in ['po_budget_code','subcat_chart','dept_chart', purchase_doc_col, pr_number_col, 'procurement_category', 'product_name', 'item_description', 'po_vendor', net_amount_col] if c in sub.columns]
                if not cols:
                    cols = [c for c in ['dept_chart', purchase_doc_col, pr_number_col, net_amount_col] if c in sub.columns]
                st.dataframe(sub[cols], use_container_width=True)

    st.subheader('Dept × Service (Cr)')
    if {'dept_chart','subcat_chart', net_amount_col}.issubset(smart.columns):
        pv = (smart.pivot_table(index='dept_chart', columns='subcat_chart', values=net_amount_col, aggfunc='sum', fill_value=0)/1e7)
        st.dataframe(pv.round(2), use_container_width=True)
        tm = pv.stack().reset_index(); tm.columns=['Department','Service','Cr']; tm = tm[tm['Cr']>0]
        if not tm.empty:
            st.plotly_chart(px.treemap(tm, path=['Department','Service'], values='Cr', title='Dept→Service Treemap (Cr ₹)'), use_container_width=True)

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

# EOF
