# Full P2P dashboard — copy/paste ready
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="P2P Dashboard — Full (Refactor)", layout="wide", initial_sidebar_state="expanded")

# ---- Helpers ----
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names into safe snake-case like names (keeps mapping simple)."""
    new_cols = {}
    for c in df.columns:
        s = str(c).strip()
        s = s.replace("\xa0", " ").replace("\\", "_").replace("/", "_")
        s = "_".join(s.split())
        s = s.lower()
        s = ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in s)
        s = "_".join([p for p in s.split('_') if p != ''])
        new_cols[c] = s
    return df.rename(columns=new_cols)

def safe_get_series(df, col, fill=None):
    if col in df.columns:
        return df[col]
    return pd.Series([fill]*len(df), index=df.index)

@st.cache_data(show_spinner=False)
def load_files(uploaded_files=None):
    """
    Load either uploaded file buffers (uploaded_files list) or default filenames in working dir.
    Attempts skiprows=1 then fallback to skiprows=0 for messy files.
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
                    continue
            # try to preserve original filename as entity tag if present
            try:
                ent = getattr(f, "name", None) or str(f)
                ent = ent.rsplit(".", 1)[0]
            except Exception:
                ent = "UPLOADED"
            df_temp["entity"] = ent
            frames.append(df_temp)
    else:
        # Default file names used in original app
        defaults = [("MEPL.xlsx","MEPL"), ("MLPL.xlsx","MLPL"), ("mmw.xlsx","MMW"), ("mmpl.xlsx","MMPL")]
        for fn, ent in defaults:
            try:
                df_temp = pd.read_excel(fn, skiprows=1)
            except Exception:
                try:
                    df_temp = pd.read_excel(fn)
                except Exception:
                    continue
            df_temp["entity"] = ent
            frames.append(df_temp)
    if not frames:
        return pd.DataFrame()
    x = pd.concat(frames, ignore_index=True, sort=False)
    x = normalize_columns(x)
    # parse likely date columns (if present)
    for d in ["pr_date_submitted","po_create_date","po_approved_date","po_delivery_date","last_po_date"]:
        if d in x.columns:
            x[d] = pd.to_datetime(x[d], errors="coerce")
    return x

# ---- Page header (always visible) ----
st.markdown(
    """
    <div style="padding:6px 0;">
      <h1 style="margin:0; font-size:34px;">P2P Dashboard — Indirect</h1>
      <div style="color:#444; margin-top:6px;">Purchase-to-Pay overview (Indirect spend focus)</div>
      <hr style="border:0; height:1px; background:#e9eef6; margin-top:10px; margin-bottom:16px;" />
    </div>
    """, unsafe_allow_html=True
)

# ---- Sidebar filters (build in logical order) ----
st.sidebar.header("Filters")

# Financial Year options (FY ranges)
FY = {
    "All Years": (pd.Timestamp("2023-04-01"), pd.Timestamp("2026-03-31")),
    "2023": (pd.Timestamp("2023-04-01"), pd.Timestamp("2024-03-31")),
    "2024": (pd.Timestamp("2024-04-01"), pd.Timestamp("2025-03-31")),
    "2025": (pd.Timestamp("2025-04-01"), pd.Timestamp("2026-03-31")),
    "2026": (pd.Timestamp("2026-04-01"), pd.Timestamp("2027-03-31")),
}
fy_key = st.sidebar.selectbox("Financial Year", list(FY.keys()), index=0)
pr_start_default, pr_end_default = FY[fy_key]

# We'll load data after the file uploader is available; to keep uploader at bottom we'll create placeholders now.
# Load data (prefer uploaded files if session_state holds them)
uploaded_files = st.session_state.get("_uploaded_files", None)
df = load_files(uploaded_files)

# If no data, show a short message but still allow uploader at bottom
if df.empty:
    st.sidebar.info("No data loaded yet. Upload Excel/CSV files at bottom of sidebar or place default files in working folder.")
# Ensure expected date columns exist (normalize mapping variations earlier)
# We'll accept either pr_date_submitted or pr_date_submitted-like cols already normalized
date_basis_candidates = []
for candidate in ["pr_date_submitted","po_create_date","pr_date","po_create_date"]:
    if candidate in df.columns:
        date_basis_candidates.append(candidate)
# prefer PR date submitted if present
date_basis = date_basis_candidates[0] if date_basis_candidates else None

# Month sub-filter (dependent on FY selection)
# We'll build month choices from the date_basis restricted to FY range (if available)
sel_month = "All Months"
month_choices = ["All Months"]
if date_basis and not df.empty:
    # apply FY range to compute months available
    s_range = FY[fy_key]
    dmask = (df[date_basis] >= s_range[0]) & (df[date_basis] <= s_range[1])
    months_periods = df.loc[dmask & df[date_basis].notna(), date_basis].dt.to_period("M").drop_duplicates().sort_values()
    if not months_periods.empty:
        month_labels = [p.strftime("%b-%Y") for p in months_periods.astype(str)]
        month_choices = ["All Months"] + month_labels
        sel_month = st.sidebar.selectbox("Month (sub-filter of FY)", month_choices, index=0)
else:
    sel_month = st.sidebar.selectbox("Month (sub-filter of FY)", month_choices, index=0)

# Date range picker (optional) - default to FY selected range
dr_default = (pr_start_default.date(), pr_end_default.date())
user_dr = st.sidebar.date_input("Date range (optional)", dr_default)
if isinstance(user_dr, tuple) and len(user_dr) == 2:
    pr_start_user = pd.to_datetime(user_dr[0])
    pr_end_user = pd.to_datetime(user_dr[1])
else:
    # Single date selected -> treat as full day
    pr_start_user = pd.to_datetime(user_dr[0])
    pr_end_user = pd.to_datetime(user_dr[-1] if isinstance(user_dr, list) else user_dr)

# At this point we also add other filter placeholders that depend on data being loaded.
# We'll prepare the filtered DataFrame 'fil' from df step-by-step, guarding for missing columns.

fil = df.copy()

# Apply FY range first (use PR date if available, else PO create date)
if date_basis and not fil.empty:
    fy_start, fy_end = FY[fy_key]
    fil = fil[(fil[date_basis] >= fy_start) & (fil[date_basis] <= fy_end)]

# Apply Month sub-filter (if chosen) — matches 'MMM-YYYY' label
if sel_month and sel_month != "All Months" and date_basis and not fil.empty:
    # Find period string in fil
    # convert months in fil to labels and filter
    fil_period = fil[date_basis].dt.to_period("M").dt.strftime("%b-%Y")
    fil = fil[fil_period == sel_month]

# Apply optional arbitrary date range (user override)
if date_basis and not fil.empty:
    fil = fil[(fil[date_basis] >= pr_start_user) & (fil[date_basis] <= pr_end_user)]

# Ensure key columns exist (create empty safe columns if missing to avoid KeyError)
for col in ['po_department','po_vendor','product_name','buyer_group','po_orderer','entity','purchase_doc','pr_number','net_amount','po_unit_rate','pending_qty','receivedqty']:
    if col not in fil.columns:
        fil[col] = pd.NA

# PO Department, Vendor, Item/Product — as multi-select dropdowns (same style)
# We place them now in the sidebar below date pickers
po_dept_choices = sorted(fil['po_department'].dropna().astype(str).unique().tolist())
sel_po_dept = st.sidebar.multiselect("PO Dept", options=po_dept_choices, default=po_dept_choices if po_dept_choices else [])
if sel_po_dept:
    fil = fil[fil['po_department'].astype(str).isin(sel_po_dept)]

vendor_choices = sorted(fil['po_vendor'].dropna().astype(str).unique().tolist())
sel_vendor = st.sidebar.multiselect("Vendor", options=vendor_choices, default=vendor_choices if vendor_choices else [])
if sel_vendor:
    fil = fil[fil['po_vendor'].astype(str).isin(sel_vendor)]

item_choices = sorted(fil['product_name'].dropna().astype(str).unique().tolist())
sel_item = st.sidebar.multiselect("Item / Product", options=item_choices, default=item_choices if item_choices else [])
if sel_item:
    fil = fil[fil['product_name'].astype(str).isin(sel_item)]

# Buyer Type (derived) and other multi-select filters
# Derive buyer_type from buyer_group if present
if 'buyer_group' in fil.columns:
    try:
        fil['buyer_group_code'] = fil['buyer_group'].astype(str).str.extract(r"(\d+)")[0].astype(float)
    except Exception:
        fil['buyer_group_code'] = np.nan
    def map_buyer_type(bg, code):
        if str(bg).upper() in ['ME_BG17','MLBG16']:
            return 'Direct'
        if (not bg) or str(bg).strip().upper() == 'NOT AVAILABLE':
            return 'Indirect'
        try:
            if not np.isnan(code) and 1 <= int(code) <= 9:
                return 'Direct'
            if not np.isnan(code) and 10 <= int(code) <= 18:
                return 'Indirect'
        except Exception:
            pass
        return 'Other'
    fil['buyer_type'] = [map_buyer_type(r, c) for r,c in zip(fil['buyer_group'], fil.get('buyer_group_code', pd.Series([np.nan]*len(fil))))]
else:
    fil['buyer_type'] = 'Unknown'

# PO Orderer mapping to friendly creators and po_buyer_type
map_orderer = {
    'mmw2324030': 'Dhruv', 'mmw2324062': 'Deepak', 'mmw2425154': 'Mukul', 'mmw2223104': 'Paurik',
    'mmw2021181': 'Nayan', 'mmw2223014': 'Aatish', 'mmw_ext_002': 'Deepakex', 'mmw2425024': 'Kamlesh',
    'mmw2021184': 'Suresh', 'n/a': 'Dilip'
}
fil['po_orderer'] = fil.get('po_orderer', fil.get('po_orderer_lc', pd.Series([pd.NA]*len(fil)))).fillna('N/A').astype(str).str.strip()
fil['po_orderer_lc'] = fil['po_orderer'].str.lower()
fil['po_creator'] = fil['po_orderer_lc'].map(map_orderer).fillna(fil['po_orderer']).replace({'N/A':'Dilip'})
indirect_people = set(['Aatish','Deepak','Deepakex','Dhruv','Dilip','Mukul','Nayan','Paurik','Kamlesh','Suresh'])
fil['po_buyer_type'] = np.where(fil['po_creator'].isin(indirect_people), 'Indirect', 'Direct')

# Multi-select filters for buyer_type, entity, po_creator, po_buyer_type
buyer_choices = sorted(fil['buyer_type'].dropna().unique().tolist())
sel_buyer = st.sidebar.multiselect("Buyer Type", options=buyer_choices, default=buyer_choices if buyer_choices else [])
if sel_buyer:
    fil = fil[fil['buyer_type'].isin(sel_buyer)]

entity_choices = sorted(fil['entity'].dropna().astype(str).unique().tolist())
sel_entity = st.sidebar.multiselect("Entity", options=entity_choices, default=entity_choices if entity_choices else [])
if sel_entity:
    fil = fil[fil['entity'].astype(str).isin(sel_entity)]

po_creator_choices = sorted(fil['po_creator'].dropna().astype(str).unique().tolist())
sel_po_creator = st.sidebar.multiselect("PO Ordered By", options=po_creator_choices, default=po_creator_choices if po_creator_choices else [])
if sel_po_creator:
    fil = fil[fil['po_creator'].astype(str).isin(sel_po_creator)]

po_buyer_type_choices = sorted(fil['po_buyer_type'].dropna().unique().tolist())
sel_po_bt = st.sidebar.multiselect("PO Buyer Type", options=po_buyer_type_choices, default=po_buyer_type_choices if po_buyer_type_choices else [])
if sel_po_bt:
    fil = fil[fil['po_buyer_type'].isin(sel_po_bt)]

# Reset Filters button (clear session state keys used for filters and uploaded files)
if st.sidebar.button("Reset Filters"):
    # list keys we use
    keys = ['_uploaded_files']
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]
    # Clear checkbox-like widgets by forcing a rerun (safe)
    st.experimental_rerun()

# Add space and a hint
st.sidebar.markdown("---")
st.sidebar.markdown("Vendor / Item / PO Department filters will appear above once data is loaded.")

# Bottom Uploader — place at the end so it appears at bottom of sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### Upload / Replace dataset (optional)")
new_uploads = st.sidebar.file_uploader("Upload Excel/CSV files (multiple allowed)", type=['xlsx','xls','csv'], accept_multiple_files=True, key="_uploader_bottom")
if new_uploads:
    # store in session and reload page to pick up dataset
    st.session_state["_uploaded_files"] = new_uploads
    st.experimental_rerun()

# ---- Tabs & main panels ----
tabs = st.tabs(["KPIs & Spend","PO/PR Timing","Delivery","Vendors","Dept & Services","Unit-rate Outliers","Forecast","Scorecards","Search","Full Data"])

# ---- KPIs & Spend (single page with KPIs and monthly spend) ----
with tabs[0]:
    st.subheader("KPIs & Spend")
    c1,c2,c3,c4,c5 = st.columns(5)
    # safe metrics
    total_prs = int(fil.get('pr_number', pd.Series(dtype=object)).nunique()) if 'pr_number' in fil.columns else 0
    total_pos = int(fil.get('purchase_doc', pd.Series(dtype=object)).nunique()) if 'purchase_doc' in fil.columns else 0
    c1.metric("Total PRs", total_prs)
    c2.metric("Total POs", total_pos)
    c3.metric("Line Items", len(fil))
    c4.metric("Entities", int(fil.get('entity', pd.Series(dtype=object)).nunique()))
    spend_val = fil.get('net_amount', pd.Series(0)).sum() if 'net_amount' in fil.columns else 0
    c5.metric("Spend (Cr ₹)", f"{spend_val/1e7:,.2f}")

    st.markdown("---")

    st.subheader("Monthly Total Spend + Cumulative")
    # choose a date column for trend chart
    date_col_for_charts = 'po_create_date' if 'po_create_date' in fil.columns else ('pr_date_submitted' if 'pr_date_submitted' in fil.columns else None)
    if date_col_for_charts and 'net_amount' in fil.columns and not fil.empty:
        tmp = fil.dropna(subset=[date_col_for_charts]).copy()
        tmp['month'] = tmp[date_col_for_charts].dt.to_period('M').dt.to_timestamp()
        m = tmp.groupby('month', as_index=False)['net_amount'].sum().sort_values('month')
        m['cr'] = m['net_amount']/1e7
        m['cumcr'] = m['cr'].cumsum()
        fig = make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_bar(x=m['month'].dt.strftime('%b-%Y'), y=m['cr'], name='Monthly Spend (Cr ₹)')
        fig.add_scatter(x=m['month'].dt.strftime('%b-%Y'), y=m['cumcr'], name='Cumulative (Cr ₹)', mode='lines+markers', secondary_y=True)
        fig.update_layout(xaxis_tickangle=-45, height=420)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Monthly Spend chart unavailable (needs date and net_amount).")

# ---- PO/PR Timing ----
with tabs[1]:
    st.subheader("SLA (PR → PO ≤ 7d)")
    if {'pr_date_submitted','po_create_date'}.issubset(fil.columns):
        ld = fil.dropna(subset=['pr_date_submitted','po_create_date']).copy()
        ld['lead_time_days'] = (ld['po_create_date'] - ld['pr_date_submitted']).dt.days
        avg = float(ld['lead_time_days'].mean()) if not ld.empty else 0.0
        max_range = max(14, avg*1.2 if avg else 14)
        gauge = go.Figure(go.Indicator(mode='gauge+number', value=avg, number={'suffix':' d'},
                                      gauge={'axis':{'range':[0,max_range]}, 'steps':[{'range':[0,7],'color':'lightgreen'},{'range':[7,max_range],'color':'lightcoral'}], 'threshold':{'line':{'color':'red','width':4}, 'value':7}}))
        st.plotly_chart(gauge, use_container_width=True)
        # distribution buckets
        bins = [0,7,15,30,60,90,999]
        labels = ['0-7','8-15','16-30','31-60','61-90','90+']
        ag = pd.cut(ld['lead_time_days'], bins=bins, labels=labels).value_counts(normalize=True).reindex(labels, fill_value=0).reset_index()
        ag.columns = ['Bucket','Pct']; ag['Pct'] = ag['Pct']*100
        st.plotly_chart(px.bar(ag, x='Bucket', y='Pct', text='Pct').update_traces(texttemplate='%{text:.1f}%', textposition='outside'), use_container_width=True)
    else:
        st.info("Need PR and PO create dates to compute lead time.")

    # PR & PO per month line
    st.subheader("PR & PO per Month")
    tmp = fil.copy()
    tmp['pr_month'] = tmp.get('pr_date_submitted', pd.Series(pd.NaT, index=tmp.index)).dt.to_period('M')
    tmp['po_month'] = tmp.get('po_create_date', pd.Series(pd.NaT, index=tmp.index)).dt.to_period('M')
    if 'pr_number' in tmp.columns and 'purchase_doc' in tmp.columns:
        ms = tmp.groupby('pr_month').agg({'pr_number':'count','purchase_doc':'count'}).reset_index()
        if not ms.empty:
            ms.columns = ['Month','PR Count','PO Count']
            ms['Month'] = ms['Month'].astype(str)
            st.line_chart(ms.set_index('Month'), use_container_width=True)

    st.subheader("Weekday Split")
    wd = fil.copy()
    wd['pr_wk'] = wd.get('pr_date_submitted', pd.Series(pd.NaT, index=wd.index)).dt.day_name().fillna('')
    wd['po_wk'] = wd.get('po_create_date', pd.Series(pd.NaT, index=wd.index)).dt.day_name().fillna('')
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    prc = wd['pr_wk'].value_counts().reindex(order, fill_value=0)
    poc = wd['po_wk'].value_counts().reindex(order, fill_value=0)
    c1,c2 = st.columns(2)
    c1.bar_chart(prc)
    c2.bar_chart(poc)

# ---- Delivery ----
with tabs[2]:
    st.subheader("Delivery Summary")
    dv = fil.rename(columns={'po_quantity':'po_qty','receivedqty':'received_qty','pending_qty':'pending_qty'}).copy()
    if {'po_qty','received_qty'}.issubset(dv.columns):
        dv['pct_received'] = np.where(dv['po_qty'].astype(float)>0, (dv['received_qty'].astype(float)/dv['po_qty'].astype(float))*100, 0.0)
        group_cols = [c for c in ['purchase_doc','po_vendor','product_name','item_description'] if c in dv.columns]
        summ = dv.groupby(group_cols, dropna=False).agg({'po_qty':'sum','received_qty':'sum','pending_qty':'sum','pct_received':'mean'}).reset_index()
        if not summ.empty:
            st.dataframe(summ.sort_values('pending_qty', ascending=False), use_container_width=True)
            st.plotly_chart(px.bar(summ.sort_values('pending_qty', ascending=False).head(20), x='purchase_doc', y='pending_qty', color='po_vendor' if 'po_vendor' in summ.columns else None, text='pending_qty').update_traces(textposition='outside'), use_container_width=True)
    else:
        st.info("Need PO Qty and Received Qty columns for delivery summary.")

# ---- Vendors ----
with tabs[3]:
    st.subheader("Top Vendors by Spend")
    if {'po_vendor','purchase_doc','net_amount'}.issubset(fil.columns):
        vs = fil.groupby('po_vendor', dropna=False).agg(Vendor_PO_Count=('purchase_doc','nunique'), Total_Spend_Cr=('net_amount', lambda s: (s.sum()/1e7).round(2))).reset_index().sort_values('Total_Spend_Cr', ascending=False)
        st.dataframe(vs.head(10), use_container_width=True)
        st.plotly_chart(px.bar(vs.head(10), x='po_vendor', y='Total_Spend_Cr', text='Total_Spend_Cr').update_traces(textposition='outside'), use_container_width=True)
    else:
        st.info("Need PO vendor, purchase doc and net amount to show top vendors.")

# ---- Dept & Services (smart mapper) ----
with tabs[4]:
    st.subheader("Dept & Services (Smart Mapper)")
    # minimal safe mapping view: show top departments by spend if dept_chart exists otherwise show PO budget code samples
    if 'dept_chart' in fil.columns and 'net_amount' in fil.columns:
        dep = fil.groupby('dept_chart', dropna=False)['net_amount'].sum().reset_index().sort_values('net_amount', ascending=False)
        dep['cr'] = dep['net_amount']/1e7
        st.plotly_chart(px.bar(dep.head(30), x='dept_chart', y='cr').update_layout(xaxis_tickangle=-45), use_container_width=True)
    else:
        st.info("Dept mapping not available; place mapping files in working folder to enable smart mapper.")

# ---- Unit-rate Outliers ----
with tabs[5]:
    st.subheader("Unit-rate Outliers vs Historical Median")
    grp_candidates = [c for c in ['product_name','item_code'] if c in fil.columns]
    if grp_candidates and 'po_unit_rate' in fil.columns:
        grp_by = st.selectbox("Group by", grp_candidates, index=0)
        z = fil[[grp_by,'po_unit_rate','purchase_doc','pr_number','po_vendor','item_description','po_create_date','net_amount']].dropna(subset=[grp_by,'po_unit_rate']).copy()
        med = z.groupby(grp_by)['po_unit_rate'].median().rename('median_rate')
        z = z.join(med, on=grp_by)
        z['pctdev'] = (z['po_unit_rate'] - z['median_rate']) / z['median_rate'].replace(0, np.nan)
        thr = st.slider('Outlier threshold (±%)', 10, 300, 50, 5)
        out = z[abs(z['pctdev']) >= thr/100.0].copy(); out['pctdev%'] = (out['pctdev']*100).round(1)
        st.dataframe(out.sort_values('pctdev%', ascending=False), use_container_width=True)
        if not z.empty:
            st.plotly_chart(px.scatter(z, x='po_create_date', y='po_unit_rate', color=np.where(abs(z['pctdev'])>=thr/100.0,'Outlier','Normal'), hover_data=[grp_by,'purchase_doc','po_vendor','median_rate']).update_layout(legend_title_text=''), use_container_width=True)
    else:
        st.info("Need 'PO Unit Rate' and a grouping column (product_name / item_code).")

# ---- Forecast ----
with tabs[6]:
    st.subheader("Forecast Next Month Spend (SMA)")
    dcol = 'po_create_date' if 'po_create_date' in fil.columns else ('pr_date_submitted' if 'pr_date_submitted' in fil.columns else None)
    if dcol and 'net_amount' in fil.columns:
        t = fil.dropna(subset=[dcol]).copy(); t['month'] = t[dcol].dt.to_period('M').dt.to_timestamp()
        m = t.groupby('month')['net_amount'].sum().sort_index(); m_cr = (m/1e7)
        k = st.slider('Window (months)', 3, 12, 6, 1)
        sma = m_cr.rolling(k).mean()
        mu = m_cr.tail(k).mean() if len(m_cr) >= k else m_cr.mean()
        sd = m_cr.tail(k).std(ddof=1) if len(m_cr) >= k else m_cr.std(ddof=1)
        n = min(k, max(1, len(m_cr)))
        se = sd/np.sqrt(n) if sd==sd else 0
        lo, hi = float(mu-1.96*se), float(mu+1.96*se)
        nxt = (m_cr.index.max() + pd.offsets.MonthBegin(1)) if len(m_cr) > 0 else pd.Timestamp.today().to_period('M').to_timestamp()
        fdf = pd.DataFrame({'Month': list(m_cr.index) + [nxt], 'SpendCr': list(m_cr.values) + [np.nan], 'SMA': list(sma.values) + [mu]})
        fig = go.Figure(); fig.add_bar(x=fdf['Month'], y=fdf['SpendCr'], name='Actual (Cr)'); fig.add_scatter(x=fdf['Month'], y=fdf['SMA'], mode='lines+markers', name=f'SMA{k}')
        fig.add_scatter(x=[nxt,nxt], y=[lo,hi], mode='lines', name='95% CI')
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Forecast for {nxt.strftime('%b-%Y')}: {mu:.2f} Cr (95% CI: {lo:.2f}–{hi:.2f})")
    else:
        st.info("Need date and Net Amount to forecast.")

# ---- Scorecards / Vendor detail ----
with tabs[7]:
    st.subheader("Vendor Scorecard")
    if 'po_vendor' in fil.columns:
        vendors_list = sorted(fil['po_vendor'].dropna().astype(str).unique().tolist())
        vendor_pick = st.selectbox("Pick Vendor", vendors_list) if vendors_list else None
        if vendor_pick:
            vd = fil[fil['po_vendor'].astype(str) == vendor_pick].copy()
            spend = vd.get('net_amount', pd.Series(0)).sum()/1e7 if 'net_amount' in vd.columns else 0
            upos = int(vd.get('purchase_doc', pd.Series(dtype=object)).nunique()) if 'purchase_doc' in vd.columns else 0
            today = pd.Timestamp.today().normalize()
            late = np.nan
            if 'po_delivery_date' in vd.columns and 'pending_qty' in vd.columns:
                late = ((pd.to_datetime(vd['po_delivery_date'], errors='coerce').dt.date < today.date()) & (vd['pending_qty'].fillna(0) > 0)).sum()
            pend_val = np.nan
            if 'pending_qty' in vd.columns and 'po_unit_rate' in vd.columns:
                vd['pending_value'] = vd['pending_qty'].fillna(0).astype(float) * vd['po_unit_rate'].fillna(0).astype(float)
                pend_val = vd['pending_value'].sum()/1e7
            k1,k2,k3,k4 = st.columns(4)
            k1.metric("Spend (Cr)", f"{spend:.2f}"); k2.metric("Unique POs", upos); k3.metric("Late PO count", None if pd.isna(late) else int(late)); k4.metric("Pending Value (Cr)", None if pd.isna(pend_val) else f"{pend_val:.2f}")
            if 'product_name' in vd.columns and 'po_unit_rate' in vd.columns:
                med = vd.groupby('product_name')['po_unit_rate'].median().rename('median_rate'); v2 = vd.join(med, on='product_name'); v2['var%'] = ((v2['po_unit_rate'] - v2['median_rate']) / v2['median_rate'].replace(0, np.nan)) * 100
                st.plotly_chart(px.box(v2, x='product_name', y='po_unit_rate', points='outliers', title='Price variance by item'), use_container_width=True)

# ---- Search ----
with tabs[8]:
    st.subheader("🔍 Keyword Search")
    valid_cols = [c for c in ['pr_number','purchase_doc','product_name','po_vendor'] if c in df.columns]
    query = st.text_input("Type vendor, product, PO, PR, etc.", "")
    cat_sel = st.multiselect("Filter by Procurement Category", sorted(df.get('procurement_category', pd.Series(dtype=object)).dropna().astype(str).unique().tolist())) if 'procurement_category' in df.columns else []
    vend_sel = st.multiselect("Filter by Vendor", sorted(df.get('po_vendor', pd.Series(dtype=object)).dropna().astype(str).unique().tolist())) if 'po_vendor' in df.columns else []
    if query and valid_cols:
        mask = pd.Series(False, index=df.index)
        q = query.lower()
        for c in valid_cols:
            mask = mask | df[c].astype(str).str.lower().str.contains(q, na=False)
        res = df[mask].copy()
        if cat_sel and 'procurement_category' in df.columns:
            res = res[res['procurement_category'].astype(str).isin(cat_sel)]
        if vend_sel and 'po_vendor' in df.columns:
            res = res[res['po_vendor'].astype(str).isin(vend_sel)]
        st.write(f"Found {len(res)} rows")
        st.dataframe(res, use_container_width=True)
        st.download_button("⬇️ Download Search Results", res.to_csv(index=False), file_name="search_results.csv", mime="text/csv", key="dl_search")
    elif not valid_cols:
        st.info("No searchable columns present.")
    else:
        st.caption("Start typing to search…")

# ---- Full Data tab (complete filtered dataset) ----
with tabs[9]:
    st.subheader("Full Data (after filters)")
    st.caption("This table shows the dataset after all sidebar filters have been applied.")
    # show a safe number of rows initially but allow download of full filtered set
    st.dataframe(fil.reset_index(drop=True), use_container_width=True)
    csv_buf = fil.to_csv(index=False)
    st.download_button("⬇️ Download Filtered Data", csv_buf, file_name="filtered_data.csv", mime="text/csv")

# End of script
