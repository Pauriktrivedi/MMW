# complete_p2p_dashboard.py
import re
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="P2P Dashboard — Full", layout="wide", initial_sidebar_state="expanded")

# ---------------- Helpers ----------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = {}
    for c in df.columns:
        s = str(c).strip()
        s = s.replace("\xa0", " ").replace("\\", "_").replace("/", "_")
        s = "_".join(s.split())
        s = s.lower()
        s = ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in s)
        s = '_'.join([p for p in s.split('_') if p != ''])
        new_cols[c] = s
    return df.rename(columns=new_cols)

def find_col(df: pd.DataFrame, *variants):
    """Find first column name that matches any variant (case-insensitive substring)."""
    if df is None:
        return None
    cols = list(df.columns)
    low = {c.lower(): c for c in cols}
    for v in variants:
        if v is None: 
            continue
        key = str(v).lower()
        # exact or substring
        for lc, orig in low.items():
            if key == lc or key in lc or lc in key:
                return orig
    # fallback: try tokens
    for v in variants:
        if isinstance(v, (list, tuple)):
            for c in cols:
                lc = c.lower()
                if all(tok.lower() in lc for tok in v):
                    return c
    return None

@st.cache_data(show_spinner=False)
def load_default_files():
    files = [("MEPL.xlsx","MEPL"),("MLPL.xlsx","MLPL"),("mmw.xlsx","MMW"),("mmpl.xlsx","MMPL")]
    frames = []
    for fn, ent in files:
        try:
            df = pd.read_excel(fn, skiprows=1)
        except Exception:
            try:
                df = pd.read_excel(fn)
            except Exception:
                continue
        df = df.copy()
        df['entity'] = ent
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    x = pd.concat(frames, ignore_index=True, sort=False)
    x = normalize_columns(x)
    return x

@st.cache_data(show_spinner=False)
def load_from_uploaded(files):
    frames = []
    for f in files:
        try:
            df = pd.read_excel(f, skiprows=1)
        except Exception:
            try:
                df = pd.read_excel(f)
            except Exception:
                try:
                    df = pd.read_csv(f)
                except Exception:
                    continue
        df = df.copy()
        name = getattr(f, "name", "uploaded")
        df['entity'] = name.rsplit(".",1)[0]
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    x = pd.concat(frames, ignore_index=True, sort=False)
    x = normalize_columns(x)
    return x

# ---------------- Load data (bottom uploader persisted) ----------------
# If user previously uploaded via bottom uploader, prefer that
uploaded_session = st.session_state.get("_bottom_uploaded", None)
if uploaded_session:
    df = load_from_uploaded(uploaded_session)
else:
    df = load_default_files()

# keep a copy of raw df for search and fallback
df_raw = df.copy()

# ---------- Smart column names we use (normalized) ----------
# We'll try multiple likely names and gracefully handle missing columns.
col_try = {
    'pr_date_submitted': ['pr_date_submitted','pr_date','pr_date_submitted','pr_date_submitted'],
    'po_create_date': ['po_create_date','po create date','po_created','po_date','po_create'],
    'po_approved_date': ['po_approved_date','po approved date'],
    'po_delivery_date': ['po_delivery_date','po delivery date','delivery_date'],
    'pr_number': ['pr_number','pr number','pr_no','prno'],
    'purchase_doc': ['purchase_doc','purchase doc','po_number','po number','po'],
    'net_amount': ['net_amount','net amount','amount','value'],
    'po_vendor': ['po_vendor','po vendor','vendor','supplier'],
    'product_name': ['product_name','product name','item_name','product'],
    'po_unit_rate': ['po_unit_rate','po unit rate','unit_rate','unit rate'],
    'po_quantity': ['po_quantity','po quantity','po_qty','quantity'],
    'receivedqty': ['receivedqty','received_qty','received qty','received'],
    'pending_qty': ['pending_qty','pending qty','pendingqty','pending'],
    'po_orderer': ['po_orderer','po orderer','orderer'],
    'buyer_group': ['buyer_group','buyer group'],
    'procurement_category': ['procurement_category','procurement category','category'],
    'po_department': ['po_department','po department','department','po_dept'],
}

# build mapping of canonical -> actual column name in df
map_col = {}
for canon, tries in col_try.items():
    found = None
    for t in tries:
        found = find_col(df, t)
        if found:
            map_col[canon] = found
            break
    # missing ones simply not included

def C(k):
    return map_col.get(k)

# parse dates where present
for d in ['pr_date_submitted','po_create_date','po_approved_date','po_delivery_date']:
    c = C(d)
    if c and c in df.columns:
        df[c] = pd.to_datetime(df[c], errors='coerce')

# ---------- Ensure df exists (but DON'T st.stop early) ----------
if df is None or df.empty:
    st.warning("No data loaded yet. Use the bottom uploader to upload your Excel/CSV files — uploader is at the bottom of this page.")
    # continue: we still render sidebar and bottom uploader so user can upload files

# ---------------- Sidebar filters ----------------
st.sidebar.header("Filters")

# Financial years (you can edit ranges)
FY = {
    "All Years": (pd.Timestamp("2023-04-01"), pd.Timestamp("2026-03-31")),
    "2023": (pd.Timestamp("2023-04-01"), pd.Timestamp("2024-03-31")),
    "2024": (pd.Timestamp("2024-04-01"), pd.Timestamp("2025-03-31")),
    "2025": (pd.Timestamp("2025-04-01"), pd.Timestamp("2026-03-31")),
}
fy_key = st.sidebar.selectbox("Financial Year", list(FY.keys()), index=0)
pr_start, pr_end = FY[fy_key]

# Month dropdown (sub-filter of FY) — built from data inside the selected FY
pr_col = C('pr_date_submitted') if C('pr_date_submitted') else None
po_col = C('po_create_date') if C('po_create_date') else None
date_basis = pr_col if pr_col and pr_col in df.columns else (po_col if po_col and po_col in df.columns else None)

# Default filtered frame to apply incremental filters
fil = df.copy() if df is not None else pd.DataFrame()

# Apply financial year filter if we have a date basis
if date_basis and date_basis in fil.columns:
    fil = fil[(fil[date_basis] >= pr_start) & (fil[date_basis] <= pr_end)]

# Build months list from filtered data
sel_month = "All Months"
if date_basis and date_basis in fil.columns and not fil[date_basis].dropna().empty:
    months = fil[date_basis].dt.to_period("M").astype(str).unique().tolist()
    months = sorted(months, key=lambda s: pd.Period(s))
    month_labels = [pd.Period(m).strftime("%b-%Y") for m in months]
    label_to_period = {pd.Period(m).strftime("%b-%Y'): m for m in months} if months else {}
    # provide a simple selectbox of months (All Months + available)
    sel_month = st.sidebar.selectbox("Month (sub-filter of Year)", ["All Months"] + month_labels, index=0)
    if sel_month != "All Months":
        # filter to the selected month period
        target_period = label_to_period.get(sel_month)
        if target_period:
            fil = fil[fil[date_basis].dt.to_period("M").astype(str) == target_period]
else:
    # still present a month selector but disabled via info
    st.sidebar.info("Month dropdown will populate after data is loaded for the selected FY.")

# Date range (optional) — after FY and month
if date_basis and date_basis in fil.columns and not fil[date_basis].dropna().empty:
    _min = fil[date_basis].dropna().min().date()
    _max = fil[date_basis].dropna().max().date()
    dr = st.sidebar.date_input("Date range (optional)", ( _min, _max ))
    if isinstance(dr, (list, tuple)) and len(dr) == 2:
        s,e = pd.to_datetime(dr[0]), pd.to_datetime(dr[1])
        fil = fil[(fil[date_basis] >= s) & (fil[date_basis] <= e)]

# Ensure some columns exist for filters (create empty if missing, to avoid KeyErrors)
for safe_col in ['Buyer.Type','Entity','PO.Creator','PO.BuyerType']:
    if safe_col not in fil.columns:
        fil[safe_col] = ""

# Map buyer / creator columns when possible
if C('buyer_group') and C('buyer_group') in df.columns:
    try:
        fil['Buyer Group Code'] = fil[C('buyer_group')].astype(str).str.extract(r'(\d+)')[0].astype(float)
    except Exception:
        fil['Buyer Group Code'] = np.nan
    def _bg_val(row):
        bg = row.get(C('buyer_group'), '')
        code = row.get('Buyer Group Code', np.nan)
        try:
            if bg in ["ME_BG17","MLBG16"]:
                return "Direct"
            if bg in ["Not Available"] or pd.isna(bg) or str(bg).strip()=="":
                return "Indirect"
            if pd.notna(code) and 1 <= int(code) <= 9:
                return "Direct"
            if pd.notna(code) and 10 <= int(code) <= 18:
                return "Indirect"
        except Exception:
            pass
        return "Other"
    fil['Buyer.Type'] = fil.apply(_bg_val, axis=1)
else:
    fil['Buyer.Type'] = fil.get('Buyer.Type', 'Unknown')

# PO orderer -> PO.Creator (simple mapping fallback)
po_orderer_col = C('po_orderer')
map_orderer = {"mmw2324030":"Dhruv","mmw2324062":"Deepak","mmw2425154":"Mukul","mmw2223104":"Paurik","mmw2021181":"Nayan","mmw2223014":"Aatish","mmw_ext_002":"Deepakex","mmw2425024":"Kamlesh","mmw2021184":"Suresh","n/a":"Dilip"}
if po_orderer_col and po_orderer_col in fil.columns:
    fil['PO Orderer Clean'] = fil[po_orderer_col].fillna("N/A").astype(str).str.strip()
else:
    fil['PO Orderer Clean'] = "N/A"
fil['PO.Creator'] = fil['PO Orderer Clean'].str.lower().map(lambda v: map_orderer.get(str(v).lower(), None))
fil['PO.Creator'] = fil['PO.Creator'].fillna(fil['PO Orderer Clean']).replace({"N/A":"Dilip"})
fil['PO.BuyerType'] = np.where(fil['PO.Creator'].isin(list(map_orderer.values())), "Indirect", "Direct")

# Build choices for filters
buyer_choices = sorted(fil['Buyer.Type'].dropna().unique().tolist()) if 'Buyer.Type' in fil.columns else []
entity_choices = sorted(fil['entity'].dropna().unique().tolist()) if 'entity' in fil.columns else []
po_creator_choices = sorted(fil['PO.Creator'].dropna().unique().tolist()) if 'PO.Creator' in fil.columns else []
po_buyertype_choices = sorted(fil['PO.BuyerType'].dropna().unique().tolist()) if 'PO.BuyerType' in fil.columns else []

vendor_col = C('po_vendor')
product_col = C('product_name')
po_dept_col = C('po_department')

vendor_choices = sorted(fil[vendor_col].dropna().unique().tolist()) if vendor_col and vendor_col in fil.columns else []
product_choices = sorted(fil[product_col].dropna().unique().tolist()) if product_col and product_col in fil.columns else []
dept_choices = sorted(fil[po_dept_col].dropna().unique().tolist()) if po_dept_col and po_dept_col in fil.columns else []

# Sidebar selection widgets (multi-select for vendor & item)
sel_b = st.sidebar.multiselect("Buyer Type", buyer_choices, default=buyer_choices)
sel_e = st.sidebar.multiselect("Entity", entity_choices, default=entity_choices)
sel_o = st.sidebar.multiselect("PO Ordered By", po_creator_choices, default=po_creator_choices)
sel_p = st.sidebar.multiselect("PO Buyer Type", po_buyertype_choices, default=po_buyertype_choices)
sel_v = st.sidebar.multiselect("Vendor (pick one or more)", vendor_choices, default=vendor_choices) if vendor_choices else []
sel_i = st.sidebar.multiselect("Item / Product (pick one or more)", product_choices, default=product_choices) if product_choices else []
sel_dept = st.sidebar.multiselect("PO Department", ["All Departments"] + dept_choices, default=["All Departments"]) if dept_choices else ["All Departments"]

# Apply the sidebar filters to the 'fil' dataframe
if sel_b:
    fil = fil[fil['Buyer.Type'].isin(sel_b)]
if sel_e:
    fil = fil[fil['entity'].isin(sel_e)]
if sel_o:
    fil = fil[fil['PO.Creator'].isin(sel_o)]
if sel_p:
    fil = fil[fil['PO.BuyerType'].isin(sel_p)]
if sel_v and vendor_col and vendor_col in fil.columns:
    fil = fil[fil[vendor_col].astype(str).isin(sel_v)]
if sel_i and product_col and product_col in fil.columns:
    fil = fil[fil[product_col].astype(str).isin(sel_i)]
if sel_dept and "All Departments" not in sel_dept and po_dept_col and po_dept_col in fil.columns:
    fil = fil[fil[po_dept_col].astype(str).isin(sel_dept)]

# Reset filters: only clear keys we use and rerun safely
if st.sidebar.button("Reset Filters"):
    keys_to_clear = ['_bottom_uploaded', '_bottom_uploader_initial', '_bottom_uploader_bottom', 'sidebar_date_range']
    for k in keys_to_clear:
        if k in st.session_state:
            try:
                del st.session_state[k]
            except Exception:
                pass
    try:
        st.experimental_rerun()
    except Exception:
        st.sidebar.info("Filters reset — please refresh if UI didn't update automatically.")

# ---------------- Tabs ----------------
tabs = st.tabs(["KPIs & Spend","PO/PR Timing","Delivery","Vendors","Dept & Services","Unit-rate Outliers","Forecast","Scorecards","Search","Full Data"])

# ---------- KPIs & Spend (combined) ----------
with tabs[0]:
    st.header("P2P Dashboard — Indirect")
    st.caption("Purchase-to-Pay overview (Indirect spend focus)")
    col1,col2,col3,col4,col5 = st.columns(5)
    # counts (safe access)
    total_prs = int(fil[C('pr_number')].nunique()) if C('pr_number') and C('pr_number') in fil.columns else int(fil.get('pr_number', pd.Series(dtype=object)).nunique()) if 'pr_number' in fil.columns else 0
    total_pos = int(fil[C('purchase_doc')].nunique()) if C('purchase_doc') and C('purchase_doc') in fil.columns else int(fil.get('purchase_doc', pd.Series(dtype=object)).nunique()) if 'purchase_doc' in fil.columns else 0
    col1.metric("Total PRs", total_prs)
    col2.metric("Total POs", total_pos)
    col3.metric("Line Items", len(fil))
    col4.metric("Entities", int(fil.get('entity', pd.Series(dtype=object)).nunique()))
    net_col = C('net_amount')
    spend_val = fil[net_col].sum() if net_col and net_col in fil.columns else 0
    col5.metric("Spend (Cr ₹)", f"{spend_val/1e7:,.2f}")

    st.markdown("---")
    # Monthly spend + cumulative
    dcol = C('po_create_date') if C('po_create_date') and C('po_create_date') in fil.columns else (C('pr_date_submitted') if C('pr_date_submitted') and C('pr_date_submitted') in fil.columns else None)
    st.subheader("Monthly Total Spend + Cumulative")
    if dcol and net_col and dcol in fil.columns:
        t = fil.dropna(subset=[dcol]).copy()
        t['month_ts'] = t[dcol].dt.to_period("M").dt.to_timestamp()
        t['month_str'] = t['month_ts'].dt.strftime("%b-%Y")
        agg = t.groupby(['month_ts','month_str'], as_index=False)[net_col].sum().sort_values('month_ts')
        agg['cr'] = agg[net_col]/1e7
        agg['cumcr'] = agg['cr'].cumsum()
        fig = make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_bar(x=agg['month_str'], y=agg['cr'], name='Monthly Spend (Cr ₹)')
        fig.add_scatter(x=agg['month_str'], y=agg['cumcr'], name='Cumulative (Cr ₹)', mode='lines+markers', secondary_y=True)
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Monthly spend needs a valid date column and Net Amount.")

    st.subheader("Entity Trend")
    if dcol and net_col and dcol in fil.columns:
        x = fil.dropna(subset=[dcol]).copy()
        x['month_ts'] = x[dcol].dt.to_period("M").dt.to_timestamp()
        g = x.groupby(['month_ts','entity'], as_index=False)[net_col].sum()
        g['cr'] = g[net_col]/1e7
        if not g.empty:
            st.plotly_chart(px.line(g, x=g['month_ts'].dt.strftime('%b-%Y'), y='cr', color='entity', markers=True, labels={'x':'Month','cr':'Cr ₹'}).update_layout(xaxis_tickangle=-45), use_container_width=True)

# ---------- PO/PR Timing ----------
with tabs[1]:
    st.subheader("SLA (PR→PO ≤7d)")
    if C('po_create_date') and C('pr_date_submitted') and C('po_create_date') in fil.columns and C('pr_date_submitted') in fil.columns:
        ld = fil.dropna(subset=[C('po_create_date'),C('pr_date_submitted')]).copy()
        ld['lead_days'] = (ld[C('po_create_date')] - ld[C('pr_date_submitted')]).dt.days
        avg = float(ld['lead_days'].mean()) if not ld.empty else 0.0
        fig = go.Figure(go.Indicator(mode='gauge+number', value=avg, number={'suffix':' d'}, gauge={'axis':{'range':[0,max(14,avg*1.2 if avg else 14)]}, 'bar':{'color':'darkblue'}, 'steps':[{'range':[0,7],'color':'lightgreen'},{'range':[7,max(14,avg*1.2 if avg else 14)],'color':'lightcoral'}], 'threshold':{'line':{'color':'red','width':4}, 'value':7}}))
        st.plotly_chart(fig, use_container_width=True)
        bins=[0,7,15,30,60,90,999]; labels=['0-7','8-15','16-30','31-60','61-90','90+']
        ag = pd.cut(ld['lead_days'], bins=bins, labels=labels).value_counts(normalize=True).reindex(labels, fill_value=0).reset_index()
        ag.columns=['Bucket','Pct']; ag['Pct']*=100
        st.plotly_chart(px.bar(ag, x='Bucket', y='Pct', text='Pct').update_traces(texttemplate='%{text:.1f}%', textposition='outside'), use_container_width=True)
    else:
        st.info("Need PR Date and PO create Date to compute SLA.")

    st.subheader("PR & PO per Month")
    tmp = fil.copy()
    if C('pr_date_submitted') and C('purchase_doc') in fil.columns:
        tmp['pr_month'] = tmp[C('pr_date_submitted')].dt.to_period('M')
        tmp['po_month'] = tmp[C('po_create_date')].dt.to_period('M') if C('po_create_date') in fil.columns else pd.NaT
        ms = tmp.groupby('pr_month').agg({C('pr_number'):'count', C('purchase_doc'):'count'}).reset_index() if C('pr_number') and C('purchase_doc') else pd.DataFrame()
        if not ms.empty:
            ms.columns=['Month','PR Count','PO Count']; ms['Month']=ms['Month'].astype(str)
            st.line_chart(ms.set_index('Month'), use_container_width=True)

# ---------- Delivery ----------
with tabs[2]:
    st.subheader("Delivery Summary")
    dv = fil.rename(columns={C('po_quantity'):'po_qty', C('receivedqty'):'received_qty', C('pending_qty'):'pending_qty'}).copy()
    if 'po_qty' in dv.columns and 'received_qty' in dv.columns:
        dv['pct_received'] = np.where(dv['po_qty'].astype(float)>0, (dv['received_qty'].astype(float)/dv['po_qty'].astype(float))*100, 0.0)
        group_cols = [C('purchase_doc') if C('purchase_doc') else 'purchase_doc', C('po_vendor') if C('po_vendor') else 'po_vendor', C('product_name') if C('product_name') else 'product_name', 'item_description']
        agcols = [c for c in group_cols if c in dv.columns]
        agg_map = {}
        if 'po_qty' in dv.columns: agg_map['po_qty']='sum'
        if 'received_qty' in dv.columns: agg_map['received_qty']='sum'
        if 'pending_qty' in dv.columns: agg_map['pending_qty']='sum'
        if agg_map and agcols:
            summ = dv.groupby(agcols, dropna=False).agg(agg_map).reset_index().sort_values('pending_qty', ascending=False)
            st.dataframe(summ, use_container_width=True)
    else:
        st.info("Delivery needs PO Qty and Received Qty columns.")

# ---------- Vendors ----------
with tabs[3]:
    st.subheader("Top Vendors by Spend")
    if C('po_vendor') and C('purchase_doc') and net_col and C('po_vendor') in fil.columns:
        vs = fil.groupby(C('po_vendor'), dropna=False).agg(Vendor_PO_Count=(C('purchase_doc'),'nunique'), Total_Spend_Cr=(net_col, lambda s: (s.sum()/1e7).round(2))).reset_index().sort_values('Total_Spend_Cr', ascending=False)
        st.dataframe(vs.head(10), use_container_width=True)
        st.plotly_chart(px.bar(vs.head(10), x=C('po_vendor'), y='Total_Spend_Cr', text='Total_Spend_Cr', title='Top 10 Vendors (Cr ₹)').update_traces(textposition='outside'), use_container_width=True)

# ---------- Dept & Services (placeholder) ----------
with tabs[4]:
    st.subheader("Dept & Services (Smart Mapper)")
    st.info("If you want the full 'LOVELY' mapping logic pasted here (budget-code mapping + fuzzy suggestions), tell me and I will paste it in.")

# ---------- Unit-rate Outliers ----------
with tabs[5]:
    st.subheader("Unit-rate Outliers vs Historical Median")
    grp_opts = [c for c in [C('product_name'), 'item_code'] if c and c in fil.columns]
    if grp_opts and C('po_unit_rate') and C('po_unit_rate') in fil.columns:
        grp_by = st.selectbox("Group by", grp_opts, index=0)
        zcols = [grp_by, C('po_unit_rate'), C('purchase_doc') if C('purchase_doc') else 'purchase_doc', C('pr_number') if C('pr_number') else 'pr_number', C('po_vendor') if C('po_vendor') else 'po_vendor', 'item_description']
        z = fil[[c for c in zcols if c in fil.columns]].dropna(subset=[grp_by, C('po_unit_rate')]).copy()
        if not z.empty:
            med = z.groupby(grp_by)[C('po_unit_rate')].median().rename('median_rate')
            z = z.join(med, on=grp_by)
            z['pctdev'] = (z[C('po_unit_rate')] - z['median_rate'])/z['median_rate'].replace(0, np.nan)
            thr = st.slider("Outlier threshold (±%)", 10, 300, 50, 5)
            out = z[abs(z['pctdev']) >= thr/100.0].copy()
            out['pctdev%'] = (out['pctdev']*100).round(1)
            st.dataframe(out.sort_values('pctdev%', ascending=False), use_container_width=True)
    else:
        st.info("Need PO Unit Rate and grouping column (Product Name / Item Code).")

# ---------- Forecast ----------
with tabs[6]:
    st.subheader("Forecast Next Month Spend (SMA)")
    dcol = C('po_create_date') if C('po_create_date') and C('po_create_date') in fil.columns else (C('pr_date_submitted') if C('pr_date_submitted') and C('pr_date_submitted') in fil.columns else None)
    if dcol and net_col and dcol in fil.columns:
        t = fil.dropna(subset=[dcol]).copy(); t['month'] = t[dcol].dt.to_period('M').dt.to_timestamp()
        m = t.groupby('month')[net_col].sum().sort_index(); m_cr = (m/1e7)
        k = st.slider('Window (months)', 3, 12, 6, 1)
        sma = m_cr.rolling(k).mean()
        mu = m_cr.tail(k).mean() if len(m_cr) >= k else m_cr.mean()
        sd = m_cr.tail(k).std(ddof=1) if len(m_cr) >= k else m_cr.std(ddof=1)
        n = min(k, max(1, len(m_cr)))
        se = sd/np.sqrt(n) if sd == sd else 0
        lo, hi = float(mu-1.96*se), float(mu+1.96*se)
        nxt = (m_cr.index.max() + pd.offsets.MonthBegin(1)) if len(m_cr) > 0 else pd.Timestamp.today().to_period('M').to_timestamp()
        fdf = pd.DataFrame({'Month': list(m_cr.index) + [nxt], 'SpendCr': list(m_cr.values) + [np.nan], 'SMA': list(sma.values) + [mu]})
        fig = go.Figure()
        fig.add_bar(x=fdf['Month'], y=fdf['SpendCr'], name='Actual (Cr)')
        fig.add_scatter(x=fdf['Month'], y=fdf['SMA'], mode='lines+markers', name=f'SMA{k}')
        fig.add_scatter(x=[nxt,nxt], y=[lo,hi], mode='lines', name='95% CI')
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Forecast for {nxt.strftime('%b-%Y')}: {mu:.2f} Cr (95% CI: {lo:.2f}–{hi:.2f})")
    else:
        st.info("Need a date column and Net Amount to forecast.")

# ---------- Scorecards ----------
with tabs[7]:
    st.subheader("Vendor Scorecard")
    if C('po_vendor') and C('po_vendor') in fil.columns:
        vendor_list = sorted(fil[C('po_vendor')].dropna().astype(str).unique().tolist())
        if vendor_list:
            vendor = st.selectbox("Pick Vendor", vendor_list)
            vd = fil[fil[C('po_vendor')].astype(str) == str(vendor)].copy()
            spend = vd.get(net_col, pd.Series(0)).sum()/1e7 if net_col and net_col in vd.columns else 0
            k1,k2,k3,k4 = st.columns(4)
            k1.metric("Spend (Cr)", f"{spend:.2f}")
            k2.metric("Unique POs", int(vd.get(C('purchase_doc'), pd.Series(dtype=object)).nunique()) if C('purchase_doc') in vd.columns else 0)

# ---------- Search ----------
with tabs[8]:
    st.subheader("🔍 Keyword Search")
    valid_cols = [c for c in [C('pr_number'), C('purchase_doc'), C('product_name'), C('po_vendor')] if c and c in df_raw.columns]
    query = st.text_input("Type vendor, product, PO, PR, etc.", "")
    cat_col = C('procurement_category') if C('procurement_category') and C('procurement_category') in df_raw.columns else None
    cat_sel = st.multiselect("Filter by Procurement Category", sorted(df_raw.get(cat_col, pd.Series(dtype=object)).dropna().astype(str).unique().tolist())) if cat_col else []
    vend_sel = st.multiselect("Filter by Vendor", sorted(df_raw.get(C('po_vendor') if C('po_vendor') else 'po_vendor', pd.Series(dtype=object)).dropna().astype(str).unique().tolist())) if C('po_vendor') and C('po_vendor') in df_raw.columns else []
    if query and valid_cols:
        mask = pd.Series(False, index=df_raw.index)
        q = query.lower()
        for c in valid_cols:
            mask = mask | df_raw[c].astype(str).str.lower().str.contains(q, na=False)
        res = df_raw[mask].copy()
        if cat_sel and cat_col:
            res = res[res[cat_col].astype(str).isin(cat_sel)]
        if vend_sel and C('po_vendor') in df_raw.columns:
            res = res[res[C('po_vendor')].astype(str).isin(vend_sel)]
        st.write(f"Found {len(res)} rows")
        st.dataframe(res, use_container_width=True)
        st.download_button("⬇️ Download Search Results", res.to_csv(index=False), file_name="search_results.csv", mime="text/csv", key="dl_search")
    elif not valid_cols:
        st.info("No searchable columns detected in raw data.")
    else:
        st.caption("Start typing to search…")

# ---------- Full Data ----------
with tabs[9]:
    st.subheader("Full Filtered Dataset")
    st.write("This shows the full dataset after applying sidebar filters. Use the download button to export CSV.")
    st.dataframe(fil.reset_index(drop=True), use_container_width=True)
    st.download_button("⬇️ Download filtered dataset (CSV)", fil.to_csv(index=False), file_name='filtered_dataset.csv', mime='text/csv')

# ---------- Bottom uploader (always shown) ----------
st.markdown("---")
st.markdown("### Upload files (bottom) — upload replaces the dataset used by the dashboard")
new_files = st.file_uploader("Upload Excel/CSV files here (bottom uploader)", type=['xlsx','xls','csv'], accept_multiple_files=True, key='_bottom_uploader_bottom')
if new_files:
    st.session_state['_bottom_uploaded'] = new_files
    try:
        st.experimental_rerun()
    except Exception:
        st.info("Files uploaded — refresh the page if necessary.")
