# p2p_dashboard_full.py
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
    """Normalize dataframe column names to simple snake-like names."""
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

def find_col_by_variants(df, *variants):
    """Return first df column that matches any variant (case-insensitive substring)."""
    if df is None:
        return None
    cols = list(df.columns)
    lc_map = {c.lower(): c for c in cols}
    for v in variants:
        if not v:
            continue
        key = str(v).lower()
        # exact match or substring
        for lc, orig in lc_map.items():
            if key == lc or key in lc or lc in key:
                return orig
    # fallback: token match
    for v in variants:
        if isinstance(v, (list, tuple)):
            toks = [t.lower() for t in v]
            for c in cols:
                lc = c.lower()
                if all(tok in lc for tok in toks):
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

# ---------------- Load data (prefer session uploaded files if present) ----------------
uploaded_session = st.session_state.get("_bottom_uploaded", None)
if uploaded_session:
    df = load_from_uploaded(uploaded_session)
else:
    df = load_default_files()

# Keep raw for search fallback
df_raw = df.copy() if df is not None else pd.DataFrame()

# Defensive: ensure df exists
if df is None:
    df = pd.DataFrame()

# ---------------- column hints (canonical keys -> possible variants) ----------------
col_candidates = {
    "pr_date_submitted": ["pr_date_submitted","pr date submitted","pr date","pr_date"],
    "po_create_date": ["po_create_date","po create date","po date","po created","po_create"],
    "po_delivery_date": ["po_delivery_date","po delivery date","delivery_date"],
    "pr_number": ["pr_number","pr number","pr no","prno"],
    "purchase_doc": ["purchase_doc","purchase doc","po_number","po number","po"],
    "net_amount": ["net_amount","net amount","amount","netamount","value"],
    "po_vendor": ["po_vendor","po vendor","vendor","supplier"],
    "product_name": ["product_name","product name","product","item_name","item code"],
    "po_unit_rate": ["po_unit_rate","po unit rate","unit_rate","unit rate","rate"],
    "po_quantity": ["po_quantity","po quantity","po_qty","quantity"],
    "receivedqty": ["receivedqty","received_qty","received qty","received"],
    "pending_qty": ["pending_qty","pending qty","pendingqty","pending"],
    "po_orderer": ["po_orderer","po orderer","orderer"],
    "buyer_group": ["buyer_group","buyer group"],
    "procurement_category": ["procurement_category","procurement category","category"],
    "po_department": ["po_department","po department","po_dept","department"]
}

# Build a mapping: canonical -> actual column name (or None)
map_col = {}
for canon, tries in col_candidates.items():
    found = find_col_by_variants(df, *tries)
    if found:
        map_col[canon] = found

def C(k):
    return map_col.get(k)

# Parse important dates safely
for d in ['pr_date_submitted','po_create_date','po_delivery_date']:
    c = C(d)
    if c and c in df.columns:
        df[c] = pd.to_datetime(df[c], errors='coerce')

# ---------------- Sidebar: Filters ----------------
st.sidebar.header("Filters")

FY = {
    "All Years": (pd.Timestamp("2023-04-01"), pd.Timestamp("2026-03-31")),
    "2023": (pd.Timestamp("2023-04-01"), pd.Timestamp("2024-03-31")),
    "2024": (pd.Timestamp("2024-04-01"), pd.Timestamp("2025-03-31")),
    "2025": (pd.Timestamp("2025-04-01"), pd.Timestamp("2026-03-31")),
}
fy_key = st.sidebar.selectbox("Financial Year", list(FY.keys()), index=0)
pr_start, pr_end = FY[fy_key]

# Working frame to apply filters incrementally
fil = df.copy()

# pick date basis (prefer PR date then PO create date)
date_basis = None
if C("pr_date_submitted") and C("pr_date_submitted") in fil.columns:
    date_basis = C("pr_date_submitted")
elif C("po_create_date") and C("po_create_date") in fil.columns:
    date_basis = C("po_create_date")

# Apply FY filter if we have a date basis
if date_basis:
    fil = fil[(fil[date_basis] >= pr_start) & (fil[date_basis] <= pr_end)]

# Months subfilter (built from data inside selected FY)
sel_month = "All Months"
if date_basis and (date_basis in fil.columns) and (not fil[date_basis].dropna().empty):
    months = fil[date_basis].dt.to_period("M").astype(str).unique().tolist()
    months = sorted(months, key=lambda s: pd.Period(s))
    month_labels = [pd.Period(m).strftime("%b-%Y") for m in months]
    label_to_period = {pd.Period(m).strftime("%b-%Y"): m for m in months} if months else {}
    sel_month = st.sidebar.selectbox("Month (sub-filter of FY)", ["All Months"] + month_labels, index=0)
    if sel_month != "All Months":
        target_period = label_to_period.get(sel_month)
        if target_period:
            fil = fil[fil[date_basis].dt.to_period("M").astype(str) == target_period]
else:
    # show hint, but still allow other filters
    st.sidebar.info("Month dropdown will populate after data is present for the selected FY.")

# Optional explicit date range (applies after FY + month)
if date_basis and date_basis in fil.columns and (not fil[date_basis].dropna().empty):
    _min = fil[date_basis].dropna().min().date()
    _max = fil[date_basis].dropna().max().date()
    dr = st.sidebar.date_input("Date range (optional)", ( _min, _max ))
    if isinstance(dr, (list, tuple)) and len(dr) == 2:
        s,e = pd.to_datetime(dr[0]), pd.to_datetime(dr[1])
        fil = fil[(fil[date_basis] >= s) & (fil[date_basis] <= e)]

# Buyer type derivation (safe)
if C("buyer_group") and C("buyer_group") in fil.columns:
    try:
        fil['buyer_group_code'] = fil[C("buyer_group")].astype(str).str.extract(r'(\d+)')[0].astype(float)
    except Exception:
        fil['buyer_group_code'] = np.nan
    def map_buyer(row):
        bg = row.get(C("buyer_group"), '')
        code = row.get('buyer_group_code', np.nan)
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
    fil['buyer_type'] = fil.apply(map_buyer, axis=1)
else:
    fil['buyer_type'] = fil.get('buyer_type', 'Unknown')

# PO orderer -> PO.Creator (map a few known IDs)
po_ord_col = C("po_orderer") if C("po_orderer") else None
map_orderer = {"mmw2324030":"Dhruv","mmw2324062":"Deepak","mmw2425154":"Mukul","mmw2223104":"Paurik","mmw2021181":"Nayan","mmw2223014":"Aatish","mmw_ext_002":"Deepakex","mmw2425024":"Kamlesh","mmw2021184":"Suresh","n/a":"Dilip"}
if po_ord_col and po_ord_col in fil.columns:
    fil['po_orderer_clean'] = fil[po_ord_col].fillna("N/A").astype(str).str.strip()
else:
    fil['po_orderer_clean'] = "N/A"
fil['po_creator'] = fil['po_orderer_clean'].str.lower().map(lambda v: map_orderer.get(str(v).lower(), None))
fil['po_creator'] = fil['po_creator'].fillna(fil['po_orderer_clean']).replace({"N/A":"Dilip"})
fil['po_buyer_type'] = np.where(fil['po_creator'].isin(list(map_orderer.values())), "Indirect", "Direct")

# Build filter choices
buyer_choices = sorted(fil['buyer_type'].dropna().unique().tolist()) if 'buyer_type' in fil.columns else []
entity_choices = sorted(fil['entity'].dropna().unique().tolist()) if 'entity' in fil.columns else []
po_creator_choices = sorted(fil['po_creator'].dropna().unique().tolist()) if 'po_creator' in fil.columns else []
po_buyertype_choices = sorted(fil['po_buyer_type'].dropna().unique().tolist()) if 'po_buyer_type' in fil.columns else []

vendor_col = C("po_vendor")
product_col = C("product_name")
po_dept_col = C("po_department")

vendor_choices = sorted(fil[vendor_col].dropna().unique().tolist()) if vendor_col and vendor_col in fil.columns else []
product_choices = sorted(fil[product_col].dropna().unique().tolist()) if product_col and product_col in fil.columns else []
dept_choices = sorted(fil[po_dept_col].dropna().unique().tolist()) if po_dept_col and po_dept_col in fil.columns else []

# Sidebar widgets
sel_b = st.sidebar.multiselect("Buyer Type", buyer_choices, default=buyer_choices)
sel_e = st.sidebar.multiselect("Entity", entity_choices, default=entity_choices)
sel_o = st.sidebar.multiselect("PO Ordered By", po_creator_choices, default=po_creator_choices)
sel_p = st.sidebar.multiselect("PO Buyer Type", po_buyertype_choices, default=po_buyertype_choices)
sel_v = st.sidebar.multiselect("Vendor (pick one or more)", vendor_choices, default=vendor_choices) if vendor_choices else []
sel_i = st.sidebar.multiselect("Item / Product (pick one or more)", product_choices, default=product_choices) if product_choices else []
sel_dept = st.sidebar.multiselect("PO Department", ["All Departments"] + dept_choices, default=["All Departments"]) if dept_choices else ["All Departments"]

# Apply filter selections
if sel_b:
    fil = fil[fil['buyer_type'].isin(sel_b)]
if sel_e:
    fil = fil[fil['entity'].isin(sel_e)]
if sel_o:
    fil = fil[fil['po_creator'].isin(sel_o)]
if sel_p:
    fil = fil[fil['po_buyer_type'].isin(sel_p)]
if sel_v and vendor_col and vendor_col in fil.columns:
    fil = fil[fil[vendor_col].astype(str).isin(sel_v)]
if sel_i and product_col and product_col in fil.columns:
    fil = fil[fil[product_col].astype(str).isin(sel_i)]
if sel_dept and "All Departments" not in sel_dept and po_dept_col and po_dept_col in fil.columns:
    fil = fil[fil[po_dept_col].astype(str).isin(sel_dept)]

# Reset filters (safe)
if st.sidebar.button("Reset Filters"):
    keys_to_clear = ['_bottom_uploaded', '_bottom_uploader_initial', '_bottom_uploader_bottom']
    for k in keys_to_clear:
        if k in st.session_state:
            try:
                del st.session_state[k]
            except Exception:
                pass
    try:
        st.experimental_rerun()
    except Exception:
        st.sidebar.info("Filters reset — refresh if UI didn't update.")

# ---------------- Tabs ----------------
tabs = st.tabs(["KPIs & Spend","PO/PR Timing","Delivery","Vendors","Dept & Services","Unit-rate Outliers","Forecast","Scorecards","Search","Full Data"])

# ---------- KPIs & Spend (same page) ----------
with tabs[0]:
    st.header("P2P Dashboard — Indirect")
    st.caption("Purchase-to-Pay overview (Indirect spend focus)")
    a,b,c,d,e = st.columns(5)
    pr_col = C('pr_number')
    po_col = C('purchase_doc')
    net_col = C('net_amount')
    total_prs = int(fil[pr_col].nunique()) if pr_col and pr_col in fil.columns else int(fil.get('pr_number', pd.Series(dtype=object)).nunique()) if 'pr_number' in fil.columns else 0
    total_pos = int(fil[po_col].nunique()) if po_col and po_col in fil.columns else int(fil.get('purchase_doc', pd.Series(dtype=object)).nunique()) if 'purchase_doc' in fil.columns else 0
    a.metric("Total PRs", total_prs)
    b.metric("Total POs", total_pos)
    c.metric("Line Items", len(fil))
    d.metric("Entities", int(fil.get('entity', pd.Series(dtype=object)).nunique()))
    spend_val = fil[net_col].sum() if net_col and net_col in fil.columns else 0
    e.metric("Spend (Cr ₹)", f"{spend_val/1e7:,.2f}")

    st.markdown("---")
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
        st.info("Monthly Spend requires a date column and Net Amount.")

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
        st.info("Need PR Date and PO create Date for SLA.")

    st.subheader("PR & PO per Month")
    tmp = fil.copy()
    if C('pr_date_submitted') and C('purchase_doc') and C('pr_date_submitted') in tmp.columns:
        tmp['pr_month'] = tmp[C('pr_date_submitted')].dt.to_period('M')
        tmp['po_month'] = tmp[C('po_create_date')].dt.to_period('M') if C('po_create_date') and C('po_create_date') in tmp.columns else pd.NaT
        if C('pr_number') and C('purchase_doc'):
            ms = tmp.groupby('pr_month').agg({C('pr_number'):'count', C('purchase_doc'):'count'}).reset_index()
            if not ms.empty:
                ms.columns=['Month','PR Count','PO Count']; ms['Month']=ms['Month'].astype(str)
                st.line_chart(ms.set_index('Month'), use_container_width=True)

# ---------- Delivery ----------
with tabs[2]:
    st.subheader("Delivery Summary")
    dv = fil.rename(columns={C('po_quantity'):'po_qty', C('receivedqty'):'received_qty', C('pending_qty'):'pending_qty'}).copy()
    if 'po_qty' in dv.columns and 'received_qty' in dv.columns:
        dv['pct_received'] = np.where(dv['po_qty'].astype(float) > 0, (dv['received_qty'].astype(float)/dv['po_qty'].astype(float))*100, 0.0)
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
        st.info("Delivery view requires PO Qty and Received Qty (or Pending Qty).")

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
    st.info("Ask me if you want the full budget-code mapping logic (fuzzy suggestions + downloads) added here.")

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
        st.info("Need PO Unit Rate and a grouping column (Product Name / Item Code).")

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
        vlist = sorted(fil[C('po_vendor')].dropna().astype(str).unique().tolist())
        if vlist:
            vendor = st.selectbox("Pick Vendor", vlist)
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

# ---------- Full Data tab (filtered dataset) ----------
with tabs[9]:
    st.subheader("Full Filtered Dataset")
    st.write("The table below reflects the dataset after applying sidebar filters.")
    st.dataframe(fil.reset_index(drop=True), use_container_width=True)
    st.download_button("⬇️ Download filtered dataset (CSV)", fil.to_csv(index=False), file_name="filtered_dataset.csv", mime="text/csv")

# ---------- Bottom uploader (always visible) ----------
st.markdown("---")
st.markdown("### Upload files (bottom) — these replace the dataset used by the dashboard")
new_files = st.file_uploader("Upload Excel/CSV files here (bottom uploader)", type=['xlsx','xls','csv'], accept_multiple_files=True, key='_bottom_uploader_bottom')
if new_files:
    st.session_state['_bottom_uploaded'] = new_files
    try:
        st.experimental_rerun()
    except Exception:
        st.info("Files uploaded — refresh the page if necessary.")
