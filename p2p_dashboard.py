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
        s = "_".join(s.split()).lower()
        s = ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in s)
        s = '_'.join([p for p in s.split('_') if p != ''])
        new_cols[c] = s
    return df.rename(columns=new_cols)

@st.cache_data(show_spinner=False)
def load_default_files():
    files = [("MEPL.xlsx","MEPL"),("MLPL.xlsx","MLPL"),("mmw.xlsx","MMW"),("mmpl.xlsx","MMPL")]
    frames = []
    for fn, ent in files:
        try:
            tmp = pd.read_excel(fn, skiprows=1)
        except Exception:
            try:
                tmp = pd.read_excel(fn)
            except Exception:
                continue
        tmp['entity'] = ent
        frames.append(tmp)
    if not frames:
        return pd.DataFrame()
    x = pd.concat(frames, ignore_index=True, sort=False)
    return normalize_columns(x)

@st.cache_data(show_spinner=False)
def load_from_uploaded(uploaded_files):
    frames = []
    for f in uploaded_files:
        try:
            tmp = pd.read_excel(f, skiprows=1)
        except Exception:
            try:
                tmp = pd.read_excel(f)
            except Exception:
                try:
                    tmp = pd.read_csv(f)
                except Exception:
                    continue
        tmp['entity'] = getattr(f, "name", "uploaded")
        frames.append(tmp)
    if not frames:
        return pd.DataFrame()
    x = pd.concat(frames, ignore_index=True, sort=False)
    return normalize_columns(x)


def safe_col(df, patterns, default=None):
    """Find first column in df matching any token in patterns (substring match)."""
    if df is None or df.empty:
        return default
    cols = df.columns.astype(str).tolist()
    for p in patterns:
        for c in cols:
            if p.lower() in c.lower():
                return c
    return default

# ---------------- Load dataset ----------------
uploaded_state = st.session_state.get("_uploaded_files", None)

if uploaded_state:
    df = load_from_uploaded(uploaded_state)
else:
    df = load_default_files()

if df is None:
    df = pd.DataFrame()

if not df.empty:
    df = normalize_columns(df)

# identify commonly used columns robustly
pr_col = safe_col(df, ["pr_date_submitted","pr date submitted","pr date"])
po_create_col = safe_col(df, ["po_create_date","po create date","po date","po_create"])
po_delivery_col = safe_col(df, ["po_delivery_date","po delivery date"])
net_amount_col = safe_col(df, ["net_amount","net amount","amount"])
purchase_doc_col = safe_col(df, ["purchase_doc","purchase doc","purchase_doc","po_number","po number"])
pr_number_col = safe_col(df, ["pr_number","pr number","pr_no","pr no"])
po_vendor_col = safe_col(df, ["po_vendor","po vendor","vendor"])
po_unit_rate_col = safe_col(df, ["po_unit_rate","po unit rate","unit rate"])
pending_qty_col = safe_col(df, ["pending_qty","pending qty","pending"])
received_qty_col = safe_col(df, ["receivedqty","received_qty","received qty"])
po_department_col = safe_col(df, ["po_department","po department","po_dept","po dept","department"])
product_col = safe_col(df, ["product_name","product name","product"])
buyer_group_col = safe_col(df, ["buyer_group","buyer group"])
po_orderer_col = safe_col(df, ["po_orderer","po orderer","po_orderer","orderer"])

# parse dates safely
for c in [pr_col, po_create_col, po_delivery_col]:
    if c and c in df.columns:
        df[c] = pd.to_datetime(df[c], errors="coerce")

# compute buyer type mapping if buyer_group present
if buyer_group_col and buyer_group_col in df.columns:
    try:
        df['buyer_group_code'] = df[buyer_group_col].astype(str).str.extract(r"(\d+)")[0].astype(float)
    except Exception:
        df['buyer_group_code'] = np.nan
    def map_buyer(row):
        bg = str(row.get(buyer_group_col, "")).strip()
        code = row.get('buyer_group_code', np.nan)
        try:
            if bg in ["ME_BG17","MLBG16"]: return "Direct"
            if bg in ["Not Available"] or bg == "" or pd.isna(bg): return "Indirect"
            if not pd.isna(code) and 1 <= int(code) <= 9: return "Direct"
            if not pd.isna(code) and 10 <= int(code) <= 18: return "Indirect"
        except Exception:
            pass
        return "Other"
    df['buyer_type'] = df.apply(map_buyer, axis=1)
else:
    df['buyer_type'] = 'Unknown'

# map PO creator/orderer into names if possible (safe fallback)
map_orderer = {
    'mmw2324030': 'Dhruv', 'mmw2324062': 'Deepak', 'mmw2425154': 'Mukul', 'mmw2223104': 'Paurik',
    'mmw2021181': 'Nayan', 'mmw2223014': 'Aatish', 'mmw_ext_002': 'Deepakex', 'mmw2425024': 'Kamlesh',
    'mmw2021184': 'Suresh', 'n/a': 'Dilip'
}
if po_orderer_col and po_orderer_col in df.columns:
    df['po_orderer_clean'] = df[po_orderer_col].fillna('N/A').astype(str).str.strip().str.lower()
    df['po_creator'] = df['po_orderer_clean'].map({k.lower():v for k,v in map_orderer.items()}).fillna(df[po_orderer_col].fillna('N/A')).replace({'N/A':'Dilip'})
else:
    df['po_creator'] = ''

# ---------------- Sidebar filters ----------------
st.sidebar.header("Filters")

FY = {
    "All Years": (pd.Timestamp("2000-01-01"), pd.Timestamp("2099-12-31")),
    "2023": (pd.Timestamp("2023-04-01"), pd.Timestamp("2024-03-31")),
    "2024": (pd.Timestamp("2024-04-01"), pd.Timestamp("2025-03-31")),
    "2025": (pd.Timestamp("2025-04-01"), pd.Timestamp("2026-03-31")),
}
fy_key = st.sidebar.selectbox("Financial Year", list(FY.keys()), index=0)
fy_start, fy_end = FY[fy_key]

fil = df.copy()

primary_date_col = pr_col if pr_col and pr_col in fil.columns else (po_create_col if po_create_col and po_create_col in fil.columns else None)
if primary_date_col:
    fil = fil[(fil[primary_date_col] >= fy_start) & (fil[primary_date_col] <= fy_end)]

# Month sub-filter of FY
sel_month = "All Months"
if primary_date_col and primary_date_col in fil.columns:
    months = fil[primary_date_col].dropna().dt.to_period("M").astype(str).unique().tolist()
    months = sorted(months, key=lambda s: pd.Period(s))
    month_labels = [pd.Period(m).strftime("%b-%Y") for m in months]
    label_to_period = {pd.Period(m).strftime("%b-%Y"): m for m in months}
    if month_labels:
        sel_month = st.sidebar.selectbox("Month (sub-filter of FY)", ["All Months"] + month_labels, index=0)
        if sel_month != "All Months":
            target_period = label_to_period[sel_month]
            fil = fil[fil[primary_date_col].dt.to_period("M").astype(str) == target_period]

# Optional date range (applies after FY->Month)
if primary_date_col and primary_date_col in fil.columns and not fil[primary_date_col].dropna().empty:
    min_dt = fil[primary_date_col].dropna().min().date()
    max_dt = fil[primary_date_col].dropna().max().date()
    dr = st.sidebar.date_input("Date range (optional)", (min_dt, max_dt), key="sidebar_range")
    if isinstance(dr, (list, tuple)) and len(dr) == 2:
        sdt = pd.to_datetime(dr[0])
        edt = pd.to_datetime(dr[1])
        fil = fil[(fil[primary_date_col] >= sdt) & (fil[primary_date_col] <= edt)]

# PO Dept / Vendor / Item multi-selects
st.sidebar.markdown("")
col_a, col_b, col_c = st.sidebar.columns([1,1,1])

with col_a:
    if po_department_col and po_department_col in df.columns:
        dept_choices = sorted(df[po_department_col].dropna().astype(str).unique().tolist())
        sel_dept = st.multiselect("PO Dept", ["All Departments"] + dept_choices, default=["All Departments"])
        if sel_dept and "All Departments" not in sel_dept:
            fil = fil[fil[po_department_col].astype(str).isin(sel_dept)]
    else:
        st.write("")

with col_b:
    if po_vendor_col and po_vendor_col in df.columns:
        vendor_choices = sorted(df[po_vendor_col].dropna().astype(str).unique().tolist())
        sel_vendor = st.multiselect("Vendor", ["All Vendors"] + vendor_choices, default=["All Vendors"])
        if sel_vendor and "All Vendors" not in sel_vendor:
            fil = fil[fil[po_vendor_col].astype(str).isin(sel_vendor)]
    else:
        st.write("")

with col_c:
    if product_col and product_col in df.columns:
        item_choices = sorted(df[product_col].dropna().astype(str).unique().tolist())
        sel_item = st.multiselect("Item / Product", ["All Items"] + item_choices, default=["All Items"])
        if sel_item and "All Items" not in sel_item:
            fil = fil[fil[product_col].astype(str).isin(sel_item)]
    else:
        st.write("")

# Additional multi-selects
for c in ['buyer_type','entity','po_creator','po_buyer_type']:
    if c not in fil.columns:
        fil[c] = ''

bt_choices = sorted(fil['buyer_type'].dropna().astype(str).unique().tolist())
ent_choices = sorted(fil['entity'].dropna().astype(str).unique().tolist())
pc_choices = sorted(fil['po_creator'].dropna().astype(str).unique().tolist())
pbt_choices = sorted(fil['po_buyer_type'].dropna().astype(str).unique().tolist())

sel_bt = st.sidebar.multiselect("Buyer Type", bt_choices, default=bt_choices)
sel_ent = st.sidebar.multiselect("Entity", ent_choices, default=ent_choices)
sel_pc = st.sidebar.multiselect("PO Ordered By", pc_choices, default=pc_choices)
sel_pbt = st.sidebar.multiselect("PO Buyer Type", pbt_choices, default=pbt_choices)

# apply these filters
if sel_bt:
    fil = fil[fil['buyer_type'].astype(str).isin(sel_bt)]
if sel_ent:
    fil = fil[fil['entity'].astype(str).isin(sel_ent)]
if sel_pc:
    fil = fil[fil['po_creator'].astype(str).isin(sel_pc)]
if sel_pbt:
    fil = fil[fil['po_buyer_type'].astype(str).isin(sel_pbt)]

# Reset filters
if st.sidebar.button("Reset Filters"):
    if "_uploaded_files" in st.session_state:
        del st.session_state["_uploaded_files"]
    if "_bottom_uploader" in st.session_state:
        del st.session_state["_bottom_uploader"]
    st.experimental_rerun()

st.sidebar.markdown("---")
st.sidebar.info("Vendor / Item / PO Department filters are multi-selects. Use Reset Filters to return to defaults.")

# ---------------- Tabs ----------------
tabs = st.tabs(["KPIs & Spend","PO/PR Timing","Delivery","Vendors","Unit-rate Outliers","Forecast","Scorecards","Search","Full Data"])

# ---------------- KPIs & Spend ----------------
with tabs[0]:
    st.subheader("P2P Dashboard — Indirect")
    col1, col2, col3, col4, col5 = st.columns(5)
    total_prs = int(fil.get(pr_number_col, pd.Series(dtype=object)).nunique()) if pr_number_col and pr_number_col in fil.columns else 0
    total_pos = int(fil.get(purchase_doc_col, pd.Series(dtype=object)).nunique()) if purchase_doc_col and purchase_doc_col in fil.columns else 0
    col1.metric("Total PRs", total_prs)
    col2.metric("Total POs", total_pos)
    col3.metric("Line Items", len(fil))
    col4.metric("Entities", int(fil.get('entity', pd.Series(dtype=object)).nunique()))
    spend_val = fil.get(net_amount_col, pd.Series(0)).sum() if net_amount_col in fil.columns else 0
    col5.metric("Spend (Cr ₹)", f"{spend_val/1e7:,.2f}")

    st.markdown("---")
    dcol = po_create_col if (po_create_col and po_create_col in fil.columns) else primary_date_col
    st.subheader("Monthly Total Spend + Cumulative")
    if dcol and net_amount_col and dcol in fil.columns:
        t = fil.dropna(subset=[dcol]).copy()
        t["month_ts"] = t[dcol].dt.to_period("M").dt.to_timestamp()
        m = t.groupby("month_ts", as_index=False)[net_amount_col].sum().sort_values("month_ts")
        m["cr"] = m[net_amount_col]/1e7
        m["cumcr"] = m["cr"].cumsum()
        fig = make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_bar(x=m["month_ts"].dt.strftime("%b-%Y"), y=m["cr"], name="Monthly Spend (Cr ₹)")
        fig.add_scatter(x=m["month_ts"].dt.strftime("%b-%Y"), y=m["cumcr"], name="Cumulative (Cr ₹)", mode="lines+markers", secondary_y=True)
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Monthly Spend requires a date column (PR/PO) and Net Amount.")

# ---------------- PO/PR Timing ----------------
with tabs[1]:
    st.subheader("SLA (PR→PO ≤7d)")
    if pr_col and po_create_col and pr_col in fil.columns and po_create_col in fil.columns:
        ld = fil.dropna(subset=[pr_col, po_create_col]).copy()
        ld['lead_days'] = (ld[po_create_col] - ld[pr_col]).dt.days
        avg = float(ld['lead_days'].mean()) if not ld.empty else 0.0
        max_range = max(14, avg*1.2 if avg else 14)
        gauge = go.Figure(go.Indicator(mode="gauge+number", value=avg, number={"suffix":" d"},
                                       gauge={"axis":{"range":[0,max_range]}, "bar":{"color":"darkblue"},
                                              "steps":[{"range":[0,7],"color":"lightgreen"},{"range":[7,max_range],"color":"lightcoral"}],
                                              "threshold":{"line":{"color":"red","width":4},"value":7}}))
        st.plotly_chart(gauge, use_container_width=True)
    else:
        st.info("PR→PO timing metrics need both PR date and PO create date columns.")

# ---------------- Delivery ----------------
with tabs[2]:
    st.subheader("Delivery Summary")
    dv = fil.copy()
    if 'po_quantity' in dv.columns and 'receivedqty' in dv.columns:
        dv = dv.rename(columns={'po_quantity':'po_qty','receivedqty':'received_qty'})
    if 'po_qty' in dv.columns and 'received_qty' in dv.columns:
        dv['pct_received'] = np.where(dv['po_qty'].astype(float)>0, (dv['received_qty'].astype(float)/dv['po_qty'].astype(float))*100, 0.0)
        group_cols = [c for c in ['purchase_doc','po_vendor','product_name','item_description'] if c in dv.columns]
        if group_cols:
            summ = dv.groupby(group_cols, dropna=False).agg({'po_qty':'sum','received_qty':'sum'}).reset_index()
            st.dataframe(summ.sort_values('po_qty', ascending=False).head(200), use_container_width=True)
        else:
            st.info("Delivery grouping not possible; missing expected columns.")
    else:
        st.info("Delivery summary requires PO Qty and Received Qty columns (try: 'po_quantity', 'receivedqty').")

# ---------------- Vendors ----------------
with tabs[3]:
    st.subheader("Top Vendors by Spend")
    if po_vendor_col and net_amount_col and po_vendor_col in fil.columns and net_amount_col in fil.columns:
        vs = fil.groupby(po_vendor_col, dropna=False)[net_amount_col].sum().reset_index()
        vs['cr'] = vs[net_amount_col]/1e7
        vs = vs.sort_values('cr', ascending=False).head(15)
        st.dataframe(vs[[po_vendor_col,'cr']].rename(columns={po_vendor_col:'Vendor','cr':'Spend (Cr)'}), use_container_width=True)
        st.plotly_chart(px.bar(vs, x=po_vendor_col, y='cr', text='cr').update_traces(texttemplate='%{text:.2f}', textposition='outside'), use_container_width=True)
    else:
        st.info("Top Vendor view requires PO Vendor and Net Amount columns.")

# ---------------- Unit-rate Outliers ----------------
with tabs[4]:
    st.subheader("Unit-rate Outliers vs Historical Median")
    grp_candidates = [c for c in [product_col, safe_col(df, ["item_code","item code"]) ] if c in fil.columns]
    if grp_candidates:
        grp_by = st.selectbox("Group by", grp_candidates, index=0)
    else:
        grp_by = None
    if grp_by and po_unit_rate_col and grp_by in fil.columns and po_unit_rate_col in fil.columns:
        z = fil[[grp_by, po_unit_rate_col, purchase_doc_col, pr_number_col, po_vendor_col, 'item_description', po_create_col, net_amount_col]].dropna(subset=[grp_by, po_unit_rate_col]).copy()
        med = z.groupby(grp_by)[po_unit_rate_col].median().rename('median_rate')
        z = z.join(med, on=grp_by)
        z['pctdev'] = (z[po_unit_rate_col] - z['median_rate']) / z['median_rate'].replace(0,np.nan)
        thr = st.slider("Outlier threshold (±%)", 10, 300, 50, 5)
        out = z[abs(z['pctdev']) >= thr/100.0].copy()
        if not out.empty:
            out['pctdev%'] = (out['pctdev']*100).round(1)
            st.dataframe(out.sort_values('pctdev%', ascending=False), use_container_width=True)
            st.plotly_chart(px.scatter(z, x=po_create_col, y=po_unit_rate_col, color=np.where(abs(z['pctdev'])>=thr/100.0,'Outlier','Normal'), hover_data=[grp_by, purchase_doc_col, po_vendor_col, 'median_rate']).update_layout(legend_title_text=''), use_container_width=True)
        else:
            st.info("No outliers found with the selected threshold.")
    else:
        st.info("Need PO Unit Rate and grouping column (product/item code).")

# ---------------- Forecast ----------------
with tabs[5]:
    st.subheader("Forecast Next Month Spend (SMA)")
    dcol = po_create_col if po_create_col and po_create_col in fil.columns else (pr_col if pr_col and pr_col in fil.columns else None)
    if dcol and net_amount_col and dcol in fil.columns:
        t = fil.dropna(subset=[dcol]).copy()
        t['month'] = t[dcol].dt.to_period('M').dt.to_timestamp()
        m = t.groupby('month')[net_amount_col].sum().sort_index()
        if not m.empty:
            m_cr = m/1e7
            k = st.slider('Window (months)', 3, 12, 6, 1)
            sma = m_cr.rolling(k).mean()
            mu = m_cr.tail(k).mean() if len(m_cr) >= k else m_cr.mean()
            sd = m_cr.tail(k).std(ddof=1) if len(m_cr) >= k else m_cr.std(ddof=1)
            n = min(k, max(1, len(m_cr)))
            se = sd/np.sqrt(n) if sd==sd else 0
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
            st.info("Not enough monthly data to forecast.")
    else:
        st.info("Need a date column (PO/PR) and Net Amount to forecast.")

# ---------------- Vendor Scorecards ----------------
with tabs[6]:
    st.subheader("Vendor Scorecard")
    if po_vendor_col and po_vendor_col in fil.columns:
        vendor_list = sorted(fil[po_vendor_col].dropna().astype(str).unique().tolist())
        vendor = st.selectbox("Pick Vendor", vendor_list)
        vd = fil[fil[po_vendor_col].astype(str) == str(vendor)].copy()
        spend = vd.get(net_amount_col, pd.Series(0)).sum()/1e7 if net_amount_col in vd.columns else 0
        upos = int(vd.get(purchase_doc_col, pd.Series(dtype=object)).nunique()) if purchase_doc_col in vd.columns else 0
        today = pd.Timestamp.today().normalize()
        if po_delivery_col and pending_qty_col and po_delivery_col in vd.columns:
            late = ((pd.to_datetime(vd[po_delivery_col], errors='coerce').dt.date < today.date()) & (vd.get(pending_qty_col, pd.Series(0)).fillna(0) > 0)).sum()
        else:
            late = np.nan
        if pending_qty_col and po_unit_rate_col and pending_qty_col in vd.columns and po_unit_rate_col in vd.columns:
            vd['pending_value'] = vd[pending_qty_col].fillna(0).astype(float) * vd[po_unit_rate_col].fillna(0).astype(float)
            pend_val = vd['pending_value'].sum()/1e7
        else:
            pend_val = np.nan
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Spend (Cr)", f"{spend:.2f}"); k2.metric("Unique POs", upos); k3.metric("Late PO count", None if pd.isna(late) else int(late)); k4.metric("Pending Value (Cr)", None if pd.isna(pend_val) else f"{pend_val:.2f}")
        if product_col and product_col in vd.columns and po_unit_rate_col and po_unit_rate_col in vd.columns:
            med = vd.groupby(product_col)[po_unit_rate_col].median().rename('median_rate')
            v2 = vd.join(med, on=product_col)
            v2['var%'] = ((v2[po_unit_rate_col] - v2['median_rate']) / v2['median_rate'].replace(0, np.nan)) * 100
            st.plotly_chart(px.box(v2, x=product_col, y=po_unit_rate_col, points='outliers', title='Price variance by item'), use_container_width=True)
    else:
        st.info("Vendor Scorecard requires 'po_vendor' column.")

# ---------------- Search ----------------
with tabs[7]:
    st.subheader("🔍 Keyword Search")
    valid_cols = [c for c in [pr_number_col, purchase_doc_col, product_col, po_vendor_col] if c in df.columns]
    query = st.text_input("Type vendor, product, PO, PR, etc.", "")
    cat_col = safe_col(df, ["procurement_category","procurement category"])
    cat_sel = st.multiselect("Filter by Procurement Category", sorted(df[cat_col].dropna().astype(str).unique().tolist()) if cat_col and cat_col in df.columns else [])
    vend_sel = st.multiselect("Filter by Vendor (search scope)", sorted(df[po_vendor_col].dropna().astype(str).unique().tolist()) if po_vendor_col and po_vendor_col in df.columns else [])
    if query and valid_cols:
        mask = pd.Series(False, index=df.index)
        q = query.lower()
        for c in valid_cols:
            mask = mask | df[c].astype(str).str.lower().str.contains(q, na=False)
        res = df[mask].copy()
        if cat_sel and cat_col:
            res = res[res[cat_col].astype(str).isin(cat_sel)]
        if vend_sel and po_vendor_col:
            res = res[res[po_vendor_col].astype(str).isin(vend_sel)]
        st.write(f"Found {len(res):,} rows")
        st.dataframe(res, use_container_width=True)
        st.download_button("⬇️ Download Search Results", res.to_csv(index=False), "search_results.csv", "text/csv")
    elif not valid_cols:
        st.info("No searchable columns present.")
    else:
        st.caption("Start typing to search…")

# ---------------- Full Data (last tab) ----------------
with tabs[8]:
    st.subheader("Full Filtered Dataset (after applying filters)")
    st.write(f"Rows: {len(fil):,}")
    st.dataframe(fil, use_container_width=True)
    st.download_button("⬇️ Download filtered dataset", fil.to_csv(index=False), "filtered_dataset.csv", "text/csv")

# ---------------- Bottom uploader ----------------
st.markdown("---")
st.markdown("### Upload / Replace dataset (bottom uploader)")
new_files = st.file_uploader("Upload Excel/CSV files to replace dataset (optional)", type=["xlsx","xls","csv"], accept_multiple_files=True, key="_bottom_uploader")
if new_files:
    st.session_state["_uploaded_files"] = new_files
    st.experimental_rerun()
