# Full dashboard — includes PR "user-wise" (PR Prepared By) fixes
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="P2P Dashboard — Full (Refactor)", layout="wide", initial_sidebar_state="expanded")

# -------- Utilities --------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to snake_case (keeps original mapping flexible)."""
    new = {}
    for c in df.columns:
        s = str(c).strip()
        s = s.replace("\xa0", " ")
        s = s.replace("\\", "_").replace("/", "_")
        s = "_".join(s.split())
        s = s.lower()
        s = ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in s)
        s = "_".join([p for p in s.split("_") if p != ""])
        new[c] = s
    return df.rename(columns=new)

def safe_series(df, col, dtype=object):
    if col in df.columns:
        return df[col]
    return pd.Series([pd.NA]*len(df), index=df.index, dtype=dtype)

@st.cache_data(show_spinner=False)
def load_files(uploaded=None):
    """Load either uploaded files (streamlit UploadedFile) or default files."""
    frames = []
    if uploaded:
        for f in uploaded:
            try:
                tmp = pd.read_excel(f, skiprows=1)
            except Exception:
                tmp = pd.read_excel(f)
            tmp['entity'] = f.name.rsplit('.', 1)[0]
            frames.append(tmp)
    else:
        defaults = [("MEPL.xlsx","MEPL"),("MLPL.xlsx","MLPL"),("mmw.xlsx","MMW"),("mmpl.xlsx","MMPL")]
        for fn, ent in defaults:
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
    df = pd.concat(frames, ignore_index=True, sort=False)
    df = normalize_columns(df)
    # parse likely date columns if present
    for c in ['pr_date_submitted','po_create_date','po_approved_date','po_delivery_date','last_po_date']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')
    return df

# ---------- Page header ----------
st.markdown("""
<div style="padding:6px 0 12px 0;">
  <h1 style="margin:0; color:#0b1f3b;">P2P Dashboard — Indirect</h1>
  <div style="font-size:14px; color:#23395b; margin-top:4px;">
    Purchase-to-Pay overview (Indirect spend focus)
  </div>
  <hr style="border:0; height:1px; background:#e6eef6; margin-top:8px; margin-bottom:12px;" />
</div>
""", unsafe_allow_html=True)

# ---------------- Sidebar Filters ----------------
st.sidebar.header("Filters")

# Financial year choices (FY boundaries used as default)
FY = {
    "All Years": (pd.Timestamp("2000-01-01"), pd.Timestamp("2099-12-31")),
    "2023": (pd.Timestamp("2023-04-01"), pd.Timestamp("2024-03-31")),
    "2024": (pd.Timestamp("2024-04-01"), pd.Timestamp("2025-03-31")),
    "2025": (pd.Timestamp("2025-04-01"), pd.Timestamp("2026-03-31")),
    "2026": (pd.Timestamp("2026-04-01"), pd.Timestamp("2027-03-31")),
}
fy_key = st.sidebar.selectbox("Financial Year", list(FY.keys()), index=0)
fy_start, fy_end = FY[fy_key]

# File uploader — we still show top upload to help diagnostics, but primary uploader is bottom
uploaded_top = st.sidebar.file_uploader("(Optional) Quick upload (appears here too)", type=['xlsx','xls','csv'], accept_multiple_files=True, key='top_uploader')

# Load data (uploaded top has priority; if none, load defaults)
df = load_files(uploaded_top if uploaded_top else None)

if df.empty:
    st.sidebar.info("No data loaded. Use the bottom uploader or place default files next to this script (MEPL.xlsx, MLPL.xlsx, mmw.xlsx, mmpl.xlsx).")
    # Place bottom uploader visible in sidebar area as convenience as well
    st.sidebar.markdown("---")
    new_bottom = st.sidebar.file_uploader("Upload Excel/CSV files (bottom uploader)", type=['xlsx','xls','csv'], accept_multiple_files=True, key='bottom_uploader')
    if new_bottom:
        st.session_state['_uploaded_files'] = new_bottom
        st.experimental_rerun()
    st.stop()

# Ensure common columns are available (safe names)
# The user's original sheet headers map examples:
# 'PR Number','PR Date Submitted','PR Prepared By','PR Status','Purchase Doc','Po create Date','PO Vendor','Net Amount','PO Department','Product Name','PO Unit Rate','Pending QTY','ReceivedQTY','PO Orderer', etc.
# Our normalized names include: pr_number, pr_date_submitted, pr_prepared_by, pr_status, purchase_doc, po_create_date, po_vendor, net_amount, po_department, product_name, po_unit_rate, pending_qty, receivedqty, po_orderer

# Add friendly alias lookups to accept either snake-case or spaced-lowercase
def col_choice(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

pr_col = col_choice(df, ['pr_date_submitted','pr_date','pr_date_submitted'])
pr_prep_col = col_choice(df, ['pr_prepared_by','pr_prepared','pr_prepared_by_name','pr_prepared_by_name'])
pr_number_col = col_choice(df, ['pr_number','pr_no','pr_no'])
purchase_doc_col = col_choice(df, ['purchase_doc','purchase_doc_no','purchase_doc_number','purchase_doc'])
po_create_col = col_choice(df, ['po_create_date','po_create_date'])
po_vendor_col = col_choice(df, ['po_vendor','po_vendor_name','po_vendor'])
net_amount_col = col_choice(df, ['net_amount','net_amount'])
po_dept_col = col_choice(df, ['po_department','po_dept','po_department'])
product_col = col_choice(df, ['product_name','product name','product'])
po_orderer_col = col_choice(df, ['po_orderer','po_orderer'])

# Normalize missing columns to avoid KeyErrors
for c in [pr_col, pr_prep_col, pr_number_col, purchase_doc_col, po_create_col, po_vendor_col, net_amount_col, po_dept_col, product_col, po_orderer_col]:
    if c and c not in df.columns:
        df[c] = pd.NA

# Force typed columns for safety
if pr_col:
    df[pr_col] = pd.to_datetime(df[pr_col], errors='coerce')
if po_create_col:
    df[po_create_col] = pd.to_datetime(df[po_create_col], errors='coerce')

# ---------- FY / Month / Date range logic ----------
# We'll base Month on PR date if available; otherwise PO create date
month_basis = pr_col if pr_col in df.columns else (po_create_col if po_create_col in df.columns else None)

# Apply FY first (filter dataset to the FY window)
if month_basis:
    df = df[(df[month_basis] >= fy_start) & (df[month_basis] <= fy_end)]

# Build month dropdown (sub-filter of FY) — list of 'Mon-YYYY' labels
if month_basis:
    months_periods = df[month_basis].dropna().dt.to_period('M').unique()
    months_periods = sorted(months_periods)
    month_labels = [p.strftime("%b-%Y") for p in months_periods] if len(months_periods) > 0 else []
    sel_month = st.sidebar.selectbox("Month (sub-filter of FY)", ["All Months"] + month_labels, index=0)
    if sel_month != "All Months":
        # convert label back to period string to compare
        target = pd.Period(sel_month, freq='M').strftime("%Y-%m")
        # we'll compare by year-month
        df = df[df[month_basis].dt.to_period('M').astype(str) == pd.Period(sel_month, freq='M').strftime("%Y-%m")]
else:
    sel_month = "All Months"

# Optional date-range picker — applies after FY and Month
if month_basis:
    min_dt = df[month_basis].dropna().min() if not df[month_basis].dropna().empty else fy_start
    max_dt = df[month_basis].dropna().max() if not df[month_basis].dropna().empty else fy_end
    try:
        dr = st.sidebar.date_input("Select Date Range (optional)", (pd.to_datetime(min_dt).date(), pd.to_datetime(max_dt).date()))
        if isinstance(dr, tuple) and len(dr) == 2:
            s, e = pd.to_datetime(dr[0]), pd.to_datetime(dr[1])
            df = df[(df[month_basis] >= s) & (df[month_basis] <= e)]
    except Exception:
        # fallback no-op if the widget returns an unusual value
        pass

# ---------- Other filters ----------
# PO Department, Vendor, Item/Product as single-select dropdowns with "All" option
if po_dept_col in df.columns:
    dept_choices = ['All Departments'] + sorted(df[po_dept_col].dropna().astype(str).unique().tolist())
else:
    dept_choices = ['All Departments']
sel_dept = st.sidebar.selectbox("PO Dept", dept_choices, index=0)
if sel_dept != 'All Departments' and po_dept_col in df.columns:
    df = df[df[po_dept_col].astype(str) == sel_dept]

if po_vendor_col in df.columns:
    vendor_choices = ['All Vendors'] + sorted(df[po_vendor_col].dropna().astype(str).unique().tolist())
else:
    vendor_choices = ['All Vendors']
sel_vendor = st.sidebar.selectbox("Vendor", vendor_choices, index=0)
if sel_vendor != 'All Vendors' and po_vendor_col in df.columns:
    df = df[df[po_vendor_col].astype(str) == sel_vendor]

if product_col in df.columns:
    item_choices = ['All Items'] + sorted(df[product_col].dropna().astype(str).unique().tolist())
else:
    item_choices = ['All Items']
sel_item = st.sidebar.selectbox("Item / Product", item_choices, index=0)
if sel_item != 'All Items' and product_col in df.columns:
    df = df[df[product_col].astype(str) == sel_item]

# Multiselects for other data-driven filters (Buyer Type, PO Creator, Entity, and PR Prepared By)
# We'll try to find reasonable column candidates and normalize to strings
if 'buyer_type' not in df.columns:
    df['buyer_type'] = df.get('buyer.type', df.get('buyer_type', pd.Series(['Unknown']*len(df), index=df.index))).astype(str)
if 'po_creator' not in df.columns and po_orderer_col in df.columns:
    df['po_creator'] = df.get(po_orderer_col, pd.Series(['']*len(df))).astype(str)
if 'entity' not in df.columns:
    df['entity'] = df.get('entity', pd.Series(['']*len(df))).astype(str)

# PR Prepared By multiselect (this is the fix you asked for)
if pr_prep_col and pr_prep_col in df.columns:
    df['pr_prepared_by'] = df[pr_prep_col].astype(str).fillna('')
    pr_users = sorted(df['pr_prepared_by'].replace({'':np.nan}).dropna().unique().tolist())
else:
    df['pr_prepared_by'] = ''
    pr_users = []

# Provide multiselects for Buyer Type, Entity, PO Creator, and PR Prepared By
sel_buyer = st.sidebar.multiselect("Buyer Type", sorted(df['buyer_type'].replace({'':np.nan}).dropna().unique().tolist()), default=sorted(df['buyer_type'].replace({'':np.nan}).dropna().unique().tolist()))
sel_entity = st.sidebar.multiselect("Entity", sorted(df['entity'].replace({'':np.nan}).dropna().unique().tolist()), default=sorted(df['entity'].replace({'':np.nan}).dropna().unique().tolist()))
sel_po_creator = st.sidebar.multiselect("PO Ordered By", sorted(df['po_creator'].replace({'':np.nan}).dropna().unique().tolist()), default=sorted(df['po_creator'].replace({'':np.nan}).dropna().unique().tolist()))
sel_pr_users = st.sidebar.multiselect("PR Prepared By", pr_users, default=pr_users)

# Apply these multiselect filters (only if selection not empty)
if sel_buyer:
    df = df[df['buyer_type'].isin(sel_buyer)]
if sel_entity:
    df = df[df['entity'].isin(sel_entity)]
if sel_po_creator:
    df = df[df['po_creator'].isin(sel_po_creator)]
if sel_pr_users:
    df = df[df['pr_prepared_by'].isin(sel_pr_users)]

# Reset Filters button in sidebar
if st.sidebar.button("Reset Filters"):
    keys = ['top_uploader','bottom_uploader','_uploaded_files']
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]
    st.experimental_rerun()

# ---------- Tabs (KPIs & Spend first combined) ----------
tabs = st.tabs(["KPIs & Spend","PO/PR Timing","Delivery","Vendors","Dept & Services","Unit-rate Outliers","Forecast","Scorecards","Search","Full Data"])

# ---- KPIs & Spend ----
with tabs[0]:
    st.subheader("KPIs & Spend")
    c1,c2,c3,c4,c5 = st.columns(5)
    total_prs = int(df[pr_number_col].nunique()) if pr_number_col and pr_number_col in df.columns else 0
    total_pos = int(df[purchase_doc_col].nunique()) if purchase_doc_col and purchase_doc_col in df.columns else 0
    c1.metric("Total PRs", total_prs)
    c2.metric("Total POs", total_pos)
    c3.metric("Line Items", len(df))
    c4.metric("Entities", int(df['entity'].replace({'':np.nan}).dropna().nunique()))
    spend_val = df[net_amount_col].sum() if net_amount_col and net_amount_col in df.columns else 0.0
    c5.metric("Spend (Cr ₹)", f"{spend_val/1e7:,.2f}")

    # Show top PR creators (user-wise) under KPIs if PR prepared info exists
    if pr_prep_col and pr_prep_col in df.columns:
        st.markdown("**Top PR creators (PR Prepared By)**")
        pr_counts = df.groupby('pr_prepared_by').size().rename('count').reset_index().sort_values('count', ascending=False)
        if not pr_counts.empty:
            top5 = pr_counts.head(5)
            st.table(top5)

    st.markdown("---")
    # Spend charts (monthly + cumulative)
    dcol = po_create_col if po_create_col and po_create_col in df.columns else (pr_col if pr_col and pr_col in df.columns else None)
    st.subheader("Monthly Total Spend + Cumulative")
    if dcol and net_amount_col and net_amount_col in df.columns:
        t = df.dropna(subset=[dcol]).copy()
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
        st.info("Monthly Spend chart not available — need date and Net Amount columns.")

    # entity trend (on same page)
    st.subheader("Entity Trend")
    if dcol and net_amount_col and net_amount_col in df.columns:
        x = df.copy()
        x['po_month'] = x[dcol].dt.to_period('M').dt.to_timestamp()
        g = x.dropna(subset=['po_month']).groupby(['po_month','entity'], as_index=False)[net_amount_col].sum()
        if not g.empty:
            g['cr'] = g[net_amount_col]/1e7
            fig_entity = px.line(g, x=g['po_month'].dt.strftime('%b-%Y'), y='cr', color='entity', markers=True, labels={'x':'Month','cr':'Cr ₹'})
            fig_entity.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_entity, use_container_width=True)
        else:
            st.info("Entity trend not available for the selected filters.")

# ---- PO/PR Timing ----
with tabs[1]:
    st.subheader("SLA (PR → PO ≤ 7d)")
    if pr_col and po_create_col:
        ld = df.dropna(subset=[pr_col, po_create_col]).copy()
        ld['lead_time_days'] = (ld[po_create_col] - ld[pr_col]).dt.days
        if not ld.empty:
            avg = float(ld['lead_time_days'].mean())
            fig = go.Figure(go.Indicator(mode='gauge+number', value=avg, number={'suffix':' d'},
                                         gauge={'axis':{'range':[0,max(14, avg*1.2 if avg else 14)]},
                                                'steps':[{'range':[0,7],'color':'lightgreen'},{'range':[7,max(14, avg*1.2 if avg else 14)],'color':'lightcoral'}],
                                                'threshold':{'line':{'color':'red','width':4}, 'value':7}}))
            st.plotly_chart(fig, use_container_width=True)
            # lead-time by PR Prepared By
            st.markdown("**Lead time by PR Prepared By (average days)**")
            if pr_prep_col and pr_prep_col in df.columns:
                lt_by_user = ld.groupby('pr_prepared_by')['lead_time_days'].mean().round(1).reset_index().sort_values('lead_time_days')
                st.dataframe(lt_by_user, use_container_width=True)
        else:
            st.info("No rows with both PR & PO dates to compute SLAs.")
    else:
        st.info("Need both PR Date and PO Create Date to compute SLA.")

    # PR & PO per Month chart (counts)
    st.subheader("PR & PO per Month")
    tmp = df.copy()
    if pr_col in tmp.columns:
        tmp['pr_month'] = tmp[pr_col].dt.to_period('M')
    else:
        tmp['pr_month'] = pd.NaT
    if po_create_col in tmp.columns:
        tmp['po_month'] = tmp[po_create_col].dt.to_period('M')
    else:
        tmp['po_month'] = pd.NaT
    if pr_number_col and purchase_doc_col:
        ms = tmp.groupby('pr_month').agg({pr_number_col:'count', purchase_doc_col:'count'}).reset_index()
        if not ms.empty:
            ms.columns = ['Month','PR Count','PO Count']
            ms['Month'] = ms['Month'].astype(str)
            st.line_chart(ms.set_index('Month'), use_container_width=True)

# ---- Delivery ----
with tabs[2]:
    st.subheader("Delivery Summary")
    dv = df.rename(columns={'po_quantity':'po_qty','receivedqty':'received_qty','pending_qty':'pending_qty'}).copy()
    if 'po_qty' in dv.columns and 'received_qty' in dv.columns:
        dv['pct_received'] = np.where(dv['po_qty'].astype(float) > 0, (dv['received_qty'].astype(float) / dv['po_qty'].astype(float)) * 100, 0.0)
        group_cols = [purchase_doc_col, po_vendor_col, product_col, 'item_description']
        agcols = [c for c in group_cols if c in dv.columns]
        summ = dv.groupby(agcols, dropna=False).agg({'po_qty':'sum','received_qty':'sum','pending_qty':'sum','pct_received':'mean'}).reset_index()
        if not summ.empty:
            st.dataframe(summ.sort_values('pending_qty', ascending=False), use_container_width=True)
    else:
        st.info("Delivery metrics need PO Qty and Received Qty columns.")

# ---- Vendors ----
with tabs[3]:
    st.subheader("Top Vendors by Spend")
    if po_vendor_col and purchase_doc_col and net_amount_col:
        vs = df.groupby(po_vendor_col, dropna=False).agg(Vendor_PO_Count=(purchase_doc_col,'nunique'), Total_Spend_Cr=(net_amount_col, lambda s: (s.sum()/1e7).round(2))).reset_index().sort_values('Total_Spend_Cr', ascending=False)
        st.dataframe(vs.head(10), use_container_width=True)
        st.plotly_chart(px.bar(vs.head(10), x=po_vendor_col, y='Total_Spend_Cr', text='Total_Spend_Cr', title='Top 10 Vendors (Cr ₹)').update_traces(textposition='outside'), use_container_width=True)
    else:
        st.info("Vendor chart needs PO Vendor, Purchase Doc and Net Amount columns.")

# ---- Dept & Services (kept minimal here) ----
with tabs[4]:
    st.subheader("Dept & Services (summary)")
    if net_amount_col in df.columns and 'dept_chart' in df.columns:
        dep = df.groupby('dept_chart', dropna=False)[net_amount_col].sum().reset_index().sort_values(net_amount_col, ascending=False)
        dep['cr'] = dep[net_amount_col]/1e7
        st.plotly_chart(px.bar(dep.head(30), x='dept_chart', y='cr', title='Department-wise Spend (Top 30)').update_layout(xaxis_tickangle=-45), use_container_width=True)
    else:
        st.info("Dept → Services mapping not available for preview here.")

# ---- Unit-rate Outliers ----
with tabs[5]:
    st.subheader("Unit-rate Outliers vs Historical Median")
    grp_candidates = [c for c in [product_col, 'item_code'] if c in df.columns]
    if grp_candidates:
        grp_by = st.selectbox("Group by", grp_candidates, index=0)
    else:
        grp_by = None
    if grp_by and po_unit_rate_col and po_unit_rate_col in df.columns:
        z = df[[grp_by, po_unit_rate_col, purchase_doc_col, pr_number_col, po_vendor_col, 'item_description', po_create_col, net_amount_col]].dropna(subset=[grp_by, po_unit_rate_col]).copy()
        med = z.groupby(grp_by)[po_unit_rate_col].median().rename('median_rate')
        z = z.join(med, on=grp_by)
        z['pctdev'] = (z[po_unit_rate_col] - z['median_rate']) / z['median_rate'].replace(0, np.nan)
        thr = st.slider('Outlier threshold (±%)', 10, 300, 50, 5)
        out = z[abs(z['pctdev']) >= thr/100.0].copy(); out['pctdev%'] = (out['pctdev']*100).round(1)
        st.dataframe(out.sort_values('pctdev%', ascending=False), use_container_width=True)
    else:
        st.info("Need PO Unit Rate and a grouping column (product / item code).")

# ---- Forecast ----
with tabs[6]:
    st.subheader("Forecast Next Month Spend (SMA)")
    dcol = po_create_col if po_create_col and po_create_col in df.columns else (pr_col if pr_col and pr_col in df.columns else None)
    if dcol and net_amount_col and net_amount_col in df.columns:
        t = df.dropna(subset=[dcol]).copy()
        t['month'] = t[dcol].dt.to_period('M').dt.to_timestamp()
        m = t.groupby('month')[net_amount_col].sum().sort_index()
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
        st.info("Need date and Net Amount to forecast.")

# ---- Scorecards (vendor focused) ----
with tabs[7]:
    st.subheader("Vendor Scorecard")
    if po_vendor_col and po_vendor_col in df.columns:
        vendor_list = sorted(df[po_vendor_col].dropna().astype(str).unique().tolist())
        vendor_choice = st.selectbox("Pick Vendor", vendor_list)
        vd = df[df[po_vendor_col].astype(str) == str(vendor_choice)].copy()
        spend = vd[net_amount_col].sum()/1e7 if net_amount_col and net_amount_col in vd.columns else 0.0
        upos = int(vd[purchase_doc_col].nunique()) if purchase_doc_col and purchase_doc_col in vd.columns else 0
        st.metric("Spend (Cr)", f"{spend:.2f}")
        st.metric("Unique POs", upos)
    else:
        st.info("No PO Vendor column available for scorecards.")

# ---- Search ----
with tabs[8]:
    st.subheader("🔍 Keyword Search")
    valid_cols = [c for c in [pr_number_col, purchase_doc_col, product_col, po_vendor_col, 'pr_prepared_by'] if c and c in df.columns]
    query = st.text_input("Type vendor, product, PO, PR, PR prepared by, etc.", "")
    cat_sel = st.multiselect("Filter by Procurement Category", sorted(df.get('procurement_category', pd.Series(dtype=object)).dropna().astype(str).unique().tolist())) if 'procurement_category' in df.columns else []
    vend_sel = st.multiselect("Filter by Vendor", sorted(df.get(po_vendor_col, pd.Series(dtype=object)).dropna().astype(str).unique().tolist())) if po_vendor_col in df.columns else []
    if query and valid_cols:
        mask = pd.Series(False, index=df.index)
        q = query.lower()
        for c in valid_cols:
            mask = mask | df[c].astype(str).str.lower().str.contains(q, na=False)
        res = df[mask].copy()
        if cat_sel:
            res = res[res['procurement_category'].astype(str).isin(cat_sel)]
        if vend_sel:
            res = res[res[po_vendor_col].astype(str).isin(vend_sel)]
        st.write(f"Found {len(res)} rows")
        st.dataframe(res, use_container_width=True)
        st.download_button("⬇️ Download Search Results", res.to_csv(index=False), file_name="search_results.csv", mime="text/csv")
    else:
        st.caption("Start typing to search…")

# ---- Full Data tab (the complete filtered dataset) ----
with tabs[9]:
    st.subheader("Full Data (filtered view)")
    if not df.empty:
        st.write(f"Rows: {len(df)}")
        st.dataframe(df.reset_index(drop=True), use_container_width=True)
        st.download_button("⬇️ Download filtered dataset (CSV)", df.to_csv(index=False), file_name="filtered_dataset.csv", mime="text/csv")
    else:
        st.info("No data to show.")

# ---- Bottom uploader (primary place to upload files) ----
st.markdown("---")
st.markdown("### Upload & Debug (bottom of page)")
new_files = st.file_uploader("Upload Excel/CSV files here (bottom uploader)", type=['xlsx','xls','csv'], accept_multiple_files=True, key='_bottom_uploader')
if new_files:
    # Save to session and reload app with newly uploaded files
    st.session_state['_uploaded_files'] = new_files
    st.experimental_rerun()
