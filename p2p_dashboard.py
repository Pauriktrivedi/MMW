import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="P2P Dashboard — Full (Refactor)", layout="wide", initial_sidebar_state="expanded")

# ---------- Helpers ----------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to safe snake_case."""
    new_cols = {}
    for c in df.columns:
        s = str(c).strip()
        # replace common NBSP and backslash safely
        s = s.replace('\xa0', ' ')
        s = s.replace('\\u00A0', ' ')
        s = s.replace("\\", "_")   # escaped backslash
        s = s.replace("/", "_")
        s = "_".join(s.split())
        s = s.lower()
        s = ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in s)
        s = '_'.join([p for p in s.split('_') if p != ''])
        new_cols[c] = s
    return df.rename(columns=new_cols)


@st.cache_data(show_spinner=False)
def load_all_from_files(uploaded_files=None, skiprows_first=1):
    """
    Load uploaded files (or default filenames). Try reading with skiprows_first then fallback.
    Returns concatenated dataframe with normalized columns.
    """
    frames = []
    if uploaded_files:
        files = [(f, f.name.rsplit('.', 1)[0]) for f in uploaded_files]
    else:
        files = [("MEPL.xlsx","MEPL"),("MLPL.xlsx","MLPL"),("mmw.xlsx","MMW"),("mmpl.xlsx","MMPL")]

    for f, ent in files:
        try:
            # f may be a path (str) or UploadedFile
            if hasattr(f, "read"):
                df_temp = pd.read_excel(f, skiprows=skiprows_first)
            else:
                df_temp = pd.read_excel(f, skiprows=skiprows_first)
            df_temp['entity'] = ent
            frames.append(df_temp)
        except Exception:
            try:
                if hasattr(f, "read"):
                    df_temp = pd.read_excel(f)
                else:
                    df_temp = pd.read_excel(f)
                df_temp['entity'] = ent
                frames.append(df_temp)
            except Exception:
                # skip unreadable file
                continue

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True, sort=False)
    df = normalize_columns(df)
    return df

# ---------- UI Header ----------
st.markdown(
    """
    <div style="padding:6px 0 12px 0;">
      <h1 style="font-size:34px; margin:0; color:#0b1f3b;">P2P Dashboard — Indirect</h1>
      <div style="font-size:14px; color:#23395b; margin-top:4px;">Purchase-to-Pay overview (Indirect spend focus)</div>
      <hr style="border:0; height:1px; background:#e6eef6; margin-top:8px; margin-bottom:12px;" />
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar filters (static top group) ----------
st.sidebar.header('Filters')

FY = {
    'All Years': (pd.Timestamp('2023-04-01'), pd.Timestamp('2026-03-31')),
    '2023': (pd.Timestamp('2023-04-01'), pd.Timestamp('2024-03-31')),
    '2024': (pd.Timestamp('2024-04-01'), pd.Timestamp('2025-03-31')),
    '2025': (pd.Timestamp('2025-04-01'), pd.Timestamp('2026-03-31')),
}
fy_key = st.sidebar.selectbox('Financial Year', list(FY.keys()), index=0)
pr_start, pr_end = FY[fy_key]

# small uploader in sidebar for quick tests
uploaded_top = st.sidebar.file_uploader('Upload Excel/CSV files (optional)', type=['xlsx','xls','csv'], accept_multiple_files=True)

# ---------- Load Data ----------
if uploaded_top:
    df = load_all_from_files(uploaded_top)
else:
    df = load_all_from_files()

# If no data loaded: show bottom uploader and stop
if df.empty:
    st.warning('No data loaded — put default Excel files (MEPL.xlsx, MLPL.xlsx, mmw.xlsx, mmpl.xlsx) next to script OR upload files using the bottom uploader.')
    st.markdown('---')
    st.markdown('### Upload files (bottom of page)')
    new_files = st.file_uploader('Upload Excel/CSV files here', type=['xlsx','xls','csv'], accept_multiple_files=True, key='_bottom_uploader')
    if new_files:
        st.session_state['_uploaded_files'] = new_files
        st.experimental_rerun()
    st.stop()

# ---------- Identify common columns (normalized) ----------
# Candidate columns (normalized)
pr_col = next((c for c in ['pr_date_submitted','pr_date','pr_date_submitted'] if c in df.columns), None)
po_create_col = next((c for c in ['po_create_date','po_create_date','po_create_date'] if c in df.columns), None)
po_delivery_col = next((c for c in ['po_delivery_date','po_delivery_date'] if c in df.columns), None)

# parse date columns defensively
for d in [pr_col, po_create_col, po_delivery_col]:
    if d and d in df.columns:
        df[d] = pd.to_datetime(df[d], errors='coerce')

net_amount_col = next((c for c in ['net_amount','net amount'] if c in df.columns), None)
pr_number_col = next((c for c in ['pr_number','pr number','pr no'] if c in df.columns), None)
purchase_doc_col = next((c for c in ['purchase_doc','purchase doc','purchase_doc'] if c in df.columns), None)
po_vendor_col = next((c for c in ['po_vendor','po vendor'] if c in df.columns), None)
po_unit_rate_col = next((c for c in ['po_unit_rate','po unit rate'] if c in df.columns), None)
received_qty_col = next((c for c in ['receivedqty','received qty'] if c in df.columns), None)
pending_qty_col = next((c for c in ['pending_qty','pending qty','pendingqty'] if c in df.columns), None)
entity_col = 'entity' if 'entity' in df.columns else None
po_department_col = next((c for c in ['po_department','po department'] if c in df.columns), None)
product_col = next((c for c in ['product_name','product name'] if c in df.columns), None)

# Derived pr month/year for building month filter values
if pr_col and pr_col in df.columns:
    df['__pr_month'] = df[pr_col].dt.to_period('M')
    df['__pr_year'] = df[pr_col].dt.year
else:
    df['__pr_month'] = pd.NaT
    df['__pr_year'] = pd.NaT

# ---------- Sidebar dynamic controls (after data loaded) ----------
# Month list (sub-filter of FY) — build from PR date (fallback to PO create date)
month_basis = pr_col if pr_col else (po_create_col if po_create_col else None)
months = ['All Months']
if month_basis and df[month_basis].notna().any():
    months_periods = df.dropna(subset=[month_basis])[month_basis].dt.to_period('M').sort_values().unique()
    month_labels = [p.strftime('%b-%Y') for p in months_periods]
    months = ['All Months'] + month_labels

sel_month = st.sidebar.selectbox('Month (sub-filter of FY)', months, index=0)

# Date range picker — default clipped to data bounds
if month_basis and df[month_basis].notna().any():
    min_date = df[month_basis].min().date()
    max_date = df[month_basis].max().date()
else:
    min_date = pr_start.date()
    max_date = pr_end.date()

try:
    dr = st.sidebar.date_input('Date range (optional)', value=(max(pr_start.date(), min_date), min(pr_end.date(), max_date)))
    if isinstance(dr, tuple) and len(dr) == 2:
        pr_start, pr_end = pd.to_datetime(dr[0]), pd.to_datetime(dr[1])
except Exception:
    pass

# apply FY window first (on month_basis)
if month_basis and month_basis in df.columns:
    df = df[(df[month_basis] >= pr_start) & (df[month_basis] <= pr_end)]

# apply month subfilter if selected
if sel_month and sel_month != 'All Months' and month_basis and month_basis in df.columns:
    try:
        chosen = pd.Period(pd.to_datetime(sel_month, format='%b-%Y'))
        df = df[df[month_basis].dt.to_period('M') == chosen]
    except Exception:
        df = df[df[month_basis].dt.strftime('%b-%Y') == sel_month]

# PO Dept / Vendor / Item — single-select with All option
if po_department_col and po_department_col in df.columns:
    dept_choices = ['All Departments'] + sorted(df[po_department_col].dropna().astype(str).unique().tolist())
else:
    dept_choices = ['All Departments']
sel_dept = st.sidebar.selectbox('PO Department', dept_choices, index=0)
if sel_dept != 'All Departments' and po_department_col in df.columns:
    df = df[df[po_department_col].astype(str) == str(sel_dept)]

if po_vendor_col and po_vendor_col in df.columns:
    vendor_choices = ['All Vendors'] + sorted(df[po_vendor_col].dropna().astype(str).unique().tolist())
else:
    vendor_choices = ['All Vendors']
sel_vendor = st.sidebar.selectbox('Vendor', vendor_choices, index=0)
if sel_vendor != 'All Vendors' and po_vendor_col in df.columns:
    df = df[df[po_vendor_col].astype(str) == str(sel_vendor)]

if product_col and product_col in df.columns:
    item_choices = ['All Items'] + sorted(df[product_col].dropna().astype(str).unique().tolist())
else:
    item_choices = ['All Items']
sel_item = st.sidebar.selectbox('Item / Product', item_choices, index=0)
if sel_item != 'All Items' and product_col in df.columns:
    df = df[df[product_col].astype(str) == str(sel_item)]

# Additional multi-selects (data-driven)
# buyer type (try a few possible column names)
for col in ['buyer_type','buyer.type','buyer_type_unified','buyer.type_unified']:
    if col in df.columns:
        df['__buyer_type'] = df[col].astype(str)
        break
else:
    df['__buyer_type'] = 'Indirect'

# entity
if entity_col and entity_col in df.columns:
    df['__entity'] = df[entity_col].astype(str)
else:
    df['__entity'] = 'Unknown'

# po creator
for col in ['po_creator','po.creator','po_orderer','po orderer']:
    if col in df.columns:
        df['__po_creator'] = df[col].astype(str)
        break
else:
    df['__po_creator'] = ''

choices_bt = sorted(df['__buyer_type'].dropna().unique().tolist())
choices_ent = sorted(df['__entity'].dropna().unique().tolist())
choices_po_c = sorted(df['__po_creator'].dropna().unique().tolist())

sel_b = st.sidebar.multiselect('Buyer Type', choices_bt, default=choices_bt)
sel_e = st.sidebar.multiselect('Entity', choices_ent, default=choices_ent)
sel_o = st.sidebar.multiselect('PO Ordered By', choices_po_c, default=choices_po_c)

if sel_b:
    df = df[df['__buyer_type'].isin(sel_b)]
if sel_e:
    df = df[df['__entity'].isin(sel_e)]
if sel_o:
    df = df[df['__po_creator'].isin(sel_o)]

# Reset Filters
if st.sidebar.button('Reset Filters'):
    # clear known session keys (only those we set)
    for k in list(st.session_state.keys()):
        if k.startswith('_') or k.startswith('filter_'):
            try:
                del st.session_state[k]
            except Exception:
                pass
    # rerun to reset UI
    st.experimental_rerun()

# ---------- Tabs ----------
T = st.tabs(['KPIs & Spend','PO/PR Timing','Delivery','Vendors','Dept & Services','Unit-rate Outliers','Forecast','Scorecards','Search','Full Data'])

# ---------- KPIs & Spend (Tab 0) ----------
with T[0]:
    st.subheader('KPIs & Spend')
    c1,c2,c3,c4,c5 = st.columns(5)
    total_prs = int(df[pr_number_col].nunique()) if pr_number_col and pr_number_col in df.columns else 0
    total_pos = int(df[purchase_doc_col].nunique()) if purchase_doc_col and purchase_doc_col in df.columns else 0
    c1.metric('Total PRs', total_prs)
    c2.metric('Total POs', total_pos)
    c3.metric('Line Items', len(df))
    c4.metric('Entities', int(df[entity_col].nunique()) if entity_col and entity_col in df.columns else 0)
    spend_val = df[net_amount_col].sum() if net_amount_col and net_amount_col in df.columns else 0
    c5.metric('Spend (Cr ₹)', f"{spend_val/1e7:,.2f}")

    st.markdown('---')
    # Monthly Spend + Cumulative
    dcol = po_create_col if po_create_col and po_create_col in df.columns else (pr_col if pr_col and pr_col in df.columns else None)
    st.subheader('Monthly Total Spend + Cumulative')
    if dcol and net_amount_col and net_amount_col in df.columns:
        t = df.dropna(subset=[dcol]).copy()
        t['month_ts'] = t[dcol].dt.to_period('M').dt.to_timestamp()
        m = t.groupby('month_ts', as_index=False)[net_amount_col].sum().sort_values('month_ts')
        m['cr'] = m[net_amount_col]/1e7
        m['cumcr'] = m['cr'].cumsum()
        fig = make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_bar(x=m['month_ts'].dt.strftime('%b-%Y'), y=m['cr'], name='Monthly Spend (Cr ₹)')
        fig.add_scatter(x=m['month_ts'].dt.strftime('%b-%Y'), y=m['cumcr'], name='Cumulative (Cr ₹)', mode='lines+markers', secondary_y=True)
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('Monthly Spend chart not available — need date and Net Amount columns.')

    # Entity trend
    st.subheader('Entity Trend')
    if dcol and net_amount_col and entity_col and all([c in df.columns for c in [dcol, net_amount_col, entity_col]]):
        g = df.dropna(subset=[dcol]).groupby([pd.Grouper(key=dcol, freq='M'), entity_col], as_index=False)[net_amount_col].sum()
        g['cr'] = g[net_amount_col]/1e7
        if not g.empty:
            g['month'] = g[dcol].dt.strftime('%b-%Y')
            st.plotly_chart(px.line(g, x='month', y='cr', color=entity_col, markers=True, labels={'month':'Month','cr':'Cr ₹'}).update_layout(xaxis_tickangle=-45), use_container_width=True)
        else:
            st.info('Entity trend not available for selected filters.')
    else:
        st.info('Entity trend not available — need date, Net Amount and Entity columns.')

    # Open PRs summary
    st.subheader('Open PRs')
    if 'pr_status' in df.columns:
        op = df[df['pr_status'].astype(str).str.lower().isin(['approved','inreview','in review','open'])]
    else:
        op = df[df.get(purchase_doc_col, pd.Series()).isna()]
    if not op.empty and pr_col and pr_col in df.columns:
        op = op.copy()
        op['pending_age_d'] = (pd.Timestamp.now().normalize() - op[pr_col]).dt.days
        cols = [c for c in [pr_number_col, pr_col, 'pending_age_d', 'procurement_category', product_col, net_amount_col, 'po_budget_code', 'pr_status', entity_col, 'po_creator', purchase_doc_col] if c in op.columns]
        st.dataframe(op[cols].head(50), use_container_width=True)
    else:
        st.info('No Open PRs found for current filters.')

# ---------- PO/PR Timing (Tab 1) ----------
with T[1]:
    # This block mirrors your requested PR report exactly
    st.subheader("SLA (PR→PO ≤7d)")
    if {po_create_col, pr_col} <= set(df.columns):
        ld = df.dropna(subset=[po_create_col, pr_col]).copy()
        ld["Lead Time (Days)"] = (ld[po_create_col] - ld[pr_col]).dt.days
        avg = float(ld["Lead Time (Days)"].mean()) if not ld.empty else 0.0
        fig = go.Figure(go.Indicator(mode="gauge+number", value=avg, number={"suffix":" d"},
                                     gauge={"axis":{"range":[0,max(14,avg*1.2 if avg else 14)]},
                                            "bar":{"color":"darkblue"},
                                            "steps":[{"range":[0,7],"color":"lightgreen"},{"range":[7,max(14,avg*1.2 if avg else 14)],"color":"lightcoral"}],
                                            "threshold":{"line":{"color":"red","width":4},"value":7}}))
        st.plotly_chart(fig, use_container_width=True)

        bins=[0,7,15,30,60,90,999]; labels=["0-7","8-15","16-30","31-60","61-90","90+"]
        ag = pd.cut(ld["Lead Time (Days)"], bins=bins, labels=labels).value_counts(normalize=True).reindex(labels, fill_value=0).reset_index()
        ag.columns=["Bucket","Pct"]; ag["Pct"] = ag["Pct"]*100
        st.plotly_chart(px.bar(ag, x="Bucket", y="Pct", text="Pct").update_traces(texttemplate="%{text:.1f}%", textposition="outside"), use_container_width=True)
    else:
        st.info("Need both PR Date and PO Create Date to compute SLA.")

    st.subheader("PR & PO per Month")
    tmp = df.copy()
    tmp["PR Month"] = tmp.get(pr_col, pd.Series(pd.NaT, index=tmp.index)).dt.to_period("M")
    tmp["PO Month"] = tmp.get(po_create_col, pd.Series(pd.NaT, index=tmp.index)).dt.to_period("M")
    if pr_number_col and purchase_doc_col and pr_number_col in tmp.columns and purchase_doc_col in tmp.columns:
        ms = tmp.groupby("PR Month").agg({pr_number_col:'count', purchase_doc_col:'count'}).reset_index()
        if not ms.empty:
            ms.columns = ["Month","PR Count","PO Count"]
            ms["Month"] = ms["Month"].astype(str)
            st.line_chart(ms.set_index("Month"), use_container_width=True)

    st.subheader("Weekday Split")
    wd = df.copy()
    wd["PR Wk"] = wd.get(pr_col, pd.Series(pd.NaT, index=wd.index)).dt.day_name() if pr_col in wd.columns else ""
    wd["PO Wk"] = wd.get(po_create_col, pd.Series(pd.NaT, index=wd.index)).dt.day_name() if po_create_col in wd.columns else ""
    prc = wd["PR Wk"].value_counts().reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"], fill_value=0)
    poc = wd["PO Wk"].value_counts().reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"], fill_value=0)
    c1,c2 = st.columns(2); c1.bar_chart(prc); c2.bar_chart(poc)

    # Lead time by Buyer Type & Buyer
    st.subheader("Lead Time by Buyer Type & Buyer")
    if {po_create_col, pr_col} <= set(df.columns):
        ld2 = df.dropna(subset=[po_create_col, pr_col]).copy()
        ld2["Lead Time (Days)"] = (ld2[po_create_col] - ld2[pr_col]).dt.days
        if "__buyer_type" in ld2.columns:
            st.dataframe(ld2.groupby("__buyer_type")["Lead Time (Days)"].mean().round(1).reset_index().sort_values("Lead Time (Days)"), use_container_width=True)
        if "__po_creator" in ld2.columns:
            st.dataframe(ld2.groupby("__po_creator")["Lead Time (Days)"].mean().round(1).reset_index().sort_values("Lead Time (Days)"), use_container_width=True)

    # Daily PR Trends
    st.subheader("Daily PR Submissions")
    if pr_col and pr_col in df.columns:
        daily = df.copy(); daily["PR Date"] = pd.to_datetime(daily[pr_col], errors="coerce")
        dtrend = daily.groupby("PR Date").size().reset_index(name="PR Count")
        if not dtrend.empty:
            st.plotly_chart(px.line(dtrend, x="PR Date", y="PR Count", title="Daily PRs"), use_container_width=True)

    # Monthly Unique PO Generation
    st.subheader("Monthly Unique PO Generation")
    if purchase_doc_col and po_create_col and purchase_doc_col in df.columns and po_create_col in df.columns:
        pm = df.dropna(subset=[po_create_col, purchase_doc_col]).copy()
        pm["PO Month"] = pm[po_create_col].dt.to_period("M")
        mcount = pm.groupby("PO Month")[purchase_doc_col].nunique().reset_index(name="Unique PO Count")
        if not mcount.empty:
            mcount["PO Month"] = mcount["PO Month"].astype(str)
            st.plotly_chart(px.bar(mcount, x="PO Month", y="Unique PO Count", text="Unique PO Count", title="Unique POs per Month").update_traces(textposition="outside"), use_container_width=True)

# ---------- Delivery (Tab 2) ----------
with T[2]:
    st.subheader('Delivery Summary')
    dv = df.rename(columns={received_qty_col:'received_qty', pending_qty_col:'pending_qty', 'po_quantity':'po_qty'})
    if 'po_qty' in dv.columns and 'received_qty' in dv.columns:
        dv['pct_received'] = np.where(dv['po_qty'].astype(float) > 0, (dv['received_qty'].astype(float)/dv['po_qty'].astype(float))*100, 0.0)
        group_cols = [purchase_doc_col, po_vendor_col, product_col, 'item_description']
        agcols = [c for c in group_cols if c in dv.columns]
        summ = dv.groupby(agcols, dropna=False).agg({'po_qty':'sum','received_qty':'sum','pending_qty':'sum','pct_received':'mean'}).reset_index()
        if not summ.empty:
            st.dataframe(summ.sort_values('pending_qty', ascending=False), use_container_width=True)
            st.plotly_chart(px.bar(summ.sort_values('pending_qty', ascending=False).head(20), x=purchase_doc_col or 'purchase_doc', y='pending_qty', color=po_vendor_col if po_vendor_col in summ.columns else None, text='pending_qty', title='Top 20 Pending Qty').update_traces(textposition='outside'), use_container_width=True)

    st.subheader('Top Pending Lines by Value')
    if 'pending_qty' in dv.columns and po_unit_rate_col and po_unit_rate_col in dv.columns:
        dv['pending_value'] = dv['pending_qty'].astype(float) * dv[po_unit_rate_col].astype(float)
        keep = [c for c in ['pr_number', purchase_doc_col, 'procurement_category', 'buying_legal_entity', 'pr_budget_description', product_col, 'item_description', 'pending_qty', po_unit_rate_col, 'pending_value'] if c in dv.columns]
        st.dataframe(dv.sort_values('pending_value', ascending=False).head(50)[keep], use_container_width=True)

# ---------- Vendors (Tab 3) ----------
with T[3]:
    st.subheader('Top Vendors by Spend')
    if po_vendor_col and purchase_doc_col and net_amount_col and all(c in df.columns for c in [po_vendor_col, purchase_doc_col, net_amount_col]):
        vs = df.groupby(po_vendor_col, dropna=False).agg(Vendor_PO_Count=(purchase_doc_col,'nunique'), Total_Spend_Cr=(net_amount_col, lambda s:(s.sum()/1e7).round(2))).reset_index().sort_values('Total_Spend_Cr', ascending=False)
        st.dataframe(vs.head(10), use_container_width=True)
        st.plotly_chart(px.bar(vs.head(10), x=po_vendor_col, y='Total_Spend_Cr', text='Total_Spend_Cr', title='Top 10 Vendors (Cr ₹)').update_traces(textposition='outside'), use_container_width=True)

    st.subheader('Vendor Delivery Performance (Top 10 by Spend)')
    if po_vendor_col and purchase_doc_col and po_delivery_col and pending_qty_col and all(c in df.columns for c in [po_vendor_col, purchase_doc_col, po_delivery_col, pending_qty_col]):
        today = pd.Timestamp.today().normalize().date()
        vdf = df.copy()
        vdf['pendingqtyfill'] = vdf[pending_qty_col].fillna(0).astype(float)
        vdf['is_fully_delivered'] = vdf['pendingqtyfill'] == 0
        vdf[po_delivery_col] = pd.to_datetime(vdf[po_delivery_col], errors='coerce')
        vdf['is_late'] = vdf[po_delivery_col].dt.date.notna() & (vdf[po_delivery_col].dt.date < today) & (vdf['pendingqtyfill'] > 0)
        perf = vdf.groupby(po_vendor_col, dropna=False).agg(Total_PO_Count=(purchase_doc_col,'nunique'), Fully_Delivered_PO_Count=('is_fully_delivered','sum'), Late_PO_Count=('is_late','sum')).reset_index()
        perf['Pct_Fully_Delivered'] = (perf['Fully_Delivered_PO_Count'] / perf['Total_PO_Count'] * 100).round(1)
        perf['Pct_Late'] = (perf['Late_PO_Count'] / perf['Total_PO_Count'] * 100).round(1)
        if net_amount_col in df.columns:
            spend = df.groupby(po_vendor_col, dropna=False)[net_amount_col].sum().rename('Spend').reset_index()
            perf = perf.merge(spend, left_on=po_vendor_col, right_on=po_vendor_col, how='left').fillna({'Spend':0})
        top10 = perf.sort_values('Spend', ascending=False).head(10)
        if not top10.empty:
            st.dataframe(top10[[po_vendor_col,'Total_PO_Count','Fully_Delivered_PO_Count','Late_PO_Count','Pct_Fully_Delivered','Pct_Late']], use_container_width=True)
            melt = top10.melt(id_vars=[po_vendor_col], value_vars=['Pct_Fully_Delivered','Pct_Late'], var_name='Metric', value_name='Percentage')
            st.plotly_chart(px.bar(melt, x=po_vendor_col, y='Percentage', color='Metric', barmode='group', title='% Fully Delivered vs % Late (Top 10 by Spend)'), use_container_width=True)

# ---------- Dept & Services (Tab 4) ----------
with T[4]:
    st.subheader('Dept & Services (Smart Mapper)')
    st.info('Dept & Services mapping and treemap live here (kept minimal in this copy).')

# ---------- Unit-rate Outliers (Tab 5) ----------
with T[5]:
    st.subheader('Unit-rate Outliers vs Historical Median')
    grp_candidates = [c for c in [product_col, 'item_code'] if c in df.columns]
    grp_by = grp_candidates[0] if grp_candidates else None
    if grp_by and po_unit_rate_col and po_unit_rate_col in df.columns:
        z = df[[grp_by, po_unit_rate_col, purchase_doc_col, pr_number_col, po_vendor_col, 'item_description', po_create_col, net_amount_col]].dropna(subset=[grp_by, po_unit_rate_col]).copy()
        med = z.groupby(grp_by)[po_unit_rate_col].median().rename('median_rate')
        z = z.join(med, on=grp_by)
        z['pctdev'] = (z[po_unit_rate_col] - z['median_rate']) / z['median_rate'].replace(0, np.nan)
        thr = st.slider('Outlier threshold (±%)', 10, 300, 50, 5)
        out = z[abs(z['pctdev']) >= thr/100.0].copy(); out['pctdev%'] = (out['pctdev']*100).round(1)
        st.dataframe(out.sort_values('pctdev%', ascending=False), use_container_width=True)
        if not z.empty:
            st.plotly_chart(px.scatter(z, x=po_create_col, y=po_unit_rate_col, color=np.where(abs(z['pctdev'])>=thr/100.0,'Outlier','Normal'), hover_data=[grp_by, purchase_doc_col, po_vendor_col, 'median_rate']).update_layout(legend_title_text=''), use_container_width=True)
    else:
        st.info("Need 'PO Unit Rate' and a grouping column (Product / Item code).")

# ---------- Forecast (Tab 6) ----------
with T[6]:
    st.subheader('Forecast Next Month Spend (SMA)')
    dcol = po_create_col if po_create_col and po_create_col in df.columns else (pr_col if pr_col and pr_col in df.columns else None)
    if dcol and net_amount_col and net_amount_col in df.columns:
        t = df.dropna(subset=[dcol]).copy(); t['month'] = t[dcol].dt.to_period('M').dt.to_timestamp()
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

# ---------- Scorecards (Tab 7) ----------
with T[7]:
    st.subheader('Vendor Scorecard')
    if po_vendor_col and po_vendor_col in df.columns:
        vendor = st.selectbox('Pick Vendor', sorted(df[po_vendor_col].dropna().astype(str).unique().tolist()))
        vd = df[df[po_vendor_col].astype(str) == str(vendor)].copy()
        spend = vd.get(net_amount_col, pd.Series(0)).sum()/1e7 if net_amount_col and net_amount_col in vd.columns else 0
        upos = int(vd.get(purchase_doc_col, pd.Series(dtype=object)).nunique()) if purchase_doc_col and purchase_doc_col in vd.columns else 0
        today = pd.Timestamp.today().normalize()
        if po_delivery_col and pending_qty_col and all(c in vd.columns for c in [po_delivery_col, pending_qty_col]):
            late = ((pd.to_datetime(vd[po_delivery_col], errors='coerce').dt.date < today.date()) & (vd[pending_qty_col].fillna(0) > 0)).sum()
        else:
            late = np.nan
        if pending_qty_col and po_unit_rate_col and all(c in vd.columns for c in [pending_qty_col, po_unit_rate_col]):
            vd['pending_value'] = vd[pending_qty_col].fillna(0).astype(float) * vd[po_unit_rate_col].fillna(0).astype(float)
            pend_val = vd['pending_value'].sum()/1e7
        else:
            pend_val = np.nan
        k1,k2,k3,k4 = st.columns(4)
        k1.metric('Spend (Cr)', f"{spend:.2f}"); k2.metric('Unique POs', upos); k3.metric('Late PO count', None if pd.isna(late) else int(late)); k4.metric('Pending Value (Cr)', None if pd.isna(pend_val) else f"{pend_val:.2f}")

# ---------- Search (Tab 8) ----------
with T[8]:
    st.subheader('🔍 Keyword Search')
    valid_cols = [c for c in [pr_number_col, purchase_doc_col, product_col, po_vendor_col] if c in df.columns]
    query = st.text_input('Type vendor, product, PO, PR, etc.', '')
    cat_sel = st.multiselect('Filter by Procurement Category', sorted(df.get('procurement_category', pd.Series(dtype=object)).dropna().astype(str).unique().tolist())) if 'procurement_category' in df.columns else []
    vend_sel = st.multiselect('Filter by Vendor', sorted(df.get(po_vendor_col, pd.Series(dtype=object)).dropna().astype(str).unique().tolist())) if po_vendor_col in df.columns else []
    if query and valid_cols:
        mask = pd.Series(False, index=df.index)
        q = query.lower()
        for c in valid_cols:
            mask = mask | df[c].astype(str).str.lower().str.contains(q, na=False)
        res = df[mask].copy()
        if cat_sel and 'procurement_category' in df.columns:
            res = res[res['procurement_category'].astype(str).isin(cat_sel)]
        if vend_sel and po_vendor_col in df.columns:
            res = res[res[po_vendor_col].astype(str).isin(vend_sel)]
        st.write(f'Found {len(res)} rows')
        st.dataframe(res, use_container_width=True)
        st.download_button('⬇️ Download Search Results', res.to_csv(index=False), file_name='search_results.csv', mime='text/csv', key='dl_search')
    elif not valid_cols:
        st.info('No searchable columns present.')
    else:
        st.caption('Start typing to search…')

# ---------- Full Data (Tab 9) ----------
with T[9]:
    st.subheader('Full Data (after filters)')
    st.write('This tab shows the full dataset after all filters — useful for exporting.')
    st.dataframe(df, use_container_width=True)
    st.download_button('⬇️ Download filtered dataset (CSV)', df.to_csv(index=False), file_name='filtered_data.csv', mime='text/csv')

# ---------- Bottom uploader (always present) ----------
st.markdown('---')
st.markdown('### Upload Excel/CSV files (bottom of page)')
new_files = st.file_uploader('Upload Excel/CSV files here', type=['xlsx','xls','csv'], accept_multiple_files=True, key='_bottom_uploader')
if new_files:
    st.session_state['_uploaded_files'] = new_files
    st.experimental_rerun()
