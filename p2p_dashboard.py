# Full P2P Dashboard — copy/paste entire file
import re
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="P2P Dashboard — Full", layout="wide", initial_sidebar_state="expanded")

# ---------- Utility helpers ----------
def normalize_text(s: str) -> str:
    return re.sub(r'\s+', ' ', str(s).strip())

def find_col(df: pd.DataFrame, *variants):
    """Return the first column name in df that matches any variant (case-insensitive, trimmed)."""
    if df is None:
        return None
    cols = list(df.columns)
    lower_map = {c.lower().strip(): c for c in cols}
    for v in variants:
        if v is None:
            continue
        if isinstance(v, (list,tuple)):
            for c in cols:
                low = c.lower()
                ok = True
                for sub in v:
                    if str(sub).lower() not in low:
                        ok = False
                        break
                if ok:
                    return c
        else:
            key = str(v).lower().strip()
            if key in lower_map:
                return lower_map[key]
            for lc, orig in lower_map.items():
                if key in lc:
                    return orig
    return None

@st.cache_data(show_spinner=False)
def load_default_files():
    fns = [("MEPL.xlsx","MEPL"),("MLPL.xlsx","MLPL"),("mmw.xlsx","MMW"),("mmpl.xlsx","MMPL")]
    frames = []
    for fn, ent in fns:
        try:
            df = pd.read_excel(fn, skiprows=1)
        except Exception:
            try:
                df = pd.read_excel(fn)
            except Exception:
                continue
        df = df.copy()
        df["Entity"] = ent
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    x = pd.concat(frames, ignore_index=True, sort=False)
    x.columns = [normalize_text(c).replace("\xa0"," ") for c in x.columns]
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
        df["Entity"] = name.rsplit(".",1)[0]
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    x = pd.concat(frames, ignore_index=True, sort=False)
    x.columns = [normalize_text(c).replace("\xa0"," ") for c in x.columns]
    return x

# ---------- Load data (respect bottom uploader session) ----------
uploaded_files_session = st.session_state.get("_bottom_uploaded", None)
if uploaded_files_session:
    df = load_from_uploaded(uploaded_files_session)
else:
    df = load_default_files()

if df is None or df.empty:
    st.warning("No data found. Place default files (MEPL.xlsx, MLPL.xlsx, mmw.xlsx, mmpl.xlsx) next to this script OR use the bottom uploader (at the bottom of the page) to upload files.")
    st.markdown("---")
    st.markdown("### Upload files (bottom of page)")
    new_files = st.file_uploader("Upload Excel/CSV files here", type=['xlsx','xls','csv'], accept_multiple_files=True, key='_bottom_uploader')
    if new_files:
        st.session_state['_bottom_uploaded'] = new_files
        st.experimental_rerun()
    st.stop()

# ---------- Smart column mapping ----------
col_map = {}
col_map['PR Date Submitted'] = find_col(df, "PR Date Submitted", ("pr date","submitted"))
col_map['Po create Date']   = find_col(df, "Po create Date", ("po create",))
col_map['PO Approved Date'] = find_col(df, "PO Approved Date", ("po approved",))
col_map['PO Delivery Date'] = find_col(df, "PO Delivery Date", ("po delivery","delivery"))
col_map['PR Number']        = find_col(df, "PR Number", ("pr number","pr no"))
col_map['Purchase Doc']     = find_col(df, "Purchase Doc", ("purchase doc","po number","po_no"))
col_map['Net Amount']       = find_col(df, "Net Amount", ("net amount","amount","value"))
col_map['PO Vendor']        = find_col(df, "PO Vendor", ("vendor","supplier"))
col_map['Product Name']     = find_col(df, "Product Name", ("product name","item"))
col_map['PO Unit Rate']     = find_col(df, "PO Unit Rate", ("unit rate","rate"))
col_map['PO Quantity']      = find_col(df, "PO Quantity", ("quantity",))
col_map['ReceivedQTY']      = find_col(df, "ReceivedQTY", ("received","receivedqty"))
col_map['Pending QTY']      = find_col(df, "Pending QTY", ("pending","pendingqty"))
col_map['PO Budget Code']   = find_col(df, "PO Budget Code", ("po budget code","budget"))
col_map['PR Budget Code']   = find_col(df, "PR Budget Code", ("pr budget code",))
col_map['Buyer Group']      = find_col(df, "Buyer Group", ("buyer group",))
col_map['PO Orderer']       = find_col(df, "PO Orderer", ("po orderer",))
col_map['Entity']           = find_col(df, "Entity", "Entity")
col_map['PR Status']        = find_col(df, "PR Status", ("pr status",))
col_map['Procurement Category'] = find_col(df, "Procurement Category", ("procurement category",))
col_map['PO Department']    = find_col(df, "PO Department", ("po department","department","dept"))

def C(k):
    return col_map.get(k) if (k in col_map and col_map.get(k) in df.columns) else None

# normalize date columns
for dcol in ['PR Date Submitted','Po create Date','PO Approved Date','PO Delivery Date']:
    c = C(dcol)
    if c:
        try:
            df[c] = pd.to_datetime(df[c], errors='coerce')
        except Exception:
            df[c] = pd.to_datetime(df[c].astype(str), errors='coerce')

# ---------- Buyer/Creator logic ----------
if C('Buyer Group'):
    try:
        df['Buyer Group Code'] = df[C('Buyer Group')].astype(str).str.extract(r"(\d+)")[0].astype(float)
    except Exception:
        df['Buyer Group Code'] = np.nan
    def _bg(r):
        bg = r.get(C('Buyer Group'), '')
        code = r.get('Buyer Group Code', np.nan)
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
    df['Buyer.Type'] = df.apply(_bg, axis=1)
else:
    df['Buyer.Type'] = "Unknown"

map_orderer = {"mmw2324030":"Dhruv","mmw2324062":"Deepak","mmw2425154":"Mukul","mmw2223104":"Paurik","mmw2021181":"Nayan","mmw2223014":"Aatish","mmw_ext_002":"Deepakex","mmw2425024":"Kamlesh","mmw2021184":"Suresh","n/a":"Dilip"}
po_ord_col = C('PO Orderer')
if po_ord_col:
    df['PO Orderer Clean'] = df[po_ord_col].fillna("N/A").astype(str).str.strip()
else:
    df['PO Orderer Clean'] = "N/A"
df['PO.Creator'] = df['PO Orderer Clean'].str.lower().map(lambda v: map_orderer.get(v.lower(), None))
df['PO.Creator'] = df['PO.Creator'].fillna(df['PO Orderer Clean']).replace({"N/A":"Dilip"})
indirect_names = set(["Aatish","Deepak","Deepakex","Dhruv","Dilip","Mukul","Nayan","Paurik","Kamlesh","Suresh"])
df['PO.BuyerType'] = np.where(df['PO.Creator'].isin(indirect_names), "Indirect", "Direct")

# ---------- Sidebar (Filters) ----------
st.sidebar.header("Filters")
FY = {
    "All Years": (pd.Timestamp("2023-04-01"), pd.Timestamp("2026-03-31")),
    "2023": (pd.Timestamp("2023-04-01"), pd.Timestamp("2024-03-31")),
    "2024": (pd.Timestamp("2024-04-01"), pd.Timestamp("2025-03-31")),
    "2025": (pd.Timestamp("2025-04-01"), pd.Timestamp("2026-03-31")),
    "2026": (pd.Timestamp("2026-04-01"), pd.Timestamp("2027-03-31")),
}
fy_key = st.sidebar.selectbox("Financial Year", list(FY.keys()), index=0)
pr_start, pr_end = FY[fy_key]
fil = df.copy()
pr_date_col = C('PR Date Submitted')
if pr_date_col:
    fil = fil[(fil[pr_date_col] >= pr_start) & (fil[pr_date_col] <= pr_end)]

# ---- Month dropdown (under FY) ----
month_basis = pr_date_col if pr_date_col else (C('Po create Date') if C('Po create Date') else None)
sel_month = "All Months"
if month_basis:
    months = fil[month_basis].dropna().dt.to_period("M").astype(str).unique().tolist()
    months = sorted(months, key=lambda s: pd.Period(s))
    month_labels = [pd.Period(m).strftime("%b-%Y") for m in months]
    label_to_period = {pd.Period(m).strftime("%b-%Y"): m for m in months}
    if month_labels:
        sel_month = st.sidebar.selectbox("Month (applies when Year selected)", ["All Months"] + month_labels, index=0)
        if sel_month != "All Months":
            target_period = label_to_period[sel_month]
            fil = fil[fil[month_basis].dt.to_period("M").astype(str) == target_period]

# ---- Optional calendar date range (applies after FY+Month) ----
if month_basis:
    _mindt = fil[month_basis].dropna().min() if not fil[month_basis].dropna().empty else pr_start
    _maxdt = fil[month_basis].dropna().max() if not fil[month_basis].dropna().empty else pr_end
    if pd.notna(_mindt) and pd.notna(_maxdt):
        dr = st.sidebar.date_input("Date range (picker)", (pd.Timestamp(_mindt).date(), pd.Timestamp(_maxdt).date()), key="sidebar_date_range")
        if isinstance(dr, (list,tuple)) and len(dr) == 2:
            _s, _e = pd.to_datetime(dr[0]), pd.to_datetime(dr[1])
            fil = fil[(fil[month_basis] >= _s) & (fil[month_basis] <= _e)]

for col in ['Buyer.Type','Entity','PO.Creator','PO.BuyerType']:
    if col not in fil.columns:
        fil[col] = ""
    fil[col] = fil[col].astype(str).str.strip()

buyer_choices = sorted(fil['Buyer.Type'].dropna().unique().tolist())
entity_choices = sorted(fil['Entity'].dropna().unique().tolist()) if 'Entity' in fil.columns else []
po_creator_choices = sorted(fil['PO.Creator'].dropna().unique().tolist())
po_buyertype_choices = sorted(fil['PO.BuyerType'].dropna().unique().tolist())

vendor_col = C('PO Vendor')
product_col = C('Product Name')
po_dept_col = C('PO Department')

vendor_choices = sorted(fil[vendor_col].dropna().unique().tolist()) if vendor_col else []
item_choices = sorted(fil[product_col].dropna().unique().tolist()) if product_col else []
dept_choices = sorted(fil[po_dept_col].dropna().unique().tolist()) if po_dept_col else []

sel_b = st.sidebar.multiselect("Buyer Type", buyer_choices, default=buyer_choices)
sel_e = st.sidebar.multiselect("Entity", entity_choices, default=entity_choices if entity_choices else entity_choices)
sel_o = st.sidebar.multiselect("PO Ordered By", po_creator_choices, default=po_creator_choices)
sel_p = st.sidebar.multiselect("PO Buyer Type", po_buyertype_choices, default=po_buyertype_choices)
sel_v = st.sidebar.multiselect("Vendor (pick one or more)", vendor_choices, default=vendor_choices) if vendor_choices else []
sel_i = st.sidebar.multiselect("Item / Product (pick one or more)", item_choices, default=item_choices) if item_choices else []
sel_dept = st.sidebar.multiselect("PO Department", ["All Departments"] + dept_choices, default=["All Departments"]) if dept_choices else ["All Departments"]

if sel_b:
    fil = fil[fil['Buyer.Type'].isin(sel_b)]
if sel_e:
    fil = fil[fil['Entity'].isin(sel_e)]
if sel_o:
    fil = fil[fil['PO.Creator'].isin(sel_o)]
if sel_p:
    fil = fil[fil['PO.BuyerType'].isin(sel_p)]
if sel_v and vendor_col:
    fil = fil[fil[vendor_col].astype(str).isin(sel_v)]
if sel_i and product_col:
    fil = fil[fil[product_col].astype(str).isin(sel_i)]
if sel_dept and "All Departments" not in sel_dept and po_dept_col:
    fil = fil[fil[po_dept_col].astype(str).isin(sel_dept)]

if st.sidebar.button("Reset Filters"):
    for k in list(st.session_state.keys()):
        if k.startswith("sidebar_") or k in ["_bottom_uploader", "_bottom_uploaded"]:
            try:
                del st.session_state[k]
            except Exception:
                pass
    st.experimental_rerun()

# ---------- Tabs: KPI & Spend combined ----------
T = st.tabs(["KPIs & Spend","PO/PR Timing","Delivery","Vendors","Dept & Services","Unit-rate Outliers","Forecast","Scorecards","Search","Full Data"])

# ---------- KPIs & Spend (combined) ----------
with T[0]:
    # KPIs
    c1,c2,c3,c4,c5 = st.columns(5)
    total_prs = int(fil.get(C('PR Number'), pd.Series(dtype=object)).nunique()) if C('PR Number') else int(fil.get('PR Number', pd.Series(dtype=object)).nunique()) if 'PR Number' in fil.columns else 0
    total_pos = int(fil.get(C('Purchase Doc'), pd.Series(dtype=object)).nunique()) if C('Purchase Doc') else int(fil.get('Purchase Doc', pd.Series(dtype=object)).nunique()) if 'Purchase Doc' in fil.columns else 0
    c1.metric("Total PRs", total_prs)
    c2.metric("Total POs", total_pos)
    c3.metric("Line Items", len(fil))
    c4.metric("Entities", int(fil.get('Entity', pd.Series(dtype=object)).nunique()))
    net_col = C('Net Amount')
    spend_val = fil[net_col].sum() if net_col and net_col in fil.columns else 0
    c5.metric("Spend (Cr ₹)", f"{spend_val/1e7:,.2f}")

    st.markdown('---')
    # Spend charts
    dcol = C('Po create Date') if C('Po create Date') else (C('PR Date Submitted') if C('PR Date Submitted') else None)
    st.subheader("Monthly Total Spend + Cumulative")
    if dcol and net_col:
        t = fil.dropna(subset=[dcol]).copy()
        t["PO_Month"] = t[dcol].dt.to_period("M").dt.to_timestamp()
        t["Month_Str"] = t["PO_Month"].dt.strftime("%b-%Y")
        m = t.groupby(["PO_Month","Month_Str"],as_index=False)[net_col].sum().sort_values("PO_Month")
        m["Cr"] = m[net_col]/1e7; m["CumCr"] = m["Cr"].cumsum()
        fig = make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_bar(x=m["Month_Str"], y=m["Cr"], name="Monthly Spend (Cr ₹)")
        fig.add_scatter(x=m["Month_Str"], y=m["CumCr"], name="Cumulative (Cr ₹)", mode="lines+markers", secondary_y=True)
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Monthly Spend chart not available — need a date column and Net Amount column.")

    st.subheader("Entity Trend")
    if dcol and net_col:
        x = fil.copy(); x["PO Month"] = x[dcol].dt.to_period("M").dt.to_timestamp()
        g = x.dropna(subset=["PO Month"]).groupby(["PO Month","Entity"], as_index=False)[net_col].sum(); g["Cr"] = g[net_col]/1e7
        if not g.empty:
            st.plotly_chart(px.line(g, x=g["PO Month"].dt.strftime("%b-%Y"), y="Cr", color="Entity", markers=True, labels={"x":"Month","Cr":"Cr ₹"}).update_layout(xaxis_tickangle=-45), use_container_width=True)

# ---------- PO/PR Timing ----------
with T[1]:
    st.subheader("SLA (PR→PO ≤7d)")
    if C('Po create Date') and C('PR Date Submitted'):
        ld = fil.dropna(subset=[C('Po create Date'), C('PR Date Submitted')]).copy()
        ld["Lead Time (Days)"] = (ld[C('Po create Date')] - ld[C('PR Date Submitted')]).dt.days
        avg = float(ld["Lead Time (Days)"].mean()) if not ld.empty else 0.0
        fig = go.Figure(go.Indicator(mode="gauge+number", value=avg, number={"suffix":" d"}, gauge={"axis":{"range":[0,max(14,avg*1.2 if avg else 14)]},"bar":{"color":"darkblue"},"steps":[{"range":[0,7],"color":"lightgreen"},{"range":[7,max(14,avg*1.2 if avg else 14)],"color":"lightcoral"}],"threshold":{"line":{"color":"red","width":4},"value":7}}))
        st.plotly_chart(fig, use_container_width=True)
        bins=[0,7,15,30,60,90,999]; labels=["0-7","8-15","16-30","31-60","61-90","90+"]
        ag = pd.cut(ld["Lead Time (Days)"], bins=bins, labels=labels).value_counts(normalize=True).reindex(labels, fill_value=0).reset_index()
        ag.columns=["Bucket","Pct"]; ag["Pct"] = ag["Pct"]*100
        st.plotly_chart(px.bar(ag, x="Bucket", y="Pct", text="Pct").update_traces(texttemplate="%{text:.1f}%", textposition="outside"), use_container_width=True)
    else:
        st.info("Need both PR Date Submitted and Po create Date to compute SLA.")

    st.subheader("PR & PO per Month")
    tmp = fil.copy(); tmp["PR Month"] = tmp.get(C('PR Date Submitted'), pd.Series(pd.NaT, index=tmp.index)).dt.to_period("M"); tmp["PO Month"] = tmp.get(C('Po create Date'), pd.Series(pd.NaT, index=tmp.index)).dt.to_period("M")
    if C('PR Number') and C('Purchase Doc'):
        ms = tmp.groupby("PR Month").agg({C('PR Number'):'count', C('Purchase Doc'):'count'}).reset_index()
    else:
        ms = pd.DataFrame()
    if not ms.empty:
        ms.columns=["Month","PR Count","PO Count"]; ms["Month"] = ms["Month"].astype(str)
        st.line_chart(ms.set_index("Month"), use_container_width=True)

    st.subheader("Weekday Split")
    wd = fil.copy(); wd["PR Wk"] = wd.get(C('PR Date Submitted'), pd.Series(pd.NaT, index=wd.index)).dt.day_name(); wd["PO Wk"] = wd.get(C('Po create Date'), pd.Series(pd.NaT, index=wd.index)).dt.day_name()
    order=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    prc = wd["PR Wk"].value_counts().reindex(order, fill_value=0)
    poc = wd["PO Wk"].value_counts().reindex(order, fill_value=0)
    c1,c2 = st.columns(2); c1.bar_chart(prc); c2.bar_chart(poc)

    st.subheader("Open PRs")
    if C('PR Status'):
        op = fil[fil[C('PR Status')].isin(["Approved","InReview"])].copy()
        if not op.empty and C('PR Date Submitted') in op.columns:
            op["Pending Age (d)"] = (pd.Timestamp.today().normalize() - op[C('PR Date Submitted')]).dt.days
            show_cols = [c for c in [C('PR Number'), C('PR Date Submitted'), "Pending Age (d)", "Procurement Category", C('Product Name'), C('Net Amount'), C('PO Budget Code'), C('PR Status'), "Entity", "PO.Creator", C('Purchase Doc')] if c in op.columns]
            st.dataframe(op[show_cols], use_container_width=True)

# ---------- Delivery ----------
with T[2]:
    st.subheader("Delivery Summary")
    dv = fil.rename(columns={C('PO Quantity'):'PO Qty', C('ReceivedQTY'):'Received Qty', C('Pending QTY'):'Pending Qty'}).copy()
    if {'PO Qty','Received Qty'}.issubset(dv.columns):
        dv['% Received'] = np.where(dv['PO Qty'].astype(float) > 0, (dv['Received Qty'].astype(float)/dv['PO Qty'].astype(float))*100, 0.0)
        group_cols = [C('Purchase Doc') if C('Purchase Doc') else 'Purchase Doc', C('PO Vendor') if C('PO Vendor') else 'PO Vendor', C('Product Name') if C('Product Name') else 'Product Name', 'Item Description']
        agcols = [c for c in group_cols if c in dv.columns]
        agg_map = {}
        if 'PO Qty' in dv.columns: agg_map['PO Qty'] = 'sum'
        if 'Received Qty' in dv.columns: agg_map['Received Qty'] = 'sum'
        if 'Pending Qty' in dv.columns: agg_map['Pending Qty'] = 'sum'
        if agg_map and agcols:
            summ = dv.groupby(agcols, dropna=False).agg(agg_map).reset_index()
            st.dataframe(summ.sort_values('Pending Qty', ascending=False), use_container_width=True)
    else:
        st.info("Delivery view needs PO Qty and Received Qty (or Pending Qty) present.")

# ---------- Vendors ----------
with T[3]:
    st.subheader("Top Vendors by Spend")
    if C('PO Vendor') and C('Purchase Doc') and net_col:
        vs = fil.groupby(C('PO Vendor'), dropna=False).agg(Vendor_PO_Count=(C('Purchase Doc'),'nunique'), Total_Spend_Cr=(net_col, lambda s: (s.sum()/1e7).round(2))).reset_index().sort_values("Total_Spend_Cr", ascending=False)
        st.dataframe(vs.head(10), use_container_width=True)
        st.plotly_chart(px.bar(vs.head(10), x=C('PO Vendor'), y='Total_Spend_Cr', text='Total_Spend_Cr', title='Top 10 Vendors (Cr ₹)').update_traces(textposition='outside'), use_container_width=True)

# ---------- Dept & Services ----------
with T[4]:
    st.subheader("Dept & Services (Smart Mapper)")
    st.info("Dept mapping section preserved from LOVELY script. See code for mapping logic and optional mapping file support.")

# ---------- Unit-rate Outliers ----------
with T[5]:
    st.subheader("Unit-rate Outliers vs Historical Median")
    grp_options = [c for c in [C('Product Name'), 'Item Code'] if c and c in fil.columns]
    if grp_options and C('PO Unit Rate'):
        grp_by = st.selectbox("Group by", grp_options, index=0)
        zcols = [grp_by, C('PO Unit Rate'), C('Purchase Doc') if C('Purchase Doc') else 'Purchase Doc', C('PR Number') if C('PR Number') else 'PR Number', C('PO Vendor') if C('PO Vendor') else 'PO Vendor', 'Item Description', C('Po create Date') if C('Po create Date') else 'Po create Date', net_col]
        z = fil[[c for c in zcols if c in fil.columns]].dropna(subset=[grp_by, C('PO Unit Rate')]).copy()
        if not z.empty:
            med = z.groupby(grp_by)[C('PO Unit Rate')].median().rename("MedianRate")
            z = z.join(med, on=grp_by)
            z["PctDev"] = (z[C('PO Unit Rate')] - z["MedianRate"]) / z["MedianRate"].replace(0, np.nan)
            thr = st.slider("Outlier threshold (±%)", 10, 300, 50, 5)
            out = z[abs(z["PctDev"]) >= thr/100.0].copy(); out["PctDev%"] = (out["PctDev"]*100).round(1)
            st.dataframe(out.sort_values("PctDev%", ascending=False), use_container_width=True)
    else:
        st.info("Need 'PO Unit Rate' and a grouping column (Product Name / Item Code).")

# ---------- Forecast ----------
with T[6]:
    st.subheader("Forecast Next Month Spend (SMA)")
    dcol = C('Po create Date') if C('Po create Date') else (C('PR Date Submitted') if C('PR Date Submitted') else None)
    if dcol and net_col:
        t = fil.dropna(subset=[dcol]).copy(); t["Month"] = t[dcol].dt.to_period("M").dt.to_timestamp()
        m = t.groupby("Month")[net_col].sum().sort_index(); m_cr = (m/1e7)
        k = st.slider("Window (months)", 3, 12, 6, 1)
        sma = m_cr.rolling(k).mean()
        mu = m_cr.tail(k).mean() if len(m_cr) >= k else m_cr.mean()
        sd = m_cr.tail(k).std(ddof=1) if len(m_cr) >= k else m_cr.std(ddof=1)
        n = min(k, max(1, len(m_cr)))
        se = sd/np.sqrt(n) if sd == sd else 0
        lo, hi = float(mu-1.96*se), float(mu+1.96*se)
        nxt = (m_cr.index.max() + pd.offsets.MonthBegin(1)) if len(m_cr) > 0 else pd.Timestamp.today().to_period("M").to_timestamp()
        fdf = pd.DataFrame({"Month": list(m_cr.index) + [nxt], "SpendCr": list(m_cr.values) + [np.nan], "SMA": list(sma.values) + [mu]})
        fig = go.Figure()
        fig.add_bar(x=fdf["Month"], y=fdf["SpendCr"], name="Actual (Cr)")
        fig.add_scatter(x=fdf["Month"], y=fdf["SMA"], mode="lines+markers", name=f"SMA{k}")
        fig.add_scatter(x=[nxt,nxt], y=[lo,hi], mode="lines", name="95% CI")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Forecast for {nxt.strftime('%b-%Y')}: {mu:.2f} Cr (95% CI: {lo:.2f}–{hi:.2f})")
    else:
        st.info("Need date and Net Amount to forecast.")

# ---------- Scorecards / Vendor Scorecard ----------
with T[7]:
    st.subheader("Vendor Scorecard")
    if C('PO Vendor') in fil.columns:
        vendor_list = sorted(fil[C('PO Vendor')].dropna().astype(str).unique().tolist())
        if vendor_list:
            vendor = st.selectbox("Pick Vendor", vendor_list)
            vd = fil[fil[C('PO Vendor')].astype(str) == str(vendor)].copy()
            spend = vd.get(net_col, pd.Series(0)).sum()/1e7 if net_col else 0
            upos = int(vd.get(C('Purchase Doc'), pd.Series(dtype=object)).nunique()) if C('Purchase Doc') else 0
            today = pd.Timestamp.today().normalize()
            st.metric("Spend (Cr)", f"{spend:.2f}")

# ---------- Search ----------
with T[8]:
    st.subheader("🔍 Keyword Search")
    valid_cols = [c for c in [C('PR Number'), C('Purchase Doc'), C('Product Name'), C('PO Vendor')] if c and c in df.columns]
    query = st.text_input("Type vendor, product, PO, PR, etc.", "")
    cat_sel = st.multiselect("Filter by Procurement Category", sorted(df.get(C('Procurement Category') if C('Procurement Category') else 'Procurement Category', pd.Series(dtype=object)).dropna().astype(str).unique().tolist())) if C('Procurement Category') or 'Procurement Category' in df.columns else []
    vend_sel = st.multiselect("Filter by Vendor", sorted(df.get(C('PO Vendor') if C('PO Vendor') else 'PO Vendor', pd.Series(dtype=object)).dropna().astype(str).unique().tolist())) if C('PO Vendor') in df.columns else []
    if query and valid_cols:
        mask = pd.Series(False, index=df.index)
        q = query.lower()
        for c in valid_cols:
            mask = mask | df[c].astype(str).str.lower().str.contains(q, na=False)
        res = df[mask].copy()
        if cat_sel and (C('Procurement Category') in df.columns or 'Procurement Category' in df.columns):
            colname = C('Procurement Category') if C('Procurement Category') in df.columns else 'Procurement Category'
            res = res[res[colname].astype(str).isin(cat_sel)]
        if vend_sel and C('PO Vendor') in df.columns:
            res = res[res[C('PO Vendor')].astype(str).isin(vend_sel)]
        st.write(f"Found {len(res)} rows")
        st.dataframe(res, use_container_width=True)
        st.download_button("⬇️ Download Search Results", res.to_csv(index=False), file_name="search_results.csv", mime="text/csv", key="dl_search")
    elif not valid_cols:
        st.info("No searchable columns present.")
    else:
        st.caption("Start typing to search…")

# ---------- Full Data tab (reflects current filters) ----------
with T[9]:
    st.subheader("Full Filtered Dataset")
    st.write("This table shows the complete dataset after applying all sidebar filters (FY, Month, Date range, Buyer/Entity/PO Creator/Vendor/Item/Dept).")
    st.dataframe(fil.reset_index(drop=True), use_container_width=True)
    csv_buf = fil.to_csv(index=False)
    st.download_button('⬇️ Download filtered dataset (CSV)', csv_buf, file_name='filtered_dataset.csv', mime='text/csv')

# ----------------- Bottom uploader (persist across runs) -----------------
st.markdown("---")
st.markdown("### Upload & Debug (bottom of page)")
new_files = st.file_uploader("Upload Excel/CSV files here (bottom uploader)", type=['xlsx','xls','csv'], accept_multiple_files=True, key='_bottom_uploader_bottom')
if new_files:
    st.session_state['_bottom_uploaded'] = new_files
    st.experimental_rerun()
