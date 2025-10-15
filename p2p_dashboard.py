# app.py — P2P Dashboard (LOVELY style, copy/paste ready)
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
    """Return the first column name in df that matches any variant (case-insensitive, trimmed).
       variants should be strings or tuples of substrings to check."""
    if df is None:
        return None
    cols = list(df.columns)
    lower_map = {c.lower().strip(): c for c in cols}
    for v in variants:
        if v is None:
            continue
        if isinstance(v, (list,tuple)):
            # if tuple/list, check if *all* substrings present in column
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
            # exact match first
            if key in lower_map:
                return lower_map[key]
            # loose contains match
            for lc, orig in lower_map.items():
                if key in lc:
                    return orig
    return None

@st.cache_data(show_spinner=False)
def load_default_files():
    """Load default workbook files in working folder (attempt skiprows=1 then fallback)."""
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
    # clean column names minimally (trim NBSP)
    x.columns = [normalize_text(c).replace("\xa0"," ") for c in x.columns]
    return x

@st.cache_data(show_spinner=False)
def load_from_uploaded(files):
    """Load uploaded files list (Streamlit UploadedFile objects)"""
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
        # entity from uploaded filename
        name = getattr(f, "name", "uploaded")
        df["Entity"] = name.rsplit(".",1)[0]
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    x = pd.concat(frames, ignore_index=True, sort=False)
    x.columns = [normalize_text(c).replace("\xa0"," ") for c in x.columns]
    return x

# ---------- Load data (respect bottom uploader session) ----------
# If user uploaded files via the bottom uploader in a previous run, use them; else try to load defaults.
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

# ---------- Smart column mapping (case-insensitive + tolerant) ----------
# We'll map the LOVELY script canonical names to actual df columns if present.
col_map = {}
# PR / PO / dates and amounts
col_map['PR Date Submitted'] = find_col(df, "PR Date Submitted", ("pr date","submitted"), ("pr date","submitted"))
col_map['Po create Date']   = find_col(df, "Po create Date", ("po create",), ("po created", "create"))
col_map['PO Approved Date'] = find_col(df, "PO Approved Date", ("po approved",))
col_map['PO Delivery Date'] = find_col(df, "PO Delivery Date", ("po delivery","delivery"))
col_map['PR Number']        = find_col(df, "PR Number", ("pr number","pr no","pr_no"))
col_map['Purchase Doc']     = find_col(df, "Purchase Doc", ("purchase doc","purchase_doc","po number","po_no","po number"))
col_map['Net Amount']       = find_col(df, "Net Amount", ("net amount","net_amount","amount","value"))
col_map['PO Vendor']        = find_col(df, "PO Vendor", ("vendor","supplier","po vendor"))
col_map['Product Name']     = find_col(df, "Product Name", ("product name","item","product_name"))
col_map['PO Unit Rate']     = find_col(df, "PO Unit Rate", ("unit rate","unit_rate","rate"))
col_map['PO Quantity']      = find_col(df, "PO Quantity", ("po quantity","quantity"))
col_map['ReceivedQTY']      = find_col(df, "ReceivedQTY", ("received","receivedqty","received qty"))
col_map['Pending QTY']      = find_col(df, "Pending QTY", ("pending","pendingqty","pending qty"))
col_map['PO Budget Code']   = find_col(df, "PO Budget Code", ("po budget code","po_budget_code","budget"))
col_map['PR Budget Code']   = find_col(df, "PR Budget Code", ("pr budget code","pr_budget_code"))
col_map['Buyer Group']      = find_col(df, "Buyer Group", ("buyer group","buyer_group"))
col_map['PO Orderer']       = find_col(df, "PO Orderer", ("po orderer","po_orderer","orderer"))
col_map['Entity']           = find_col(df, "Entity", "Entity")
col_map['PR Status']        = find_col(df, "PR Status", ("pr status","pr_status"))
col_map['Procurement Category'] = find_col(df, "Procurement Category", ("procurement category","procurement_category"))
col_map['PO Department']    = find_col(df, "PO Department", ("po department","po_department","department","dept"))

# Make an easy accessor to check presence
def C(k): 
    return col_map.get(k) if (k in col_map and col_map.get(k) in df.columns) else None

# ---------- Normalize dates for columns we have ----------
for dcol in ['PR Date Submitted','Po create Date','PO Approved Date','PO Delivery Date']:
    c = C(dcol)
    if c:
        try:
            df[c] = pd.to_datetime(df[c], errors='coerce')
        except Exception:
            df[c] = pd.to_datetime(df[c].astype(str), errors='coerce')

# ---------- Buyer/Creator logic (LOVELY mapping) ----------
# create Buyer.Type similar to your script
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

# PO Orderer mapping (case-insensitive keys)
map_orderer = {
    "mmw2324030":"Dhruv","mmw2324062":"Deepak","mmw2425154":"Mukul","mmw2223104":"Paurik",
    "mmw2021181":"Nayan","mmw2223014":"Aatish","mmw_ext_002":"Deepakex","mmw2425024":"Kamlesh",
    "mmw2021184":"Suresh","n/a":"Dilip"
}
po_ord_col = C('PO Orderer')
if po_ord_col:
    df['PO Orderer Clean'] = df[po_ord_col].fillna("N/A").astype(str).str.strip()
else:
    df['PO Orderer Clean'] = "N/A"

df['PO.Creator'] = df['PO Orderer Clean'].str.lower().map(lambda v: map_orderer.get(v.lower(), None))
# where mapping not found, show original cleaned (title-cased)
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

# apply FY filter (use PR date if available)
pr_date_col = C('PR Date Submitted')
if pr_date_col:
    fil = fil[(fil[pr_date_col] >= pr_start) & (fil[pr_date_col] <= pr_end)]

# ---- Month dropdown (under FY) ----
# Use PR Date Submitted primarily; if missing, fall back to Po create Date
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
            # fil uses period strings
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

# make sure common filter columns exist and are cleaned
for col in ['Buyer.Type','Entity','PO.Creator','PO.BuyerType']:
    if col not in fil.columns:
        fil[col] = ""
    fil[col] = fil[col].astype(str).str.strip()

# data-driven filter choices
buyer_choices = sorted(fil['Buyer.Type'].dropna().unique().tolist())
entity_choices = sorted(fil['Entity'].dropna().unique().tolist()) if 'Entity' in fil.columns else []
po_creator_choices = sorted(fil['PO.Creator'].dropna().unique().tolist())
po_buyertype_choices = sorted(fil['PO.BuyerType'].dropna().unique().tolist())

# Vendor & Item & PO Department filters only show if columns present
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

# apply side filters
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

# Reset Filters button
if st.sidebar.button("Reset Filters"):
    # only clear UI-like session keys
    for k in list(st.session_state.keys()):
        if k.startswith("sidebar_") or k in ["_bottom_uploader", "_bottom_uploaded"]:
            try:
                del st.session_state[k]
            except Exception:
                pass
    st.experimental_rerun()

# ---------- Tabs ----------
T = st.tabs(["KPIs","Spend","PO/PR Timing","Delivery","Vendors","Dept & Services","Unit-rate Outliers","Forecast","Scorecards","Search"])

# ---------- KPIs ----------
with T[0]:
    c1,c2,c3,c4,c5 = st.columns(5)
    total_prs = int(fil.get(col_map.get('PR Number','PR Number'), pd.Series(dtype=object)).nunique()) if col_map.get('PR Number') else int(fil.get('PR Number', pd.Series(dtype=object)).nunique())
    total_pos = int(fil.get(col_map.get('Purchase Doc','Purchase Doc'), pd.Series(dtype=object)).nunique()) if col_map.get('Purchase Doc') else int(fil.get('Purchase Doc', pd.Series(dtype=object)).nunique())
    c1.metric("Total PRs", total_prs)
    c2.metric("Total POs", total_pos)
    c3.metric("Line Items", len(fil))
    c4.metric("Entities", int(fil.get('Entity', pd.Series(dtype=object)).nunique()))
    net_col = C('Net Amount')
    spend_val = fil[net_col].sum() if net_col and net_col in fil.columns else 0
    c5.metric("Spend (Cr ₹)", f"{spend_val/1e7:,.2f}")

# ---------- Spend (Monthly + Entity) ----------
with T[1]:
    dcol = C('Po create Date') if C('Po create Date') else (C('PR Date Submitted') if C('PR Date Submitted') else None)
    st.subheader("Monthly Total Spend + Cumulative")
    if dcol and net_col:
        t = fil.dropna(subset=[dcol]).copy()
        t["PO_Month"] = t[dcol].dt.to_period("M").dt.to_timestamp()
        t["Month_Str"] = t["PO_Month"].dt.strftime("%b-%Y")
        m = t.groupby(["PO_Month","Month_Str"],as_index=False)[net_col].sum().sort_values("PO_Month")
        m["Cr"] = m[net_col]/1e7
        m["CumCr"] = m["Cr"].cumsum()
        fig = make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_bar(x=m["Month_Str"], y=m["Cr"], name="Monthly Spend (Cr ₹)")
        fig.add_scatter(x=m["Month_Str"], y=m["CumCr"], name="Cumulative (Cr ₹)", mode="lines+markers", secondary_y=True)
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Monthly Spend chart not available — need a date column (PO create / PR date) and Net Amount column.")

    st.subheader("Entity Trend")
    if dcol and net_col:
        x = fil.copy()
        x["PO Month"] = x[dcol].dt.to_period("M").dt.to_timestamp()
        g = x.dropna(subset=["PO Month"]).groupby(["PO Month","Entity"], as_index=False)[net_col].sum()
        g["Cr"] = g[net_col]/1e7
        if not g.empty:
            st.plotly_chart(px.line(g, x=g["PO Month"].dt.strftime("%b-%Y"), y="Cr", color="Entity", markers=True, labels={"x":"Month","Cr":"Cr ₹"}).update_layout(xaxis_tickangle=-45), use_container_width=True)

# ---------- PR/PO Timing ----------
with T[2]:
    st.subheader("SLA (PR→PO ≤7d)")
    if C('Po create Date') and C('PR Date Su
