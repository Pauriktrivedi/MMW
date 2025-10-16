# p2p_dashboard_full_with_sidebar_row.py
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="P2P Dashboard — Indirect", layout="wide", initial_sidebar_state="expanded")

# ---------------- Helpers ----------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = {}
    for c in df.columns:
        s = str(c).strip().replace("\xa0", " ").replace("\\", "_").replace("/", "_")
        s = "_".join(s.split()).lower()
        s = ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in s)
        new_cols[c] = s
    return df.rename(columns=new_cols)

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
        df["entity"] = ent
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return normalize_columns(pd.concat(frames, ignore_index=True, sort=False))

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
        df["entity"] = getattr(f, "name", "uploaded")
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return normalize_columns(pd.concat(frames, ignore_index=True, sort=False))

# ---------------- Load Data ----------------
# If user used bottom uploader in prior run, it will be stored in session_state key "_uploaded_files"
uploaded_session = st.session_state.get("_uploaded_files", None)
if uploaded_session:
    df = load_from_uploaded(uploaded_session)
else:
    df = load_default_files()

if df is None:
    df = pd.DataFrame()

# Show a small badge about data presence
if df.empty:
    st.warning("No data loaded. Use the bottom uploader to upload Excel/CSV files, or place default files (MEPL.xlsx, MLPL.xlsx, mmw.xlsx, mmpl.xlsx) next to this script.")
else:
    st.caption(f"Loaded {len(df):,} rows — columns: {', '.join(df.columns[:12])}...")

# Normalize certain known columns (best-effort)
date_candidates = [
    "pr_date_submitted","pr date submitted","pr date","po_create_date","po create date","po_create_date",
    "po_approved_date","po approved date","po_delivery_date","po delivery date","last_po_date"
]
# convert names to normalized if present in original raw
# (normalize_columns already applied when loading, so df columns are snake_case if loaded via helper)
for c in df.columns:
    # ensure all date-like columns get parsed if they look like dates
    pass

# Choose primary date column to drive FY/month filters (prefer PR date then PO create)
date_col = None
if "pr_date_submitted" in df.columns:
    date_col = "pr_date_submitted"
elif "po_create_date" in df.columns:
    date_col = "po_create_date"
# Ensure date_col is datetime
if date_col and date_col in df.columns:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

# ---------------- Sidebar Filters ----------------
st.sidebar.header("Filters")

FY = {
    "All Years": (pd.Timestamp("2023-04-01"), pd.Timestamp("2026-03-31")),
    "2023": (pd.Timestamp("2023-04-01"), pd.Timestamp("2024-03-31")),
    "2024": (pd.Timestamp("2024-04-01"), pd.Timestamp("2025-03-31")),
    "2025": (pd.Timestamp("2025-04-01"), pd.Timestamp("2026-03-31")),
}
fy_key = st.sidebar.selectbox("Financial Year", list(FY.keys()), index=0)
pr_start, pr_end = FY[fy_key]

# copy & filter
fil = df.copy()

# apply FY filter if we have a date column
if date_col and date_col in fil.columns:
    fil = fil[(fil[date_col] >= pr_start) & (fil[date_col] <= pr_end)]

# --- Month sub-filter (Month is sub-filter of chosen FY)
sel_month = "All Months"
if date_col and date_col in fil.columns:
    # build month list from the currently FY-filtered frame
    months = fil[date_col].dropna().dt.to_period("M").astype(str).unique().tolist()
    months = sorted(months, key=lambda s: pd.Period(s))
    month_labels = [pd.Period(m).strftime("%b-%Y") for m in months]
    label_to_period = {pd.Period(m).strftime("%b-%Y"): m for m in months}
    if month_labels:
        sel_month = st.sidebar.selectbox("Month (sub-filter of FY)", ["All Months"] + month_labels, index=0)
        if sel_month != "All Months":
            fil = fil[fil[date_col].dt.to_period("M").astype(str) == label_to_period[sel_month]]

# --- Date range picker (optional) - defaults to min/max of filtered set so far
if date_col and date_col in fil.columns and not fil[date_col].dropna().empty:
    min_dt = fil[date_col].dropna().min().date()
    max_dt = fil[date_col].dropna().max().date()
    dr = st.sidebar.date_input("Date range (optional)", (min_dt, max_dt), key="sidebar_date_range")
    if isinstance(dr, (list, tuple)) and len(dr) == 2:
        s_dt = pd.to_datetime(dr[0])
        e_dt = pd.to_datetime(dr[1])
        fil = fil[(fil[date_col] >= s_dt) & (fil[date_col] <= e_dt)]

# --- PO Department / Vendor / Item as single-row dropdowns in sidebar
# Create three equal columns inside sidebar for compact row layout
col1, col2, col3 = st.sidebar.columns([1,1,1])

# find candidate column names in normalized df
po_dept_col = next((c for c in df.columns if "po_department" in c or "po department" in c), None)
vendor_col = next((c for c in df.columns if "po_vendor" in c or "po vendor" in c), None)
item_col = next((c for c in df.columns if "product_name" in c or "product name" in c or "product" in c), None)

with col1:
    if po_dept_col and po_dept_col in df.columns:
        dept_choices = ["All Departments"] + sorted(df[po_dept_col].dropna().unique().tolist())
        sel_dept = st.selectbox("PO Dept", dept_choices, index=0)
        if sel_dept != "All Departments":
            fil = fil[fil[po_dept_col] == sel_dept]
    else:
        st.write("")  # placeholder so layout stays aligned

with col2:
    if vendor_col and vendor_col in df.columns:
        vendor_choices = ["All Vendors"] + sorted(df[vendor_col].dropna().unique().tolist())
        sel_vendor = st.selectbox("Vendor", vendor_choices, index=0)
        if sel_vendor != "All Vendors":
            fil = fil[fil[vendor_col] == sel_vendor]
    else:
        st.write("")

with col3:
    if item_col and item_col in df.columns:
        item_choices = ["All Items"] + sorted(df[item_col].dropna().unique().tolist())
        sel_item = st.selectbox("Item / Product", item_choices, index=0)
        if sel_item != "All Items":
            fil = fil[fil[item_col] == sel_item]
    else:
        st.write("")

# Additional standard filters (Buyer Type, Entity, PO Creator, PO Buyer Type) as multi-selects under the row
# attempt to find reasonable column names in normalized df
buyer_type_col = next((c for c in df.columns if c in ["buyer_type","buyer.type","buyer.type"]), None)
entity_col = "entity" if "entity" in df.columns else (next((c for c in df.columns if "entity" in c), None))
po_creator_col = next((c for c in df.columns if "po_creator" in c or "po.creator" in c or "po creator" in c), None)
po_buyer_type_col = next((c for c in df.columns if "po_buyer_type" in c or "po.buyertype" in c or "po buyer type" in c), None)

# ensure columns exist in filtered frame
for col in [buyer_type_col, entity_col, po_creator_col, po_buyer_type_col]:
    if col and col not in fil.columns:
        fil[col] = ""

# Build multiselect widgets (show all values by default)
if buyer_type_col:
    bt_choices = sorted(pd.Series(fil[buyer_type_col].dropna().unique()).astype(str).tolist())
    sel_bt = st.sidebar.multiselect("Buyer Type", bt_choices, default=bt_choices)
    if sel_bt:
        fil = fil[fil[buyer_type_col].astype(str).isin(sel_bt)]
if entity_col:
    ent_choices = sorted(pd.Series(fil[entity_col].dropna().unique()).astype(str).tolist())
    sel_ent = st.sidebar.multiselect("Entity", ent_choices, default=ent_choices)
    if sel_ent:
        fil = fil[fil[entity_col].astype(str).isin(sel_ent)]
if po_creator_col:
    pc_choices = sorted(pd.Series(fil[po_creator_col].dropna().unique()).astype(str).tolist())
    sel_pc = st.sidebar.multiselect("PO Ordered By", pc_choices, default=pc_choices)
    if sel_pc:
        fil = fil[fil[po_creator_col].astype(str).isin(sel_pc)]
if po_buyer_type_col:
    pbt_choices = sorted(pd.Series(fil[po_buyer_type_col].dropna().unique()).astype(str).tolist())
    sel_pbt = st.sidebar.multiselect("PO Buyer Type", pbt_choices, default=pbt_choices)
    if sel_pbt:
        fil = fil[fil[po_buyer_type_col].astype(str).isin(sel_pbt)]

# Reset filters: clear the bottom uploader and rerun, safe-guarded
if st.sidebar.button("Reset Filters"):
    # Remove uploader stored files (if any)
    if "_uploaded_files" in st.session_state:
        del st.session_state["_uploaded_files"]
    # clear any other temporary UI keys if you used them (none required here)
    st.experimental_rerun()

# ---------------- Tabs ----------------
tabs = st.tabs(["KPIs & Spend","PO/PR Timing","Delivery","Vendors","Forecast","Full Data"])

# ---------------- KPIs & Spend ----------------
with tabs[0]:
    st.header("P2P Dashboard — Indirect")
    # KPI row
    k1,k2,k3,k4,k5 = st.columns(5)
    pr_col = next((c for c in fil.columns if "pr_number" == c or "pr number" == c), None)
    po_col = next((c for c in fil.columns if "purchase_doc" == c or "purchase doc" == c), None)
    net_col = next((c for c in fil.columns if "net_amount" in c or "net amount" in c), None)
    k1.metric("Total PRs", int(fil[pr_col].nunique()) if pr_col else 0)
    k2.metric("Total POs", int(fil[po_col].nunique()) if po_col else 0)
    k3.metric("Line Items", len(fil))
    k4.metric("Entities", int(fil.get("entity", pd.Series(dtype=object)).nunique()))
    k5.metric("Spend (Cr ₹)", f"{fil.get(net_col, pd.Series(0)).sum()/1e7:,.2f}")

    st.markdown("---")
    # Monthly spend + cumulative
    dcol = "po_create_date" if "po_create_date" in fil.columns else date_col
    st.subheader("Monthly Total Spend + Cumulative")
    if dcol and net_col and dcol in fil.columns:
        t = fil.dropna(subset=[dcol]).copy()
        t["month"] = t[dcol].dt.to_period("M").dt.to_timestamp()
        m = t.groupby("month", as_index=False)[net_col].sum().sort_values("month")
        m["cr"] = m[net_col]/1e7
        m["cumcr"] = m["cr"].cumsum()
        fig = make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_bar(x=m["month"].dt.strftime("%b-%Y"), y=m["cr"], name="Monthly Spend (Cr ₹)")
        fig.add_scatter(x=m["month"].dt.strftime("%b-%Y"), y=m["cumcr"], name="Cumulative (Cr ₹)", mode="lines+markers", secondary_y=True)
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Monthly spend chart not available — need date & net amount columns.")

# ---------------- PO/PR Timing ----------------
with tabs[1]:
    st.subheader("SLA (PR→PO ≤7d)")
    if {"po_create_date","pr_date_submitted"}.issubset(fil.columns):
        ld = fil.dropna(subset=["po_create_date","pr_date_submitted"]).copy()
        ld["lead_days"] = (ld["po_create_date"] - ld["pr_date_submitted"]).dt.days
        avg = float(ld["lead_days"].mean()) if not ld.empty else 0.0
        fig = go.Figure(go.Indicator(mode="gauge+number",value=avg,number={"suffix":" d"},
            gauge={"axis":{"range":[0,max(14,avg*1.2 if avg else 14)]},"bar":{"color":"darkblue"},
                   "steps":[{"range":[0,7],"color":"lightgreen"},{"range":[7,max(14,avg*1.2 if avg else 14)],"color":"lightcoral"}],
                   "threshold":{"line":{"color":"red","width":4},"value":7}}))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("PR→PO timing requires 'pr_date_submitted' and 'po_create_date' columns.")

# ---------------- Delivery ----------------
with tabs[2]:
    st.subheader("Delivery Summary")
    dv = fil.copy()
    # try to harmonize common names
    if "po_quantity" in dv.columns and "receivedqty" in dv.columns:
        dv = dv.rename(columns={"po_quantity":"po_qty","receivedqty":"received_qty"})
    if "po_qty" in dv.columns and "received_qty" in dv.columns:
        dv["pct_received"] = np.where(dv["po_qty"].astype(float) > 0, (dv["received_qty"].astype(float) / dv["po_qty"].astype(float)) * 100, 0.0)
        st.dataframe(dv[["entity","po_qty","received_qty","pct_received"]].sort_values("pct_received").head(100), use_container_width=True)
    else:
        st.info("Delivery summary needs PO qty and Received qty columns (e.g., 'po_quantity', 'receivedqty').")

# ---------------- Vendors ----------------
with tabs[3]:
    st.subheader("Top Vendors by Spend")
    if vendor_col and net_col and vendor_col in fil.columns:
        vs = fil.groupby(vendor_col, dropna=False)[net_col].sum().sort_values(ascending=False).head(10)/1e7
        st.bar_chart(vs)
    else:
        st.info("Top vendor chart requires 'po_vendor' and 'net_amount' columns.")

# ---------------- Forecast ----------------
with tabs[4]:
    st.subheader("Forecast Next Month Spend (SMA)")
    dcol = "po_create_date" if "po_create_date" in fil.columns else date_col
    if dcol and net_col and dcol in fil.columns:
        t = fil.dropna(subset=[dcol]).copy()
        t["month"] = t[dcol].dt.to_period("M").dt.to_timestamp()
        m = t.groupby("month")[net_col].sum().sort_index()/1e7
        k = st.slider("Window (months)", 3, 12, 6, 1)
        sma = m.rolling(k).mean()
        mu = float(m.tail(k).mean()) if len(m) >= k else float(m.mean()) if not m.empty else 0.0
        dd = pd.DataFrame({"SpendCr":m, "SMA":sma})
        st.line_chart(dd, use_container_width=True)
        st.caption(f"Approx. forecast (SMA) for next month: {mu:.2f} Cr")
    else:
        st.info("Forecast needs date and net amount columns.")

# ---------------- Full Data (last tab) ----------------
with tabs[5]:
    st.subheader("Full Filtered Dataset")
    st.write(f"Rows: {len(fil):,}")
    st.dataframe(fil, use_container_width=True)
    st.download_button("⬇️ Download filtered dataset", fil.to_csv(index=False), "filtered_dataset.csv", "text/csv")

# ---------------- Bottom uploader ----------------
st.markdown("---")
st.markdown("### Upload new files (bottom uploader)")
new_files = st.file_uploader("Upload Excel/CSV files here (this replaces dataset)", type=["xlsx","xls","csv"], accept_multiple_files=True, key="_bottom_uploader")
if new_files:
    # save into session_state for next run
    st.session_state["_uploaded_files"] = new_files
    # rerun to pick up new files
    st.experimental_rerun()
