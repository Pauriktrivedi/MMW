# p2p_dashboard_complete.py
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
uploaded_session = st.session_state.get("_uploaded_files", None)
if uploaded_session:
    df = load_from_uploaded(uploaded_session)
else:
    df = load_default_files()

if df is None:
    df = pd.DataFrame()

# ensure lower-case normalized names exist
df = normalize_columns(df) if not df.empty else df

# pick a primary date column (pr_date_submitted preferred, else po_create_date)
date_col = None
if "pr_date_submitted" in df.columns:
    date_col = "pr_date_submitted"
elif "po_create_date" in df.columns:
    date_col = "po_create_date"

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

# base filtered frame
fil = df.copy()

# apply FY filter if we have the date column
if date_col and date_col in fil.columns:
    fil = fil[(fil[date_col] >= pr_start) & (fil[date_col] <= pr_end)]

# Month sub-filter (sub-filter of FY)
sel_month = "All Months"
if date_col and date_col in fil.columns:
    # build month list from FY-filtered frame
    months = fil[date_col].dropna().dt.to_period("M").astype(str).unique().tolist()
    months = sorted(months, key=lambda s: pd.Period(s))
    month_labels = [pd.Period(m).strftime("%b-%Y") for m in months]
    label_to_period = {pd.Period(m).strftime("%b-%Y"): m for m in months}
    if month_labels:
        sel_month = st.sidebar.selectbox("Month (sub-filter of FY)", ["All Months"] + month_labels, index=0)
        if sel_month != "All Months":
            target_period = label_to_period[sel_month]
            fil = fil[fil[date_col].dt.to_period("M").astype(str) == target_period]

# Optional calendar date-range (applies after FY + Month)
if date_col and date_col in fil.columns and not fil[date_col].dropna().empty:
    min_dt = fil[date_col].dropna().min().date()
    max_dt = fil[date_col].dropna().max().date()
    dr = st.sidebar.date_input("Date range (optional)", (min_dt, max_dt), key="sidebar_date_range")
    if isinstance(dr, (list, tuple)) and len(dr) == 2:
        s_dt = pd.to_datetime(dr[0])
        e_dt = pd.to_datetime(dr[1])
        fil = fil[(fil[date_col] >= s_dt) & (fil[date_col] <= e_dt)]

# PO Department, Vendor, Item in one row (single-select dropdowns)
col1, col2, col3 = st.sidebar.columns([1,1,1])

po_dept_col = next((c for c in df.columns if "po_department" in c or "po department" in c), None)
vendor_col = next((c for c in df.columns if "po_vendor" in c or "po vendor" in c), None)
item_col = next((c for c in df.columns if "product_name" in c or "product name" in c or "product" in c), None)

with col1:
    if po_dept_col and po_dept_col in df.columns:
        dept_choices = ["All Departments"] + sorted(df[po_dept_col].dropna().astype(str).unique().tolist())
        sel_dept = st.selectbox("PO Dept", dept_choices, index=0)
        if sel_dept != "All Departments":
            fil = fil[fil[po_dept_col].astype(str) == sel_dept]
    else:
        st.write("")

with col2:
    if vendor_col and vendor_col in df.columns:
        vendor_choices = ["All Vendors"] + sorted(df[vendor_col].dropna().astype(str).unique().tolist())
        sel_vendor = st.selectbox("Vendor", vendor_choices, index=0)
        if sel_vendor != "All Vendors":
            fil = fil[fil[vendor_col].astype(str) == sel_vendor]
    else:
        st.write("")

with col3:
    if item_col and item_col in df.columns:
        item_choices = ["All Items"] + sorted(df[item_col].dropna().astype(str).unique().tolist())
        sel_item = st.selectbox("Item / Product", item_choices, index=0)
        if sel_item != "All Items":
            fil = fil[fil[item_col].astype(str) == sel_item]
    else:
        st.write("")

# Additional multi-select filters (Buyer Type, Entity, PO Ordered By, PO Buyer Type)
buyer_type_col = next((c for c in df.columns if c in ["buyer_type","buyer.type","buyer.type"]), None)
entity_col = "entity" if "entity" in df.columns else (next((c for c in df.columns if "entity" in c and c != "entity"), None))
po_creator_col = next((c for c in df.columns if "po_creator" in c or "po.creator" in c or "po creator" in c), None)
po_buyer_type_col = next((c for c in df.columns if "po_buyer_type" in c or "po.buyertype" in c or "po buyer type" in c), None)

for col in [buyer_type_col, entity_col, po_creator_col, po_buyer_type_col]:
    if col and col not in fil.columns:
        fil[col] = ""

if buyer_type_col and buyer_type_col in fil.columns:
    bt_choices = sorted(fil[buyer_type_col].dropna().astype(str).unique().tolist())
    default_bt = bt_choices[:] if bt_choices else []
    sel_bt = st.sidebar.multiselect("Buyer Type", bt_choices, default=default_bt)
    if sel_bt:
        fil = fil[fil[buyer_type_col].astype(str).isin(sel_bt)]

if entity_col and entity_col in fil.columns:
    ent_choices = sorted(fil[entity_col].dropna().astype(str).unique().tolist())
    default_ent = ent_choices[:] if ent_choices else []
    sel_ent = st.sidebar.multiselect("Entity", ent_choices, default=default_ent)
    if sel_ent:
        fil = fil[fil[entity_col].astype(str).isin(sel_ent)]

if po_creator_col and po_creator_col in fil.columns:
    pc_choices = sorted(fil[po_creator_col].dropna().astype(str).unique().tolist())
    sel_pc = st.sidebar.multiselect("PO Ordered By", pc_choices, default=pc_choices)
    if sel_pc:
        fil = fil[fil[po_creator_col].astype(str).isin(sel_pc)]

if po_buyer_type_col and po_buyer_type_col in fil.columns:
    pbt_choices = sorted(fil[po_buyer_type_col].dropna().astype(str).unique().tolist())
    sel_pbt = st.sidebar.multiselect("PO Buyer Type", pbt_choices, default=pbt_choices)
    if sel_pbt:
        fil = fil[fil[po_buyer_type_col].astype(str).isin(sel_pbt)]

# Reset Filters button (clears uploader session and reruns)
if st.sidebar.button("Reset Filters"):
    if "_uploaded_files" in st.session_state:
        del st.session_state["_uploaded_files"]
    # rerun to reset UI
    st.experimental_rerun()

# ---------------- Tabs ----------------
tabs = st.tabs(["KPIs & Spend","PO/PR Timing","Delivery","Vendors","Search","Full Data"])

# ---------------- KPIs & Spend ----------------
with tabs[0]:
    st.header("P2P Dashboard — Indirect")
    k1,k2,k3,k4,k5 = st.columns(5)
    pr_col = next((c for c in fil.columns if "pr_number" == c or "pr number" == c), None)
    po_col = next((c for c in fil.columns if "purchase_doc" == c or "purchase doc" == c or "purchase_doc" in c), None)
    net_col = next((c for c in fil.columns if "net_amount" in c or "net amount" in c), None)
    k1.metric("Total PRs", int(fil[pr_col].nunique()) if pr_col and pr_col in fil.columns else 0)
    k2.metric("Total POs", int(fil[po_col].nunique()) if po_col and po_col in fil.columns else 0)
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
        max_range = max(14, avg*1.2 if avg else 14)
        fig = go.Figure(go.Indicator(mode="gauge+number",value=avg,number={"suffix":" d"},
            gauge={"axis":{"range":[0,max_range]},"bar":{"color":"darkblue"},
                   "steps":[{"range":[0,7],"color":"lightgreen"},{"range":[7,max_range],"color":"lightcoral"}],
                   "threshold":{"line":{"color":"red","width":4},"value":7}}))
        st.plotly_chart(fig, use_container_width=True)
        # Lead time buckets
        bins=[0,7,15,30,60,90,999]; labels=["0-7","8-15","16-30","31-60","61-90","90+"]
        ag = pd.cut(ld["lead_days"], bins=bins, labels=labels).value_counts(normalize=True).reindex(labels, fill_value=0).reset_index()
        ag.columns=["Bucket","Pct"]; ag["Pct"] = ag["Pct"] * 100
        st.plotly_chart(px.bar(ag, x="Bucket", y="Pct", text="Pct").update_traces(texttemplate="%{text:.1f}%", textposition="outside"), use_container_width=True)
    else:
        st.info("PR→PO timing requires 'pr_date_submitted' and 'po_create_date' columns.")

# ---------------- Delivery ----------------
with tabs[2]:
    st.subheader("Delivery Summary")
    dv = fil.copy()
    # harmonize common names
    if "po_quantity" in dv.columns and "receivedqty" in dv.columns:
        dv = dv.rename(columns={"po_quantity":"po_qty","receivedqty":"received_qty"})
    if "po_qty" in dv.columns and "received_qty" in dv.columns:
        dv["pct_received"] = np.where(dv["po_qty"].astype(float) > 0, (dv["received_qty"].astype(float) / dv["po_qty"].astype(float)) * 100, 0.0)
        group_cols = [c for c in ["purchase_doc","po_vendor","product_name","item_description"] if c in dv.columns]
        summ = dv.groupby(group_cols, dropna=False).agg({"po_qty":"sum","received_qty":"sum"}).reset_index() if group_cols else pd.DataFrame()
        if not summ.empty:
            st.dataframe(summ.sort_values("po_qty", ascending=False).head(200), use_container_width=True)
    else:
        st.info("Delivery summary needs PO qty and Received qty columns (e.g., 'po_quantity', 'receivedqty').")

# ---------------- Vendors ----------------
with tabs[3]:
    st.subheader("Top Vendors by Spend")
    if vendor_col and net_col and vendor_col in fil.columns:
        vs = fil.groupby(vendor_col, dropna=False)[net_col].sum().reset_index()
        vs["cr"] = vs[net_col]/1e7
        vs = vs.sort_values("cr", ascending=False).head(10)
        if not vs.empty:
            st.dataframe(vs[[vendor_col,"cr"]].rename(columns={vendor_col:"PO Vendor","cr":"Spend (Cr)"}), use_container_width=True)
            st.plotly_chart(px.bar(vs, x=vendor_col, y="cr", text="cr", labels={"cr":"Cr ₹", vendor_col:"Vendor"}).update_traces(texttemplate="%{text:.2f}", textposition="outside"), use_container_width=True)
    else:
        st.info("Top vendor chart requires 'po_vendor' and 'net_amount' columns.")

# ---------------- Search ----------------
with tabs[4]:
    st.subheader("🔍 Keyword Search")
    valid_cols = [c for c in ["pr_number","purchase_doc","product_name","po_vendor"] if c in df.columns]
    query = st.text_input("Type vendor, product, PO, PR, etc.", "")
    cat_col = next((c for c in df.columns if "procurement_category" in c or "procurement category" in c), None)
    cat_sel = st.multiselect("Filter by Procurement Category", sorted(df[cat_col].dropna().astype(str).unique().tolist()) if cat_col else [])
    vend_sel = st.multiselect("Filter by Vendor", sorted(df[vendor_col].dropna().astype(str).unique().tolist()) if vendor_col else [])
    if query and valid_cols:
        mask = pd.Series(False, index=df.index)
        q = query.lower()
        for c in valid_cols:
            mask = mask | df[c].astype(str).str.lower().str.contains(q, na=False)
        res = df[mask].copy()
        if cat_sel and cat_col:
            res = res[res[cat_col].astype(str).isin(cat_sel)]
        if vend_sel and vendor_col:
            res = res[res[vendor_col].astype(str).isin(vend_sel)]
        st.write(f"Found {len(res):,} rows")
        st.dataframe(res, use_container_width=True)
        st.download_button("⬇️ Download Search Results", res.to_csv(index=False), "search_results.csv", "text/csv")
    elif not valid_cols:
        st.info("No searchable columns present.")
    else:
        st.caption("Start typing to search…")

# ---------------- Full Data (last tab) ----------------
with tabs[5]:
    st.subheader("Full Filtered Dataset (after applying filters)")
    st.write(f"Rows: {len(fil):,}")
    st.dataframe(fil, use_container_width=True)
    st.download_button("⬇️ Download filtered dataset", fil.to_csv(index=False), "filtered_dataset.csv", "text/csv")

# ---------------- Bottom uploader ----------------
st.markdown("---")
st.markdown("### Upload new files (bottom uploader)")
new_files = st.file_uploader("Upload Excel/CSV files here (this replaces dataset)", type=["xlsx","xls","csv"], accept_multiple_files=True, key="_bottom_uploader")
if new_files:
    st.session_state["_uploaded_files"] = new_files
    st.experimental_rerun()
