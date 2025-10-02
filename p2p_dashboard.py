import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date

# ====================================
#  Procure-to-Pay Dashboard (FINAL)
#  - Includes unified Smart Budget Mapper
#  - Department & Services drilldowns
# ====================================

st.set_page_config(page_title="Procure-to-Pay Dashboard", layout="wide", initial_sidebar_state="expanded")

# ------------------------------------
#  1) Load & Combine Source Data
# ------------------------------------
@st.cache_data(show_spinner=False)
def load_and_combine_data():
    mepl_df = pd.read_excel("MEPL1.xlsx", skiprows=1)
    mlpl_df = pd.read_excel("MLPL1.xlsx", skiprows=1)
    mmw_df  = pd.read_excel("mmw1.xlsx",  skiprows=1)
    mmpl_df = pd.read_excel("mmpl1.xlsx", skiprows=1)

    mepl_df["Entity"] = "MEPL"; mlpl_df["Entity"] = "MLPL"; mmw_df["Entity"] = "MMW"; mmpl_df["Entity"] = "MMPL"
    combined = pd.concat([mepl_df, mlpl_df, mmw_df, mmpl_df], ignore_index=True)
    combined.columns = combined.columns.str.strip().str.replace("\xa0"," ", regex=False).str.replace(" +"," ", regex=True)
    return combined

df = load_and_combine_data()

# ------------------------------------
#  2) Dates
# ------------------------------------
for c in ["PR Date Submitted","Po create Date","PO Approved Date","PO Delivery Date"]:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors="coerce")

# ------------------------------------
#  3) Buyer Group Classification
# ------------------------------------
if "Buyer Group" in df.columns:
    df["Buyer Group Code"] = df["Buyer Group"].astype(str).str.extract(r"(\d+)").astype(float)
    def classify(row):
        bg = row["Buyer Group"]; code = row["Buyer Group Code"]
        if bg in ["ME_BG17","MLBG16"]: return "Direct"
        if bg in ["Not Available"] or pd.isna(bg): return "Indirect"
        if pd.notna(code) and 1 <= code <= 9: return "Direct"
        if pd.notna(code) and 10 <= code <= 18: return "Indirect"
        return "Other"
    df["Buyer.Type"] = df.apply(classify, axis=1)
else:
    df["Buyer.Type"] = "Unknown"

# ------------------------------------
#  4) PO Orderer → PO.Creator Mapping
# ------------------------------------
o_created_by_map = {
    "MMW2324030":"Dhruv","MMW2324062":"Deepak","MMW2425154":"Mukul","MMW2223104":"Paurik","MMW2021181":"Nayan",
    "MMW2223014":"Aatish","MMW_EXT_002":"Deepakex","MMW2425024":"Kamlesh","MMW2021184":"Suresh","N/A":"Dilip"
}

df["PO Orderer"] = df.get("PO Orderer", pd.Series([None]*len(df))).fillna("N/A").astype(str).str.strip()
df["PO.Creator"] = df["PO Orderer"].map(o_created_by_map).fillna(df["PO Orderer"]).replace({"N/A":"Dilip"})
indirect_buyers = ["Aatish","Deepak","Deepakex","Dhruv","Dilip","Mukul","Nayan","Paurik","Kamlesh","Suresh"]
df["PO.BuyerType"] = df["PO.Creator"].apply(lambda x: "Indirect" if x in indirect_buyers else "Direct")

# ------------------------------------
#  5) Sidebar Filters
# ------------------------------------
st.sidebar.header("🔍 Filters")
for col in ["Buyer.Type","Entity","PO.Creator","PO.BuyerType","PR Date Submitted"]:
    if col not in df.columns: df[col] = pd.NA
for col in ["Buyer.Type","Entity","PO.Creator","PO.BuyerType"]:
    df[col] = df[col].astype(str).fillna("").str.strip()

def safe_unique(s):
    return sorted([str(x).strip() for x in s.dropna().unique()])

fy = {
    "All Years":(pd.to_datetime("2023-04-01"), pd.to_datetime("2026-03-31")),
    "2023":(pd.to_datetime("2023-04-01"), pd.to_datetime("2024-03-31")),
    "2024":(pd.to_datetime("2024-04-01"), pd.to_datetime("2025-03-31")),
    "2025":(pd.to_datetime("2025-04-01"), pd.to_datetime("2026-03-31")),
    "2026":(pd.to_datetime("2026-04-01"), pd.to_datetime("2027-03-31")),
}
sel_fy = st.sidebar.selectbox("Financial Year", list(fy.keys()), index=0)
pr_start, pr_end = fy[sel_fy]

buyer_filter = st.sidebar.multiselect("Buyer Type", safe_unique(df["Buyer.Type"]), default=safe_unique(df["Buyer.Type"]))
entity_filter = st.sidebar.multiselect("Entity", safe_unique(df["Entity"]), default=safe_unique(df["Entity"]))
orderer_filter = st.sidebar.multiselect("PO Ordered By", safe_unique(df["PO.Creator"]), default=safe_unique(df["PO.Creator"]))
po_buyer_type_filter = st.sidebar.multiselect("PO Buyer Type", safe_unique(df["PO.BuyerType"]), default=safe_unique(df["PO.BuyerType"]))

filtered_df = df.copy()
if "PR Date Submitted" in filtered_df.columns:
    filtered_df = filtered_df[(filtered_df["PR Date Submitted"] >= pr_start) & (filtered_df["PR Date Submitted"] <= pr_end)]

if buyer_filter:
    filtered_df = filtered_df[filtered_df["Buyer.Type"].isin(buyer_filter)]
if entity_filter:
    filtered_df = filtered_df[filtered_df["Entity"].isin(entity_filter)]
if orderer_filter:
    filtered_df = filtered_df[filtered_df["PO.Creator"].isin(orderer_filter)]
if po_buyer_type_filter:
    filtered_df = filtered_df[filtered_df["PO.BuyerType"].isin(po_buyer_type_filter)]

st.sidebar.markdown("---")
st.sidebar.write("FY:", sel_fy)
st.sidebar.write("Rows:", len(filtered_df))

# ------------------------------------
#  6) KPIs
# ------------------------------------
st.title("📊 Procure-to-Pay Dashboard")
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Total PRs", int(filtered_df.get("PR Number", pd.Series(dtype=object)).nunique()))
c2.metric("Total POs", int(filtered_df.get("Purchase Doc", pd.Series(dtype=object)).nunique()))
c3.metric("Line Items", len(filtered_df))
c4.metric("Entities", int(filtered_df.get("Entity", pd.Series(dtype=object)).nunique()))
c5.metric("Spend (Cr ₹)", f"{filtered_df.get('Net Amount', pd.Series(0)).sum()/1e7:,.2f}")

# ------------------------------------
#  7) SLA Compliance Gauge
# ------------------------------------
st.subheader("🎯 SLA Compliance (PR → PO ≤ 7 days)")
if {"Po create Date","PR Date Submitted"}.issubset(filtered_df.columns):
    lead_df = filtered_df.dropna(subset=["Po create Date","PR Date Submitted"]).copy()
    lead_df["Lead Time (Days)"] = (lead_df["Po create Date"] - lead_df["PR Date Submitted"]).dt.days
    SLA_DAYS = 7
    avg_lead = float(lead_df["Lead Time (Days)"].mean()) if not lead_df.empty else 0.0
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=avg_lead, number={"suffix":" days"},
        gauge={"axis":{"range":[0, max(14, avg_lead*1.2 if avg_lead else 14)]},
               "bar":{"color":"darkblue"},
               "steps":[{"range":[0,SLA_DAYS],"color":"lightgreen"},{"range":[SLA_DAYS, max(14, avg_lead*1.2 if avg_lead else 14)],"color":"lightcoral"}],
               "threshold":{"line":{"color":"red","width":4},"thickness":0.75,"value":SLA_DAYS}}))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Need 'Po create Date' and 'PR Date Submitted' to compute lead time.")

# ------------------------------------
#  8) Monthly Total Spend (bar + cumulative)
# ------------------------------------
st.subheader("📊 Monthly Total Spend (with Cumulative)")
dcol = "Po create Date" if "Po create Date" in filtered_df.columns else ("PR Date Submitted" if "PR Date Submitted" in filtered_df.columns else None)
if dcol and "Net Amount" in filtered_df.columns:
    t = filtered_df.dropna(subset=[dcol]).copy()
    t["PO_Month"] = t[dcol].dt.to_period("M").dt.to_timestamp(); t["Month_Str"] = t["PO_Month"].dt.strftime("%b-%Y")
    m = t.groupby(["PO_Month","Month_Str"], as_index=False)["Net Amount"].sum().sort_values("PO_Month")
    m["Spend (Cr ₹)"] = m["Net Amount"]/1e7; m["Cumulative (Cr ₹)"] = m["Spend (Cr ₹)"].cumsum()
    fig = make_subplots(specs=[[{"secondary_y":True}]])
    fig.add_bar(x=m["Month_Str"], y=m["Spend (Cr ₹)"], name="Monthly Spend (Cr ₹)")
    fig.add_scatter(x=m["Month_Str"], y=m["Cumulative (Cr ₹)"], mode="lines+markers", name="Cumulative (Cr ₹)", secondary_y=True)
    fig.update_layout(xaxis_tickangle=-45, legend=dict(orientation="h", y=1.05, x=1))
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------
#  9) Entity Trend
# ------------------------------------
st.subheader("💹 Monthly Spend Trend by Entity")
if "Net Amount" in filtered_df.columns and dcol:
    x = filtered_df.copy(); x["PO Month"] = x[dcol].dt.to_period("M").dt.to_timestamp()
    g = x.dropna(subset=["PO Month"]).groupby(["PO Month","Entity"], as_index=False)["Net Amount"].sum(); g["Cr"] = g["Net Amount"]/1e7
    fig = px.line(g, x=g["PO Month"].dt.strftime("%b-%Y"), y="Cr", color="Entity", markers=True, labels={"x":"Month","Cr":"Spend (Cr ₹)"})
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------
# 10) PR/PO Trends, Aging, Weekday
# ------------------------------------
st.subheader("📅 Monthly PR & PO Trends")
if dcol:
    tmp = filtered_df.copy()
    tmp["PR Month"] = tmp.get("PR Date Submitted", pd.Series(pd.NaT, index=tmp.index)).dt.to_period("M")
    tmp["PO Month"] = tmp.get("Po create Date", pd.Series(pd.NaT, index=tmp.index)).dt.to_period("M")
    ms = tmp.groupby("PR Month").agg({"PR Number":"count","Purchase Doc":"count"}).reset_index()
    if not ms.empty:
        ms.columns = ["Month","PR Count","PO Count"]; ms["Month"] = ms["Month"].astype(str)
        st.line_chart(ms.set_index("Month"), use_container_width=True)

st.subheader("🧮 PR to PO Aging Buckets")
if {"Po create Date","PR Date Submitted"}.issubset(filtered_df.columns):
    la = (filtered_df["Po create Date"] - filtered_df["PR Date Submitted"]).dt.days.dropna()
    bins = [0,7,15,30,60,90,999]; labels = ["0-7","8-15","16-30","31-60","61-90","90+"]
    age = pd.cut(la, bins=bins, labels=labels).value_counts(normalize=True).sort_index().reset_index()
    age.columns = ["Aging Bucket","Percentage"]; age["Percentage"] *= 100
    fig = px.bar(age, x="Aging Bucket", y="Percentage", text="Percentage"); fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("📆 PRs and POs by Weekday")
wd = filtered_df.copy(); wd["PR Weekday"] = wd.get("PR Date Submitted", pd.Series(pd.NaT, index=wd.index)).dt.day_name(); wd["PO Weekday"] = wd.get("Po create Date", pd.Series(pd.NaT, index=wd.index)).dt.day_name()
prc = wd["PR Weekday"].value_counts().reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"], fill_value=0)
poc = wd["PO Weekday"].value_counts().reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"], fill_value=0)
c1,c2 = st.columns(2); c1.bar_chart(prc, use_container_width=True); c2.bar_chart(poc, use_container_width=True)

# ------------------------------------
# 11) Open PRs
# ------------------------------------
st.subheader("⚠️ Open PRs (Approved/InReview)")
if "PR Status" in filtered_df.columns:
    open_df = filtered_df[filtered_df["PR Status"].isin(["Approved","InReview"])].copy()
    if not open_df.empty and "PR Date Submitted" in open_df.columns:
        open_df["Pending Age (Days)"] = (pd.Timestamp.today().normalize() - open_df["PR Date Submitted"]).dt.days
        sm = (open_df.groupby("PR Number").agg({
            "PR Date Submitted":"first","Pending Age (Days)":"first","Procurement Category":"first","Product Name":"first",
            "Net Amount":"sum","PO Budget Code":"first","PR Status":"first","Buyer Group":"first","Buyer.Type":"first",
            "Entity":"first","PO.Creator":"first","Purchase Doc":"first"}).reset_index())
        st.metric("🔢 Open PRs", int(sm["PR Number"].nunique()))
        st.bar_chart(pd.to_datetime(sm["PR Date Submitted"]).dt.to_period("M").value_counts().sort_index(), use_container_width=True)
        st.dataframe(sm, use_container_width=True)

# ------------------------------------
# 12) Buyer-wise Spend & Category
# ------------------------------------
st.subheader("💰 Buyer-wise Spend (Cr ₹)")
if {"PO.Creator","Net Amount"}.issubset(filtered_df.columns):
    bs = filtered_df.groupby("PO.Creator")["Net Amount"].sum().sort_values(ascending=False).reset_index(); bs["Cr"] = bs["Net Amount"]/1e7
    fig = px.bar(bs, x="PO.Creator", y="Cr", text="Cr", labels={"Cr":"Spend (Cr ₹)","PO.Creator":"Buyer"}); fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("🧭 Spend by Category (Descending)")
if {"Procurement Category","Net Amount"}.issubset(filtered_df.columns):
    cs = filtered_df.groupby("Procurement Category")["Net Amount"].sum().sort_values(ascending=False).reset_index(); cs["Cr"] = cs["Net Amount"]/1e7
    fig = px.bar(cs, x="Procurement Category", y="Cr", labels={"Cr":"Spend (Cr ₹)"}); fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------
# 13) PO Approval & Status
# ------------------------------------
if "PO Approved Date" in filtered_df.columns:
    st.subheader("📋 PO Approval Summary")
    pad = filtered_df.dropna(subset=["Po create Date"]).copy(); pad["PO Approved Date"] = pd.to_datetime(pad["PO Approved Date"], errors="coerce")
    tot = int(pad.get("Purchase Doc", pd.Series(dtype=object)).nunique()); appr = int(pad[pad["PO Approved Date"].notna()].get("Purchase Doc", pd.Series(dtype=object)).nunique())
    pend = tot - appr; pad["PO Approval Lead Time"] = (pad["PO Approved Date"] - pad["Po create Date"]).dt.days; avg = float(pad["PO Approval Lead Time"].mean()) if not pad.empty else 0.0
    c1,c2,c3,c4 = st.columns(4); c1.metric("📦 Total POs", tot); c2.metric("✅ Approved POs", appr); c3.metric("⏳ Pending", pend); c4.metric("⏱️ Avg Lead (d)", avg)
    st.dataframe(pad[["PO.Creator","Purchase Doc","Po create Date","PO Approved Date","PO Approval Lead Time"]].sort_values(by="PO Approval Lead Time", ascending=False), use_container_width=True)

if "PO Status" in filtered_df.columns:
    st.subheader("📊 PO Status Breakdown")
    s = filtered_df["PO Status"].value_counts().reset_index(); s.columns=["PO Status","Count"]
    c1,c2 = st.columns([2,3]); c1.dataframe(s, use_container_width=True); c2.plotly_chart(px.pie(s, names="PO Status", values="Count", hole=0.3), use_container_width=True)

# ------------------------------------
# 14) Delivery Summary & Top Pending by Value
# ------------------------------------
st.subheader("🚚 PO Delivery Summary: Received vs Pending")
delv = filtered_df.rename(columns={"PO Quantity":"PO Qty","ReceivedQTY":"Received Qty","Pending QTY":"Pending Qty"}).copy()
if {"PO Qty","Received Qty"}.issubset(delv.columns):
    delv["% Received"] = (delv["Received Qty"] / delv["PO Qty"]).replace([pd.NA, pd.NaT], 0).fillna(0)*100
    summ = delv.groupby(["Purchase Doc","PO Vendor","Product Name","Item Description"], dropna=False).agg({"PO Qty":"sum","Received Qty":"sum","Pending Qty":"sum","% Received":"mean"}).reset_index()
    st.dataframe(summ.sort_values("Pending Qty", ascending=False), use_container_width=True)
    fig = px.bar(summ.sort_values("Pending Qty", ascending=False).head(20), x="Purchase Doc", y="Pending Qty", color="PO Vendor", text="Pending Qty", title="Top 20 POs Awaiting Delivery")
    fig.update_traces(textposition="outside"); st.plotly_chart(fig, use_container_width=True)
    st.markdown("### 📋 Delivery Performance Summary")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("PO Lines", len(delv)); c2.metric("Fully Delivered", int((delv["Pending Qty"]==0).sum())); c3.metric("Pending Delivery", int((delv["Pending Qty"]>0).sum())); c4.metric("Avg. Receipt %", f"{delv['% Received'].mean():.1f}%")

st.subheader("📋 Top 50 Pending Lines (by Value)")
if {"Pending Qty","PO Unit Rate"}.issubset(delv.columns):
    p = delv[delv["Pending Qty"]>0].copy(); p["Pending Value"] = p["Pending Qty"]*p["PO Unit Rate"]
    keep = ["PR Number","Purchase Doc","Procurement Category","Buying legal entity","PR Budget description","Product Name","Item Description","Pending Qty","Pending Value"]
    cols = [c for c in keep if c in p.columns]
    st.dataframe(p.sort_values("Pending Value", ascending=False).head(50)[cols], use_container_width=True)

st.subheader("🏆 Top 10 Vendors by Spend (Cr ₹)")
if {"PO Vendor","Purchase Doc","Net Amount"}.issubset(filtered_df.columns):
    vs = (filtered_df.groupby("PO Vendor", dropna=False)
          .agg(Vendor_PO_Count=("Purchase Doc","nunique"), Total_Spend_Cr=("Net Amount", lambda x:(x.sum()/1e7).round(2)))
          .reset_index().sort_values("Total_Spend_Cr", ascending=False))
    st.dataframe(vs.head(10), use_container_width=True)
    st.plotly_chart(px.bar(vs.head(10), x="PO Vendor", y="Total_Spend_Cr", text="Total_Spend_Cr", labels={"Total_Spend_Cr":"Spend (Cr ₹)","PO Vendor":"Vendor"}), use_container_width=True)

# ======================================================================
# 15) SMART BUDGET MAPPER (Unified) — Dept & Services (Subcategory)
# ======================================================================

st.subheader("🏢 Department-wise Spend — Smart Mapper (Dept + Services)")

smart = filtered_df.copy(); smart["Dept.Chart"], smart["Subcat.Chart"], smart["__Dept.MapSrc"] = pd.NA, pd.NA, "UNMAPPED"

# --- Normalizers ---
import re

def _norm_code_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.upper().str.strip()
    s = s.str.replace("\xa0", " ", regex=False)
    s = s.str.replace("&", "AND", regex=False)
    s = s.str.replace("R&D", "RANDD", regex=False)
    s = s.str.replace(r"[/\\_\-]+", ".", regex=True)  # unify separators
    s = s.str.replace(r"\s+", "", regex=True)
    s = s.str.replace(r"\.{2,}", ".", regex=True)
    s = s.str.replace(r"^\.+|\.+$", "", regex=True)
    s = s.str.replace(r"[^A-Z0-9\.]+", "", regex=True)
    return s

def _norm_one(x):
    if pd.isna(x): return ""
    return _norm_code_series(pd.Series([x])).iloc[0]

# --- Hard fallback Prefix_3 → Dept (last resort) ---
_P3_FALLBACK = {
    "CNST":"Program","PM":"Program","PRJ":"Design",
    "MFG":"Manufacturing","INF":"Infra","HR":"HR & Admin","HRP":"HR & Admin",
    "MKT":"Marketing","A&M":"Marketing","FIN":"Finance",
    "LGL":"Legal & IP","LGLF":"Legal & IP","IP":"Legal & IP","PRT":"Legal & IP",
    "SS":"SS & SCM","TLG":"SS & SCM","COG":"SS & SCM",
    "SLS":"Sales","RENT":"Rental Offices","COUR":"Rental Offices",
    "R&D":"R&D","RANDD":"R&D","PRDDEV":"R&D","DVP":"R&D","MT":"R&D","TV":"R&D","VIC":"R&D","PT":"R&D","SFL":"R&D",
    "DSN":"Design","ACCS":"Design","CS":"Customer Success",
}

# --- Load mapping file(s) ---
expanded = None
try:
    expanded = pd.read_excel("Expanded_Budget_Code_Mapping.xlsx")
except Exception:
    try:
        expanded = pd.read_excel("Final_Budget_Mapping_Completed_Verified.xlsx")
    except Exception:
        expanded = None

exact_map = {}; exact_map_sub = {}; entity_pfx_map = {}; entity_pfx_map_sub = {}; pfx_map = {}; pfx_map_sub = {}
if expanded is not None:
    m = expanded.copy(); m.columns = m.columns.astype(str).str.strip()
    code_col = next((c for c in m.columns if c.lower().strip() in ["budget code","code","budget_code"]), None)
    dept_col = next((c for c in m.columns if "department" in c.lower()), None)
    subc_col = next((c for c in m.columns if ("subcat" in c.lower()) or ("sub category" in c.lower()) or ("subcategory" in c.lower())), None)
    p3_col   = next((c for c in m.columns if "prefix_3" in c.lower()), None)
    ent_col  = next((c for c in m.columns if c.lower().strip() in ["entity","domain","company","prefix_1","brand"]), None)

    if code_col and dept_col:
        m["__code_norm"] = _norm_code_series(m[code_col])
        tmp = m.dropna(subset=["__code_norm"]).drop_duplicates(subset=["__code_norm"], keep="first")
        exact_map = dict(zip(tmp["__code_norm"], tmp[dept_col].astype(str).str.strip()))
        if subc_col: exact_map_sub = dict(zip(tmp["__code_norm"], tmp[subc_col].astype(str).str.strip()))
    if p3_col: m["__p3_norm"] = _norm_code_series(m[p3_col])
    if ent_col: m["__ent_norm"] = _norm_code_series(m[ent_col])

    if p3_col and dept_col and ent_col and not m.empty:
        grp = m.dropna(subset=["__p3_norm","__ent_norm"]).copy()
        if not grp.empty:
            mode_dept = grp.groupby(["__ent_norm","__p3_norm"])[dept_col].agg(lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0])
            entity_pfx_map = mode_dept.to_dict()
            if subc_col:
                mode_sub = grp.groupby(["__ent_norm","__p3_norm"])[subc_col].agg(lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0])
                entity_pfx_map_sub = mode_sub.to_dict()

    if p3_col and dept_col and not m.empty:
        grp2 = m.dropna(subset=["__p3_norm"]).copy()
        if not grp2.empty:
            mode_dept2 = grp2.groupby("__p3_norm")[dept_col].agg(lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0])
            pfx_map = mode_dept2.to_dict()
            if subc_col:
                mode_sub2 = grp2.groupby("__p3_norm")[subc_col].agg(lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0])
                pfx_map_sub = mode_sub2.to_dict()

# --- Mapping function ---

def _pick_p3(code: str):
    segs = code.split('.')
    for j in range(len(segs)-1, -1, -1):
        seg = segs[j]
        if (seg in pfx_map) or (seg in _P3_FALLBACK):
            return seg
    return None

def _map_one(code_raw, ent_raw):
    code = _norm_one(code_raw); ent = _norm_one(ent_raw)
    if not code: return (pd.NA, pd.NA, "UNMAPPED")

    if code in exact_map:  # EXACT
        return (exact_map.get(code), exact_map_sub.get(code, pd.NA), "EXACT")

    parts = code.split('.')  # HIER
    if len(parts) > 1:
        for i in range(1, len(parts)):
            suf = '.'.join(parts[i:])
            if suf in exact_map:
                return (exact_map.get(suf), exact_map_sub.get(suf, pd.NA), "HIER")

    p3 = _pick_p3(code)
    if p3 and ent and (ent, p3) in entity_pfx_map:  # ENTITY+PFX3
        return (entity_pfx_map.get((ent,p3)), entity_pfx_map_sub.get((ent,p3), pd.NA), "ENTITY_PFX")
    if p3 and (p3 in pfx_map):  # PFX3 only
        return (pfx_map.get(p3), pfx_map_sub.get(p3, pd.NA), "PFX3")
    if p3 and (p3 in _P3_FALLBACK):  # KEYWORD fallback
        return (_P3_FALLBACK[p3], pd.NA, "KEYWORD")
    return (pd.NA, pd.NA, "UNMAPPED")

# --- Apply mapping ---
code_cols = [c for c in ["PO Budget Code","PR Budget Code"] if c in smart.columns]
if code_cols:
    base = smart[code_cols[0]]; ent = smart.get("Entity", pd.Series([pd.NA]*len(smart)))
    mapped = pd.DataFrame([_map_one(c,e) for c,e in zip(base.tolist(), ent.tolist())], columns=["__dept","__subc","__src"], index=smart.index)
    smart["Dept.Chart"] = smart["Dept.Chart"].combine_first(mapped["__dept"]) if "Dept.Chart" in smart.columns else mapped["__dept"]
    smart["Subcat.Chart"] = smart["Subcat.Chart"].combine_first(mapped["__subc"]) if "Subcat.Chart" in smart.columns else mapped["__subc"]
    need_src = smart["__Dept.MapSrc"].isin(["UNMAPPED", pd.NA, None])
    smart.loc[need_src, "__Dept.MapSrc"] = mapped.loc[need_src, "__src"].fillna("UNMAPPED")

pre_fb = smart["Dept.Chart"].isna()
if pre_fb.all():
    for cand in ["PO Department","PO Dept","PR Department","PR Dept","Dept.Final","Department"]:
        if cand in smart.columns:
            smart["Dept.Chart"] = smart["Dept.Chart"].combine_first(smart[cand])
smart.loc[smart["__Dept.MapSrc"].eq("UNMAPPED") & smart["Dept.Chart"].notna(), "__Dept.MapSrc"] = "FALLBACK"
smart["Dept.Chart"].fillna("Unmapped / Missing", inplace=True)

# --- Canonical labels (Dept/Subcategory) & optional alias override ---
DEPT_ALIASES = {
    "HR & ADMIN": "HR & Admin",
    "HUMAN RESOURCES": "HR & Admin",
    "LEGAL": "Legal & IP",
    "LEGAL & IP": "Legal & IP",
    "PROGRAM": "Program",
    "R AND D": "R&D",
    "R&D": "R&D",
    "RANDD": "R&D",
    "RESEARCH & DEVELOPMENT": "R&D",
    "INFRASTRUCTURE": "Infra",
    "INFRA": "Infra",
    "CUSTOMER SUCCESS": "Customer Success",
    "MFG": "Manufacturing",
    "MANUFACTURING": "Manufacturing",
    "DESIGN": "Design",
    "MARKETING": "Marketing",
    "SALES": "Sales",
    "SS & SCM": "SS & SCM",
    "SUPPLY CHAIN": "SS & SCM",
    "FINANCE": "Finance",
    "RENTAL OFFICES": "Rental Offices",
}
SUBCAT_ALIASES = {
    # Keep as-is unless you want to coalesce spelling variants. Examples:
    "HOUSEKEEPING": "Admin, Housekeeping and Security",
    "ADMIN, HOUSEKEEPING AND SECURITY": "Admin, Housekeeping and Security",
    "ELECTRICITY EXPENSES": "Electricity",
    "PANTY": "Pantry and Canteen",
    "PANTRY": "Pantry and Canteen",
    "TRAVEL": "Travel & Other",
}

def _canonize(val: pd.Series, mapping: dict) -> pd.Series:
    def f(x):
        if pd.isna(x):
            return x
        t = str(x).strip()
        key = t.upper()
        return mapping.get(key, t)
    return val.apply(f)

# Sidebar loader for alias overrides (CSV/XLSX with columns: Department, Subcategory, Dept.Alias, Subcat.Alias)
with st.sidebar.expander("🔧 Optional: alias overrides"):
    alias_file = st.file_uploader("Upload alias overrides (CSV/XLSX)", type=["csv","xlsx"], key="alias_upload")
    if alias_file is not None:
        try:
            if str(alias_file.name).lower().endswith(".csv"):
                alias_df = pd.read_csv(alias_file)
            else:
                alias_df = pd.read_excel(alias_file)
            alias_df.columns = alias_df.columns.astype(str).str.strip()
            if "Department" in alias_df.columns and "Dept.Alias" in alias_df.columns:
                DEPT_ALIASES.update({str(k).upper().strip(): str(v).strip() for k,v in alias_df[["Department","Dept.Alias"]].dropna().values})
            if "Subcategory" in alias_df.columns and "Subcat.Alias" in alias_df.columns:
                SUBCAT_ALIASES.update({str(k).upper().strip(): str(v).strip() for k,v in alias_df[["Subcategory","Subcat.Alias"]].dropna().values})
            st.success("Aliases loaded.")
        except Exception as e:
            st.warning(f"Alias file could not be read: {e}")

# Apply canonicalization just before charts/drilldowns
smart["Dept.Chart"] = _canonize(smart["Dept.Chart"].astype(str), DEPT_ALIASES)
smart["Subcat.Chart"] = _canonize(smart.get("Subcat.Chart", pd.Series([pd.NA]*len(smart))), SUBCAT_ALIASES)

with st.expander("🧪 Smart Mapper QA", expanded=False):
    st.write({"counts": smart["__Dept.MapSrc"].value_counts(dropna=False).to_dict()})
    if code_cols:
        probe = smart[smart["__Dept.MapSrc"]=="UNMAPPED"].copy()
        if not probe.empty:
            probe["Code.Raw"] = probe[code_cols[0]]; probe["Code.Norm"] = _norm_code_series(probe["Code.Raw"])
            def _guess(c):
                parts = str(c).split("."); return parts[-1] if parts else ""
            probe["P3.Guess"] = probe["Code.Norm"].apply(_guess)
            probe["Entity.Norm"] = smart.get("Entity", pd.Series([pd.NA]*len(smart))).loc[probe.index].apply(_norm_one)
            probe["P3.Fallback Dept"] = probe["P3.Guess"].apply(lambda t:_P3_FALLBACK.get(t, ""))
            show = probe[["Code.Raw","Code.Norm","Entity.Norm","P3.Guess","P3.Fallback Dept"]].drop_duplicates().head(200)
            st.dataframe(show, use_container_width=True)

# --- Charts & Drilldowns (Dept → Service → Lines) ---
if "Net Amount" in smart.columns:
    dep = smart.groupby("Dept.Chart", dropna=False)["Net Amount"].sum().reset_index().sort_values("Net Amount", ascending=False)
    dep["Cr"] = dep["Net Amount"]/1e7
    st.plotly_chart(px.bar(dep.head(30), x="Dept.Chart", y="Cr", labels={"Cr":"Spend (Cr ₹)"}, title="Department-wise Spend (Top 30)").update_layout(xaxis_tickangle=-45), use_container_width=True)

    cA,cB = st.columns([2,1])
    with cA:
        dept_pick = st.selectbox("Drill down: Department", dep["Dept.Chart"].astype(str).tolist(), key="smart_dept_pick")
    with cB:
        topn = st.number_input("Top N Vendors/Services", min_value=5, max_value=100, value=20, step=5, key="smart_topn")

    det = smart[smart["Dept.Chart"].astype(str)==str(dept_pick)].copy()
    k1,k2,k3 = st.columns(3)
    k1.metric("Lines", len(det)); k2.metric("PRs", int(det.get("PR Number", pd.Series(dtype=object)).nunique())); k3.metric("Spend (Cr ₹)", f"{det.get('Net Amount', pd.Series(0)).sum()/1e7:,.2f}")

    if "Subcat.Chart" in det.columns:
        ss = det.groupby("Subcat.Chart", dropna=False)["Net Amount"].sum().sort_values(ascending=False).reset_index(); ss["Cr"] = ss["Net Amount"]/1e7
        c1,c2 = st.columns(2)
        c1.plotly_chart(px.bar(ss.head(int(topn)), x="Subcat.Chart", y="Cr", title=f"{dept_pick} — Top Services").update_layout(xaxis_tickangle=-45), use_container_width=True)
        c2.plotly_chart(px.pie(ss.head(12), names="Subcat.Chart", values="Net Amount", title=f"{dept_pick} — Service Share"), use_container_width=True)
        svc = st.selectbox("🔎 Drill Service to see lines", ss["Subcat.Chart"].astype(str).tolist(), key="smart_svc_pick")
        sub = det[det["Subcat.Chart"].astype(str)==str(svc)].copy()
        want = ["PO Budget Code","Subcat.Chart","Dept.Chart","Purchase Doc","PR Number","Procurement Category","Product Name","Item Description","PO Vendor","Net Amount"]
        cols = [c for c in want if c in sub.columns]
        if not cols:
            cols = [c for c in ["Dept.Chart","Purchase Doc","PR Number","Net Amount"] if c in sub.columns]
        st.dataframe(sub[cols], use_container_width=True)
        st.download_button("⬇️ Download Lines (CSV)", sub[cols].to_csv(index=False), file_name=f"lines_{dept_pick}_{svc}.csv", mime="text/csv", key=f"dl_lines_{hash((dept_pick,svc))}")

    st.subheader("🧩 Dept × Service (₹ Cr)")
    if {"Dept.Chart","Subcat.Chart","Net Amount"}.issubset(smart.columns):
        pivot = smart.pivot_table(index="Dept.Chart", columns="Subcat.Chart", values="Net Amount", aggfunc="sum", fill_value=0.0)
        st.dataframe((pivot/1e7).round(2), use_container_width=True)
        treedf = (pivot/1e7).stack().reset_index(); treedf.columns=["Department","Service","SpendCr"]; treedf = treedf[treedf["SpendCr"]>0]
        st.plotly_chart(px.treemap(treedf, path=["Department","Service"], values="SpendCr", title="Department → Service Treemap (Cr ₹)"), use_container_width=True)

# ------------------------------------
#  End
# ------------------------------------
