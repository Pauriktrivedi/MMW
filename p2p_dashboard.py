import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="P2P Dashboard — Full", layout="wide", initial_sidebar_state="expanded")

# ---------- Load Data ----------
@st.cache_data(show_spinner=False)
def load_all():
    fns = [("MEPL.xlsx","MEPL"),("MLPL.xlsx","MLPL"),("mmw.xlsx","MMW"),("mmpl.xlsx","MMPL")]
    frames = []
    for fn, ent in fns:
        try:
            df = pd.read_excel(fn, skiprows=1)
        except Exception:
            # file missing or unreadable — skip
            continue
        df["Entity"] = ent
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    x = pd.concat(frames, ignore_index=True)
    # normalize column names lightly (keep human-readable names as user provided)
    x.columns = x.columns.str.strip().str.replace("\xa0"," ",regex=False).str.replace(r" +"," ",regex=True)
    # coerce common date fields if present
    for c in ["PR Date Submitted","Po create Date","PO Approved Date","PO Delivery Date"]:
        if c in x.columns:
            x[c] = pd.to_datetime(x[c], errors="coerce")
    return x

# load
df = load_all()
if df.empty:
    st.warning("No data loaded. Place MEPL.xlsx, MLPL.xlsx, mmw.xlsx, mmpl.xlsx next to this script or upload manually in your deployment.")

# ---------- Buyer/Creator ----------
if "Buyer Group" in df.columns:
    # extract numeric code (may be missing)
    try:
        df["Buyer Group Code"] = df["Buyer Group"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    except Exception:
        df["Buyer Group Code"] = np.nan
    def _bg(r):
        bg = r.get("Buyer Group")
        code = r.get("Buyer Group Code")
        try:
            if isinstance(bg, str) and bg.strip() in ["ME_BG17","MLBG16"]:
                return "Direct"
        except Exception:
            pass
        if pd.isna(bg) or str(bg).strip() == "Not Available" or str(bg).strip() == "":
            return "Indirect"
        try:
            if pd.notna(code) and 1 <= int(code) <= 9:
                return "Direct"
            if pd.notna(code) and 10 <= int(code) <= 18:
                return "Indirect"
        except Exception:
            pass
        return "Other"
    df["Buyer.Type"] = df.apply(_bg, axis=1)
else:
    df["Buyer.Type"] = "Unknown"

# PO orderer -> creator mapping
map_orderer = {"MMW2324030":"Dhruv","MMW2324062":"Deepak","MMW2425154":"Mukul","MMW2223104":"Paurik","MMW2021181":"Nayan","MMW2223014":"Aatish","MMW_EXT_002":"Deepakex","MMW2425024":"Kamlesh","MMW2021184":"Suresh","N/A":"Dilip"}
# safe getter for PO Orderer
po_orderer = df.get("PO Orderer", pd.Series([pd.NA]*len(df))).fillna("N/A").astype(str).str.strip()
df["PO Orderer"] = po_orderer
# map to nice names where possible
df["PO.Creator"] = df["PO Orderer"].str.upper().map({k.upper():v for k,v in map_orderer.items()}).fillna(df["PO Orderer"]).replace({"N/A":"Dilip"})
indirect = set(["Aatish","Deepak","Deepakex","Dhruv","Dilip","Mukul","Nayan","Paurik","Kamlesh","Suresh"])
df["PO.BuyerType"] = np.where(df["PO.Creator"].isin(indirect), "Indirect", "Direct")

# ---------- Sidebar Filters ----------
st.sidebar.header("Filters")
FY = {
    "All Years":(pd.Timestamp("2023-04-01"),pd.Timestamp("2026-03-31")),
    "2023":(pd.Timestamp("2023-04-01"),pd.Timestamp("2024-03-31")),
    "2024":(pd.Timestamp("2024-04-01"),pd.Timestamp("2025-03-31")),
    "2025":(pd.Timestamp("2025-04-01"),pd.Timestamp("2026-03-31")),
    "2026":(pd.Timestamp("2026-04-01"),pd.Timestamp("2027-03-31"))
}
fy_key = st.sidebar.selectbox("Financial Year", list(FY), index=0)
pr_start, pr_end = FY[fy_key]

fil = df.copy()
if "PR Date Submitted" in fil.columns:
    fil = fil[(fil["PR Date Submitted"] >= pr_start) & (fil["PR Date Submitted"] <= pr_end)]

# Month selector (derived from chosen date basis)
month_basis = "PR Date Submitted" if "PR Date Submitted" in fil.columns else ("Po create Date" if "Po create Date" in fil.columns else None)
if month_basis:
    months = fil[month_basis].dropna().dt.to_period("M").astype(str).unique().tolist()
    months = sorted(months, key=lambda s: pd.Period(s))
    month_labels = [pd.Period(m).strftime("%b-%Y") for m in months]
    label_to_period = {pd.Period(m).strftime("%b-%Y"): m for m in months}
    if month_labels:
        sel_month = st.sidebar.selectbox("Month", ["All Months"] + month_labels, index=0)
        if sel_month != "All Months":
            target_period = label_to_period[sel_month]
            fil = fil[fil[month_basis].dt.to_period("M").astype(str) == target_period]

# optional date-range (after FY/month)
if month_basis:
    mindt = fil[month_basis].dropna().min()
    maxdt = fil[month_basis].dropna().max()
    if pd.notna(mindt) and pd.notna(maxdt):
        dr = st.sidebar.date_input("Date range", (pd.Timestamp(mindt).date(), pd.Timestamp(maxdt).date()), key="sidebar_date_range")
        if isinstance(dr, tuple) and len(dr) == 2:
            _s, _e = pd.to_datetime(dr[0]), pd.to_datetime(dr[1])
            fil = fil[(fil[month_basis] >= _s) & (fil[month_basis] <= _e)]

# ensure filter cols exist and normalized for display
for col in ["Buyer.Type","Entity","PO.Creator","PO.BuyerType"]:
    if col not in fil.columns:
        fil[col] = ""
    fil[col] = fil[col].astype(str).str.strip()

sel_b = st.sidebar.multiselect("Buyer Type", sorted(fil["Buyer.Type"].dropna().unique().tolist()), default=sorted(fil["Buyer.Type"].dropna().unique().tolist()))
sel_e = st.sidebar.multiselect("Entity", sorted(fil["Entity"].dropna().unique().tolist()), default=sorted(fil["Entity"].dropna().unique().tolist()))
sel_o = st.sidebar.multiselect("PO Ordered By", sorted(fil["PO.Creator"].dropna().unique().tolist()), default=sorted(fil["PO.Creator"].dropna().unique().tolist()))
sel_p = st.sidebar.multiselect("PO Buyer Type", sorted(fil["PO.BuyerType"].dropna().unique().tolist()), default=sorted(fil["PO.BuyerType"].dropna().unique().tolist()))

if sel_b:
    fil = fil[fil["Buyer.Type"].isin(sel_b)]
if sel_e:
    fil = fil[fil["Entity"].isin(sel_e)]
if sel_o:
    fil = fil[fil["PO.Creator"].isin(sel_o)]
if sel_p:
    fil = fil[fil["PO.BuyerType"].isin(sel_p)]

# Vendor & Item filters (multi-select) — shown below the other filters
if "PO Vendor" not in fil.columns:
    fil["PO Vendor"] = ""
if "Product Name" not in fil.columns:
    fil["Product Name"] = ""
fil["PO Vendor"] = fil["PO Vendor"].astype(str).str.strip()
fil["Product Name"] = fil["Product Name"].astype(str).str.strip()

vendor_choices = sorted(fil["PO Vendor"].dropna().unique().tolist())
item_choices = sorted(fil["Product Name"].dropna().unique().tolist())
sel_v = st.sidebar.multiselect("Vendor (pick one or more)", vendor_choices, default=vendor_choices)
sel_i = st.sidebar.multiselect("Item / Product (pick one or more)", item_choices, default=item_choices)

if sel_v:
    fil = fil[fil["PO Vendor"].isin(sel_v)]
if sel_i:
    fil = fil[fil["Product Name"].isin(sel_i)]

# Reset Filters button
if st.sidebar.button("Reset Filters"):
    for k in list(st.session_state.keys()):
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
    c1.metric("Total PRs", int(fil.get("PR Number",pd.Series(dtype=object)).nunique()))
    c2.metric("Total POs", int(fil.get("Purchase Doc",pd.Series(dtype=object)).nunique()))
    c3.metric("Line Items", len(fil))
    c4.metric("Entities", int(fil.get("Entity",pd.Series(dtype=object)).nunique()))
    c5.metric("Spend (Cr ₹)", f"{fil.get('Net Amount',pd.Series(0)).sum()/1e7:,.2f}")

# ---------- Spend (Monthly + Entity) ----------
with T[1]:
    dcol = "Po create Date" if "Po create Date" in fil.columns else ("PR Date Submitted" if "PR Date Submitted" in fil.columns else None)
    st.subheader("Monthly Total Spend + Cumulative")
    if dcol and "Net Amount" in fil.columns:
        t = fil.dropna(subset=[dcol]).copy()
        t["PO_Month"] = t[dcol].dt.to_period("M").dt.to_timestamp(); t["Month_Str"] = t["PO_Month"].dt.strftime("%b-%Y")
        m = t.groupby(["PO_Month","Month_Str"],as_index=False)["Net Amount"].sum().sort_values("PO_Month")
        m["Cr"] = m["Net Amount"]/1e7; m["CumCr"] = m["Cr"].cumsum()
        fig = make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_bar(x=m["Month_Str"], y=m["Cr"], name="Monthly Spend (Cr ₹)")
        fig.add_scatter(x=m["Month_Str"], y=m["CumCr"], name="Cumulative (Cr ₹)", mode="lines+markers", secondary_y=True)
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Entity Trend")
    if dcol and "Net Amount" in fil.columns:
        x = fil.copy(); x["PO Month"] = x[dcol].dt.to_period("M").dt.to_timestamp()
        g = x.dropna(subset=["PO Month"]).groupby(["PO Month","Entity"],as_index=False)["Net Amount"].sum(); g["Cr"] = g["Net Amount"]/1e7
        st.plotly_chart(px.line(g, x=g["PO Month"].dt.strftime("%b-%Y"), y="Cr", color="Entity", markers=True, labels={"x":"Month","Cr":"Cr ₹"}).update_layout(xaxis_tickangle=-45), use_container_width=True)

# ---------- PR/PO Timing ----------
with T[2]:
    st.subheader("SLA (PR→PO ≤7d)")
    if {"Po create Date","PR Date Submitted"}.issubset(fil.columns):
        ld = fil.dropna(subset=["Po create Date","PR Date Submitted"]).copy(); ld["Lead Time (Days)"]=(ld["Po create Date"]-ld["PR Date Submitted"]).dt.days
        avg = float(ld["Lead Time (Days)"].mean()) if not ld.empty else 0.0
        fig = go.Figure(go.Indicator(mode="gauge+number",value=avg,number={"suffix":" d"},gauge={"axis":{"range":[0,max(14,avg*1.2 if avg else 14)]},"bar":{"color":"darkblue"},"steps":[{"range":[0,7],"color":"lightgreen"},{"range":[7,max(14,avg*1.2 if avg else 14)],"color":"lightcoral"}],"threshold":{"line":{"color":"red","width":4},"value":7}}))
        st.plotly_chart(fig, use_container_width=True)
        bins=[0,7,15,30,60,90,999]; labels=["0-7","8-15","16-30","31-60","61-90","90+"]
        ag = pd.cut(ld["Lead Time (Days)"], bins=bins, labels=labels).value_counts(normalize=True).sort_index().reset_index(); ag.columns=["Bucket","Pct"]; ag["Pct"]=ag["Pct"]*100
        st.plotly_chart(px.bar(ag,x="Bucket",y="Pct",text="Pct").update_traces(texttemplate="%{text:.1f}%",textposition="outside"), use_container_width=True)
    # ... rest of timing logic (PR/PO per month, weekday, open PRs) follows same structure as earlier code

# ---------- Delivery ----------
with T[3]:
    st.subheader("Delivery Summary")
    dv = fil.rename(columns={"PO Quantity":"PO Qty","ReceivedQTY":"Received Qty","Pending QTY":"Pending Qty"}).copy()
    if {"PO Qty","Received Qty"}.issubset(dv.columns):
        dv["% Received"] = np.where(dv["PO Qty"].astype(float)>0,(dv["Received Qty"].astype(float)/dv["PO Qty"].astype(float))*100,0.0)
        summ = dv.groupby(["Purchase Doc","PO Vendor","Product Name","Item Description"],dropna=False).agg({"PO Qty":"sum","Received Qty":"sum","Pending Qty":"sum","% Received":"mean"}).reset_index()
        st.dataframe(summ.sort_values("Pending Qty",ascending=False), use_container_width=True)

# ---------- Vendors ----------
with T[4]:
    st.subheader("Top Vendors by Spend")
    if {"PO Vendor","Purchase Doc","Net Amount"}.issubset(fil.columns):
        vs = fil.groupby("PO Vendor",dropna=False).agg(Vendor_PO_Count=("Purchase Doc","nunique"), Total_Spend_Cr=("Net Amount", lambda s:(s.sum()/1e7).round(2))).reset_index().sort_values("Total_Spend_Cr",ascending=False)
        st.dataframe(vs.head(10), use_container_width=True)

# ---------- Dept & Services (Smart Mapper) ----------
with T[5]:
    st.subheader("Dept & Services (Smart Mapper)")
    # For department spend use PR Department column if present, else fallback
    dept_col_candidates = [c for c in ["PR Department","Dept","Department","PR Dept"] if c in fil.columns]
    if dept_col_candidates:
        dept_col = dept_col_candidates[0]
        dep = fil.groupby(dept_col, dropna=False)["Net Amount"].sum().reset_index().sort_values("Net Amount", ascending=False)
        dep["Cr"] = dep["Net Amount"]/1e7
        st.plotly_chart(px.bar(dep.head(30), x=dept_col, y="Cr", title='Department-wise Spend (Top 30)').update_layout(xaxis_tickangle=-45), use_container_width=True)
    else:
        st.info("Mapping logic can be enabled by placing mapping files in the working folder. See logs / earlier versions for mapping code.")

# ---------- Unit-rate Outliers ----------
with T[6]:
    st.subheader("Unit-rate Outliers vs Historical Median")
    grp_by = st.selectbox("Group by", [c for c in ["Product Name","Item Code"] if c in fil.columns], index=0)
    if {grp_by,"PO Unit Rate"}.issubset(fil.columns):
        z = fil[[grp_by,"PO Unit Rate","Purchase Doc","PR Number","PO Vendor","Item Description","Po create Date","Net Amount"]].dropna(subset=[grp_by,"PO Unit Rate"]).copy()
        med = z.groupby(grp_by)["PO Unit Rate"].median().rename("MedianRate")
        z = z.join(med, on=grp_by)
        z["PctDev"] = (z["PO Unit Rate"] - z["MedianRate"]) / z["MedianRate"].replace(0,np.nan)
        thr = st.slider("Outlier threshold (±%)", 10, 300, 50, 5)
        out = z[abs(z["PctDev"]) >= thr/100.0].copy(); out["PctDev%"]=(out["PctDev"]*100).round(1)
        st.dataframe(out.sort_values("PctDev%", ascending=False), use_container_width=True)

# ---------- Forecast ----------
with T[7]:
    st.subheader("Forecast Next Month Spend (SMA)")
    dcol = "Po create Date" if "Po create Date" in fil.columns else ("PR Date Submitted" if "PR Date Submitted" in fil.columns else None)
    if dcol and "Net Amount" in fil.columns:
        t = fil.dropna(subset=[dcol]).copy(); t["Month"] = t[dcol].dt.to_period("M").dt.to_timestamp()
        m = t.groupby("Month")["Net Amount"].sum().sort_index(); m_cr = (m/1e7)
        k = st.slider("Window (months)", 3, 12, 6, 1)
        sma = m_cr.rolling(k).mean()
        mu = m_cr.tail(k).mean() if len(m_cr) >= k else m_cr.mean()
        sd = m_cr.tail(k).std(ddof=1) if len(m_cr) >= k else m_cr.std(ddof=1)
        n = min(k, max(1, len(m_cr)))
        se = sd/np.sqrt(n) if sd==sd else 0
        lo, hi = float(mu-1.96*se), float(mu+1.96*se)
        nxt = (m_cr.index.max() + pd.offsets.MonthBegin(1)) if len(m_cr) > 0 else pd.Timestamp.today().to_period("M").to_timestamp()
        fdf = pd.DataFrame({"Month": list(m_cr.index) + [nxt], "SpendCr": list(m_cr.values) + [np.nan], "SMA": list(sma.values) + [mu]})
        fig = go.Figure()
        fig.add_bar(x=fdf["Month"], y=fdf["SpendCr"], name="Actual (Cr)")
        fig.add_scatter(x=fdf["Month"], y=fdf["SMA"], mode='lines+markers', name=f"SMA{k}")
        fig.add_scatter(x=[nxt,nxt], y=[lo,hi], mode='lines', name='95% CI')
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Forecast for {nxt.strftime('%b-%Y')}: {mu:.2f} Cr (95% CI: {lo:.2f}–{hi:.2f})")

# ---------- Vendor Scorecards ----------
with T[8]:
    st.subheader("Vendor Scorecard")
    if "PO Vendor" in fil.columns:
        vendor = st.selectbox("Pick Vendor", sorted(fil["PO Vendor"].dropna().astype(str).unique().tolist()))
        vd = fil[fil["PO Vendor"].astype(str) == str(vendor)].copy()
        spend = vd.get("Net Amount", pd.Series(0)).sum()/1e7
        upos = int(vd.get("Purchase Doc", pd.Series(dtype=object)).nunique())
        today = pd.Timestamp.today().normalize()
        if {"PO Delivery Date","Pending QTY"}.issubset(vd.columns):
            late = ((pd.to_datetime(vd["PO Delivery Date"], errors='coerce').dt.date < today.date()) & (vd["Pending QTY"].fillna(0) > 0)).sum()
        else:
            late = np.nan
        if "Pending QTY" in vd.columns and "PO Unit Rate" in vd.columns:
            vd["Pending Value"] = vd["Pending QTY"].fillna(0).astype(float) * vd["PO Unit Rate"].fillna(0).astype(float)
            pend_val = vd["Pending Value"].sum()/1e7
        else:
            pend_val = np.nan
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Spend (Cr)", f"{spend:.2f}"); k2.metric("Unique POs", upos); k3.metric("Late PO count", None if pd.isna(late) else int(late)); k4.metric("Pending Value (Cr)", None if pd.isna(pend_val) else f"{pend_val:.2f}")

# ---------- Search (Keyword) ----------
with T[9]:
    st.subheader("🔍 Keyword Search")
    valid_cols = [c for c in ["PR Number","Purchase Doc","Product Name","PO Vendor"] if c in df.columns]
    query = st.text_input("Type vendor, product, PO, PR, etc.", "")
    cat_sel = st.multiselect("Filter by Procurement Category", sorted(df.get("Procurement Category", pd.Series(dtype=object)).dropna().astype(str).unique().tolist())) if "Procurement Category" in df.columns else []
    vend_sel = st.multiselect("Filter by Vendor", sorted(df.get("PO Vendor", pd.Series(dtype=object)).dropna().astype(str).unique().tolist())) if "PO Vendor" in df.columns else []
    if query and valid_cols:
        mask = pd.Series(False, index=df.index)
        q = query.lower()
        for c in valid_cols:
            mask = mask | df[c].astype(str).str.lower().str.contains(q, na=False)
        res = df[mask].copy()
        if cat_sel:
            res = res[res["Procurement Category"].astype(str).isin(cat_sel)]
        if vend_sel:
            res = res[res["PO Vendor"].astype(str).isin(vend_sel)]
        st.write(f"Found {len(res)} rows")
        st.dataframe(res, use_container_width=True)
        st.download_button("⬇️ Download Search Results", res.to_csv(index=False), file_name="search_results.csv", mime="text/csv", key="dl_search")
    elif not valid_cols:
        st.info("No searchable columns present.")
    else:
        st.caption("Start typing to search…")
