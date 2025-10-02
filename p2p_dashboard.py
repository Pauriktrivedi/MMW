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
        df = pd.read_excel(fn, skiprows=1)
        df["Entity"] = ent
        frames.append(df)
    x = pd.concat(frames, ignore_index=True)
    x.columns = x.columns.str.strip().str.replace("\xa0"," ",regex=False).str.replace(" +"," ",regex=True)
    for c in ["PR Date Submitted","Po create Date","PO Approved Date","PO Delivery Date"]:
        if c in x.columns: x[c] = pd.to_datetime(x[c], errors="coerce")
    return x

df = load_all()

# ---------- Buyer/Creator ----------
if "Buyer Group" in df.columns:
    df["Buyer Group Code"] = df["Buyer Group"].astype(str).str.extract(r"(\d+)").astype(float)
    def _bg(r):
        bg,code = r["Buyer Group"], r["Buyer Group Code"]
        if bg in ["ME_BG17","MLBG16"]: return "Direct"
        if bg in ["Not Available"] or pd.isna(bg): return "Indirect"
        if pd.notna(code) and 1<=code<=9: return "Direct"
        if pd.notna(code) and 10<=code<=18: return "Indirect"
        return "Other"
    df["Buyer.Type"] = df.apply(_bg, axis=1)
else:
    df["Buyer.Type"] = "Unknown"
map_orderer = {"MMW2324030":"Dhruv","MMW2324062":"Deepak","MMW2425154":"Mukul","MMW2223104":"Paurik","MMW2021181":"Nayan","MMW2223014":"Aatish","MMW_EXT_002":"Deepakex","MMW2425024":"Kamlesh","MMW2021184":"Suresh","N/A":"Dilip"}
df["PO Orderer"] = df.get("PO Orderer", pd.Series([None]*len(df))).fillna("N/A").astype(str).str.strip()
df["PO.Creator"] = df["PO Orderer"].map(map_orderer).fillna(df["PO Orderer"]).replace({"N/A":"Dilip"})
indirect = set(["Aatish","Deepak","Deepakex","Dhruv","Dilip","Mukul","Nayan","Paurik","Kamlesh","Suresh"])
df["PO.BuyerType"] = np.where(df["PO.Creator"].isin(indirect), "Indirect", "Direct")

# ---------- Sidebar Filters ----------
st.sidebar.header("Filters")
FY = {"All Years":(pd.Timestamp("2023-04-01"),pd.Timestamp("2026-03-31")),"2023":(pd.Timestamp("2023-04-01"),pd.Timestamp("2024-03-31")),"2024":(pd.Timestamp("2024-04-01"),pd.Timestamp("2025-03-31")),"2025":(pd.Timestamp("2025-04-01"),pd.Timestamp("2026-03-31")),"2026":(pd.Timestamp("2026-04-01"),pd.Timestamp("2027-03-31"))}
fy_key = st.sidebar.selectbox("Financial Year", list(FY))
pr_start, pr_end = FY[fy_key]
fil = df.copy()
if "PR Date Submitted" in fil.columns:
    fil = fil[(fil["PR Date Submitted"]>=pr_start)&(fil["PR Date Submitted"]<=pr_end)]

# ---- Month dropdown (under FY) ----
# Use PR Date Submitted primarily; if missing, fall back to PO create Date
month_basis = "PR Date Submitted" if "PR Date Submitted" in fil.columns else ("Po create Date" if "Po create Date" in fil.columns else None)
sel_month = "All Months"
if month_basis:
    # Build month list from filtered rows (FY applied)
    months = fil[month_basis].dropna().dt.to_period("M").astype(str).unique().tolist()
    months = sorted(months, key=lambda s: pd.Period(s))
    month_labels = [pd.Period(m).strftime("%b-%Y") for m in months]
    label_to_period = {pd.Period(m).strftime("%b-%Y"): m for m in months}
    if month_labels:
        sel_month = st.sidebar.selectbox("Month", ["All Months"] + month_labels, index=0)
        if sel_month != "All Months":
            target_period = label_to_period[sel_month]
            fil = fil[fil[month_basis].dt.to_period("M").astype(str) == target_period]

# ---- Optional calendar date range (applies after FY + Month) ----
if month_basis:
    _mindt = fil[month_basis].dropna().min()
    _maxdt = fil[month_basis].dropna().max()
    if pd.notna(_mindt) and pd.notna(_maxdt):
        dr = st.sidebar.date_input(
            "Date range",
            (pd.Timestamp(_mindt).date(), pd.Timestamp(_maxdt).date()),
            key="sidebar_date_range",
        )
        if isinstance(dr, tuple) and len(dr) == 2:
            _s, _e = pd.to_datetime(dr[0]), pd.to_datetime(dr[1])
            fil = fil[(fil[month_basis] >= _s) & (fil[month_basis] <= _e)]
for col in ["Buyer.Type","Entity","PO.Creator","PO.BuyerType"]:
    if col not in fil.columns: fil[col] = ""
    fil[col] = fil[col].astype(str).str.strip()
sel_b = st.sidebar.multiselect("Buyer Type", sorted(fil["Buyer.Type"].dropna().unique().tolist()), default=sorted(fil["Buyer.Type"].dropna().unique().tolist()))
sel_e = st.sidebar.multiselect("Entity", sorted(fil["Entity"].dropna().unique().tolist()), default=sorted(fil["Entity"].dropna().unique().tolist()))
sel_o = st.sidebar.multiselect("PO Ordered By", sorted(fil["PO.Creator"].dropna().unique().tolist()), default=sorted(fil["PO.Creator"].dropna().unique().tolist()))
sel_p = st.sidebar.multiselect("PO Buyer Type", sorted(fil["PO.BuyerType"].dropna().unique().tolist()), default=sorted(fil["PO.BuyerType"].dropna().unique().tolist()))
if sel_b: fil = fil[fil["Buyer.Type"].isin(sel_b)]
if sel_e: fil = fil[fil["Entity"].isin(sel_e)]
if sel_o: fil = fil[fil["PO.Creator"].isin(sel_o)]
if sel_p: fil = fil[fil["PO.BuyerType"].isin(sel_p)]

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
    st.subheader("PR & PO per Month")
    tmp = fil.copy(); tmp["PR Month"]=tmp.get("PR Date Submitted",pd.Series(pd.NaT,index=tmp.index)).dt.to_period("M"); tmp["PO Month"]=tmp.get("Po create Date",pd.Series(pd.NaT,index=tmp.index)).dt.to_period("M")
    ms = tmp.groupby("PR Month").agg({"PR Number":"count","Purchase Doc":"count"}).reset_index();
    if not ms.empty:
        ms.columns=["Month","PR Count","PO Count"]; ms["Month"]=ms["Month"].astype(str)
        st.line_chart(ms.set_index("Month"), use_container_width=True)
    st.subheader("Weekday Split")
    wd = fil.copy(); wd["PR Wk"]=wd.get("PR Date Submitted",pd.Series(pd.NaT,index=wd.index)).dt.day_name(); wd["PO Wk"]=wd.get("Po create Date",pd.Series(pd.NaT,index=wd.index)).dt.day_name()
    prc = wd["PR Wk"].value_counts().reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"], fill_value=0)
    poc = wd["PO Wk"].value_counts().reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"], fill_value=0)
    c1,c2 = st.columns(2); c1.bar_chart(prc); c2.bar_chart(poc)
    st.subheader("Open PRs")
    if "PR Status" in fil.columns:
        op = fil[fil["PR Status"].isin(["Approved","InReview"])].copy()
        if not op.empty and "PR Date Submitted" in op.columns:
            op["Pending Age (d)"]=(pd.Timestamp.today().normalize()-op["PR Date Submitted"]).dt.days
            st.dataframe(op[[c for c in ["PR Number","PR Date Submitted","Pending Age (d)","Procurement Category","Product Name","Net Amount","PO Budget Code","PR Status","Entity","PO.Creator","Purchase Doc"] if c in op.columns]], use_container_width=True)

    # NEW: Lead time by Buyer Type & Buyer
    st.subheader("Lead Time by Buyer Type & Buyer")
    if {"Po create Date","PR Date Submitted"}.issubset(fil.columns):
        ld2 = fil.dropna(subset=["Po create Date","PR Date Submitted"]).copy(); ld2["Lead Time (Days)"]=(ld2["Po create Date"]-ld2["PR Date Submitted"]).dt.days
        if "Buyer.Type" in ld2.columns:
            st.dataframe(ld2.groupby("Buyer.Type")["Lead Time (Days)"].mean().round(1).reset_index().sort_values("Lead Time (Days)"), use_container_width=True)
        if "PO.Creator" in ld2.columns:
            st.dataframe(ld2.groupby("PO.Creator")["Lead Time (Days)"].mean().round(1).reset_index().sort_values("Lead Time (Days)"), use_container_width=True)

    # NEW: Daily PR Trends
    st.subheader("Daily PR Submissions")
    if "PR Date Submitted" in fil.columns:
        daily = fil.copy(); daily["PR Date"]=pd.to_datetime(daily["PR Date Submitted"], errors="coerce")
        dtrend = daily.groupby("PR Date").size().reset_index(name="PR Count")
        st.plotly_chart(px.line(dtrend, x="PR Date", y="PR Count", title="Daily PRs"), use_container_width=True)

    # NEW: Monthly Unique PO Generation
    st.subheader("Monthly Unique PO Generation")
    if {"Purchase Doc","Po create Date"}.issubset(fil.columns):
        pm = fil.dropna(subset=["Po create Date","Purchase Doc"]).copy(); pm["PO Month"]=pm["Po create Date"].dt.to_period("M")
        mcount = pm.groupby("PO Month")["Purchase Doc"].nunique().reset_index(name="Unique PO Count"); mcount["PO Month"]=mcount["PO Month"].astype(str)
        st.plotly_chart(px.bar(mcount, x="PO Month", y="Unique PO Count", text="Unique PO Count", title="Unique POs per Month").update_traces(textposition="outside"), use_container_width=True)

# ---------- Delivery ----------
with T[3]:
    st.subheader("Delivery Summary")
    dv = fil.rename(columns={"PO Quantity":"PO Qty","ReceivedQTY":"Received Qty","Pending QTY":"Pending Qty"}).copy()
    if {"PO Qty","Received Qty"}.issubset(dv.columns):
        dv["% Received"]=np.where(dv["PO Qty"].astype(float)>0,(dv["Received Qty"].astype(float)/dv["PO Qty"].astype(float))*100,0.0)
        summ = dv.groupby(["Purchase Doc","PO Vendor","Product Name","Item Description"],dropna=False).agg({"PO Qty":"sum","Received Qty":"sum","Pending Qty":"sum","% Received":"mean"}).reset_index()
        st.dataframe(summ.sort_values("Pending Qty",ascending=False), use_container_width=True)
        st.plotly_chart(px.bar(summ.sort_values("Pending Qty",ascending=False).head(20), x="Purchase Doc", y="Pending Qty", color="PO Vendor", text="Pending Qty", title="Top 20 Pending Qty").update_traces(textposition="outside"), use_container_width=True)
        st.subheader("Top Pending Lines by Value")
        if {"Pending Qty","PO Unit Rate"}.issubset(dv.columns):
            dv["Pending Value"]=dv["Pending Qty"].astype(float)*dv["PO Unit Rate"].astype(float)
            keep=["PR Number","Purchase Doc","Procurement Category","Buying legal entity","PR Budget description","Product Name","Item Description","Pending Qty","PO Unit Rate","Pending Value"]
            st.dataframe(dv.sort_values("Pending Value",ascending=False).head(50)[[c for c in keep if c in dv.columns]], use_container_width=True)

# ---------- Vendors ----------
with T[4]:
    st.subheader("Top Vendors by Spend")
    if {"PO Vendor","Purchase Doc","Net Amount"}.issubset(fil.columns):
        vs = fil.groupby("PO Vendor",dropna=False).agg(Vendor_PO_Count=("Purchase Doc","nunique"), Total_Spend_Cr=("Net Amount", lambda s:(s.sum()/1e7).round(2))).reset_index().sort_values("Total_Spend_Cr",ascending=False)
        st.dataframe(vs.head(10), use_container_width=True)
        st.plotly_chart(px.bar(vs.head(10), x="PO Vendor", y="Total_Spend_Cr", text="Total_Spend_Cr", title="Top 10 Vendors (Cr ₹)").update_traces(textposition="outside"), use_container_width=True)

    # NEW: Vendor Delivery Performance
    st.subheader("Vendor Delivery Performance (Top 10 by Spend)")
    if {"PO Vendor","Purchase Doc","PO Delivery Date","Pending QTY"}.issubset(fil.columns):
        today = pd.Timestamp.today().normalize().date()
        vdf = fil.copy()
        vdf["PendingQtyFill"] = vdf["Pending QTY"].fillna(0).astype(float)
        vdf["Is_Fully_Delivered"] = vdf["PendingQtyFill"] == 0
        vdf["PO Delivery Date"] = pd.to_datetime(vdf["PO Delivery Date"], errors="coerce")
        vdf["Is_Late"] = vdf["PO Delivery Date"].dt.date.notna() & (vdf["PO Delivery Date"].dt.date < today) & (vdf["PendingQtyFill"] > 0)
        perf = vdf.groupby("PO Vendor", dropna=False).agg(Total_PO_Count=("Purchase Doc","nunique"), Fully_Delivered_PO_Count=("Is_Fully_Delivered","sum"), Late_PO_Count=("Is_Late","sum")).reset_index()
        perf["Pct_Fully_Delivered"] = (perf["Fully_Delivered_PO_Count"] / perf["Total_PO_Count"] * 100).round(1)
        perf["Pct_Late"] = (perf["Late_PO_Count"] / perf["Total_PO_Count"] * 100).round(1)
        # join spend
        if {"PO Vendor","Net Amount"}.issubset(fil.columns):
            spend = fil.groupby("PO Vendor", dropna=False)["Net Amount"].sum().rename("Spend").reset_index()
            perf = perf.merge(spend, on="PO Vendor", how="left").fillna({"Spend":0})
        top10 = perf.sort_values("Spend", ascending=False).head(10)
        st.dataframe(top10[["PO Vendor","Total_PO_Count","Fully_Delivered_PO_Count","Late_PO_Count","Pct_Fully_Delivered","Pct_Late"]], use_container_width=True)
        melt = top10.melt(id_vars=["PO Vendor"], value_vars=["Pct_Fully_Delivered","Pct_Late"], var_name="Metric", value_name="Percentage")
        st.plotly_chart(px.bar(melt, x="PO Vendor", y="Percentage", color="Metric", barmode="group", title="% Fully Delivered vs % Late (Top 10 by Spend)"), use_container_width=True)

# ---------- Smart Budget Mapper / Dept & Services ----------
with T[5]:
    st.subheader("Dept & Services (Smart Mapper)")
    def _norm_series(s):
        s = s.astype(str).str.upper().str.strip()
        s = s.str.replace("\xa0"," ",regex=False).str.replace("&","AND",regex=False).str.replace("R&D","RANDD",regex=False)
        s = s.str.replace(r"[/\\_\-]+",".",regex=True).str.replace(r"\s+","",regex=True).str.replace(r"\.{2,}",".",regex=True)
        s = s.str.replace(r"^\.+|\.+$","",regex=True).str.replace(r"[^A-Z0-9\.]+","",regex=True)
        return s
    def _norm_one(x):
        return _norm_series(pd.Series([x])).iloc[0] if pd.notna(x) else ""
    P3F={"CNST":"Program","PM":"Program","PRJ":"Design","MFG":"Manufacturing","INF":"Infra","HR":"HR & Admin","HRP":"HR & Admin","MKT":"Marketing","A&M":"Marketing","FIN":"Finance","LGL":"Legal & IP","LGLF":"Legal & IP","IP":"Legal & IP","PRT":"Legal & IP","SS":"SS & SCM","TLG":"SS & SCM","COG":"SS & SCM","SLS":"Sales","RENT":"Rental Offices","COUR":"Rental Offices","R&D":"R&D","RANDD":"R&D","PRDDEV":"R&D","DVP":"R&D","MT":"R&D","TV":"R&D","VIC":"R&D","PT":"R&D","SFL":"R&D","DSN":"Design","ACCS":"Design","CS":"Customer Success"}
    smart = fil.copy(); smart["Dept.Chart"],smart["Subcat.Chart"],smart["__src"]=pd.NA,pd.NA,"UNMAPPED"
    exp=None
    for name in ["Expanded_Budget_Code_Mapping.xlsx","Final_Budget_Mapping_Completed_Verified.xlsx"]:
        try:
            exp=pd.read_excel(name); break
        except Exception: pass
    EM,EMs,EP,EPs,P3,P3s = {},{},{},{},{},{}
    if exp is not None:
        m=exp.copy(); m.columns=m.columns.astype(str).str.strip()
        code_col=next((c for c in m.columns if c.lower().strip() in ["budget code","code","budget_code"]),None)
        dept_col=next((c for c in m.columns if "department" in c.lower()),None)
        subc_col=next((c for c in m.columns if ("subcat" in c.lower()) or ("sub category" in c.lower()) or ("subcategory" in c.lower())),None)
        p3_col=next((c for c in m.columns if "prefix_3" in c.lower()),None)
        ent_col=next((c for c in m.columns if c.lower().strip() in ["entity","domain","company","prefix_1","brand"]),None)
        if code_col and dept_col:
            m["__code"]=_norm_series(m[code_col]); tmp=m.dropna(subset=["__code"]).drop_duplicates("__code")
            EM= dict(zip(tmp["__code"], tmp[dept_col].astype(str).str.strip()))
            if subc_col: EMs=dict(zip(tmp["__code"], tmp[subc_col].astype(str).str.strip()))
        if p3_col: m["__p3"]=_norm_series(m[p3_col])
        if ent_col: m["__ent"]=_norm_series(m[ent_col])
        if p3_col and dept_col and ent_col and not m.empty:
            g=m.dropna(subset=["__p3","__ent"]).copy();
            if not g.empty:
                EP = g.groupby(["__ent","__p3"])[dept_col].agg(lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0]).to_dict()
                if subc_col: EPs = g.groupby(["__ent","__p3"])[subc_col].agg(lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0]).to_dict()
        if p3_col and dept_col and not m.empty:
            g2=m.dropna(subset=["__p3"]).copy();
            if not g2.empty:
                P3 = g2.groupby("__p3")[dept_col].agg(lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0]).to_dict()
                if subc_col: P3s = g2.groupby("__p3")[subc_col].agg(lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0]).to_dict()
    def pick_p3(code):
        segs=code.split('.')
        for j in range(len(segs)-1,-1,-1):
            if (segs[j] in P3) or (segs[j] in P3F): return segs[j]
        return None
    def map_one(code_raw, ent_raw):
        code=_norm_one(code_raw); ent=_norm_one(ent_raw)
        if not code: return (pd.NA,pd.NA,"UNMAPPED")
        if code in EM: return (EM.get(code), EMs.get(code,pd.NA), "EXACT")
        parts=code.split('.')
        if len(parts)>1:
            for i in range(1,len(parts)):
                suf='.'.join(parts[i:])
                if suf in EM: return (EM.get(suf), EMs.get(suf,pd.NA), "HIER")
        p3=pick_p3(code)
        if p3 and ent and (ent,p3) in EP: return (EP.get((ent,p3)), EPs.get((ent,p3),pd.NA), "ENTITY_PFX")
        if p3 and (p3 in P3): return (P3.get(p3), P3s.get(p3,pd.NA), "PFX3")
        if p3 and (p3 in P3F): return (P3F[p3], pd.NA, "KEYWORD")
        return (pd.NA,pd.NA,"UNMAPPED")
    bcol=[c for c in ["PO Budget Code","PR Budget Code"] if c in smart.columns]
    if bcol:
        base=smart[bcol[0]]; ent=smart.get("Entity", pd.Series([pd.NA]*len(smart)))
        mp=pd.DataFrame([map_one(c,e) for c,e in zip(base.tolist(),ent.tolist())], columns=["Dept.Chart","Subcat.Chart","__src"], index=smart.index)
        for c in ["Dept.Chart","Subcat.Chart","__src"]:
            smart[c]=smart[c].combine_first(mp[c]) if c in smart.columns else mp[c]
    # --- Subcategory token fallback (e.g., MKT.CONS → Consulting / Agency)
SUBCAT_TOKEN_MAP = {
    # Marketing
    "CONS": "Consulting / Agency",
    "PR":   "Public Relations",
    "CBR":  "Dealer Campaigns / Market Activation",
    "TVM":  "TV / Media Buying",
    "VP":   "Vendor Promotion / Partnerships",
    "AE":   "Agency & Events",
    "A&E":  "Agency & Events",
    "ANDE": "Agency & Events",
    # DT/IT examples
    "SPMW": "Software & Productivity Tools",
    "INT":  "Internet / Connectivity",
    "ACCER":"Accessories",
    "HDW":  "Hardware",
    # Ops/Program examples
    "CNST": "Construction / Fitout",
    "EHS":  "EHS / Safety",
}

def _last_token(_raw):
    if not isinstance(_raw, str):
        return ""
    tok = _raw.split(".")[-1].upper().replace(" ", "")
    tok = tok.replace("&", "AND")
    return tok

if "Dept.Chart" in smart.columns:
    has_dept = smart["Dept.Chart"].notna()
    no_subc  = ~smart.get("Subcat.Chart", pd.Series([pd.NA]*len(smart))).notna()
    need_sub = has_dept & no_subc
    code_series = (
        smart["PO Budget Code"] if "PO Budget Code" in smart.columns else (
            smart["PR Budget Code"] if "PR Budget Code" in smart.columns else pd.Series([""]*len(smart), index=smart.index)
        )
    )
    tokens = code_series.apply(_last_token)
    subc_guess = tokens.map(lambda t: SUBCAT_TOKEN_MAP.get(t, pd.NA))
    smart.loc[need_sub & subc_guess.notna(), "Subcat.Chart"] = subc_guess[need_sub & subc_guess.notna()]
    if "__src" in smart.columns:
        smart.loc[need_sub & subc_guess.notna(), "__src"] = smart.loc[need_sub & subc_guess.notna(), "__src"].fillna("TOKEN_SUBCAT").replace("UNMAPPED", "TOKEN_SUBCAT")

smart["Dept.Chart"].fillna("Unmapped / Missing", inplace=True)
    # Canonical labels + optional overrides
# --- Canonical Department Aliases ---
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
        st.plotly_chart(px.scatter(z, x="Po create Date", y="PO Unit Rate", color=np.where(abs(z["PctDev"])>=thr/100.0,"Outlier","Normal"), hover_data=[grp_by,"Purchase Doc","PO Vendor","MedianRate"]).update_layout(legend_title_text=""), use_container_width=True)
    else:
        st.info("Need 'PO Unit Rate' and a grouping column (Product Name / Item Code)")

# ---------- Forecast ----------
with T[7]:
    st.subheader("Forecast Next Month Spend (SMA)")
    dcol = "Po create Date" if "Po create Date" in fil.columns else ("PR Date Submitted" if "PR Date Submitted" in fil.columns else None)
    if dcol and "Net Amount" in fil.columns:
        t = fil.dropna(subset=[dcol]).copy(); t["Month"]=t[dcol].dt.to_period("M").dt.to_timestamp()
        m = t.groupby("Month")["Net Amount"].sum().sort_index(); m_cr=(m/1e7)
        k = st.slider("Window (months)", 3, 12, 6, 1)
        sma = m_cr.rolling(k).mean()
        mu = m_cr.tail(k).mean() if len(m_cr)>=k else m_cr.mean()
        sd = m_cr.tail(k).std(ddof=1) if len(m_cr)>=k else m_cr.std(ddof=1)
        n = min(k, max(1,len(m_cr)))
        se = sd/np.sqrt(n) if sd==sd else 0
        lo, hi = float(mu-1.96*se), float(mu+1.96*se)
        nxt = (m_cr.index.max() + pd.offsets.MonthBegin(1)) if len(m_cr)>0 else pd.Timestamp.today().to_period("M").to_timestamp()
        fdf = pd.DataFrame({"Month": list(m_cr.index)+[nxt], "SpendCr": list(m_cr.values)+[np.nan], "SMA": list(sma.values)+[mu]})
        fig = go.Figure()
        fig.add_bar(x=fdf["Month"], y=fdf["SpendCr"], name="Actual (Cr)")
        fig.add_scatter(x=fdf["Month"], y=fdf["SMA"], mode="lines+markers", name=f"SMA{k}")
        fig.add_scatter(x=[nxt,nxt], y=[lo,hi], mode="lines", name="95% CI")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Forecast for {nxt.strftime('%b-%Y')}: {mu:.2f} Cr (95% CI: {lo:.2f}–{hi:.2f})")
    else:
        st.info("Need date and Net Amount to forecast.")

# ---------- Vendor Scorecards ----------
with T[8]:
    st.subheader("Vendor Scorecard")
    if "PO Vendor" in fil.columns:
        vendor = st.selectbox("Pick Vendor", sorted(fil["PO Vendor"].dropna().astype(str).unique().tolist()))
        vd = fil[fil["PO Vendor"].astype(str)==str(vendor)].copy()
        spend = vd.get("Net Amount",pd.Series(0)).sum()/1e7
        upos = int(vd.get("Purchase Doc",pd.Series(dtype=object)).nunique())
        today = pd.Timestamp.today().normalize()
        if {"PO Delivery Date","Pending QTY"}.issubset(vd.columns):
            late = ((pd.to_datetime(vd["PO Delivery Date"],errors="coerce").dt.date < today.date()) & (vd["Pending QTY"].fillna(0)>0)).sum()
        else:
            late = np.nan
        if "Pending QTY" in vd.columns and "PO Unit Rate" in vd.columns:
            vd["Pending Value"]=vd["Pending QTY"].fillna(0).astype(float)*vd["PO Unit Rate"].fillna(0).astype(float)
            pend_val = vd["Pending Value"].sum()/1e7
        else:
            pend_val = np.nan
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Spend (Cr)", f"{spend:.2f}"); k2.metric("Unique POs", upos); k3.metric("Late PO count", None if pd.isna(late) else int(late)); k4.metric("Pending Value (Cr)", None if pd.isna(pend_val) else f"{pend_val:.2f}")
        if {"Product Name","PO Unit Rate"}.issubset(vd.columns):
            med = vd.groupby("Product Name")["PO Unit Rate"].median().rename("MedianRate"); v2 = vd.join(med, on="Product Name"); v2["Var%"]=((v2["PO Unit Rate"]-v2["MedianRate"])/v2["MedianRate"].replace(0,np.nan))*100
            st.plotly_chart(px.box(v2, x="Product Name", y="PO Unit Rate", points="outliers", title="Price variance by item"), use_container_width=True)
        if dcol:= ("Po create Date" if "Po create Date" in vd.columns else ("PR Date Submitted" if "PR Date Submitted" in vd.columns else None)):
            vsp = vd.dropna(subset=[dcol]).groupby(pd.to_datetime(vd[dcol]).dt.to_period("M"))["Net Amount"].sum().to_timestamp()/1e7
            st.plotly_chart(px.line(vsp, labels={"value":"Spend (Cr)","index":"Month"}, title="Monthly Spend"), use_container_width=True)

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
