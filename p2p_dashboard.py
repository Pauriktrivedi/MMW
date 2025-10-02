# Create an updated Streamlit app focusing on Department ↔ Services (Subcategory) analytics
from textwrap import dedent

code = dedent('''
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date

# ====================================
#  Procure-to-Pay Dashboard (Streamlit)
#  — Dept & Services focused build (v2)
# ====================================

st.set_page_config(page_title="P2P Dashboard — Dept & Services", layout="wide", initial_sidebar_state="expanded")

# ------------------------------------
#  1) Load & Combine Source Data
# ------------------------------------
@st.cache_data(show_spinner=False)
def load_and_combine_data():
    mepl_df = pd.read_excel("MEPL1.xlsx", skiprows=1)
    mlpl_df = pd.read_excel("MLPL1.xlsx", skiprows=1)
    mmw_df  = pd.read_excel("mmw1.xlsx",  skiprows=1)
    mmpl_df = pd.read_excel("mmpl1.xlsx", skiprows=1)

    mepl_df["Entity"] = "MEPL"
    mlpl_df["Entity"] = "MLPL"
    mmw_df["Entity"]  = "MMW"
    mmpl_df["Entity"] = "MMPL"

    combined = pd.concat([mepl_df, mlpl_df, mmw_df, mmpl_df], ignore_index=True)
    combined.columns = (
        combined.columns
        .str.strip()
        .str.replace("\\xa0", " ", regex=False)
        .str.replace(" +", " ", regex=True)
    )
    combined.rename(columns=lambda c: c.strip(), inplace=True)
    return combined

df = load_and_combine_data()

# ------------------------------------
#  2) Dates
# ------------------------------------
for date_col in ["PR Date Submitted", "Po create Date"]:
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

# ------------------------------------
#  3) Sidebar Filters
# ------------------------------------
st.sidebar.header("🔍 Filters")
fy_options = {
    "All Years": (pd.to_datetime("2023-04-01"), pd.to_datetime("2026-03-31")),
    "2023": (pd.to_datetime("2023-04-01"), pd.to_datetime("2024-03-31")),
    "2024": (pd.to_datetime("2024-04-01"), pd.to_datetime("2025-03-31")),
    "2025": (pd.to_datetime("2025-04-01"), pd.to_datetime("2026-03-31")),
    "2026": (pd.to_datetime("2026-04-01"), pd.to_datetime("2027-03-31")),
}
selected_fy = st.sidebar.selectbox("Financial Year", list(fy_options.keys()), index=0)
pr_start, pr_end = fy_options[selected_fy]

filtered_df = df.copy()
if "PR Date Submitted" in filtered_df.columns:
    filtered_df = filtered_df[(filtered_df["PR Date Submitted"] >= pr_start) & (filtered_df["PR Date Submitted"] <= pr_end)]

# ------------------------------------
#  4) Smart Budget Mapper (Unified)
# ------------------------------------
def _norm_code_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.upper().str.strip()
    s = s.str.replace("\\xa0", " ", regex=False)
    s = s.str.replace("&", "AND", regex=False)
    s = s.str.replace("R&D", "RANDD", regex=False)  # freeze token
    s = s.str.replace(r"[/\\\\_\\-]+", ".", regex=True)  # unify separators
    s = s.str.replace(r"\\s+", "", regex=True)          # remove spaces
    s = s.str.replace(r"\\.{2,}", ".", regex=True)      # collapse dots
    s = s.str.replace(r"^\\.+|\\.+$", "", regex=True)   # trim edge dots
    s = s.str.replace(r"[^A-Z0-9\\.]+", "", regex=True) # keep A-Z0-9 and dots
    return s

def _norm_one(x):
    if pd.isna(x):
        return ""
    return _norm_code_series(pd.Series([x])).iloc[0]

# Fallback P3 → Department map (used only if files don't cover it)
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

smart_df = filtered_df.copy()
smart_df["Dept.Chart"], smart_df["Subcat.Chart"], smart_df["__Dept.MapSrc"] = pd.NA, pd.NA, "UNMAPPED"

# Load expanded mapping
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
    exp = expanded.copy()
    exp.columns = exp.columns.astype(str).str.strip()
    code_col = next((c for c in exp.columns if c.lower().strip() in ["budget code","code","budget_code"]), None)
    dept_col = next((c for c in exp.columns if "department" in c.lower()), None)
    subc_col = next((c for c in exp.columns if ("subcat" in c.lower()) or ("sub category" in c.lower()) or ("subcategory" in c.lower())), None)
    p3_col   = next((c for c in exp.columns if "prefix_3" in c.lower()), None)
    ent_col  = next((c for c in exp.columns if c.lower().strip() in ["entity","domain","company","prefix_1","brand"]), None)

    if code_col and dept_col:
        exp["__code_norm"] = _norm_code_series(exp[code_col])
        tmp = exp.dropna(subset=["__code_norm"]).drop_duplicates(subset=["__code_norm"], keep="first")
        exact_map = dict(zip(tmp["__code_norm"], tmp[dept_col].astype(str).str.strip()))
        if subc_col:
            exact_map_sub = dict(zip(tmp["__code_norm"], tmp[subc_col].astype(str).str.strip()))
    if p3_col:
        exp["__p3_norm"] = _norm_code_series(exp[p3_col])
    if ent_col:
        exp["__ent_norm"] = _norm_code_series(exp[ent_col])

    if p3_col and dept_col and ent_col and not exp.empty:
        grp = exp.dropna(subset=["__p3_norm","__ent_norm"]).copy()
        if not grp.empty:
            mode_dept = grp.groupby(["__ent_norm","__p3_norm"])[dept_col].agg(lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0])
            entity_pfx_map = mode_dept.to_dict()
            if subc_col:
                mode_sub = grp.groupby(["__ent_norm","__p3_norm"])[subc_col].agg(lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0])
                entity_pfx_map_sub = mode_sub.to_dict()

    if p3_col and dept_col and not exp.empty:
        grp2 = exp.dropna(subset=["__p3_norm"]).copy()
        if not grp2.empty:
            mode_dept2 = grp2.groupby("__p3_norm")[dept_col].agg(lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0])
            pfx_map = mode_dept2.to_dict()
            if subc_col:
                mode_sub2 = grp2.groupby("__p3_norm")[subc_col].agg(lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0])
                pfx_map_sub = mode_sub2.to_dict()

def _map_one(code_raw, ent_raw):
    code = _norm_one(code_raw)
    ent  = _norm_one(ent_raw)
    if not code:
        return (pd.NA, pd.NA, "UNMAPPED")

    # 1) EXACT
    if code in exact_map:
        return (exact_map.get(code), exact_map_sub.get(code, pd.NA), "EXACT")

    # 2) HIER suffix: drop left segments progressively
    parts = code.split('.')
    if len(parts) > 1:
        for i in range(1, len(parts)):
            suf = '.'.join(parts[i:])
            if suf in exact_map:
                return (exact_map.get(suf), exact_map_sub.get(suf, pd.NA), "HIER")

    # 3) ENTITY+PFX3 or PFX3 only
    def _pick_p3(c):
        segs = c.split('.')
        # prefer right-most token that is known in either map or fallback
        for j in range(len(segs)-1, -1, -1):
            seg = segs[j]
            if (seg in pfx_map) or (seg in _P3_FALLBACK):
                return seg
        return None
    p3 = _pick_p3(code)

    if p3 and ent and (ent, p3) in entity_pfx_map:
        return (entity_pfx_map.get((ent,p3)), entity_pfx_map_sub.get((ent,p3), pd.NA), "ENTITY_PFX")

    if p3 and (p3 in pfx_map):
        return (pfx_map.get(p3), pfx_map_sub.get(p3, pd.NA), "PFX3")

    # 4) P3 keyword fallback
    if p3 and (p3 in _P3_FALLBACK):
        return (_P3_FALLBACK[p3], pd.NA, "KEYWORD")

    return (pd.NA, pd.NA, "UNMAPPED")

code_cols_avail = [c for c in ["PO Budget Code","PR Budget Code"] if c in smart_df.columns]
if code_cols_avail:
    base = smart_df[code_cols_avail[0]]
    ent  = smart_df.get("Entity", pd.Series([pd.NA]*len(smart_df)))
    mapped = pd.DataFrame([_map_one(c,e) for c,e in zip(base.tolist(), ent.tolist())],
                          columns=["__dept","__subc","__src"], index=smart_df.index)
    smart_df["Dept.Chart"] = smart_df["Dept.Chart"].combine_first(mapped["__dept"]) if "Dept.Chart" in smart_df.columns else mapped["__dept"]
    smart_df["Subcat.Chart"] = smart_df["Subcat.Chart"].combine_first(mapped["__subc"]) if "Subcat.Chart" in smart_df.columns else mapped["__subc"]
    need_src = smart_df["__Dept.MapSrc"].isin(["UNMAPPED", pd.NA, None])
    smart_df.loc[need_src, "__Dept.MapSrc"] = mapped.loc[need_src, "__src"].fillna("UNMAPPED")

# Fallback to in-file dept if absolutely needed
pre_fallback_na = smart_df["Dept.Chart"].isna()
if pre_fallback_na.all():
    for cand in ["PO Department","PO Dept","PR Department","PR Dept","Dept.Final","Department"]:
        if cand in smart_df.columns:
            smart_df["Dept.Chart"] = smart_df["Dept.Chart"].combine_first(smart_df[cand])
smart_df.loc[smart_df["__Dept.MapSrc"].eq("UNMAPPED") & smart_df["Dept.Chart"].notna(), "__Dept.MapSrc"] = "FALLBACK"
smart_df["Dept.Chart"].fillna("Unmapped / Missing", inplace=True)

with st.expander("🧪 Smart Mapper QA", expanded=False):
    st.write({"counts": smart_df["__Dept.MapSrc"].value_counts(dropna=False).to_dict()})
    if code_cols_avail:
        probe = smart_df[smart_df["__Dept.MapSrc"] == "UNMAPPED"].copy()
        if not probe.empty:
            probe["Code.Raw"] = probe[code_cols_avail[0]]
            probe["Code.Norm"] = _norm_code_series(probe["Code.Raw"])
            def _p3_guess(c):
                parts = str(c).split(".")
                return parts[-1] if parts else ""
            probe["P3.Guess"] = probe["Code.Norm"].apply(_p3_guess)
            probe["Entity.Norm"] = smart_df.get("Entity", pd.Series([pd.NA]*len(smart_df))).loc[probe.index].apply(_norm_one)
            probe["P3.Fallback Dept"] = probe["P3.Guess"].apply(lambda t: _P3_FALLBACK.get(t, ""))
            show = probe[["Code.Raw","Code.Norm","Entity.Norm","P3.Guess","P3.Fallback Dept"]].drop_duplicates().head(200)
            st.dataframe(show, use_container_width=True)

# ------------------------------------
#  5) KPIs
# ------------------------------------
st.title("📊 P2P — Department & Services (Subcategory)")
k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Total PRs", int(smart_df.get("PR Number", pd.Series(dtype=object)).nunique()))
k2.metric("Total POs", int(smart_df.get("Purchase Doc", pd.Series(dtype=object)).nunique()))
k3.metric("Line Items", len(smart_df))
k4.metric("Entities", int(smart_df.get("Entity", pd.Series(dtype=object)).nunique()))
k5.metric("Spend (Cr ₹)", f"{smart_df.get('Net Amount', pd.Series(0)).sum()/1e7:,.2f}")

# ------------------------------------
#  6) Department-wise Spend + Drilldown
# ------------------------------------
st.subheader("🏢 Department-wise Spend (Smart)")
if "Net Amount" in smart_df.columns:
    dept_spend = (
        smart_df.groupby("Dept.Chart", dropna=False)["Net Amount"].sum()
        .sort_values(ascending=False).reset_index()
    )
    dept_spend["Spend (Cr ₹)"] = dept_spend["Net Amount"]/1e7
    st.plotly_chart(px.bar(dept_spend.head(30), x="Dept.Chart", y="Spend (Cr ₹)", title="Top Departments by Spend")
                    .update_layout(xaxis_tickangle=-45), use_container_width=True)

    dd1, dd2 = st.columns([2,1])
    with dd1:
        dept_pick = st.selectbox("Drill down: Department", dept_spend["Dept.Chart"].tolist(), key="dept_pick_main")
    with dd2:
        topn = st.number_input("Top N Vendors/Services", min_value=5, max_value=100, value=20, step=5, key="dept_topn")

    detail = smart_df[smart_df["Dept.Chart"].astype(str) == str(dept_pick)].copy()

    c1,c2,c3 = st.columns(3)
    c1.metric("Lines", len(detail))
    c2.metric("PRs", int(detail.get("PR Number", pd.Series(dtype=object)).nunique()))
    c3.metric("Spend (Cr ₹)", f"{detail.get('Net Amount', pd.Series(0)).sum()/1e7:,.2f}")

    # Service (Subcategory) split inside picked dept
    if "Subcat.Chart" in detail.columns:
        sub_spend = (detail.groupby("Subcat.Chart", dropna=False)["Net Amount"].sum()
                          .sort_values(ascending=False).reset_index())
        sub_spend["Spend (Cr ₹)"] = sub_spend["Net Amount"]/1e7
        cA, cB = st.columns(2)
        cA.plotly_chart(px.bar(sub_spend.head(int(topn)), x="Subcat.Chart", y="Spend (Cr ₹)",
                               title=f"{dept_pick} — Top Services (Subcategories)")
                        .update_layout(xaxis_tickangle=-45), use_container_width=True)
        cB.plotly_chart(px.pie(sub_spend.head(12), names="Subcat.Chart", values="Net Amount",
                               title=f"{dept_pick} — Service Share (Top 12)"), use_container_width=True)

        # optional second-level drilldown: choose a service to see line items
        svc = st.selectbox("🔎 Drill further: Pick a Service (Subcategory) to see lines",
                           sub_spend["Subcat.Chart"].astype(str).tolist(), key="svc_pick")
        sub_detail = detail[detail["Subcat.Chart"].astype(str) == str(svc)].copy()

        wanted = ["PO Budget Code","Subcat.Chart","Dept.Chart","Purchase Doc","PR Number","Procurement Category","Product Name","Item Description","PO Vendor","Net Amount"]
        cols_present = [c for c in wanted if c in sub_detail.columns]
        st.dataframe(sub_detail[cols_present], use_container_width=True)
        st.download_button("⬇️ Download Lines (CSV)",
                           sub_detail[cols_present].to_csv(index=False),
                           file_name=f"lines_{dept_pick}_{svc}.csv",
                           mime="text/csv",
                           key=f"dl_lines_{hash((dept_pick,svc))}")

# ------------------------------------
#  7) Department × Service Matrix (Pivot) + Heatmap
# ------------------------------------
st.subheader("🧩 Department × Service (₹ Cr) — Pivot & Heatmap")
if {"Dept.Chart","Subcat.Chart","Net Amount"}.issubset(smart_df.columns):
    mat = (smart_df
           .pivot_table(index="Dept.Chart", columns="Subcat.Chart", values="Net Amount", aggfunc="sum", fill_value=0.0)
           .sort_index())
    mat_cr = mat / 1e7
    st.dataframe(mat_cr.round(2), use_container_width=True)
    # Heatmap-like using treemap or imshow-like bubbles
    heat_df = mat_cr.stack().reset_index()
    heat_df.columns = ["Department","Service","SpendCr"]
    heat_df = heat_df[heat_df["SpendCr"] > 0]
    st.plotly_chart(
        px.treemap(heat_df, path=["Department","Service"], values="SpendCr", title="Department → Service Treemap (Cr ₹)"),
        use_container_width=True
    )
    st.download_button("⬇️ Download Dept×Service (CSV)",
                       heat_df.to_csv(index=False),
                       file_name="dept_service_matrix.csv",
                       mime="text/csv",
                       key="dl_dept_service_matrix")

# ------------------------------------
#  8) Top Services overall + Vendor view
# ------------------------------------
st.subheader("🏷️ Top Services (Subcategories) — Overall")
if {"Subcat.Chart","Net Amount"}.issubset(smart_df.columns):
    svc_all = (smart_df.groupby("Subcat.Chart", dropna=False)["Net Amount"].sum()
                      .sort_values(ascending=False).reset_index())
    svc_all["Spend (Cr ₹)"] = svc_all["Net Amount"]/1e7
    st.plotly_chart(px.bar(svc_all.head(25), x="Subcat.Chart", y="Spend (Cr ₹)", title="Top Services by Spend (Overall)")
                    .update_layout(xaxis_tickangle=-45), use_container_width=True)

    st.markdown("#### 🔧 Vendor split for a picked Service")
    svc_pick2 = st.selectbox("Pick Service", svc_all["Subcat.Chart"].astype(str).tolist(), key="svc_pick2")
    svc_det = smart_df[smart_df["Subcat.Chart"].astype(str) == str(svc_pick2)].copy()
    if {"PO Vendor","Net Amount"}.issubset(svc_det.columns):
        vend_sp = (svc_det.groupby("PO Vendor", dropna=False)["Net Amount"].sum()
                         .sort_values(ascending=False).reset_index())
        vend_sp["Spend (Cr ₹)"] = vend_sp["Net Amount"]/1e7
        st.plotly_chart(px.bar(vend_sp.head(20), x="PO Vendor", y="Spend (Cr ₹)",
                               title=f"Vendors for Service: {svc_pick2}")
                        .update_layout(xaxis_tickangle=-45), use_container_width=True)

# ------------------------------------
#  End
# ------------------------------------
''')

path = "/mnt/data/p2p_dashboard_dept_service.py"
with open(path, "w", encoding="utf-8") as f:
    f.write(code)

print("Wrote file:", path)
