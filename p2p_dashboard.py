def _norm_code_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip().str.upper()
    s = s.str.replace("\xa0", " ", regex=False)
    s = s.str.replace("&", "AND", regex=False)
    s = s.str.replace(r"\s+", " ", regex=True)
    s = s.str.replace(r"\.+$", "", regex=True)  # remove trailing dots
    s = s.str.replace(r"\.{2,}", ".", regex=True)  # collapse multiple dots
    s = s.str.replace(" ", "")  # remove stray internal spaces
    return s


# ------------------------------------
# 33) Department-wise Spend — robust (mapped if possible) + Mapping QA
# ------------------------------------
st.subheader("🏢 Department-wise Spend")

# Always create a working column inside filtered_df
filtered_df = filtered_df.copy()
filtered_df["Dept.Chart"], filtered_df["Subcat.Chart"] = pd.NA, pd.NA
filtered_df["__Dept.MapSrc"] = "UNMAPPED"  # EXACT | SUF2 | FALLBACK | UNMAPPED

# Try mapping via Budget Code → Department/Subcategory from the mapping file
try:
    bm = pd.read_excel("Final_Budget_Mapping_Completed_Verified.xlsx")
    bm.columns = bm.columns.astype(str).str.strip()
    if ("Budget Code" in bm.columns):
        bm_u = bm.dropna(subset=["Budget Code"]).drop_duplicates(subset=["Budget Code"], keep="first").copy()

        # Normalize mapping codes
        bm_u["__code_norm"] = _norm_code_series(bm_u["Budget Code"]) if "Budget Code" in bm_u.columns else pd.Series(pd.NA, index=bm_u.index)
        dept_col = next((c for c in bm_u.columns if "dept" in c.lower()), None)
        subc_col = next((c for c in bm_u.columns if "subcat" in c.lower() or "sub category" in c.lower() or "subcategory" in c.lower()), None)
        bm_u["__dept"] = bm_u[dept_col].astype(str).str.strip() if dept_col else ""
        bm_u["__subc"] = bm_u[subc_col].astype(str).str.strip() if subc_col else ""

        dept_map = dict(zip(bm_u["__code_norm"], bm_u["__dept"])) if dept_col else {}
        subc_map = dict(zip(bm_u["__code_norm"], bm_u["__subc"])) if subc_col else {}

        # Helper: hierarchy-based match (ME.R&D.M2.BTRSWP → R&D.M2.BTRSWP → M2.BTRSWP → BTRSWP)
        def _hierarchy_lookup(code_norm: str):
            if pd.isna(code_norm) or str(code_norm) == "":
                return (pd.NA, pd.NA, "UNMAPPED")
            if code_norm in dept_map:
                return (dept_map.get(code_norm), subc_map.get(code_norm, pd.NA), "EXACT")
            parts = str(code_norm).split(".")
            for i in range(1, len(parts)):
                sub = ".".join(parts[i:])
                if sub in dept_map:
                    return (dept_map.get(sub), subc_map.get(sub, pd.NA), "HIER")
            return (pd.NA, pd.NA, "UNMAPPED")

        # Normalize codes from filtered_df
        if "PO Budget Code" in filtered_df.columns:
            codes = _norm_code_series(filtered_df["PO Budget Code"])  # type: ignore
        elif "PR Budget Code" in filtered_df.columns:
            codes = _norm_code_series(filtered_df["PR Budget Code"])  # type: ignore
        else:
            codes = pd.Series([pd.NA] * len(filtered_df), index=filtered_df.index)

        # Apply hierarchical lookup vectorized via apply→Series
        look = codes.apply(_hierarchy_lookup).apply(pd.Series)
        look.columns = ["__dept_out", "__subc_out", "__src_out"]

        # Assign back
        filtered_df["Dept.Chart"] = filtered_df["Dept.Chart"].combine_first(look["__dept_out"]) if "Dept.Chart" in filtered_df.columns else look["__dept_out"]
        filtered_df["Subcat.Chart"] = filtered_df["Subcat.Chart"].combine_first(look["__subc_out"]) if "Subcat.Chart" in filtered_df.columns else look["__subc_out"]
        # Only set source for those we mapped here; keep any prior tags if already set
        need_src = filtered_df["__Dept.MapSrc"].isin(["UNMAPPED", pd.NA, None])
        filtered_df.loc[need_src, "__Dept.MapSrc"] = look.loc[need_src, "__src_out"].fillna("UNMAPPED")
except Exception as e:
    st.warning(f"Mapping failed: {e}")

# Fallback chain from in-file columns if still NA
pre_fallback_na = filtered_df["Dept.Chart"].isna()
if pre_fallback_na.all():
    for cand in ["PO Department", "PO Dept", "PR Department", "PR Dept", "Dept.Final", "Department"]:
        if cand in filtered_df.columns:
            filtered_df["Dept.Chart"] = filtered_df["Dept.Chart"].combine_first(filtered_df[cand])
# mark fallback fills
fallback_filled = pre_fallback_na & filtered_df["Dept.Chart"].notna() & (filtered_df["__Dept.MapSrc"] == "UNMAPPED")
filtered_df.loc[fallback_filled, "__Dept.MapSrc"] = "FALLBACK"

# --- Prefix_3 heuristic mapping (entity-agnostic) ---
try:
    bm_p = pd.read_excel("Final_Budget_Mapping_Completed_Verified.xlsx")
    bm_p.columns = bm_p.columns.astype(str).str.strip()
    p3_col = next((c for c in bm_p.columns if c.strip().lower() == 'prefix_3' or 'prefix_3' in c.strip().lower()), None)
    dept_full_col = next((c for c in bm_p.columns if 'department' in c.lower()), None)
    subc_col2 = next((c for c in bm_p.columns if ('subcat' in c.lower()) or ('sub category' in c.lower()) or ('subcategory' in c.lower())), None)
    if p3_col and dept_full_col:
        bm_p["__p3_norm"] = _norm_code_series(bm_p[p3_col])
        bm_p["__dept_full"] = bm_p[dept_full_col].astype(str).str.strip()
        if subc_col2:
            bm_p["__subc2"] = bm_p[subc_col2].astype(str).str.strip()
        mode_dept = bm_p.groupby("__p3_norm")["__dept_full"].agg(lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0])
        p3_to_dept = mode_dept.to_dict()
        if subc_col2:
            mode_subc = bm_p.groupby("__p3_norm")["__subc2"].agg(lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0])
            p3_to_subc = mode_subc.to_dict()
        else:
            p3_to_subc = {}
        p3_keys = set(p3_to_dept.keys())
        def _match_prefix3(code_norm: str):
            if pd.isna(code_norm) or str(code_norm)=="":
                return (pd.NA, pd.NA)
            parts = str(code_norm).split('.')
            for i in range(len(parts)-1, -1, -1):
                seg = parts[i]
                if seg in p3_keys:
                    return (p3_to_dept.get(seg, pd.NA), p3_to_subc.get(seg, pd.NA))
            return (pd.NA, pd.NA)
        code_cols_avail = [c for c in ["PO Budget Code", "PR Budget Code"] if c in filtered_df.columns]
        if code_cols_avail:
            base_code = _norm_code_series(filtered_df[code_cols_avail[0]])
            needs_p3 = filtered_df["Dept.Chart"].isna() | (filtered_df["Dept.Chart"]=="") | (filtered_df["Dept.Chart"]=="Unmapped / Missing")
            p3_lookup = base_code[needs_p3].apply(_match_prefix3).apply(pd.Series)
            if not p3_lookup.empty:
                p3_lookup.columns = ["__p3_dept","__p3_subc"]
                idxs = p3_lookup.index
                found = p3_lookup["__p3_dept"].notna() & (p3_lookup["__p3_dept"].astype(str) != "")
                filtered_df.loc[idxs[found], "Dept.Chart"] = p3_lookup.loc[idxs[found], "__p3_dept"]
                filtered_df.loc[idxs[found] & (filtered_df["__Dept.MapSrc"].isin(["UNMAPPED", "FALLBACK"])) , "__Dept.MapSrc"] = "PFX3"
                has_subc = subc_col2 and p3_lookup["__p3_subc"].notna() & (p3_lookup["__p3_subc"].astype(str) != "")
                if subc_col2:
                    filtered_df.loc[idxs[has_subc], "Subcat.Chart"] = p3_lookup.loc[idxs[has_subc], "__p3_subc"]
except Exception:
    pass

# Final safety: if still all NA, label as "Unmapped / Missing"
filtered_df["Dept.Chart"].fillna("Unmapped / Missing", inplace=True)

# --- Mapping QA ---
with st.expander("🧪 Mapping QA (exact vs suffix vs fallback)", expanded=False):
    src_counts = filtered_df["__Dept.MapSrc"].value_counts(dropna=False).to_dict()
    st.write({"counts": src_counts})

    # Unmapped list with sample budget codes (normalized) and frequency
    code_cols_avail = [c for c in ["PO Budget Code", "PR Budget Code"] if c in filtered_df.columns]
    if code_cols_avail:
        base_code = filtered_df[code_cols_avail[0]].copy()
        unm = filtered_df[filtered_df["__Dept.MapSrc"] == "UNMAPPED"].copy()
        unm["Budget Code (normalized)"] = _norm_code_series(unm[code_cols_avail[0]])
        summary_unm = unm["Budget Code (normalized)"].value_counts().reset_index()
        summary_unm.columns = ["Budget Code (normalized)", "Lines"]
        st.dataframe(summary_unm.head(100), use_container_width=True)
        st.download_button(
            "⬇️ Download Unmapped Budget Codes",
            summary_unm.to_csv(index=False),
            "unmapped_budget_codes.csv",
            "text/csv",
            key="dl_unmapped_codes",
        )
    else:
        st.caption("No Budget Code column present to summarize unmapped.")

# Build chart
if "Net Amount" in filtered_df.columns:
    dept_spend = (
        filtered_df.groupby("Dept.Chart", dropna=False)["Net Amount"].sum().reset_index()
        .sort_values("Net Amount", ascending=False)
    )
    dept_spend["Spend (Cr ₹)"] = dept_spend["Net Amount"] / 1e7
    st.plotly_chart(
        px.bar(dept_spend.head(30), x="Dept.Chart", y="Spend (Cr ₹)", title="Department-wise Spend (Top 30)")
        .update_layout(xaxis_tickangle=-45),
        use_container_width=True,
    )

    # --- Drilldown selector ---
    dd_col1, dd_col2 = st.columns([2,1])
    with dd_col1:
        dept_pick = st.selectbox("Drill down: choose a department", dept_spend["Dept.Chart"].tolist())
    with dd_col2:
        topn = st.number_input("Show top N vendors/items", min_value=5, max_value=100, value=20, step=5)

    detail = filtered_df[filtered_df["Dept.Chart"].astype(str) == str(dept_pick)].copy()
    st.markdown(f"### 🔎 Details for **{dept_pick}**")

    # KPIs
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Lines", len(detail))
    k2.metric("PRs", int(detail["PR Number"].nunique()) if "PR Number" in detail.columns else 0)
    k3.metric("POs", int(detail["Purchase Doc"].nunique()) if "Purchase Doc" in detail.columns else 0)
    k4.metric("Spend (Cr ₹)", f"{(detail.get('Net Amount', pd.Series(0)).sum()/1e7):,.2f}")

    # Top vendors & items
    c1,c2 = st.columns(2)
    if {"PO Vendor","Net Amount"}.issubset(detail.columns):
        top_v = (
            detail.groupby("PO Vendor", dropna=False)["Net Amount"].sum().sort_values(ascending=False).head(int(topn)).reset_index()
        )
        top_v["Spend (Cr ₹)"] = top_v["Net Amount"]/1e7
        c1.plotly_chart(px.bar(top_v, x="PO Vendor", y="Spend (Cr ₹)", title="Top Vendors (Cr ₹)").update_layout(xaxis_tickangle=-45), use_container_width=True)
    if {"Product Name","Net Amount"}.issubset(detail.columns):
        top_i = (
            detail.groupby("Product Name", dropna=False)["Net Amount"].sum().sort_values(ascending=False).head(int(topn)).reset_index()
        )
        top_i["Spend (Cr ₹)"] = top_i["Net Amount"]/1e7
        c2.plotly_chart(px.bar(top_i, x="Product Name", y="Spend (Cr ₹)", title="Top Items (Cr ₹)").update_layout(xaxis_tickangle=-45), use_container_width=True)

    # Optional time trend for the picked department
    dcol = "Po create Date" if "Po create Date" in detail.columns else ("PR Date Submitted" if "PR Date Submitted" in detail.columns else None)
    if dcol and "Net Amount" in detail.columns:
        detail[dcol] = pd.to_datetime(detail[dcol], errors="coerce")
        m = detail.dropna(subset=[dcol]).groupby(detail[dcol].dt.to_period('M'))['Net Amount'].sum().to_timestamp()
        st.plotly_chart(px.line(m/1e7, labels={'value':'Spend (Cr ₹)','index':'Month'}, title=f"{dept_pick} — Monthly Spend"), use_container_width=True)

    # Line-level detail with mapped Subcategory/Department if available
    st.markdown("#### 📄 Line-level detail")
    dept_col = None
    for cand in ["Dept.Chart", "Dept.Final", "PO Department", "PO Dept", "PR Department", "PR Dept", "Department"]:
        if cand in detail.columns:
            dept_col = cand
            break
    subcat_col = None
    for cand in ["Subcat.Chart", "Subcat.Final", "Subcategory", "Sub Category", "Sub-Category"]:
        if cand in detail.columns:
            subcat_col = cand
            break

    desired_order = [
        "PO Budget Code",
        subcat_col if subcat_col else "",
        dept_col if dept_col else "",
        "Purchase Doc",
        "PR Number",
        "Procurement Category",
        "Product Name",
        "Item Description",
    ]
    show_cols = [c for c in desired_order if c and c in detail.columns]
    disp = detail[show_cols].rename(columns={
        subcat_col: "Subcategory" if subcat_col else "Subcategory",
        dept_col: "Department" if dept_col else "Department",
    }) if show_cols else detail

    st.dataframe(disp, use_container_width=True)
    st.download_button(
        "⬇️ Download Department Lines (CSV)",
        disp.to_csv(index=False),
        file_name=f"dept_drilldown_{str(dept_pick).replace(' ','_')}.csv",
        mime="text/csv",
        key=f"dl_dept_lines_{str(dept_pick)}",
    )

    with st.expander("View table / download"):
        st.dataframe(dept_spend, use_container_width=True)
        st.download_button(
            "⬇️ Download Department Spend (CSV)",
            dept_spend.to_csv(index=False),
            "department_spend.csv",
            "text/csv",
            key="dl_dept_spend_csv",
        )
else:
    st.info("Net Amount column not present, cannot compute department spend.")

# ------------------------------------
# 34) End of Dashboard
# ------------------------------------


# ------------------------------------
# 33b) Department-wise Spend — Smart Budget Mapper (Unified) [NEW]
# ------------------------------------
# This new block supersedes the older Dept mapping logic above.
# It uses an expanded mapping file and unified matching tiers.

st.subheader("🏢 Department-wise Spend — Smart Mapper [NEW]")

_df = filtered_df.copy()
_df["Dept.Chart"], _df["Subcat.Chart"], _df["__Dept.MapSrc"] = pd.NA, pd.NA, "UNMAPPED"

# ---------- Load mapping ----------
expanded = None
try:
    expanded = pd.read_excel("Expanded_Budget_Code_Mapping.xlsx")
except Exception:
    try:
        expanded = pd.read_excel("Final_Budget_Mapping_Completed_Verified.xlsx")
    except Exception:
        expanded = None

# ---------- Normalization helpers ----------
def _norm_one_val(x):
    if pd.isna(x):
        return ""
    s = str(x).strip().upper()
    s = s.replace(" ", " ").replace(" ", "")
    s = s.replace("&", "AND")
    while ".." in s:
        s = s.replace("..", ".")
    if s.endswith('.'):
        s = s[:-1]
    return s

def _norm_series(s):
    return s.apply(_norm_one_val)

# ---------- Build maps from expanded file ----------
exact_map = {}
exact_map_sub = {}
entity_pfx_map = {}
entity_pfx_map_sub = {}
pfx_map = {}
pfx_map_sub = {}

if expanded is not None:
    exp = expanded.copy()
    exp.columns = exp.columns.astype(str).str.strip()

    code_col = next((c for c in exp.columns if c.lower().strip() in ["budget code","code","budget_code"]), None)
    dept_col = next((c for c in exp.columns if "department" in c.lower()), None)
    subc_col = next((c for c in exp.columns if ("subcat" in c.lower()) or ("sub category" in c.lower()) or ("subcategory" in c.lower())), None)
    p3_col   = next((c for c in exp.columns if "prefix_3" in c.lower()), None)
    ent_col  = next((c for c in exp.columns if c.lower().strip() in ["entity","domain","company","prefix_1","brand"]), None)

    if code_col and dept_col:
        exp["__code_norm"] = _norm_series(exp[code_col])
        tmp = exp.dropna(subset=["__code_norm"]).drop_duplicates(subset=["__code_norm"], keep="first")
        exact_map = dict(zip(tmp["__code_norm"], tmp[dept_col].astype(str).str.strip()))
        if subc_col:
            exact_map_sub = dict(zip(tmp["__code_norm"], tmp[subc_col].astype(str).str.strip()))

    if p3_col:
        exp["__p3_norm"] = _norm_series(exp[p3_col])
    if ent_col:
        exp["__ent_norm"] = _norm_series(exp[ent_col])

    if p3_col and dept_col and ent_col and not exp.empty:
        grp = exp.dropna(subset=["__p3_norm","__ent_norm"]).copy()
        if not grp.empty:
            mode_dept = grp.groupby(["__ent_norm","__p3_norm"]) [dept_col].agg(lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0])
            entity_pfx_map = mode_dept.to_dict()
            if subc_col:
                mode_sub = grp.groupby(["__ent_norm","__p3_norm"]) [subc_col].agg(lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0])
                entity_pfx_map_sub = mode_sub.to_dict()

    if p3_col and dept_col and not exp.empty:
        grp2 = exp.dropna(subset=["__p3_norm"]).copy()
        if not grp2.empty:
            mode_dept2 = grp2.groupby("__p3_norm")[dept_col].agg(lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0])
            pfx_map = mode_dept2.to_dict()
            if subc_col:
                mode_sub2 = grp2.groupby("__p3_norm")[subc_col].agg(lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0])
                pfx_map_sub = mode_sub2.to_dict()

# ---------- Mapping function ----------
def map_one(code_raw, ent_raw):
    code = _norm_one_val(code_raw)
    ent  = _norm_one_val(ent_raw)
    if not code:
        return (pd.NA, pd.NA, "UNMAPPED")

    # Tier 1: EXACT
    if code in exact_map:
        return (exact_map.get(code), exact_map_sub.get(code, pd.NA), "EXACT")

    # Tier 2: HIER — progressively drop left segments
    parts = code.split('.')
    if len(parts) > 1:
        for i in range(1, len(parts)):
            suf = '.'.join(parts[i:])
            if suf in exact_map:
                return (exact_map.get(suf), exact_map_sub.get(suf, pd.NA), "HIER")

    # Derive P3 candidate from rightmost segment that exists in pfx_map
    def pick_p3(c):
        segs = c.split('.')
        for j in range(len(segs)-1, -1, -1):
            seg = segs[j]
            if seg in pfx_map:
                return seg
        return None

    p3 = pick_p3(code)

    # Tier 3: ENTITY + PREFIX_3
    if p3 and ent and (ent, p3) in entity_pfx_map:
        return (entity_pfx_map.get((ent,p3)), entity_pfx_map_sub.get((ent,p3), pd.NA), "ENTITY_PFX")

    # Tier 4: PREFIX_3 only
    if p3 and (p3 in pfx_map):
        return (pfx_map.get(p3), pfx_map_sub.get(p3, pd.NA), "PFX3")

    # Tier 5: simple keyword nudge (no regex)
    tokens = set(parts)
    if {"MFG"} & tokens:
        return ("Manufacturing", pd.NA, "KEYWORD")
    if {"R&D","RANDD","PRDDEV","TV","VIC","PT"} & tokens:
        return ("R&D", pd.NA, "KEYWORD")
    if {"HR","HRP","ADM","OTHADM","PLANT","PLNT"} & tokens:
        return ("HR & Admin", pd.NA, "KEYWORD")
    if {"MKT","A&M"} & tokens:
        return ("Marketing", pd.NA, "KEYWORD")
    if {"INF","PUNE","CONCOR","SAF"} & tokens:
        return ("Infra", pd.NA, "KEYWORD")
    if {"CNST","PM","PRJ"} & tokens:
        return ("Program", pd.NA, "KEYWORD")
    if {"FIN"} & tokens:
        return ("Finance", pd.NA, "KEYWORD")
    if {"LGL","LGLF","IP","PRT"} & tokens:
        return ("Legal & IP", pd.NA, "KEYWORD")
    if {"SS","COG","TLG"} & tokens:
        return ("SS & SCM", pd.NA, "KEYWORD")
    if {"SLS"} & tokens:
        return ("Sales", pd.NA, "KEYWORD")
    if {"RENT","COUR"} & tokens:
        return ("Rental Offices", pd.NA, "KEYWORD")

    return (pd.NA, pd.NA, "UNMAPPED")

# ---------- Apply mapping ----------
code_cols = [c for c in ["PO Budget Code","PR Budget Code"] if c in _df.columns]
if code_cols:
    base = _df[code_cols[0]]
    ent  = _df.get("Entity", pd.Series([pd.NA]*len(_df)))
    mapped_rows = pd.DataFrame([map_one(c,e) for c,e in zip(base.tolist(), ent.tolist())], columns=["__dept","__subc","__src"], index=_df.index)
    _df["Dept.Chart"] = _df["Dept.Chart"].combine_first(mapped_rows["__dept"]) if "Dept.Chart" in _df.columns else mapped_rows["__dept"]
    _df["Subcat.Chart"] = _df["Subcat.Chart"].combine_first(mapped_rows["__subc"]) if "Subcat.Chart" in _df.columns else mapped_rows["__subc"]
    need_src = _df["__Dept.MapSrc"].isin(["UNMAPPED", pd.NA, None])
    _df.loc[need_src, "__Dept.MapSrc"] = mapped_rows.loc[need_src, "__src"].fillna("UNMAPPED")

# Final NA fill
_df["Dept.Chart"].fillna("Unmapped / Missing", inplace=True)

# ---------- QA ----------
with st.expander("🧪 Smart Mapper QA", expanded=False):
    st.write({"counts": _df["__Dept.MapSrc"].value_counts(dropna=False).to_dict()})
    if code_cols:
        unm = _df[_df["__Dept.MapSrc"] == "UNMAPPED"].copy()
        if not unm.empty:
            unm["Budget Code (normalized)"] = _norm_series(unm[code_cols[0]])
            summary_unm = unm["Budget Code (normalized)"].value_counts().reset_index()
            summary_unm.columns = ["Budget Code (normalized)", "Lines"]
            st.dataframe(summary_unm.head(200), use_container_width=True)
            st.download_button("⬇️ Download Unmapped Budget Codes", summary_unm.to_csv(index=False), "unmapped_budget_codes.csv", "text/csv", key="dl_unmapped_codes_new")
        else:
            st.caption("All lines mapped by Smart Mapper.")

# ---------- Charts ----------
if "Net Amount" in _df.columns:
    dept_spend_new = (
        _df.groupby("Dept.Chart", dropna=False)["Net Amount"].sum().reset_index()
        .sort_values("Net Amount", ascending=False)
    )
    dept_spend_new["Spend (Cr ₹)"] = dept_spend_new["Net Amount"] / 1e7
    st.plotly_chart(
        px.bar(dept_spend_new.head(30), x="Dept.Chart", y="Spend (Cr ₹)", title="Department-wise Spend (Top 30) — NEW")
        .update_layout(xaxis_tickangle=-45),
        use_container_width=True,
    )

    # Drilldown
    c1, c2 = st.columns([2,1])
    with c1:
        dept_pick_new = st.selectbox("Drill down (NEW): choose a department", dept_spend_new["Dept.Chart"].tolist(), key="dept_pick_new")
    with c2:
        topn_new = st.number_input("Show top N vendors/items", min_value=5, max_value=100, value=20, step=5, key="topn_new")

    det = _df[_df["Dept.Chart"].astype(str) == str(dept_pick_new)].copy()
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Lines", len(det))
    k2.metric("PRs", int(det["PR Number"].nunique()) if "PR Number" in det.columns else 0)
    k3.metric("POs", int(det["Purchase Doc"].nunique()) if "Purchase Doc" in det.columns else 0)
    k4.metric("Spend (Cr ₹)", f"{(det.get('Net Amount', pd.Series(0)).sum()/1e7):,.2f}")

    c3,c4 = st.columns(2)
    if {"PO Vendor","Net Amount"}.issubset(det.columns):
        tv = det.groupby("PO Vendor", dropna=False)["Net Amount"].sum().sort_values(ascending=False).head(int(topn_new)).reset_index()
        tv["Spend (Cr ₹)"] = tv["Net Amount"]/1e7
        c3.plotly_chart(px.bar(tv, x="PO Vendor", y="Spend (Cr ₹)", title="Top Vendors (Cr ₹) — NEW").update_layout(xaxis_tickangle=-45), use_container_width=True)
    if {"Product Name","Net Amount"}.issubset(det.columns):
        ti = det.groupby("Product Name", dropna=False)["Net Amount"].sum().sort_values(ascending=False).head(int(topn_new)).reset_index()
        ti["Spend (Cr ₹)"] = ti["Net Amount"]/1e7
        c4.plotly_chart(px.bar(ti, x="Product Name", y="Spend (Cr ₹)", title="Top Items (Cr ₹) — NEW").update_layout(xaxis_tickangle=-45), use_container_width=True)

    dcol = "Po create Date" if "Po create Date" in det.columns else ("PR Date Submitted" if "PR Date Submitted" in det.columns else None)
    if dcol and "Net Amount" in det.columns:
        det[dcol] = pd.to_datetime(det[dcol], errors="coerce")
        m = det.dropna(subset=[dcol]).groupby(det[dcol].dt.to_period('M'))['Net Amount'].sum().to_timestamp()
        st.plotly_chart(px.line(m/1e7, labels={'value':'Spend (Cr ₹)','index':'Month'}, title=f"{dept_pick_new} — Monthly Spend — NEW"), use_container_width=True)

# End NEW block
