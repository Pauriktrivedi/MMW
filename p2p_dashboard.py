import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="P2P Dashboard — Indirect", layout="wide", initial_sidebar_state="expanded")

# ---------- Header ----------
st.markdown(
    """
    <div style="padding:8px 0 12px 0; margin-bottom:10px;">
      <h1 style="font-size:34px; margin:0; color:#0b1f3b;">P2P Dashboard — Indirect</h1>
      <div style="font-size:13px; color:#23395b; margin-top:6px;">Purchase-to-Pay overview (Indirect spend focus)</div>
      <hr style="border:0; height:1px; background:#e6eef6; margin-top:10px; margin-bottom:12px;" />
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Helpers ----------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to snake_case lowercase and remove NBSPs.

    This version carefully escapes backslashes when replacing path separators.
    """
    new_cols = {}
    for c in df.columns:
        s = str(c).strip()
        s = s.replace(" ", " ")
        # escape backslash correctly in Python strings
        s = s.replace("\\\\", "_").replace("/", "_")
        # collapse whitespace to single underscore
        s = "_".join(s.split())
        s = s.lower()
        # keep only alphanumeric and underscore
        s = ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in s)
        # remove duplicate underscores
        s = '_'.join([p for p in s.split('_') if p != ''])
        new_cols[c] = s
    return df.rename(columns=new_cols)

@st.cache_data(show_spinner=False, ttl=3600)
def load_all(fns=None):
    # load default files if not provided
    if fns is None:
        fns = [("MEPL.xlsx","MEPL"),("MLPL.xlsx","MLPL"),("mmw.xlsx","MMW"),("mmpl.xlsx","MMPL")]
    frames = []
    for fn, ent in fns:
        try:
            tmp = pd.read_excel(fn, skiprows=1)
            tmp['entity'] = ent
            frames.append(tmp)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    x = pd.concat(frames, ignore_index=True)
    x = normalize_columns(x)
    for dcol in ['pr_date_submitted','po_create_date','po_approved_date','po_delivery_date']:
        if dcol in x.columns:
            x[dcol] = pd.to_datetime(x[dcol], errors='coerce')
    return x

# load data (cached)
df = load_all()
if df.empty:
    st.warning("No data loaded. Put MEPL.xlsx, MLPL.xlsx, mmw.xlsx, mmpl.xlsx next to this script or update load_all().")

# --- canonical column names used below ---
pr_col = 'pr_date_submitted' if 'pr_date_submitted' in df.columns else None
po_create_col = 'po_create_date' if 'po_create_date' in df.columns else None
net_amount_col = 'net_amount' if 'net_amount' in df.columns else None
purchase_doc_col = 'purchase_doc' if 'purchase_doc' in df.columns else None
pr_number_col = 'pr_number' if 'pr_number' in df.columns else None
po_vendor_col = 'po_vendor' if 'po_vendor' in df.columns else None
product_col = 'product_name' if 'product_name' in df.columns else None
entity_col = 'entity'

TODAY = pd.Timestamp.now().normalize()

# ---------- Sidebar filters ----------
st.sidebar.header('Filters')
FY = {
    'All Years': (pd.Timestamp('2023-04-01'), pd.Timestamp('2026-03-31')),
    '2023': (pd.Timestamp('2023-04-01'), pd.Timestamp('2024-03-31')),
    '2024': (pd.Timestamp('2024-04-01'), pd.Timestamp('2025-03-31')),
    '2025': (pd.Timestamp('2025-04-01'), pd.Timestamp('2026-03-31'))
}
fy_key = st.sidebar.selectbox('Financial Year', list(FY), index=0)
pr_start, pr_end = FY[fy_key]

fil = df.copy()
if pr_col:
    fil = fil[(fil[pr_col] >= pr_start) & (fil[pr_col] <= pr_end)]

# ensure vendor/product cols exist
if po_vendor_col not in fil.columns:
    fil['po_vendor'] = ''
    po_vendor_col = 'po_vendor'
if product_col not in fil.columns:
    fil['product_name'] = ''
    product_col = 'product_name'

# default to empty selections (faster). User picks values to filter.
vendor_choices = sorted(fil[po_vendor_col].dropna().unique().tolist()) if po_vendor_col in fil.columns else []
product_choices = sorted(fil[product_col].dropna().unique().tolist()) if product_col in fil.columns else []

sel_v = st.sidebar.multiselect('Vendor (pick one or more)', vendor_choices, default=None)
sel_p = st.sidebar.multiselect('Item / Product (pick one or more)', product_choices, default=None)

# simple reset button
if st.sidebar.button('Reset Filters'):
    for k in list(st.session_state.keys()):
        try:
            del st.session_state[k]
        except Exception:
            pass
    st.experimental_rerun()

# apply vendor/product filters only if user selected values
if sel_v:
    fil = fil[fil[po_vendor_col].isin(sel_v)]
if sel_p:
    fil = fil[fil[product_col].isin(sel_p)]

# ---------- Tabs ----------
T = st.tabs(['KPIs & Spend','PO/PR Timing','Delivery','Vendors','Dept & Services','Unit-rate Outliers','Forecast','Scorecards','Search'])

# ---------- KPIs & Spend (combined) ----------
with T[0]:
    c1,c2,c3,c4,c5 = st.columns(5)
    total_prs = int(fil[pr_number_col].nunique()) if pr_number_col in fil.columns else 0
    total_pos = int(fil[purchase_doc_col].nunique()) if purchase_doc_col in fil.columns else 0
    line_items = len(fil)
    entities = int(fil[entity_col].nunique()) if entity_col in fil.columns else 0
    spend_val = fil[net_amount_col].sum() if net_amount_col in fil.columns else 0
    c1.metric('Total PRs', total_prs)
    c2.metric('Total POs', total_pos)
    c3.metric('Line Items', line_items)
    c4.metric('Entities', entities)
    c5.metric('Spend (Cr ₹)', f"{spend_val/1e7:,.2f}")

    st.markdown('---')
    # Monthly spend + cumulative (last 24 months to avoid huge plots)
    if po_create_col and net_amount_col in fil.columns:
        t = fil.dropna(subset=[po_create_col]).copy()
        t['month'] = t[po_create_col].dt.to_period('M').dt.to_timestamp()
        m = t.groupby('month')[net_amount_col].sum().sort_index()
        m = m.tail(24)
        dfm = m.reset_index().rename(columns={net_amount_col:'amount'})
        dfm['cr'] = dfm['amount']/1e7
        dfm['cumcr'] = dfm['cr'].cumsum()
        fig = make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_bar(x=dfm['month'], y=dfm['cr'], name='Monthly Spend (Cr ₹)')
        fig.add_scatter(x=dfm['month'], y=dfm['cumcr'], name='Cumulative (Cr ₹)', mode='lines+markers', secondary_y=True)
        fig.update_layout(xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('Monthly spend requires Po create Date and Net Amount columns.')

    st.markdown('---')
    # Entity trend
    if po_create_col and net_amount_col in fil.columns and entity_col in fil.columns:
        x = fil.dropna(subset=[po_create_col, net_amount_col])[[po_create_col, entity_col, net_amount_col]].copy()
        x['month'] = x[po_create_col].dt.to_period('M').dt.to_timestamp()
        g = x.groupby(['month', entity_col])[net_amount_col].sum().reset_index()
        g['cr'] = g[net_amount_col]/1e7
        # limit to top 4 entities by spend to keep chart readable
        top_entities = g.groupby(entity_col)['cr'].sum().nlargest(4).index.tolist()
        g = g[g[entity_col].isin(top_entities)]
        fig2 = px.line(g, x=g['month'].dt.strftime('%b-%Y'), y='cr', color=entity_col, markers=True, labels={'cr':'Cr ₹','x':'Month'})
        fig2.update_layout(xaxis_tickangle=-45, height=350)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info('Entity trend requires Po create Date, Net Amount and Entity.')

# ---------- PO/PR Timing ----------
with T[1]:
    st.subheader('SLA (PR→PO ≤7d)')
    if pr_col and po_create_col in fil.columns:
        ld = fil.dropna(subset=[pr_col, po_create_col]).copy()
        ld['lead_time_days'] = (ld[po_create_col] - ld[pr_col]).dt.days
        avg = float(ld['lead_time_days'].mean()) if not ld.empty else 0.0
        max_range = max(14, avg*1.5 if avg else 14)
        gauge = go.Figure(go.Indicator(mode='gauge+number',value=avg,number={'suffix':' d'},gauge={'axis':{'range':[0,max_range]},'steps':[{'range':[0,7],'color':'lightgreen'},{'range':[7,max_range],'color':'lightcoral'}],'threshold':{'line':{'color':'red','width':4},'value':7}}))
        st.plotly_chart(gauge, use_container_width=True)
        bins=[0,7,15,30,60,90,999]
        labels=['0-7','8-15','16-30','31-60','61-90','90+']
        ag = pd.cut(ld['lead_time_days'], bins=bins, labels=labels).value_counts(normalize=True).reindex(labels, fill_value=0).reset_index()
        ag.columns=['Bucket','Pct']
        ag['Pct'] = ag['Pct']*100
        st.plotly_chart(px.bar(ag, x='Bucket', y='Pct', text='Pct').update_traces(texttemplate='%{text:.1f}%', textposition='outside'), use_container_width=True)
    else:
        st.info('PR→PO timing requires PR Date Submitted and Po create Date columns.')

# ---------- Delivery ----------
with T[2]:
    st.subheader('Delivery Summary')
    dv = fil.copy()
    # normalize some common columns
    if 'po_quantity' in dv.columns and 'receivedqty' in dv.columns:
        dv['po_qty'] = pd.to_numeric(dv['po_quantity'], errors='coerce')
        dv['received_qty'] = pd.to_numeric(dv['receivedqty'], errors='coerce')
        dv['%_received'] = np.where(dv['po_qty']>0, dv['received_qty']/dv['po_qty']*100, 0)
        summ = dv.groupby(['purchase_doc','po_vendor','product_name'], dropna=False).agg({'po_qty':'sum','received_qty':'sum','%_received':'mean'}).reset_index()
        st.dataframe(summ.sort_values('po_qty', ascending=False).head(50), use_container_width=True)
    else:
        st.info('Delivery view needs PO Quantity and ReceivedQTY columns.')

# ---------- Vendors ----------
with T[3]:
    st.subheader('Top Vendors by Spend')
    if po_vendor_col in fil.columns and net_amount_col in fil.columns:
        vs = fil.groupby(po_vendor_col, dropna=False)[net_amount_col].sum().reset_index().sort_values(net_amount_col, ascending=False)
        vs['Cr'] = vs[net_amount_col]/1e7
        st.plotly_chart(px.bar(vs.head(15), x=po_vendor_col, y='Cr', text='Cr').update_traces(textposition='outside'), use_container_width=True)
    else:
        st.info('Vendor view requires PO Vendor and Net Amount.')

# ---------- Dept & Services (placeholder) ----------
with T[4]:
    st.subheader('Dept & Services (PR Department)')

    # Use PR department column (check a list of likely column names created by normalization)
    pr_dept_candidates = [c for c in ['pr_department','pr_dept','pr_dept_name','pr_department_name','pr_department_code','dept_chart','department'] if c in fil.columns]
    if pr_dept_candidates:
        pr_dept_col = pr_dept_candidates[0]
        st.write(f"Using department column: **{pr_dept_col}** for Dept spend aggregation")
        d = fil.copy()
        d[pr_dept_col] = d[pr_dept_col].astype(str).str.strip().replace({'nan':pd.NA})
        if net_amount_col in d.columns:
            dep = d.groupby(pr_dept_col, dropna=False)[net_amount_col].sum().reset_index().sort_values(net_amount_col, ascending=False)
            dep['Cr'] = dep[net_amount_col]/1e7
            # show top 30 departments as bar chart
            topn = min(30, len(dep))
            if topn > 0:
                st.plotly_chart(px.bar(dep.head(topn), x=pr_dept_col, y='Cr', title='Department-wise Spend (Top departments)')
                              .update_layout(xaxis_tickangle=-45, yaxis_title='Cr ₹'), use_container_width=True)
                st.dataframe(dep.head(200).assign(Spend_Cr=lambda df: (df[net_amount_col]/1e7).round(2)), use_container_width=True)
            else:
                st.info('No departments with spend found.')
        else:
            st.info('Net Amount column not present — cannot compute department spend.')
    else:
        st.info('No PR department-like column found in data. Expected columns: pr_department, pr_dept, pr_dept_name, dept_chart, etc.')

with T[5]:
    st.subheader('Unit-rate Outliers vs Historical Median')
    grp_candidates = [c for c in ['product_name','item_code'] if c in fil.columns]
    if grp_candidates and 'po_unit_rate' in fil.columns:
        grp_by = st.selectbox('Group by', grp_candidates)
        z = fil[[grp_by,'po_unit_rate','purchase_doc','pr_number','po_vendor','item_description','po_create_date',net_amount_col]].dropna(subset=[grp_by,'po_unit_rate']).copy()
        med = z.groupby(grp_by)['po_unit_rate'].median().rename('median_rate')
        z = z.join(med, on=grp_by)
        z['pctdev'] = (z['po_unit_rate'] - z['median_rate']) / z['median_rate'].replace(0,np.nan)
        thr = st.slider('Outlier threshold (±%)', 10, 300, 50, 5)
        out = z[abs(z['pctdev']) >= thr/100.0].copy()
        out['pctdev%'] = (out['pctdev']*100).round(1)
        st.dataframe(out.sort_values('pctdev%', ascending=False).head(200), use_container_width=True)
    else:
        st.info("Need 'PO Unit Rate' and grouping column (product_name / item_code) for outlier detection.")

# ---------- Forecast ----------
with T[6]:
    st.subheader('Forecast Next Month Spend (SMA)')
    if po_create_col and net_amount_col in fil.columns:
        t = fil.dropna(subset=[po_create_col]).copy()
        t['month'] = t[po_create_col].dt.to_period('M').dt.to_timestamp()
        m = t.groupby('month')[net_amount_col].sum().sort_index()/1e7
        m = m.tail(24)
        k = st.slider('Window (months)', 3, 12, 6)
        sma = m.rolling(k).mean()
        mu = m.tail(k).mean() if len(m) >= k else m.mean()
        sd = m.tail(k).std(ddof=1) if len(m) >= k else m.std(ddof=1)
        n = min(k, max(1, len(m)))
        se = sd/np.sqrt(n) if not np.isnan(sd) else 0
        lo, hi = float(mu-1.96*se), float(mu+1.96*se)
        nxt = (m.index.max() + pd.offsets.MonthBegin(1)) if len(m)>0 else pd.Timestamp.now().to_period('M').to_timestamp()
        fdf = pd.DataFrame({'Month': list(m.index)+[nxt],'SpendCr': list(m.values)+[np.nan],'SMA': list(sma.values)+[mu]})
        fig = go.Figure()
        fig.add_bar(x=fdf['Month'], y=fdf['SpendCr'], name='Actual (Cr)')
        fig.add_scatter(x=fdf['Month'], y=fdf['SMA'], mode='lines+markers', name=f'SMA{k}')
        fig.add_scatter(x=[nxt,nxt], y=[lo,hi], mode='lines', name='95% CI')
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Forecast for {nxt.strftime('%b-%Y')}: {mu:.2f} Cr (95% CI: {lo:.2f}–{hi:.2f})")
    else:
        st.info('Need Po create Date and Net Amount to forecast.')

# ---------- Scorecards / Search ----------
with T[8]:
    st.subheader('🔍 Keyword Search')
    valid_cols = [c for c in [pr_number_col, purchase_doc_col, product_col, po_vendor_col] if c in fil.columns]
    q = st.text_input('Search Vendor/Product/PO/PR')
    if q and valid_cols:
        mask = pd.Series(False, index=fil.index)
        for c in valid_cols:
            mask = mask | fil[c].astype(str).str.contains(q, case=False, na=False)
        res = fil[mask]
        st.write(f'Found {len(res)} rows — showing first 200')
        st.dataframe(res.head(200), use_container_width=True)

# EOF
