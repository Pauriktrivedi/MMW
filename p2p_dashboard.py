# ----- Clickable PR Budget Code / PR Budget Description charts with drill-down -----
import plotly.express as px

# helper: try import plotly_events for click handling; fallback to None
try:
    from streamlit_plotly_events import plotly_events
except Exception:
    plotly_events = None

# ensure columns exist (use normalized column names used in the app)
pr_code_col = 'pr_budget_code' if 'pr_budget_code' in dept_df.columns else ('pr budget code' if 'pr budget code' in dept_df.columns else None)
pr_desc_col = 'pr_budget_description' if 'pr_budget_description' in dept_df.columns else ('pr budget description' if 'pr budget description' in dept_df.columns else None)

# prepare a clean DataFrame for grouping
if net_amount_col in dept_df.columns:
    df_codes = dept_df.copy()
    # safe string conversions
    if pr_code_col:
        df_codes['__pr_code'] = df_codes[pr_code_col].astype(str).fillna('').str.strip()
    else:
        df_codes['__pr_code'] = ''
    if pr_desc_col:
        df_codes['__pr_desc'] = df_codes[pr_desc_col].astype(str).fillna('').str.strip()
    else:
        df_codes['__pr_desc'] = ''

    # aggregate top 30 PR Budget Codes
    agg_code = (df_codes.groupby('__pr_code', dropna=False)[net_amount_col]
                      .sum()
                      .reset_index()
                      .rename(columns={net_amount_col: 'spend'})
                      .sort_values('spend', ascending=False))
    agg_code['cr'] = agg_code['spend'] / 1e7
    top_code = agg_code.head(30).copy()

    # aggregate top 30 PR Budget Descriptions
    agg_desc = (df_codes.groupby('__pr_desc', dropna=False)[net_amount_col]
                      .sum()
                      .reset_index()
                      .rename(columns={net_amount_col: 'spend'})
                      .sort_values('spend', ascending=False))
    agg_desc['cr'] = agg_desc['spend'] / 1e7
    top_desc = agg_desc.head(30).copy()

    # Layout: show two charts stacked (Code then Description)
    st.markdown('### PR Budget Code Spend (Top 30)')
    fig_code = px.bar(top_code, x='__pr_code', y='cr', labels={'__pr_code':'PR Budget Code','cr':'Cr'}, hover_data={'spend':True})
    fig_code.update_layout(xaxis_tickangle=-45, height=420, margin=dict(t=40,b=120))

    # If plotly_events is available, use it to capture clicks; else show the chart and a selectbox fallback
    selected_code = None
    if plotly_events is not None:
        # show and capture click
        selected = plotly_events(fig_code, click_event=True, hover_event=False)
        # selected is a list of dicts; 'x' holds the category value
        if selected:
            selected_code = selected[0].get('x')
        st.plotly_chart(fig_code, use_container_width=True)
    else:
        st.plotly_chart(fig_code, use_container_width=True)
        # fallback: dropdown to pick a code
        code_options = ['(click a bar)'] + top_code['__pr_code'].astype(str).tolist()
        sel = st.selectbox('Pick PR Budget Code to drill (fallback)', code_options, index=0)
        selected_code = None if sel == '(click a bar)' else sel

    st.markdown('### PR Budget Description Spend (Top 30)')
    fig_desc = px.bar(top_desc, x='__pr_desc', y='cr', labels={'__pr_desc':'PR Budget Description','cr':'Cr'}, hover_data={'spend':True})
    fig_desc.update_layout(xaxis_tickangle=-45, height=420, margin=dict(t=40,b=120))

    selected_desc = None
    if plotly_events is not None:
        selected_d = plotly_events(fig_desc, click_event=True, hover_event=False)
        if selected_d:
            selected_desc = selected_d[0].get('x')
        st.plotly_chart(fig_desc, use_container_width=True)
    else:
        st.plotly_chart(fig_desc, use_container_width=True)
        desc_options = ['(click a bar)'] + top_desc['__pr_desc'].astype(str).tolist()
        sel_d = st.selectbox('Pick PR Budget Description to drill (fallback)', desc_options, index=0)
        selected_desc = None if sel_d == '(click a bar)' else sel_d

    # Show the related rows (priority: code click -> description click -> nothing)
    if selected_code:
        st.markdown(f"**Showing rows matching PR Budget Code = `{selected_code}`**")
        mask = df_codes['__pr_code'].astype(str) == str(selected_code)
        rows = df_codes[mask].copy()
        if not rows.empty:
            cols = [c for c in ['pr_number', 'purchase_doc', 'product_name', 'po_vendor', net_amount_col, pr_code_col, pr_desc_col] if c in rows.columns]
            st.dataframe(rows[cols].reset_index(drop=True), use_container_width=True)
            st.download_button('Download drilled rows (CSV)', rows.to_csv(index=False), file_name='drill_pr_code.csv', mime='text/csv')
        else:
            st.info('No rows found for the selected PR Budget Code.')
    elif selected_desc:
        st.markdown(f"**Showing rows matching PR Budget Description = `{selected_desc}`**")
        mask = df_codes['__pr_desc'].astype(str) == str(selected_desc)
        rows = df_codes[mask].copy()
        if not rows.empty:
            cols = [c for c in ['pr_number', 'purchase_doc', 'product_name', 'po_vendor', net_amount_col, pr_code_col, pr_desc_col] if c in rows.columns]
            st.dataframe(rows[cols].reset_index(drop=True), use_container_width=True)
            st.download_button('Download drilled rows (CSV)', rows.to_csv(index=False), file_name='drill_pr_desc.csv', mime='text/csv')
        else:
            st.info('No rows found for the selected PR Budget Description.')
    else:
        st.info('Click a bar on the charts above (or use the fallback dropdown) to view the related rows here.')
else:
    st.info('Dept data or Net Amount missing; cannot build PR Budget charts.')
