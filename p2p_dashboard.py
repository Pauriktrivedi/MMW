# ----------------- Savings (REPLACEMENT BLOCK) -----------------
import traceback

with T[8]:
    st.subheader('Savings — PR → PO (defensive)')

    # Detect PR/PO candidate columns using safe_col as before
    pr_qty_col = safe_col(fil, ['pr_quantity','pr qty','pr_quantity','pr quantity','pr_quantity'])
    pr_unit_rate_col = safe_col(fil, ['unit_rate','pr_unit_rate','pr unit rate','pr_unit_rate'])
    pr_value_col = safe_col(fil, ['pr_value','pr value','pr_value'])
    po_qty_col = safe_col(fil, ['po_quantity','po qty','po_quantity','po quantity'])
    po_unit_rate_col = safe_col(fil, ['po_unit_rate','po unit rate','po_unit_rate'])
    net_col = safe_col(fil, ['net_amount','net amount','net_amount_inr','net_amount'])

    # Quick diagnostics for you in-app (uncomment if you want)
    # st.write(dict(pr_qty_col=pr_qty_col, pr_unit_rate_col=pr_unit_rate_col, pr_value_col=pr_value_col,
    #                po_qty_col=po_qty_col, po_unit_rate_col=po_unit_rate_col, net_col=net_col))

    if (pr_qty_col or pr_unit_rate_col or pr_value_col) and (po_qty_col or po_unit_rate_col or net_col):
        try:
            def build_savings():
                z = fil.copy()

                # ensure any categorical columns used are converted to safe types first
                for col in [pr_qty_col, pr_unit_rate_col, pr_value_col, po_qty_col, po_unit_rate_col, net_col]:
                    if col and col in z.columns and pd.api.types.is_categorical_dtype(z[col]):
                        z[col] = z[col].astype(object)

                # compute PR line value: prefer PR Value if present else PR Qty * PR Unit Rate
                if pr_value_col and pr_value_col in z.columns:
                    z['pr_line_value'] = pd.to_numeric(z[pr_value_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0.0)
                else:
                    # safe multiplication with defaults
                    pr_q = pd.to_numeric(z.get(pr_qty_col, 0).astype(str).str.replace(',', ''), errors='coerce').fillna(0.0)
                    pr_r = pd.to_numeric(z.get(pr_unit_rate_col, 0).astype(str).str.replace(',', ''), errors='coerce').fillna(0.0)
                    z['pr_line_value'] = pr_q * pr_r

                # compute PO line net: prefer Net Amount if present else PO Qty * PO Unit Rate
                if net_col and net_col in z.columns:
                    z['po_line_value'] = pd.to_numeric(z[net_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0.0)
                else:
                    po_q = pd.to_numeric(z.get(po_qty_col, 0).astype(str).str.replace(',', ''), errors='coerce').fillna(0.0)
                    po_r = pd.to_numeric(z.get(po_unit_rate_col, 0).astype(str).str.replace(',', ''), errors='coerce').fillna(0.0)
                    z['po_line_value'] = po_q * po_r

                # unit rates numeric where available
                z['pr_unit_rate_f'] = pd.to_numeric(z.get(pr_unit_rate_col, np.nan), errors='coerce')
                z['po_unit_rate_f'] = pd.to_numeric(z.get(po_unit_rate_col, np.nan), errors='coerce')

                # savings absolute and percent (use pr_line_value as denominator when >0)
                z['savings_abs'] = z['pr_line_value'] - z['po_line_value']
                z['savings_pct'] = np.where(z['pr_line_value'] > 0, (z['savings_abs'] / z['pr_line_value']) * 100.0, np.nan)

                # per-unit pct if unit rates present
                z['unit_rate_pct_saved'] = np.where(z['pr_unit_rate_f'] > 0,
                                                    (z['pr_unit_rate_f'] - z['po_unit_rate_f']) / z['pr_unit_rate_f'] * 100.0,
                                                    np.nan)

                # select display columns (only those that exist)
                disp_cols = [
                    pr_number_col, purchase_doc_col,
                    pr_qty_col, pr_unit_rate_col, pr_value_col,
                    po_qty_col, po_unit_rate_col, net_col,
                    'pr_line_value', 'po_line_value',
                    'savings_abs', 'savings_pct', 'unit_rate_pct_saved',
                    'po_vendor', 'buyer_display', 'entity', 'procurement_category'
                ]
                disp_cols = [c for c in disp_cols if c and c in z.columns]
                return z[disp_cols].copy()

            # compute and memoize
            savings_df = memoized_compute('savings', filter_signature, build_savings)

            if savings_df.empty:
                st.info('No matching PR/PO rows found to compute savings.')
            else:
                # KPIs
                total_pr_value = float(savings_df['pr_line_value'].sum())
                total_po_value = float(savings_df['po_line_value'].sum())
                total_savings = total_pr_value - total_po_value
                pct_saved_overall = (total_savings / total_pr_value * 100.0) if total_pr_value > 0 else np.nan
                k1,k2,k3,k4 = st.columns(4)
                k1.metric('Total PR Value (Cr)', f"{total_pr_value/1e7:,.2f}")
                k2.metric('Total PO Value (Cr)', f"{total_po_value/1e7:,.2f}")
                k3.metric('Total Savings (Cr)', f"{total_savings/1e7:,.2f}")
                k4.metric('% Saved vs PR', f"{pct_saved_overall:.2f}%" if pct_saved_overall==pct_saved_overall else 'N/A')

                st.markdown('---')
                # Histogram % saved
                st.subheader('Distribution of % Saved (per line)')
                fig_hist = px.histogram(savings_df, x='savings_pct', nbins=50, title='% Saved per Line (PR→PO)', labels={'savings_pct':'% Saved'})
                st.plotly_chart(fig_hist, use_container_width=True)

                # Top savings by absolute value
                st.subheader('Top Savings — Absolute (Cr)')
                top_abs = savings_df.sort_values('savings_abs', ascending=False).head(20).copy()
                top_abs['savings_cr'] = top_abs['savings_abs']/1e7
                x_axis_for_top = 'po_vendor' if 'po_vendor' in top_abs.columns else purchase_doc_col
                fig_top_abs = px.bar(top_abs, x=x_axis_for_top, y='savings_cr', hover_data=['pr_line_value','po_line_value'], title='Top 20 Savings by Absolute Value (Cr)')
                st.plotly_chart(fig_top_abs, use_container_width=True)

                # Category level
                st.subheader('Savings by Procurement Category')
                if 'procurement_category' in savings_df.columns:
                    pc = savings_df.groupby('procurement_category', dropna=False)[['pr_line_value','po_line_value','savings_abs']].sum().reset_index()
                    pc['savings_cr'] = pc['savings_abs']/1e7
                    pc['pct_saved'] = np.where(pc['pr_line_value']>0, pc['savings_abs']/pc['pr_line_value']*100.0, np.nan)
                    fig_pc = px.bar(pc.sort_values('savings_cr', ascending=False), x='procurement_category', y='savings_cr', text='pct_saved', title='Procurement Category — Savings (Cr)')
                    fig_pc.update_traces(texttemplate='%{text:.2f}%')
                    fig_pc.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_pc, use_container_width=True)
                else:
                    st.info('Procurement Category not available for category breakdown.')

                # PR unit vs PO unit scatter
                if 'pr_unit_rate_f' in savings_df.columns and 'po_unit_rate_f' in savings_df.columns:
                    st.subheader('PR Unit Rate vs PO Unit Rate (scatter)')
                    sc = savings_df.dropna(subset=['pr_unit_rate_f','po_unit_rate_f']).copy()
                    fig_sc = px.scatter(sc, x='pr_unit_rate_f', y='po_unit_rate_f', size='pr_line_value', hover_data=[pr_number_col, purchase_doc_col, 'po_vendor'], title='PR Unit Rate vs PO Unit Rate')
                    st.plotly_chart(fig_sc, use_container_width=True)

                st.markdown('---')
                st.subheader('Detailed Savings List')
                st.dataframe(savings_df.sort_values('savings_abs', ascending=False).reset_index(drop=True), use_container_width=True)
                try:
                    st.download_button('⬇️ Download Savings CSV', savings_df.to_csv(index=False), file_name='savings_detail.csv', mime='text/csv')
                except Exception:
                    pass

        except Exception as ex:
            st.error('Error while computing Savings — full traceback below:')
            st.text(traceback.format_exc())
    else:
        st.info('Required PR/PO quantity/unit/value columns or Net Amount not present — cannot compute savings.')
