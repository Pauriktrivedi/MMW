# ------------------------------------
# 13) Procurement Category Spend (Ascending Order)
# ------------------------------------
st.subheader("📦 Procurement Category Spend")
if "Procurement Category" in filtered_df.columns:
    # 1) Aggregate total Net Amount by category
    cat_spend = (
        filtered_df
        .groupby("Procurement Category")["Net Amount"]
        .sum()
        .reset_index()
    )

    # 2) Convert to crores
    cat_spend["Spend (Cr ₹)"] = cat_spend["Net Amount"] / 1e7

    # 3) Sort ascending by spend
    cat_spend = cat_spend.sort_values("Spend (Cr ₹)", ascending=True)

    # 4) Plot as a bar chart
    fig_cat = px.bar(
        cat_spend,
        x="Procurement Category",
        y="Spend (Cr ₹)",
        title="Spend by Category (Ascending)",
        labels={"Spend (Cr ₹)": "Spend (Cr ₹)", "Procurement Category": "Category"},
        text="Spend (Cr ₹)"
    )
    fig_cat.update_layout(xaxis_tickangle=-45)  # rotate x-labels if needed
    fig_cat.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    st.plotly_chart(fig_cat, use_container_width=True)
else:
    st.info("ℹ️ No 'Procurement Category' column found.")
