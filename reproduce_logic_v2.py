import pandas as pd
import numpy as np
from app import load_all, preprocess_data, safe_col

# Load and preprocess data
df_raw = load_all()
df = preprocess_data(df_raw)

print(f"Loaded dataframe with shape: {df.shape}")

# Simulate global variables/configs
po_unit_rate_col = safe_col(df, ['po_unit_rate', 'po unit rate', 'po_unit_price'])
net_amount_col = safe_col(df, ['net_amount', 'net amount', 'net_amount_inr', 'amount'])
purchase_doc_col = safe_col(df, ['purchase_doc', 'purchase_doc_number', 'purchase doc'])
pr_number_col = safe_col(df, ['pr_number', 'pr number', 'pr_no'])
po_vendor_col = safe_col(df, ['po_vendor', 'vendor', 'po vendor'])
po_create_col = safe_col(df, ['po_create_date', 'po create date', 'po_created_date'])

# --- 2. Savings Logic Check (New Robust Logic) ---
print("\n--- Savings (New Logic) ---")
pr_qty_col = safe_col(df, ['pr_quantity','pr qty','pr_quantity','pr quantity','pr quantity','pr_quantity'])
pr_unit_rate_col = safe_col(df, ['unit_rate','pr_unit_rate','pr unit rate','pr_unit_rate'])
pr_value_col = safe_col(df, ['pr_value','pr value','pr_value'])
po_qty_col = safe_col(df, ['po_quantity','po qty','po_quantity','po quantity'])
# po_unit_rate_col already defined
# net_col already defined

def compute_savings_robust(d):
    z = d.copy()
    
    # 1. Try PR Value column
    val_from_col = pd.Series(0.0, index=z.index)
    if pr_value_col and pr_value_col in z.columns:
        val_from_col = pd.to_numeric(z[pr_value_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0.0)
    
    # 2. Try Qty * Rate calculation
    pr_q_series = z.get(pr_qty_col, pd.Series(0, index=z.index))
    pr_r_series = z.get(pr_unit_rate_col, pd.Series(0, index=z.index))
    
    pr_q = pd.to_numeric(pr_q_series.astype(str).str.replace(',', ''), errors='coerce').fillna(0.0)
    pr_r = pd.to_numeric(pr_r_series.astype(str).str.replace(',', ''), errors='coerce').fillna(0.0)
    val_calc = pr_q * pr_r

    # 3. Use calculated if column is 0 but calculated is > 0 (data fix)
    #    Otherwise trust column if present, else calc.
    if pr_value_col and pr_value_col in z.columns:
        z['pr_line_value'] = np.where((val_from_col == 0) & (val_calc > 0), val_calc, val_from_col)
    else:
        z['pr_line_value'] = val_calc

    # compute PO line net: prefer Net Amount if present else PO Qty * PO Unit Rate
    if net_amount_col and net_amount_col in z.columns:
        z['po_line_value'] = pd.to_numeric(z[net_amount_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0.0)
    else:
        po_q_series = z.get(po_qty_col, pd.Series(0, index=z.index))
        po_r_series = z.get(po_unit_rate_col, pd.Series(0, index=z.index))
        po_q = pd.to_numeric(po_q_series.astype(str).str.replace(',', ''), errors='coerce').fillna(0.0)
        po_r = pd.to_numeric(po_r_series.astype(str).str.replace(',', ''), errors='coerce').fillna(0.0)
        z['po_line_value'] = po_q * po_r

    z['savings_abs'] = z['pr_line_value'] - z['po_line_value']
    
    # savings percent: exclude cases where PR Value is 0 (Unbudgeted/Unestimated)
    z['savings_pct'] = np.where(z['pr_line_value'] > 0, (z['savings_abs'] / z['pr_line_value']) * 100.0, np.nan)
    
    # Flag for unestimated spend (PR=0, PO>0)
    z['is_unestimated'] = (z['pr_line_value'] == 0) & (z['po_line_value'] > 0)
    
    return z

savings_df = compute_savings_robust(df)
print(f"Savings DF shape: {savings_df.shape}")
print(savings_df[['pr_line_value', 'po_line_value', 'savings_abs', 'savings_pct', 'is_unestimated']].describe())

unest_count = savings_df['is_unestimated'].sum()
print(f"\nUnestimated Count: {unest_count}")
print(f"Unestimated Spend: {savings_df.loc[savings_df['is_unestimated'], 'po_line_value'].sum()}")
