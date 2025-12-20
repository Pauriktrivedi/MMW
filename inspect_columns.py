import pandas as pd
import numpy as np
from pathlib import Path

# Load MEPL.xlsx to inspect columns
try:
    df = pd.read_excel("MEPL.xlsx", skiprows=1)
    print("Columns found in MEPL.xlsx:")
    for c in df.columns:
        print(f"- {c}")
    
    # Check for specific value columns
    pr_val_candidates = [c for c in df.columns if 'pr' in str(c).lower() and ('val' in str(c).lower() or 'amount' in str(c).lower() or 'price' in str(c).lower())]
    pr_qty_candidates = [c for c in df.columns if 'pr' in str(c).lower() and ('qty' in str(c).lower() or 'quantity' in str(c).lower())]
    pr_rate_candidates = [c for c in df.columns if 'pr' in str(c).lower() and ('rate' in str(c).lower() or 'price' in str(c).lower())]
    
    po_val_candidates = [c for c in df.columns if 'net' in str(c).lower() or ('po' in str(c).lower() and ('val' in str(c).lower() or 'amount' in str(c).lower()))]
    
    print("\nPotential PR Value Columns:", pr_val_candidates)
    print("Potential PR Qty Columns:", pr_qty_candidates)
    print("Potential PR Rate Columns:", pr_rate_candidates)
    print("Potential PO Value Columns:", po_val_candidates)
    
    # Sample data check
    print("\nSample Data (First 5 rows for relevant columns):")
    cols_to_check = pr_val_candidates + pr_qty_candidates + pr_rate_candidates + po_val_candidates
    # Deduplicate
    cols_to_check = list(set(cols_to_check))
    
    if cols_to_check:
        print(df[cols_to_check].head())
        
        # Check calculation consistency if possible
        # Assume specific columns based on output or common names
        # Adjust these based on the actual print output above if running interactively, but here I'll try to guess generic ones to check
        
except Exception as e:
    print(f"Error reading file: {e}")
