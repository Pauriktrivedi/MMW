import pandas as pd
from app import load_all, preprocess_data, safe_col

# Ensure we have data
try:
    df_raw = load_all()
    df = preprocess_data(df_raw)
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

print(f"Dataframe shape: {df.shape}")

po_vendor_col = safe_col(df, ['po_vendor', 'vendor', 'po vendor'])

if po_vendor_col:
    vendors = df[po_vendor_col].dropna().unique()
    print(f"\nUnique Vendors (First 50): {vendors[:50]}")
    
    # Fuzzy search
    print("\nSearching for 'Gujarat' or 'Irrigation' in Vendors...")
    matches = [v for v in vendors if 'gujarat' in str(v).lower() or 'irrigation' in str(v).lower()]
    print("Matches found:", matches)
else:
    print("PO Vendor column not found.")
