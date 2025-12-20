import pandas as pd
import numpy as np

try:
    df = pd.read_parquet('p2p_data.parquet')
    print("Loaded parquet.")
except:
    print("Parquet not found, trying excel...")
    # fallback code not needed for this environment usually, assuming parquet exists from previous steps
    exit()

cols_to_check = ['item_code', 'item_description', 'procurement_category', 'product_name']
existing_cols = [c for c in cols_to_check if c in df.columns]

print(f"Checking columns: {existing_cols}")

# Sample distinct combinations
print("\n--- Procurement Categories ---")
if 'procurement_category' in df.columns:
    print(df['procurement_category'].unique())

print("\n--- Item Code Samples ---")
if 'item_code' in df.columns:
    print(df['item_code'].dropna().sample(20).tolist())
    # Check if item codes starting with 99 exist (Services)
    services_99 = df[df['item_code'].astype(str).str.startswith('99', na=False)]
    if not services_99.empty:
        print(f"\nFound {len(services_99)} rows with Item Code starting with '99'. Sample:")
        print(services_99[existing_cols].head(5))

print("\n--- Product Name Samples ---")
if 'product_name' in df.columns:
    print(df['product_name'].dropna().unique()[:20])

print("\n--- Correlation Check ---")
# See what categories map to item codes starting with 99
if 'item_code' in df.columns and 'procurement_category' in df.columns:
    s99 = df[df['item_code'].astype(str).str.startswith('99', na=False)]
    print("Categories for Item Code '99...':")
    print(s99['procurement_category'].value_counts())

    non99 = df[~df['item_code'].astype(str).str.startswith('99', na=False)]
    print("\nCategories for other Item Codes:")
    print(non99['procurement_category'].value_counts().head(10))
