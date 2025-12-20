import pandas as pd
import numpy as np

try:
    df = pd.read_parquet("p2p_data.parquet")
    print("Columns:", df.columns.tolist())
    
    if 'entity' in df.columns:
        print("Unique Entities:", df['entity'].unique())
    elif 'entity_source_file' in df.columns:
        print("Unique Source Files:", df['entity_source_file'].unique())
    else:
        print("No entity column found.")

    po_vendor_col = None
    for c in ['po_vendor', 'vendor', 'po vendor']:
        if c in df.columns:
            po_vendor_col = c
            break
            
    if po_vendor_col:
        print(f"Vendor Column: {po_vendor_col}")
        # Search for Gujarat Irrigation
        mask = df[po_vendor_col].astype(str).str.contains('Gujarat', case=False, na=False) & \
               df[po_vendor_col].astype(str).str.contains('Irrigation', case=False, na=False)
        matches = df[mask][po_vendor_col].unique()
        print("Matches for 'Gujarat Irrigation':", matches)
    else:
        print("No vendor column found.")

except Exception as e:
    print(f"Error: {e}")
