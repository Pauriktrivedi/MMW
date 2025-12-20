import pandas as pd
import numpy as np

# Mock the logic from app.py
def build_savings_mock(df):
    z = df.copy()
    po_vendor_col = 'po_vendor'
    
    # EXCLUSION LOGIC
    if po_vendor_col and po_vendor_col in z.columns:
        mask_exclude = z[po_vendor_col].astype(str).str.contains('Gujarat', case=False, na=False) & \
                       z[po_vendor_col].astype(str).str.contains('Erection', case=False, na=False)
        if mask_exclude.any():
            print(f"Excluding {mask_exclude.sum()} rows matching 'Gujarat' + 'Erection'")
            z = z[~mask_exclude]
    
    return z

def test_exclusion():
    data = {
        'po_vendor': [
            'Regular Vendor A',
            'Gujarat Erection Contractor',
            'GUJARAT ERECTION CONTRACTOR',
            'Some Other Contractor',
            'Gujarat Irrigation Works', # Should NOT be excluded now
            'Erection Services Ltd'     # Should NOT be excluded (missing Gujarat)
        ],
        'net_amount': [100, 200, 300, 400, 500, 600]
    }
    df = pd.DataFrame(data)
    
    print("Original DataFrame:")
    print(df)
    
    result = build_savings_mock(df)
    
    print("\nProcessed DataFrame:")
    print(result)
    
    # Assertions
    assert len(result) == 4, f"Expected 4 rows, got {len(result)}"
    excluded_names = ['Gujarat Erection Contractor', 'GUJARAT ERECTION CONTRACTOR']
    for name in excluded_names:
        assert name not in result['po_vendor'].values, f"Failed to exclude {name}"
        
    assert 'Gujarat Irrigation Works' in result['po_vendor'].values
    print("\nSUCCESS: Exclusion logic verified for 'Gujarat Erection'!")

if __name__ == "__main__":
    test_exclusion()
