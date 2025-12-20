import pandas as pd
from app import load_all, preprocess_data, safe_col

df_raw = load_all()
df = preprocess_data(df_raw)

search_term = "Gujarat Irrigation Contractor"
found = False

print(f"Searching for '{search_term}'...")

for col in df.columns:
    if df[col].dtype == object or df[col].dtype.name == 'category':
        # Check if term exists in this column
        # Use str conversion to be safe
        matches = df[df[col].astype(str).str.contains(search_term, case=False, na=False)]
        if not matches.empty:
            print(f"Found '{search_term}' in column: '{col}'")
            print(f"Exact values found: {matches[col].unique()}")
            found = True

if not found:
    print(f"'{search_term}' not found in any column.")
