import pandas as pd
from pathlib import Path
from IPython.display import display


# Load processed CSV
processed_file = Path("data/processed/WHO-COVID-19-global-table-data.csv")
df = pd.read_csv(processed_file)
print(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns")

# Overview
display(df.head())
print("\nColumn info:")
print(df.info())
print("\nMissing values per column:")
print(df.isnull().sum())

# ------------------------
# Derived metrics check
# ------------------------
metrics_to_check = ["Case_Fatality_Rate", "Weekly_Case_Growth_%"]

for metric in metrics_to_check:
    if metric in df.columns:
        print(f"\nSummary of {metric}:")
        print(df[metric].describe())
    else:
        print(f"\n⚠️ {metric} not found — cannot display summary (maybe filtered out).")

# ------------------------
# Extreme / suspicious values
# ------------------------
if "Case_Fatality_Rate" in df.columns:
    extreme_fatality = df[df["Case_Fatality_Rate"] > 50]
    if not extreme_fatality.empty:
        print("\nRows with Case Fatality Rate > 50%:")
        display(extreme_fatality)
    else:
        print("\nNo rows with Case Fatality Rate > 50%")

if "Weekly_Case_Growth_%" in df.columns:
    negative_growth = df[df["Weekly_Case_Growth_%"] < 0]
    if not negative_growth.empty:
        print("\nRows with negative Weekly Case Growth %:")
        display(negative_growth)
    else:
        print("\nNo rows with negative Weekly Case Growth %")
