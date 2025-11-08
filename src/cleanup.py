import os
import pandas as pd
from pathlib import Path

# ==========================================================
# 🧾 CONFIGURATION
# ==========================================================

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

IMPORTANT_COLUMNS = [
    "Name",
    "WHO_Region",
    "Cases_cumulative_total",
    "Cases_newly_reported_in_last_7_days",
    "Cases_newly_reported_in_last_24_hours",
    "Deaths_cumulative_total",
    "Deaths_newly_reported_in_last_7_days",
    "Deaths_newly_reported_in_last_24_hours",
]


# ==========================================================
# 🧹 CLEANING FUNCTIONS
# ==========================================================

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    print("🔤 Cleaning column names...")
    df.columns = [
        c.strip()
        .replace(" ", "_")
        .replace("-", "")
        .replace("/", "")
        .replace("(", "")
        .replace(")", "")
        .replace("__", "_")
        for c in df.columns
    ]
    return df


def simplify_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    print("🔢 Converting numeric columns where possible...")
    for col in df.columns:
        before_dtype = df[col].dtype
        df[col] = pd.to_numeric(df[col], errors="ignore")
        after_dtype = df[col].dtype
        if before_dtype != after_dtype:
            print(f"   ✅ Converted '{col}' from {before_dtype} → {after_dtype}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    print("🧩 Handling missing values...")
    missing_info = df.isnull().sum()
    total_missing = missing_info.sum()
    if total_missing == 0:
        print("   ✅ No missing values found.")
    else:
        print("   ⚠️ Found missing values:")
        print(missing_info[missing_info > 0])
        print("   → Filling numeric NaNs with 0, text NaNs with 'Unknown'")
        df = df.fillna(
            {
                col: 0 if pd.api.types.is_numeric_dtype(df[col]) else "Unknown"
                for col in df.columns
            }
        )
    return df


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    print("➕ Calculating derived metrics...")

    if "Deaths_cumulative_total" in df.columns and "Cases_cumulative_total" in df.columns:
        df["Case_Fatality_Rate"] = df.apply(
            lambda x: (x["Deaths_cumulative_total"] / x["Cases_cumulative_total"]) * 100
            if x["Cases_cumulative_total"] > 0 else 0,
            axis=1
        )
        df["Case_Fatality_Rate"] = df["Case_Fatality_Rate"].round(2)
        print("   ✅ Added 'Case_Fatality_Rate'")

    if "Cases_newly_reported_in_last_7_days" in df.columns and "Cases_cumulative_total" in df.columns:
        df["Weekly_Case_Growth_%"] = df.apply(
            lambda x: (x["Cases_newly_reported_in_last_7_days"] / x["Cases_cumulative_total"]) * 100
            if x["Cases_cumulative_total"] > 0 else 0,
            axis=1
        )
        df["Weekly_Case_Growth_%"] = df["Weekly_Case_Growth_%"].round(2)
        print("   ✅ Added 'Weekly_Case_Growth_%'")

    return df


def confirm_and_keep_columns(df: pd.DataFrame) -> pd.DataFrame:
    print("\n📊 Columns available in dataset:")
    for col in df.columns:
        print("  •", col)

    confirm = input(
        "\nWould you like to keep ONLY the key WHO columns? (y/n): "
    ).strip().lower()

    if confirm == "y":
        print("✂️ Keeping only the selected WHO columns...")
        df = df[[col for col in IMPORTANT_COLUMNS if col in df.columns]]
    else:
        print("✅ Keeping all columns (no filtering applied).")

    return df


# ==========================================================
# 🪶 DOCUMENTATION
# ==========================================================

def generate_markdown_summary(df: pd.DataFrame, file_name: str):
    print("📝 Generating column summary markdown...")

    summary_lines = [
        f"# 📊 Data Summary: {file_name}",
        "",
        f"**Rows:** {len(df)}  |  **Columns:** {len(df.columns)}",
        "",
        "| Column Name | Data Type | Missing Values (%) | Unique Values | Example Values | Stats |",
        "|--------------|------------|--------------------|----------------|----------------|-------|",
    ]

    for col in df.columns:
        dtype = df[col].dtype
        missing_pct = (df[col].isnull().sum() / len(df)) * 100
        unique_count = df[col].nunique()
        sample_vals = df[col].dropna().unique()[:3]
        example_vals = ", ".join(map(str, sample_vals)) if len(sample_vals) > 0 else "None"

        # Add numeric stats if numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            mean = df[col].mean()
            median = df[col].median()
            min_val = df[col].min()
            max_val = df[col].max()
            stats_str = f"Mean={mean:.2f}, Median={median:.2f}, Min={min_val:.2f}, Max={max_val:.2f}"
        else:
            stats_str = ""

        summary_lines.append(
            f"| {col} | {dtype} | {missing_pct:.2f}% | {unique_count} | {example_vals} | {stats_str} |"
        )

    summary_text = "\n".join(summary_lines)
    summary_path = PROCESSED_DIR / f"{file_name.replace('.csv', '')}_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"✅ Summary file saved to: {summary_path.resolve()}\n")


# ==========================================================
# 🚀 MAIN LOGIC
# ==========================================================

def process_dataset(file_path: Path):
    print(f"\n📂 Processing file: {file_path.name}")

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"❌ Failed to read {file_path.name}: {e}")
        return

    print(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns")

    # Cleaning steps
    df = clean_column_names(df)
    df = simplify_numeric_columns(df)
    df = handle_missing_values(df)
    df = add_derived_metrics(df)
    df = confirm_and_keep_columns(df)

    # Save processed version
    out_path = PROCESSED_DIR / file_path.name
    df.to_csv(out_path, index=False)
    print(f"💾 Saved cleaned file to: {out_path.resolve()}")

    # Generate Markdown documentation
    generate_markdown_summary(df, file_path.name)


if __name__ == "__main__":
    print("🧹 Starting dataset cleanup process...\n")

    csv_files = list(RAW_DIR.glob("*.csv"))
    if not csv_files:
        print("⚠️ No CSV files found in data/raw/")
    else:
        for f in csv_files:
            process_dataset(f)

    print("🎉 All done! Processed files and summaries saved in data/processed/")
