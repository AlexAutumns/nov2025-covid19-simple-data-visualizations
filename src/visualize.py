# ==========================================================
# visualize.py
# Generates WHO COVID-19 visualizations (with & without Global)
# ==========================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---------------------------
# Configuration
# ---------------------------
base_data_path = Path("data/processed")
data_with_global = base_data_path / "WHO-COVID-19-global-table-data_with_global.csv"
data_without_global = base_data_path / "WHO-COVID-19-global-table-data_without_global.csv"

output_base = Path("src/visualization")
output_with_global = output_base / "WHO_COVID19_WithGlobal"
output_without_global = output_base / "WHO_COVID19_NoGlobal"

for folder in [output_with_global, output_without_global]:
    folder.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Load datasets
# ---------------------------
def load_dataset(path: Path):
    try:
        df = pd.read_csv(path)
        print(f"✅ Loaded {path.name} ({len(df)} rows, {len(df.columns)} columns)")
        return df
    except FileNotFoundError:
        print(f"❌ File not found: {path}")
        exit(1)

df_with = load_dataset(data_with_global)
df_without = load_dataset(data_without_global)

# ---------------------------
# Derived metrics
# ---------------------------
def ensure_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure derived metrics exist and are valid."""
    if "Weekly_Case_Growth_%" not in df.columns:
        df["Weekly_Case_Growth_%"] = (
            df["Cases_newly_reported_in_last_7_days"] /
            df["Cases_cumulative_total"].replace(0, pd.NA)
        ) * 100
    df["Weekly_Case_Growth_%"] = df["Weekly_Case_Growth_%"].fillna(0)

    if "Case_Fatality_Rate" not in df.columns:
        df["Case_Fatality_Rate"] = (
            df["Deaths_cumulative_total"] /
            df["Cases_cumulative_total"].replace(0, pd.NA)
        ) * 100
    df["Case_Fatality_Rate"] = df["Case_Fatality_Rate"].fillna(0)

    return df

df_with = ensure_metrics(df_with)
df_without = ensure_metrics(df_without)

# ---------------------------
# Helper to save plots
# ---------------------------
def save_plot(fig, filename, folder):
    path = folder / filename
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"💾 Saved plot: {path.relative_to(output_base)}")

# ---------------------------
# Function to generate all visualizations for a dataset
# ---------------------------
def generate_visualizations(df: pd.DataFrame, output_folder: Path, label: str):
    print(f"\n🧭 Generating visualizations for {label} dataset...")

    # 1️⃣ Top 10 countries by total cases
    top_cases = df.sort_values("Cases_cumulative_total", ascending=False).head(10)

    fig = plt.figure(figsize=(12,6))
    sns.barplot(
        data=top_cases,
        y="Name",
        x="Cases_cumulative_total",
        palette="Blues_r"
    )
    plt.xlabel("Total Cases")
    plt.ylabel("Country")
    plt.title("Top 10 Countries by Total COVID-19 Cases")
    save_plot(fig, "top_10_total_cases_bar.png", output_folder)

    fig, ax = plt.subplots(figsize=(12,6))
    ax.hlines(y=top_cases["Name"], xmin=0, xmax=top_cases["Cases_cumulative_total"],
              color="skyblue", linewidth=2)
    ax.plot(top_cases["Cases_cumulative_total"], top_cases["Name"], "o", color="navy")
    ax.set_xlabel("Total Cases")
    ax.set_ylabel("Country")
    ax.set_title("Top 10 Countries by Total COVID-19 Cases (Lollipop)")
    save_plot(fig, "top_10_total_cases_lollipop.png", output_folder)

    # 2️⃣ Total deaths by WHO region
    region_deaths = df.groupby("WHO_Region", dropna=False)["Deaths_cumulative_total"].sum().sort_values(ascending=False)

    fig = plt.figure(figsize=(10,5))
    sns.barplot(
        x=region_deaths.index,
        y=region_deaths.values,
        palette="crest"
    )
    plt.ylabel("Total Deaths")
    plt.xlabel("WHO Region")
    plt.title("Total COVID-19 Deaths by WHO Region")
    plt.xticks(rotation=45)
    save_plot(fig, "total_deaths_by_region_bar.png", output_folder)

    # Grouped variant
    fig, ax = plt.subplots(figsize=(10,5))
    sns.barplot(
        x="WHO_Region",
        y="Deaths_cumulative_total",
        data=df,
        estimator=sum,
        palette="crest"
    )
    plt.ylabel("Total Deaths")
    plt.xlabel("WHO Region")
    plt.title("Total COVID-19 Deaths by WHO Region (Grouped)")
    plt.xticks(rotation=45)
    save_plot(fig, "total_deaths_by_region_grouped.png", output_folder)

    # 3️⃣ Top 10 countries by weekly case growth (%)
    top_growth = df[df["Weekly_Case_Growth_%"] > 0].sort_values("Weekly_Case_Growth_%", ascending=False).head(10)

    fig = plt.figure(figsize=(12,6))
    sns.barplot(
        data=top_growth,
        y="Name",
        x="Weekly_Case_Growth_%",
        palette="flare"
    )
    plt.xlabel("Weekly Case Growth (%)")
    plt.ylabel("Country")
    plt.title("Top 10 Countries by Weekly Case Growth (%)")
    save_plot(fig, "top_10_weekly_growth_bar.png", output_folder)

    fig = plt.figure(figsize=(12,6))
    sns.scatterplot(
        data=top_growth,
        x="Cases_cumulative_total",
        y="Weekly_Case_Growth_%",
        hue="Name",
        palette="tab10",
        s=100,
        legend=False
    )
    plt.xscale("log")
    plt.xlabel("Total Cases (log scale)")
    plt.ylabel("Weekly Case Growth (%)")
    plt.title("Top 10 Countries: Weekly Case Growth vs Total Cases")
    save_plot(fig, "top_10_weekly_growth_scatter.png", output_folder)

    # 4️⃣ Total cases vs deaths
    fig = plt.figure(figsize=(10,6))
    sns.scatterplot(
        data=df,
        x="Cases_cumulative_total",
        y="Deaths_cumulative_total",
        hue="WHO_Region",
        palette="tab10",
        s=80,
        alpha=0.8
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Total Cases (log scale)")
    plt.ylabel("Total Deaths (log scale)")
    plt.title("Total Cases vs Deaths by Country")
    plt.legend(title="WHO Region", bbox_to_anchor=(1.05,1), loc="upper left")
    save_plot(fig, "cases_vs_deaths_scatter.png", output_folder)

    fig = plt.figure(figsize=(10,6))
    sns.regplot(
        data=df,
        x="Cases_cumulative_total",
        y="Deaths_cumulative_total",
        scatter_kws={"s":50, "alpha":0.5},
        line_kws={"color":"red"}
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Total Cases (log scale)")
    plt.ylabel("Total Deaths (log scale)")
    plt.title("Total Cases vs Deaths with Trend Line")
    save_plot(fig, "cases_vs_deaths_trend.png", output_folder)

    # 5️⃣ Top 10 countries by case fatality rate (%)
    top_fatality = df[df["Case_Fatality_Rate"] > 0].sort_values("Case_Fatality_Rate", ascending=False).head(10)

    fig = plt.figure(figsize=(12,6))
    sns.barplot(
        data=top_fatality,
        y="Name",
        x="Case_Fatality_Rate",
        palette="rocket"
    )
    plt.xlabel("Case Fatality Rate (%)")
    plt.ylabel("Country")
    plt.title("Top 10 Countries by Case Fatality Rate (%)")
    save_plot(fig, "top_10_case_fatality_bar.png", output_folder)

    fig, ax = plt.subplots(figsize=(12,6))
    ax.hlines(y=top_fatality["Name"], xmin=0, xmax=top_fatality["Case_Fatality_Rate"], color="salmon", linewidth=2)
    ax.plot(top_fatality["Case_Fatality_Rate"], top_fatality["Name"], "o", color="red")
    ax.set_xlabel("Case Fatality Rate (%)")
    ax.set_ylabel("Country")
    ax.set_title("Top 10 Countries by Case Fatality Rate (%) (Lollipop)")
    save_plot(fig, "top_10_case_fatality_lollipop.png", output_folder)

    print(f"✅ Finished visualizations for {label} dataset\n")

# ---------------------------
# Run for both datasets
# ---------------------------
generate_visualizations(df_with, output_with_global, label="WITH Global")
generate_visualizations(df_without, output_without_global, label="WITHOUT Global")

print("🎉 All visualizations generated and saved successfully!")
