# ==========================================================
# visualize.py
# Generates WHO COVID-19 visualizations (enhanced) and saves as PNGs
# ==========================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---------------------------
# Configuration
# ---------------------------
processed_csv = Path("data/processed/WHO-COVID-19-global-table-data.csv")
output_folder = Path("src/visualization/WHO_COVID19")
output_folder.mkdir(parents=True, exist_ok=True)

print(f"📂 Output folder: {output_folder.resolve()}")
print(f"📄 Loading processed dataset from: {processed_csv.resolve()}")

# ---------------------------
# Load processed dataset
# ---------------------------
try:
    df = pd.read_csv(processed_csv)
    print(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns")
except FileNotFoundError:
    print(f"❌ Processed CSV not found: {processed_csv}")
    exit(1)

# ---------------------------
# Helper function to save plots
# ---------------------------
def save_plot(fig, filename):
    path = output_folder / filename
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"💾 Saved plot: {path.name}")

# ---------------------------
# Ensure derived metrics exist
# ---------------------------
# Weekly Case Growth (%)
if "Weekly_Case_Growth_%" not in df.columns or df["Weekly_Case_Growth_%"].sum() == 0:
    df["Weekly_Case_Growth_%"] = (df["Cases_newly_reported_in_last_7_days"] /
                                  df["Cases_cumulative_total"]) * 100
df["Weekly_Case_Growth_%"] = df["Weekly_Case_Growth_%"].fillna(0)

# Case Fatality Rate (%)
if "Case_Fatality_Rate" not in df.columns or df["Case_Fatality_Rate"].sum() == 0:
    df["Case_Fatality_Rate"] = (df["Deaths_cumulative_total"] /
                                df["Cases_cumulative_total"]) * 100
df["Case_Fatality_Rate"] = df["Case_Fatality_Rate"].fillna(0)

# ---------------------------
# 1️⃣ Top 10 countries by total cases
# ---------------------------
print("\n[1/5] Top 10 Countries by Total COVID-19 Cases")

top_cases = df.sort_values("Cases_cumulative_total", ascending=False).head(10)

# Horizontal bar chart
fig = plt.figure(figsize=(12,6))
sns.barplot(
    data=top_cases,
    y="Name",
    x="Cases_cumulative_total",
    palette=sns.color_palette("coolwarm", len(top_cases))
)
plt.xlabel("Total Cases")
plt.ylabel("Country")
plt.title("Top 10 Countries by Total COVID-19 Cases")
save_plot(fig, "top_10_total_cases_bar.png")

# Lollipop chart variant
fig, ax = plt.subplots(figsize=(12,6))
ax.hlines(y=top_cases["Name"], xmin=0, xmax=top_cases["Cases_cumulative_total"], color="skyblue", linewidth=2)
ax.plot(top_cases["Cases_cumulative_total"], top_cases["Name"], "o", color="blue")
ax.set_xlabel("Total Cases")
ax.set_ylabel("Country")
ax.set_title("Top 10 Countries by Total COVID-19 Cases (Lollipop)")
save_plot(fig, "top_10_total_cases_lollipop.png")

# ---------------------------
# 2️⃣ Total deaths by WHO region
# ---------------------------
print("\n[2/5] Total Deaths by WHO Region (excluding Global)")

# Exclude global row
df_region = df[df["Name"] != "Global"].copy()
region_deaths = df_region.groupby("WHO_Region")["Deaths_cumulative_total"].sum().sort_values(ascending=False)

# Horizontal bar chart
fig = plt.figure(figsize=(10,5))
sns.barplot(
    x=region_deaths.index,
    y=region_deaths.values,
    palette=sns.color_palette("viridis", len(region_deaths))
)
plt.ylabel("Total Deaths")
plt.xlabel("WHO Region")
plt.title("Total COVID-19 Deaths by WHO Region")
plt.xticks(rotation=45)
save_plot(fig, "total_deaths_by_region_bar.png")

# Grouped bar chart variant (optional: show "Unknown" contribution)
fig, ax = plt.subplots(figsize=(10,5))
sns.barplot(
    x="WHO_Region",
    y="Deaths_cumulative_total",
    data=df_region,
    palette="viridis",
    estimator=sum
)
plt.ylabel("Total Deaths")
plt.xlabel("WHO Region")
plt.title("Total COVID-19 Deaths by WHO Region (Grouped)")
plt.xticks(rotation=45)
save_plot(fig, "total_deaths_by_region_grouped.png")

# ---------------------------
# 3️⃣ Top 10 countries by weekly case growth (%)
# ---------------------------
print("\n[3/5] Top 10 Countries by Weekly Case Growth (%)")

top_growth = df[df["Weekly_Case_Growth_%"] > 0].sort_values("Weekly_Case_Growth_%", ascending=False).head(10)

# Horizontal bar chart
fig = plt.figure(figsize=(12,6))
sns.barplot(
    data=top_growth,
    y="Name",
    x="Weekly_Case_Growth_%",
    palette=sns.color_palette("plasma", len(top_growth))
)
plt.xlabel("Weekly Case Growth (%)")
plt.ylabel("Country")
plt.title("Top 10 Countries by Weekly Case Growth (%)")
save_plot(fig, "top_10_weekly_growth_bar.png")

# Scatter: Total cases vs weekly growth
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
save_plot(fig, "top_10_weekly_growth_scatter.png")

# ---------------------------
# 4️⃣ Total cases vs deaths
# ---------------------------
print("\n[4/5] Relationship Between Total Cases and Deaths")

# Scatter plot
fig = plt.figure(figsize=(10,6))
sns.scatterplot(
    data=df,
    x="Cases_cumulative_total",
    y="Deaths_cumulative_total",
    hue="WHO_Region",
    palette="tab10",
    s=100,
    alpha=0.7
)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Total Cases (log scale)")
plt.ylabel("Total Deaths (log scale)")
plt.title("Total Cases vs Deaths by Country")
plt.legend(title="WHO Region", bbox_to_anchor=(1.05,1), loc="upper left")
save_plot(fig, "cases_vs_deaths_scatter.png")

# Scatter + trend line
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
save_plot(fig, "cases_vs_deaths_trend.png")

# ---------------------------
# 5️⃣ Top 10 countries by case fatality rate (%)
# ---------------------------
print("\n[5/5] Top 10 Countries by Case Fatality Rate (%)")

top_fatality = df[df["Case_Fatality_Rate"] > 0].sort_values("Case_Fatality_Rate", ascending=False).head(10)

# Horizontal bar chart
fig = plt.figure(figsize=(12,6))
sns.barplot(
    data=top_fatality,
    y="Name",
    x="Case_Fatality_Rate",
    palette=sns.color_palette("magma", len(top_fatality))
)
plt.xlabel("Case Fatality Rate (%)")
plt.ylabel("Country")
plt.title("Top 10 Countries by Case Fatality Rate (%)")
save_plot(fig, "top_10_case_fatality_bar.png")

# Lollipop chart variant
fig, ax = plt.subplots(figsize=(12,6))
ax.hlines(y=top_fatality["Name"], xmin=0, xmax=top_fatality["Case_Fatality_Rate"], color="pink", linewidth=2)
ax.plot(top_fatality["Case_Fatality_Rate"], top_fatality["Name"], "o", color="red")
ax.set_xlabel("Case Fatality Rate (%)")
ax.set_ylabel("Country")
ax.set_title("Top 10 Countries by Case Fatality Rate (%) (Lollipop)")
save_plot(fig, "top_10_case_fatality_lollipop.png")

# ---------------------------
# Done
# ---------------------------
print("\n🎉 All visualizations generated and saved successfully!")
