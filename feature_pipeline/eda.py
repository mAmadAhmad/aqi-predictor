from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns

DATA_PATH = Path(__file__).parent.parent / "data" / "islamabad_features_backfill.csv"
OUT_DIR = Path(__file__).parent.parent / "data" / "eda_outputs"


def _save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}")


def print_summary(df: pd.DataFrame) -> None:
    duplicates = df["time"].duplicated().sum()
    out_of_order = (df["time"].diff().dropna() < pd.Timedelta(0)).sum()
    print(f"Rows       : {len(df)}")
    print(f"Date range : {df['time'].min()} → {df['time'].max()}")
    print(f"Duplicate timestamps : {duplicates}")
    print(f"Out-of-order timestamps : {out_of_order}")


def plot_timeseries(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(18, 4))
    ax.plot(df["time"], df["us_aqi"], linewidth=0.6, color="#e07b39", alpha=0.9)

    data_start, data_end = df["time"].min(), df["time"].max()
    years = range(data_start.year, data_end.year + 2)
    for year in years:
        oct_start = pd.Timestamp(year, 10, 1)
        feb_end = pd.Timestamp(year + 1, 3, 1)
        if oct_start >= data_end or feb_end <= data_start:
            continue
        ax.axvspan(
            max(oct_start, data_start),
            min(feb_end, data_end),
            color="#4a90d9",
            alpha=0.08,
        )

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.set_title("US AQI — Full Time Series (shaded: Oct–Feb)", fontsize=13)
    ax.set_xlabel("Time")
    ax.set_ylabel("US AQI")
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    _save(fig, "eda_aqi_timeseries.png")


def plot_monthly_boxplot(df: pd.DataFrame) -> None:
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig, ax = plt.subplots(figsize=(12, 5))
    data_by_month = [df.loc[df["month"] == m, "us_aqi"].dropna().values for m in range(1, 13)]
    bp = ax.boxplot(data_by_month, patch_artist=True, medianprops={"color": "white", "linewidth": 1.5})
    for patch in bp["boxes"]:
        patch.set_facecolor("#4a90d9")
        patch.set_alpha(0.75)
    ax.set_xticklabels(month_labels)
    ax.set_title("US AQI Distribution by Calendar Month (all years pooled)", fontsize=13)
    ax.set_xlabel("Month")
    ax.set_ylabel("US AQI")
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    _save(fig, "eda_aqi_monthly_boxplot.png")


def plot_by_hour(df: pd.DataFrame) -> None:
    hourly = df.groupby("hour")["us_aqi"].mean()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(hourly.index, hourly.values, color="#e07b39", alpha=0.85)
    ax.set_title("Average US AQI by Hour of Day", fontsize=13)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Mean US AQI")
    ax.set_xticks(range(24))
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    _save(fig, "eda_aqi_by_hour.png")


def plot_by_dow(df: pd.DataFrame) -> None:
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    daily = df.groupby("day_of_week")["us_aqi"].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(daily.index, daily.values, color="#4a90d9", alpha=0.85)
    ax.set_title("Average US AQI by Day of Week", fontsize=13)
    ax.set_xlabel("Day")
    ax.set_ylabel("Mean US AQI")
    ax.set_xticks(range(7))
    ax.set_xticklabels(dow_labels)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    _save(fig, "eda_aqi_by_dow.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    numeric = df.select_dtypes(include="number")
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.4,
        annot_kws={"size": 7},
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=13)
    fig.tight_layout()
    _save(fig, "eda_correlation_heatmap.png")


def print_low_aqi_anomalies(df: pd.DataFrame) -> None:
    low = df[df["us_aqi"] < 30].sort_values("time")
    print(f"Rows with us_aqi < 30: {len(low)}")
    for _, row in low.iterrows():
        pos = df.index.get_loc(row.name)
        before = df.iloc[max(0, pos - 3) : pos][["time", "us_aqi"]]
        after = df.iloc[pos + 1 : pos + 4][["time", "us_aqi"]]
        print(f"\n  [{row['time']}] us_aqi={row['us_aqi']}")
        print("    before:")
        for _, r in before.iterrows():
            print(f"      {r['time']}  us_aqi={r['us_aqi']}")
        print("    after:")
        for _, r in after.iterrows():
            print(f"      {r['time']}  us_aqi={r['us_aqi']}")


def print_physical_validity(df: pd.DataFrame) -> None:
    out_of_range_aqi = ((df["us_aqi"] < 0) | (df["us_aqi"] > 500)).sum()
    pollutants = ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone"]
    neg_mask = (df[pollutants] < 0).any(axis=1)
    neg_pollutant_rows = neg_mask.sum()
    neg_precip_rows = (df["precipitation"] < 0).sum()
    print(f"us_aqi outside 0–500        : {out_of_range_aqi}")
    print(f"Rows with any negative pollutant : {neg_pollutant_rows}")
    print(f"Rows with negative precipitation : {neg_precip_rows}")
    if neg_pollutant_rows > 0:
        inspect_cols = ["time"] + pollutants + ["us_aqi"]
        print("\nNegative-pollutant rows (full detail):")
        print(df.loc[neg_mask, inspect_cols].to_string(index=False))


def plot_pm25_vs_aqi_by_hour(df: pd.DataFrame) -> None:
    hourly_pm25 = df.groupby("hour")["pm2_5"].mean()
    hourly_aqi = df.groupby("hour")["us_aqi"].mean()
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax2 = ax1.twinx()
    ax1.plot(hourly_pm25.index, hourly_pm25.values, color="#4a90d9", marker="o", markersize=4, label="Mean PM2.5")
    ax2.plot(hourly_aqi.index, hourly_aqi.values, color="#e07b39", marker="s", markersize=4, label="Mean US AQI")
    ax1.set_xlabel("Hour of Day")
    ax1.set_ylabel("Mean PM2.5 (µg/m³)", color="#4a90d9")
    ax2.set_ylabel("Mean US AQI", color="#e07b39")
    ax1.set_xticks(range(24))
    ax1.tick_params(axis="y", labelcolor="#4a90d9")
    ax2.tick_params(axis="y", labelcolor="#e07b39")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
    ax1.set_title("Mean PM2.5 and Mean US AQI by Hour of Day", fontsize=13)
    ax1.grid(axis="y", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    _save(fig, "eda_pm25_vs_aqi_by_hour.png")


def plot_aqi_by_year(df: pd.DataFrame) -> None:
    yearly = df.groupby(df["time"].dt.year)["us_aqi"].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(yearly.index.astype(str), yearly.values, color="#4a90d9", alpha=0.85)
    ax.set_title("Mean US AQI by Calendar Year", fontsize=13)
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean US AQI")
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    _save(fig, "eda_aqi_by_year.png")


def main() -> None:
    df = pd.read_csv(DATA_PATH, parse_dates=["time"])

    print("=== Summary ===")
    print_summary(df)
    print()

    print("=== Anomaly investigation: us_aqi < 30 ===")
    print_low_aqi_anomalies(df)
    print()

    print("=== Physical validity checks ===")
    print_physical_validity(df)
    print()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Generating plots ===")
    plot_timeseries(df)
    plot_monthly_boxplot(df)
    plot_by_hour(df)
    plot_by_dow(df)
    plot_correlation_heatmap(df)
    plot_pm25_vs_aqi_by_hour(df)
    plot_aqi_by_year(df)
    print(f"\nAll outputs saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
