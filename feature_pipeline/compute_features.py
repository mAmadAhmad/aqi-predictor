from pathlib import Path

import pandas as pd

AQ_PATH = Path("data/islamabad_aq_raw.csv")
WEATHER_PATH = Path("data/islamabad_weather_raw.csv")
OUTPUT_PATH = Path("data/islamabad_features_sample.csv")


def compute_features(aq_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    aq = aq_df.copy()
    weather = weather_df.copy()

    aq["time"] = pd.to_datetime(aq["time"])
    weather["time"] = pd.to_datetime(weather["time"])

    aq_rows, weather_rows = len(aq), len(weather)
    df = aq.merge(weather, on="time", how="inner")
    merged_rows = len(df)

    print(f"aq rows={aq_rows}, weather rows={weather_rows}, merged rows={merged_rows}")
    if merged_rows < min(aq_rows, weather_rows):
        print(
            f"WARNING: merged row count ({merged_rows}) is less than the smaller "
            f"input ({min(aq_rows, weather_rows)}). Possible time alignment problem."
        )

    pollutants = ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone"]
    df[pollutants] = df[pollutants].clip(lower=0)

    df["hour"] = df["time"].dt.hour
    df["day"] = df["time"].dt.day
    df["month"] = df["time"].dt.month
    df["day_of_week"] = df["time"].dt.dayofweek

    df["us_aqi_lag_24h"] = df["us_aqi"].shift(24)
    df["us_aqi_lag_168h"] = df["us_aqi"].shift(168)
    print(f"us_aqi_lag_24h NaNs: {df['us_aqi_lag_24h'].isna().sum()}")
    print(f"us_aqi_lag_168h NaNs: {df['us_aqi_lag_168h'].isna().sum()}")

    time_delta = df["time"].diff()
    exactly_one_hour = time_delta == pd.Timedelta(hours=1)
    aqi_diff = df["us_aqi"].diff()
    df["aqi_change_rate"] = aqi_diff.where(exactly_one_hour)
    nan_count = df["aqi_change_rate"].isna().sum()
    print(f"aqi_change_rate NaNs (gap ≠ 1 h or no previous row): {nan_count}")

    return df


def main() -> None:
    aq_df = pd.read_csv(AQ_PATH)
    weather_df = pd.read_csv(WEATHER_PATH)

    df = compute_features(aq_df, weather_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df)} rows → {OUTPUT_PATH}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
