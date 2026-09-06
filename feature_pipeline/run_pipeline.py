"""
Hourly feature ingestion. Fetches the last 400 h from Open-Meteo, computes
features, and upserts into the Hopsworks feature group.

400 h rather than 168 h (the longest lag) gives the pipeline enough runway
to stay correct through API gaps or a missed hourly run.
"""
import argparse
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from feature_pipeline.compute_features import compute_features
from feature_pipeline.constants import CITIES
from feature_pipeline.hopsworks_utils import get_feature_store

AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
AQ_VARIABLES = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "us_aqi",
]
WEATHER_VARIABLES = ["temperature_2m", "precipitation"]
WINDOW_HOURS = 400
FEATURE_GROUP_NAME = "islamabad_aqi_features"
FEATURE_GROUP_VERSION = 1


def _window() -> tuple[datetime, datetime]:
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(hours=WINDOW_HOURS)
    return start, end


def _fetch_aq(lat: float, lon: float, start: datetime, end: datetime) -> pd.DataFrame:
    fmt = "%Y-%m-%dT%H:%M"
    response = requests.get(
        AQ_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(AQ_VARIABLES),
            "start_hour": start.strftime(fmt),
            "end_hour": end.strftime(fmt),
        },
        timeout=60,
    )
    response.raise_for_status()
    hourly = response.json()["hourly"]
    df = pd.DataFrame({"time": pd.to_datetime(hourly["time"])})
    for var in AQ_VARIABLES:
        df[var] = hourly[var]
    return df


def _fetch_weather(lat: float, lon: float, start: datetime, end: datetime) -> pd.DataFrame:
    response = requests.get(
        WEATHER_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(WEATHER_VARIABLES),
            "start_date": start.date().isoformat(),
            "end_date": end.date().isoformat(),
        },
        timeout=60,
    )
    response.raise_for_status()
    hourly = response.json()["hourly"]
    df = pd.DataFrame({"time": pd.to_datetime(hourly["time"])})
    for var in WEATHER_VARIABLES:
        df[var] = hourly[var]
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true", help="Insert result into Hopsworks feature group")
    args = parser.parse_args()

    lat, lon = CITIES["islamabad"]
    start, end = _window()

    print(f"Window: {start} → {end}")

    print("Fetching AQ data...")
    aq_df = _fetch_aq(lat, lon, start, end)

    print("Fetching weather data...")
    weather_df = _fetch_weather(lat, lon, start, end)

    df = compute_features(aq_df, weather_df)
    df = df.dropna(subset=["us_aqi_lag_24h", "us_aqi_lag_168h"]).reset_index(drop=True)

    print(f"\n{len(df)} rows ready, columns: {list(df.columns)}")

    if args.push:
        fs = get_feature_store()
        fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
        fg.insert(df, write_options={"wait_for_job": False})
        print(f"Pushed {len(df)} rows to '{FEATURE_GROUP_NAME}' v{FEATURE_GROUP_VERSION}")


if __name__ == "__main__":
    main()
