"""
Historical backfill. Meant to be run once (or after a data gap).

Queries the feature group for its latest timestamp before fetching so reruns
are gap-fills rather than full re-ingestions from 2022.
"""
import argparse
from datetime import date, timedelta
from pathlib import Path

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
BACKFILL_START = date(2022, 9, 1)
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "islamabad_features_backfill.csv"
FEATURE_GROUP_NAME = "islamabad_aqi_features"
FEATURE_GROUP_VERSION = 1
MAX_RETRIES = 3


def _yearly_chunks(start: date, end: date) -> list[tuple[date, date]]:
    chunks = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(date(chunk_start.year + 1, chunk_start.month, chunk_start.day) - timedelta(days=1), end)
        chunks.append((chunk_start, chunk_end))
        chunk_start = chunk_end + timedelta(days=1)
    return chunks


def _fetch_with_retry(url: str, params: dict) -> dict:
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            print(f"  attempt {attempt}/{MAX_RETRIES} failed: {exc}")
            last_exc = exc
    raise RuntimeError(f"All {MAX_RETRIES} attempts failed for {url}") from last_exc


def _fetch_chunk(url: str, variables: list[str], lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    data = _fetch_with_retry(
        url,
        {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(variables),
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
        },
    )
    hourly = data["hourly"]
    df = pd.DataFrame({"time": pd.to_datetime(hourly["time"])})
    for var in variables:
        df[var] = hourly[var]
    return df


def _fetch_all_chunks(url: str, variables: list[str], lat: float, lon: float, start: date) -> pd.DataFrame:
    today = date.today()
    chunks = _yearly_chunks(start, today)
    frames = []
    for s, e in chunks:
        print(f"  fetching {s} → {e}")
        frames.append(_fetch_chunk(url, variables, lat, lon, s, e))
    df = pd.concat(frames).sort_values("time").drop_duplicates(subset="time", keep="first").reset_index(drop=True)

    full_range = pd.date_range(start=df["time"].min(), end=df["time"].max(), freq="h")
    missing = full_range.difference(df["time"])
    print(f"  rows={len(df)}, range={df['time'].min()} → {df['time'].max()}, missing hours={len(missing)}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true", help="Insert result into Hopsworks feature group")
    args = parser.parse_args()

    lat, lon = CITIES["islamabad"]

    effective_start = BACKFILL_START
    fg = None
    try:
        fs = get_feature_store()
        fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
        last_ts = fg.select(["time"]).read()["time"].max()
        if pd.notna(last_ts):
            effective_start = (pd.Timestamp(last_ts) + timedelta(hours=1)).date()
            print(f"Gap-fill from {effective_start} (last known: {last_ts})")
        else:
            print(f"Full backfill from {effective_start} (feature group is empty)")
    except Exception as exc:
        print(f"Could not query feature store ({exc.__class__.__name__}), using full backfill from {effective_start}")

    print("Fetching AQ data...")
    aq_df = _fetch_all_chunks(AQ_URL, AQ_VARIABLES, lat, lon, effective_start)

    print("Fetching weather data...")
    weather_df = _fetch_all_chunks(WEATHER_URL, WEATHER_VARIABLES, lat, lon, effective_start)

    df = compute_features(aq_df, weather_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df)} rows → {OUTPUT_PATH}")
    print(f"Columns: {list(df.columns)}")

    if args.push and fg is not None:
        fg.insert(df, write_options={"wait_for_job": False})
        print(f"Pushed {len(df)} rows to '{FEATURE_GROUP_NAME}' v{FEATURE_GROUP_VERSION}")
    elif args.push:
        print("Cannot push: Hopsworks connection failed during gap-detection.")


if __name__ == "__main__":
    main()
