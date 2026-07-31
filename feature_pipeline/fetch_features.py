from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from feature_pipeline.constants import CITIES

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
HOURLY_VARIABLES = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "us_aqi",
]
OUTPUT_PATH = Path("data/islamabad_aq_raw.csv")


def _hour_range() -> tuple[str, str]:
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = now - timedelta(hours=24)
    fmt = "%Y-%m-%dT%H:%M"
    return start.strftime(fmt), now.strftime(fmt)


def fetch_islamabad_air_quality() -> pd.DataFrame:
    lat, lon = CITIES["islamabad"]
    start_hour, end_hour = _hour_range()

    response = requests.get(
        AIR_QUALITY_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(HOURLY_VARIABLES),
            "start_hour": start_hour,
            "end_hour": end_hour,
        },
        timeout=30,
    )
    response.raise_for_status()
    hourly = response.json()["hourly"]

    df = pd.DataFrame({"time": pd.to_datetime(hourly["time"])})
    for var in HOURLY_VARIABLES:
        df[var] = hourly[var]
    return df


def main() -> None:
    df = fetch_islamabad_air_quality()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} rows → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
