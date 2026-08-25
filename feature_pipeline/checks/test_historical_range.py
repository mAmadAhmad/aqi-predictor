from datetime import date, timedelta

import pandas as pd
import requests

from feature_pipeline.constants import CITIES

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
VARIABLES = ["pm10", "pm2_5", "us_aqi"]


def fetch_window(start_date: str, end_date: str) -> pd.DataFrame:
    lat, lon = CITIES["islamabad"]
    response = requests.get(
        AIR_QUALITY_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(VARIABLES),
            "start_date": start_date,
            "end_date": end_date,
        },
        timeout=30,
    )
    response.raise_for_status()
    hourly = response.json()["hourly"]
    df = pd.DataFrame({"time": pd.to_datetime(hourly["time"])})
    for var in VARIABLES:
        df[var] = hourly[var]
    return df


def report(label: str, start_date: str, end_date: str) -> None:
    print(f"\n{'─' * 50}")
    print(f"  {label}")
    print(f"  {start_date} → {end_date}")
    print(f"{'─' * 50}")
    df = fetch_window(start_date, end_date)
    print(f"  Rows: {len(df)}")
    nan_counts = df[VARIABLES].isna().sum()
    for var, count in nan_counts.items():
        print(f"  NaN {var:<8}: {count} / {len(df)}")
    print(df.head(3).to_string(index=False))


def main() -> None:
    today = date.today()

    windows = [
        ("6 months ago", today - timedelta(days=182), today - timedelta(days=179)),
        ("2 years ago",  today - timedelta(days=730), today - timedelta(days=727)),
        ("2022-09-01 (just after documented global data start)", date(2022, 9, 1), date(2022, 9, 3)),
    ]

    for label, start, end in windows:
        report(label, start.isoformat(), end.isoformat())

    print(f"\n{'─' * 50}")


if __name__ == "__main__":
    main()
