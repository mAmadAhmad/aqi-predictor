from pathlib import Path

import pandas as pd

from hopsworks_utils import get_feature_store

DATA_PATH = Path(__file__).parent.parent / "data" / "islamabad_features_sample.csv"
FEATURE_GROUP_NAME = "islamabad_aqi_features"
FEATURE_GROUP_VERSION = 1
PRIMARY_KEY = ["time"]
EVENT_TIME = "time"


def main() -> None:
    df = pd.read_csv(DATA_PATH, parse_dates=[EVENT_TIME])

    fs = get_feature_store()

    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        primary_key=PRIMARY_KEY,
        event_time=EVENT_TIME,
        online_enabled=False,
        time_travel_format="HUDI",
        statistics_config=False,
    )

    fg.statistics_config = False
    fg.update_statistics_config()

    fg.insert(df, write_options={"wait_for_job": False})

    print(f"Wrote {len(df)} rows to feature group '{FEATURE_GROUP_NAME}' v{FEATURE_GROUP_VERSION}")
    print(f"Online enabled: {fg.online_enabled}")


if __name__ == "__main__":
    main()
