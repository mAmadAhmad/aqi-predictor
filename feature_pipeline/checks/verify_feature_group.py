from feature_pipeline.hopsworks_utils import get_feature_store

FEATURE_GROUP_NAME = "islamabad_aqi_features"
FEATURE_GROUP_VERSION = 1


def main() -> None:
    fs = get_feature_store()
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    df = fg.read()
    print(f"Row count: {len(df)}")
    print(df.head(3))
    print(df["time"].min(), df["time"].max())
    print(df["time"].duplicated().sum())


if __name__ == "__main__":
    main()
