import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from feature_pipeline.hopsworks_utils import get_project

FEATURE_GROUP_NAME = "islamabad_aqi_features"
FEATURE_GROUP_VERSION = 1
TEST_DAYS = 30
RESULTS_PATH = Path(__file__).parent.parent / "data" / "training_results.csv"
MODELS_DIR = Path(__file__).parent.parent / "data" / "models"

FEATURE_COLS = [
    "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
    "sulphur_dioxide", "ozone", "us_aqi",
    "temperature_2m", "precipitation",
    "hour", "day", "month", "day_of_week",
    "us_aqi_lag_24h", "us_aqi_lag_168h", "aqi_change_rate",
]

DAILY_TARGETS = ["avg_aqi_next_24h", "avg_aqi_24_48h", "avg_aqi_48_72h"]
HOURLY_TARGETS = [f"us_aqi_h{h}" for h in range(1, 25)]


def _load_from_hopsworks(project: Any) -> pd.DataFrame:
    fg = project.get_feature_store().get_feature_group(
        name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION
    )
    df = fg.read()
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True)


def _compute_targets(df: pd.DataFrame) -> pd.DataFrame:
    aqi = df["us_aqi"]

    for h in range(1, 25):
        df[f"us_aqi_h{h}"] = aqi.shift(-h)

    # Rolling mean on reversed series avoids materialising 24/48/72 arrays at once.
    aqi_rev = aqi[::-1]
    df["avg_aqi_next_24h"] = aqi_rev.rolling(24).mean().shift(1)[::-1].values
    df["avg_aqi_24_48h"]   = aqi_rev.rolling(24).mean().shift(25)[::-1].values
    df["avg_aqi_48_72h"]   = aqi_rev.rolling(24).mean().shift(49)[::-1].values

    return df


def _split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = df["time"].max() - pd.Timedelta(days=TEST_DAYS)
    train = df[df["time"] < cutoff].copy()
    test = df[df["time"] >= cutoff]
    return train, test


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _persistence_metrics(test: pd.DataFrame, target: str) -> dict[str, float]:
    valid = test[["us_aqi", target]].dropna()
    return _metrics(valid[target].values, valid["us_aqi"].values)


def _train_daily(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
) -> tuple[list[dict], dict[str, Any]]:
    rows = []
    tr = train[FEATURE_COLS + [target]].dropna()
    te = test[FEATURE_COLS + [target]].dropna()

    X_tr, y_tr = tr[FEATURE_COLS].values, tr[target].values
    X_te, y_te = te[FEATURE_COLS].values, te[target].values

    candidates = [
        ("RandomForest", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=2)),
        ("Ridge", Ridge(alpha=1.0)),
        ("XGBoost", XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6,
                                  random_state=42, verbosity=0, n_jobs=2)),
    ]
    fitted: dict[str, Any] = {}
    for name, model in candidates:
        model.fit(X_tr, y_tr)
        fitted[name] = model
        m = _metrics(y_te, model.predict(X_te))
        rows.append({"target": target, "model": name, **m})

    pm = _persistence_metrics(test, target)
    rows.append({"target": target, "model": "Persistence", **pm})
    return rows, fitted


def _train_hourly(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[list[dict], Any]:
    rows = []
    all_cols = FEATURE_COLS + HOURLY_TARGETS
    tr = train[all_cols].dropna()
    te = test[all_cols].dropna()

    X_tr, Y_tr = tr[FEATURE_COLS].values, tr[HOURLY_TARGETS].values
    X_te, Y_te = te[FEATURE_COLS].values, te[HOURLY_TARGETS].values

    # RF handles multi-output natively — no wrapper needed, no 24 separate fits.
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=2)
    model.fit(X_tr, Y_tr)
    Y_pred = model.predict(X_te)

    for i, target in enumerate(HOURLY_TARGETS):
        m = _metrics(Y_te[:, i], Y_pred[:, i])
        rows.append({"target": target, "model": "MultiOutputRF", **m})

    for i, target in enumerate(HOURLY_TARGETS):
        pm = _persistence_metrics(test, target)
        rows.append({"target": target, "model": "Persistence", **pm})

    return rows, model


def _pick_best(results_df: pd.DataFrame, target: str) -> tuple[str, dict[str, float]]:
    candidate_models = ["RandomForest", "Ridge", "XGBoost"]
    subset = results_df[
        (results_df["target"] == target) & (results_df["model"].isin(candidate_models))
    ]
    best_row = subset.loc[subset["rmse"].idxmin()]
    return best_row["model"], {"rmse": best_row["rmse"], "mae": best_row["mae"], "r2": best_row["r2"]}


def _assert_beats_persistence(
    model_name: str, model_rmse: float, baseline_rmse: float, target: str
) -> None:
    assert model_rmse < baseline_rmse, (
        f"[REGRESSION] {model_name} RMSE {model_rmse:.3f} does NOT beat "
        f"persistence baseline {baseline_rmse:.3f} for target '{target}'. "
        "Model is not better than naive — do not push to registry."
    )


def _save_and_push(
    model: Any,
    registry_name: str,
    local_filename: str,
    metrics: dict[str, float],
    project: Any,
) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    local_path = MODELS_DIR / local_filename
    joblib.dump(model, local_path)

    mr = project.get_model_registry()
    hw_model = mr.python.create_model(name=registry_name, metrics=metrics)
    hw_model.save(str(local_path))
    print(f"  pushed '{registry_name}' -> version {hw_model.version}  (RMSE={metrics['rmse']:.3f})")


def _print_table(results: pd.DataFrame) -> None:
    print(f"\n{'target':<22} {'model':<16} {'RMSE':>8} {'MAE':>8} {'R2':>8}")
    print("-" * 66)
    for _, r in results.iterrows():
        print(f"{r['target']:<22} {r['model']:<16} {r['rmse']:>8.2f} {r['mae']:>8.2f} {r['r2']:>8.3f}")


def _verify(df: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame) -> None:
    assert train["time"].max() < test["time"].min(), "Train/test time overlap"
    assert not any(c in FEATURE_COLS for c in DAILY_TARGETS + HOURLY_TARGETS), "Target leaked into features"
    assert train.index.max() < test.index.min(), "Index ordering violated"

    pm_day1 = _persistence_metrics(test, "avg_aqi_next_24h")["r2"]
    pm_day3 = _persistence_metrics(test, "avg_aqi_48_72h")["r2"]
    print(f"\n[check] Persistence R2: day1={pm_day1:.3f}, day3={pm_day3:.3f}", end="")
    if pm_day3 < pm_day1:
        print(" (day3 < day1 as expected)")
    else:
        print(" WARNING: day3 R2 >= day1 -- inspect target computation")

    print(f"[check] Train rows: {len(train)}, Test rows: {len(test)}")
    print(f"[check] Train ends: {train['time'].max()}  |  Test starts: {test['time'].min()}")
    print(f"[check] No shuffle: {'yes' if train['time'].max() < test['time'].min() else 'NO - BROKEN'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", metavar="CSV", help="Use local CSV instead of Hopsworks (for dry-run)")
    args = parser.parse_args()

    project = None
    if args.local:
        print(f"Loading from local CSV: {args.local}")
        df = pd.read_csv(args.local, parse_dates=["time"])
        df = df.sort_values("time").reset_index(drop=True)
    else:
        print("Connecting to Hopsworks...")
        project = get_project()
        df = _load_from_hopsworks(project)

    print(f"Loaded {len(df)} rows | {df['time'].min()} --> {df['time'].max()}")

    df["aqi_change_rate"] = df["aqi_change_rate"].fillna(0)
    df = df.dropna(subset=["us_aqi_lag_24h", "us_aqi_lag_168h"]).reset_index(drop=True)

    df = _compute_targets(df)
    df = df.dropna(subset=["avg_aqi_48_72h"]).reset_index(drop=True)

    train, test = _split(df)
    _verify(df, train, test)
    del df  # project object is independent — registry push unaffected

    print("\nTraining daily-average models...")
    results = []
    fitted_daily: dict[str, dict[str, Any]] = {}
    for target in DAILY_TARGETS:
        print(f"  {target}")
        rows, fitted = _train_daily(train, test, target)
        results.extend(rows)
        fitted_daily[target] = fitted

    print("Training hourly multi-output model...")
    hourly_rows, hourly_model = _train_hourly(train, test)
    results.extend(hourly_rows)

    results_df = pd.DataFrame(results)

    print("\n--- Daily-average targets ---")
    _print_table(results_df[results_df["target"].isin(DAILY_TARGETS)])

    print("\n--- Hourly targets (MultiOutputRF vs Persistence, averaged over h1-h24) ---")
    hourly_summary = (
        results_df[results_df["target"].isin(HOURLY_TARGETS)]
        .groupby("model")[["rmse", "mae", "r2"]]
        .mean()
        .round(3)
    )
    print(hourly_summary.to_string())

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\nFull results saved --> {RESULTS_PATH}")

    # --- Select best, assert beats persistence, save + push ---
    print("\nSelecting best models and validating against persistence baseline...")

    hourly_rf_rows = results_df[
        results_df["target"].isin(HOURLY_TARGETS) & (results_df["model"] == "MultiOutputRF")
    ]
    hourly_persistence_rows = results_df[
        results_df["target"].isin(HOURLY_TARGETS) & (results_df["model"] == "Persistence")
    ]
    multorf_rmse = hourly_rf_rows["rmse"].mean()
    persistence_rmse_hourly = hourly_persistence_rows["rmse"].mean()
    _assert_beats_persistence("MultiOutputRF", multorf_rmse, persistence_rmse_hourly, "hourly")

    best_models: list[tuple[str, str, Any, dict[str, float]]] = []
    for target in DAILY_TARGETS:
        best_name, best_metrics = _pick_best(results_df, target)
        persistence_rmse = results_df[
            (results_df["target"] == target) & (results_df["model"] == "Persistence")
        ]["rmse"].iloc[0]
        _assert_beats_persistence(best_name, best_metrics["rmse"], persistence_rmse, target)
        best_models.append((target, best_name, fitted_daily[target][best_name], best_metrics))
        print(f"  {target}: best={best_name}  RMSE={best_metrics['rmse']:.3f} < persistence {persistence_rmse:.3f} ✓")

    hourly_metrics = {
        "rmse": round(multorf_rmse, 4),
        "mae": round(hourly_rf_rows["mae"].mean(), 4),
        "r2": round(hourly_rf_rows["r2"].mean(), 4),
    }
    print(f"  hourly: MultiOutputRF  RMSE={multorf_rmse:.3f} < persistence {persistence_rmse_hourly:.3f} ✓")

    if project is None:
        print("\n[dry-run] Skipping joblib save and registry push (--local mode).")
        return

    print("\nSaving models and pushing to Hopsworks Model Registry...")
    for target, best_name, model_obj, metrics in best_models:
        registry_name = f"aqi_{target}_model"
        _save_and_push(model_obj, registry_name, f"{registry_name}.joblib", metrics, project)

    _save_and_push(
        hourly_model, "aqi_hourly_model", "aqi_hourly_model.joblib", hourly_metrics, project
    )


if __name__ == "__main__":
    main()
