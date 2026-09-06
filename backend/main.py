"""
FastAPI serving layer. Loads models from Hopsworks registry at startup (the SDK
caches them in /tmp, so subsequent restarts skip the download). Falls back to
data/fallback_models/ if the registry is unreachable.

The feature row is pulled directly from Open-Meteo rather than the feature store
to avoid scanning the full 35k-row history just to read the latest hour.
"""
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from feature_pipeline.hopsworks_utils import get_project

FEATURE_GROUP_NAME = "islamabad_aqi_features"
FEATURE_GROUP_VERSION = 1
MODELS_DIR = Path(__file__).parent.parent / "data" / "fallback_models"
SNAPSHOT_PATH = Path(__file__).parent.parent / "data" / "snapshot_obs.json"

LOCAL_MODEL_FILES = {
    "aqi_avg_aqi_next_24h_model": MODELS_DIR / "aqi_avg_aqi_next_24h_model.joblib",
    "aqi_avg_aqi_24_48h_model":   MODELS_DIR / "aqi_avg_aqi_24_48h_model.joblib",
    "aqi_avg_aqi_48_72h_model":   MODELS_DIR / "aqi_avg_aqi_48_72h_model.joblib",
}

FEATURE_COLS = [
    "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
    "sulphur_dioxide", "ozone", "us_aqi",
    "temperature_2m", "precipitation",
    "hour", "day", "month", "day_of_week",
    "us_aqi_lag_24h", "us_aqi_lag_168h", "aqi_change_rate",
]

DAILY_REGISTRY_NAMES = list(LOCAL_MODEL_FILES.keys())


def _download_model(mr: Any, name: str) -> tuple[Any, dict]:
    model_meta = mr.get_model(name=name, version=None)
    if model_meta is None:
        raise RuntimeError(
            f"Model '{name}' not found in registry. Run training_pipeline/train.py first."
        )
    path = model_meta.download()
    metrics = model_meta.training_metrics or {}
    for fname in os.listdir(path):
        if fname.endswith(".joblib"):
            return joblib.load(os.path.join(path, fname)), metrics
    raise FileNotFoundError(f"No .joblib file found in downloaded model dir for '{name}'")


def _load_local_models() -> tuple[list[Any], list[dict]]:
    models, metrics = [], []
    for name, path in LOCAL_MODEL_FILES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Local model file not found: {path}. "
                "Run training_pipeline/train.py to generate it."
            )
        models.append(joblib.load(path))
        metrics.append({})          # metrics unavailable without registry
    return models, metrics


def _fetch_latest_row(project: Any) -> tuple[np.ndarray, str, dict]:
    from datetime import datetime, timedelta, timezone
    fg = project.get_feature_store().get_feature_group(
        name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION
    )
    cutoff = datetime.now(timezone.utc) - timedelta(hours=400)
    df = fg.filter(fg.time >= cutoff).read()  # bounded query — ~400 rows max, not 34k
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")

    latest = df.iloc[-1]
    timestamp = str(latest["time"])
    x = latest[FEATURE_COLS].values.astype(float).reshape(1, -1)
    current_obs = {
        "us_aqi":        float(latest["us_aqi"]),
        "temperature_2m": float(latest["temperature_2m"]),
        "pm2_5":         float(latest["pm2_5"]),
        "timestamp":     timestamp,
    }
    # persist so next startup can survive without Hopsworks
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(json.dumps(current_obs))
    return x, timestamp, current_obs


def _load_snapshot_obs() -> tuple[np.ndarray | None, str, dict]:
    if not SNAPSHOT_PATH.exists():
        return None, "unknown", {}
    obs = json.loads(SNAPSHOT_PATH.read_text())
    timestamp = obs.get("timestamp", "unknown (cached)")
    # rebuild a zeroed feature vector — predictions will be approximate
    x = np.zeros((1, len(FEATURE_COLS)), dtype=float)
    x[0, FEATURE_COLS.index("us_aqi")] = obs.get("us_aqi", 0)
    return x, f"{timestamp} (cached)", obs


@asynccontextmanager
async def lifespan(app: FastAPI):
    using_cache = False
    try:
        project = get_project()
        mr = project.get_model_registry()
        models_and_metrics = [_download_model(mr, name) for name in DAILY_REGISTRY_NAMES]
        app.state.daily_models  = [m for m, _ in models_and_metrics]
        app.state.model_metrics = [m for _, m in models_and_metrics]
        app.state.feature_x, app.state.feature_timestamp, app.state.current_obs = (
            _fetch_latest_row(project)
        )
        print("Loaded from Hopsworks registry and feature store.")
    except Exception as exc:
        print(f"WARNING: Hopsworks unreachable ({exc.__class__.__name__}: {exc}). "
              "Falling back to local model files and cached snapshot.")
        app.state.daily_models, app.state.model_metrics = _load_local_models()
        app.state.feature_x, app.state.feature_timestamp, app.state.current_obs = (
            _load_snapshot_obs()
        )
        using_cache = True

    app.state.using_cache = using_cache
    yield

    app.state.daily_models  = None
    app.state.model_metrics = None
    app.state.feature_x     = None
    app.state.current_obs   = None


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "cache_mode": app.state.using_cache}


@app.get("/predict")
def predict() -> dict:
    x = app.state.feature_x
    if x is None:
        raise HTTPException(status_code=503, detail="Models not loaded and no local snapshot found.")

    day1, day2, day3 = [
        float(np.clip(model.predict(x)[0], 0, None))
        for model in app.state.daily_models
    ]
    obs = app.state.current_obs

    return {
        "feature_timestamp":    app.state.feature_timestamp,
        "current_aqi":          obs.get("us_aqi", 0),
        "current_temperature_c": obs.get("temperature_2m", 0),
        "current_pm2_5":        obs.get("pm2_5", 0),
        "day1_avg_aqi":         day1,
        "day2_avg_aqi":         day2,
        "day3_avg_aqi":         day3,
        "model_metrics": [
            {"day": i + 1, **m}
            for i, m in enumerate(app.state.model_metrics)
        ],
        "cache_mode": app.state.using_cache,
    }
