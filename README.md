# Pearls AQI Predictor

3-day rolling average AQI forecast for Islamabad, built for 10 Pearls.

**Live dashboard →** `<https://aqi-predictor-5xbyxairurkcd5zvxat8dz.streamlit.app/>`  
---

## Architecture

```
Open-Meteo API
     │  hourly fetch (GitHub Actions)
     ▼
Hopsworks Feature Store        ← single source of truth for features
     │  full read at training time
     ▼
Training Pipeline              ← GitHub Actions, daily at 00:00 UTC
     │  3 XGBoost models + compressed fallback copies committed to repo
     ▼
Hopsworks Model Registry       ← versioned model storage
     │  downloaded at backend startup (SDK caches in /tmp)
     ▼
FastAPI on Render              ← /predict endpoint
     │  feature row fetched live from Open-Meteo (not the feature store)
     ▼
Streamlit Cloud Dashboard      ← consumer-facing, calls /predict every 30 min
```

---

## Repository Structure

```
feature_pipeline/
  run_pipeline.py     # hourly ingestion — fetches last 400 h, upserts to feature store
  backfill.py         # one-shot historical backfill from 2022-09-01 (or last known row)
  compute_features.py # lag features, calendar fields, change rate
  constants.py        # city coordinates
  hopsworks_utils.py  # shared login helpers

training_pipeline/
  train.py            # trains RF/Ridge/XGBoost per target, picks best, pushes to registry
  check_registry.py   # verifies all 3 models exist; prints version + metrics

backend/
  main.py             # FastAPI app — loads models at startup, /health + /predict

dashboard/
  app.py              # Streamlit consumer dashboard
  requirements.txt    # Streamlit Cloud dependency file

data/
  fallback_models/    # compressed .joblib files committed by training workflow
                      # used as fallback when Hopsworks registry is unreachable

.github/workflows/
  hourly.yml          # runs feature_pipeline.run_pipeline --push every hour
  training.yml        # runs training_pipeline.train daily, commits fallback models
```

---

## Local Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file (or export directly):

```bash
export HOPSWORKS_API_KEY=<your-key>
export HOPSWORKS_PROJECT=<your-project-name>
```

---

## Running Pipelines Locally

### Feature pipeline — gap-fill and push
```bash
python -m feature_pipeline.backfill --push
```
Queries the feature store for the last known timestamp, fetches only the gap from Open-Meteo, and upserts. On first run, backfills from 2022-09-01.

### Feature pipeline — dry run (no push)
```bash
python -m feature_pipeline.backfill
```
Fetches and saves to `data/islamabad_features_backfill.csv` without touching Hopsworks.

### Hourly pipeline (same as GitHub Actions)
```bash
python -m feature_pipeline.run_pipeline --push
```

### Training
```bash
python -m training_pipeline.train
```
Add `--local path/to/features.csv` to train from a local CSV and skip all Hopsworks I/O.

### Check model registry
```bash
python -m training_pipeline.check_registry
```
Prints the latest version and RMSE/MAE/R² for each of the 3 required models.

### Backend
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### Dashboard (requires backend running)
```bash
BACKEND_URL=http://localhost:8000 streamlit run dashboard/app.py
```

---

## GitHub Actions

| Workflow | Trigger | What it does |
|----------|---------|--------------|
| `hourly.yml` | Every hour + `workflow_dispatch` | Runs `run_pipeline --push` to keep the feature store current |
| `training.yml` | Daily 00:00 UTC + `workflow_dispatch` | Retrains models, pushes new versions to registry, commits updated fallback models to repo |

### Required Secrets

Set these in **Settings → Secrets → Actions** on the GitHub repository:

| Secret | Description |
|--------|-------------|
| `HOPSWORKS_API_KEY` | From Hopsworks project settings |
| `HOPSWORKS_PROJECT` | Hopsworks project name |

`training.yml` also needs **Settings → Actions → General → Workflow permissions** set to **Read and write** (to push the fallback model commit).

---

## Deployment

### Backend — Render

1. Create a new **Web Service**, connect the GitHub repo.
2. Set **Start command**: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
3. Add environment variables: `HOPSWORKS_API_KEY`, `HOPSWORKS_PROJECT`
4. Render uses `requirements.txt` for dependencies.

At startup the backend downloads the latest model versions from the Hopsworks registry. If the registry is unreachable, it falls back to `data/fallback_models/` (committed to the repo by the training workflow).

### Dashboard — Streamlit Cloud

1. Connect the GitHub repo, set **Main file path** to `dashboard/app.py`.
2. Streamlit Cloud uses `dashboard/requirements.txt` (separate from the root one — no Hopsworks dependency).
3. Add secret: `BACKEND_URL = https://<your-render-service>.onrender.com`

---

## API Reference

### `GET /health`
```json
{ "status": "ok" }
```

### `GET /predict`
```json
{
  "feature_timestamp": "2026-09-06 23:00:00+00:00",
  "current_aqi": 117,
  "current_temperature_c": 22.5,
  "current_pm2_5": 43.3,
  "day1_avg_aqi": 125.4,
  "day2_avg_aqi": 123.8,
  "day3_avg_aqi": 123.5,
  "model_metrics": [
    { "day": 1, "rmse": 10.858, "mae": 8.76, "r2": 0.296 },
    { "day": 2, "rmse": 16.201, "mae": 13.40, "r2": -0.549 },
    { "day": 3, "rmse": 17.161, "mae": 14.14, "r2": -0.668 }
  ]
}
```

`feature_timestamp` is the timestamp of the Open-Meteo reading used for prediction. `day1_avg_aqi` is the predicted rolling 24-hour average AQI for the next 24 h, day2 for 24–48 h, day3 for 48–72 h.
