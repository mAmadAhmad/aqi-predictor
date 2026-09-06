"""
Verify that the 3 required daily models exist and are complete in Hopsworks Model Registry.
Run: python -m training_pipeline.check_registry
"""
from feature_pipeline.hopsworks_utils import get_project

REQUIRED_MODELS = [
    "aqi_avg_aqi_next_24h_model",
    "aqi_avg_aqi_24_48h_model",
    "aqi_avg_aqi_48_72h_model",
]


def main() -> None:
    project = get_project()
    mr = project.get_model_registry()

    all_ok = True
    print(f"\n{'model':<35} {'version':>8} {'RMSE':>8} {'MAE':>8} {'R2':>8}  status")
    print("-" * 82)

    for name in REQUIRED_MODELS:
        model = mr.get_model(name=name, version=None)
        if model is None:
            print(f"{name:<35} {'—':>8} {'—':>8} {'—':>8} {'—':>8}  MISSING")
            all_ok = False
            continue

        metrics = model.training_metrics or {}
        rmse = metrics.get("rmse", "?")
        mae  = metrics.get("mae",  "?")
        r2   = metrics.get("r2",   "?")
        rmse_str = f"{rmse:.3f}" if isinstance(rmse, float) else str(rmse)
        mae_str  = f"{mae:.3f}"  if isinstance(mae,  float) else str(mae)
        r2_str   = f"{r2:.3f}"   if isinstance(r2,   float) else str(r2)
        print(f"{name:<35} {model.version:>8} {rmse_str:>8} {mae_str:>8} {r2_str:>8}  OK")

    print()
    if all_ok:
        print("✓ All 3 models present in registry — backend is ready to serve.")
    else:
        print("✗ One or more models missing. Run: python -m training_pipeline.train")


if __name__ == "__main__":
    main()
