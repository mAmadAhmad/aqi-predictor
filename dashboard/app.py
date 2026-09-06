"""
Streamlit consumer dashboard for the Pearls AQI Predictor (10 Pearls).
Calls the backend /predict endpoint and presents a 3-day AQI forecast with
health guidance. Cache TTL is 30 min — backend data refreshes on the same cadence.
"""
import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
CITY = "Islamabad"
LAT, LON = 33.68, 73.04

AQI_GUIDE = [
    (0,   50,  "Good",                       "#27AE60", "#1a1a1a",
     "Air quality is satisfactory. Safe for all groups — enjoy outdoor activities."),
    (51,  100, "Moderate",                   "#F39C12", "#1a1a1a",
     "Acceptable air quality. Unusually sensitive individuals should reduce prolonged outdoor exertion."),
    (101, 150, "Unhealthy · Sensitive Groups","#E67E22", "#fff",
     "People with heart/lung disease, elderly, and children should reduce prolonged outdoor exertion."),
    (151, 200, "Unhealthy",                  "#C0392B", "#fff",
     "Everyone may begin to experience effects. Avoid prolonged outdoor exertion."),
    (201, 300, "Very Unhealthy",             "#8E44AD", "#fff",
     "Health alert. Everyone should avoid outdoor exertion."),
]


def _aqi_color(aqi: float) -> str:
    for lo, hi, _, color, *_ in AQI_GUIDE:
        if aqi <= hi:
            return color
    return "#8E44AD"


def _aqi_label(aqi: float) -> str:
    for lo, hi, label, *_ in AQI_GUIDE:
        if aqi <= hi:
            return label
    return "Very Unhealthy"


def _text_color(aqi: float) -> str:
    return "#fff" if aqi >= 100 else "#1a1a1a"


@st.cache_data(ttl=1800)
def _fetch() -> dict:
    resp = requests.get(f"{BACKEND_URL}/predict", timeout=15)
    resp.raise_for_status()
    return resp.json()


def _forecast_card(label: str, aqi: float, rmse: float) -> str:
    bg = _aqi_color(aqi)
    fg = _text_color(aqi)
    return f"""
<div style="
    background:{bg};
    border-radius:16px;
    padding:28px 16px 22px;
    text-align:center;
    box-shadow:0 4px 14px rgba(0,0,0,0.13);
">
  <div style="font-size:11px;font-weight:700;color:{fg};letter-spacing:.08em;opacity:.8;text-transform:uppercase;">{label}</div>
  <div style="font-size:56px;font-weight:800;color:{fg};line-height:1.05;margin:8px 0 2px;">{aqi:.0f}</div>
  <div style="font-size:11px;font-weight:500;color:{fg};opacity:.7;margin-bottom:6px;">± {rmse:.0f} AQI</div>
  <div style="font-size:12px;font-weight:600;color:{fg};opacity:.9;">{_aqi_label(aqi)}</div>
</div>"""


def _render_aqi_guide() -> None:
    rows = []
    for lo, hi, label, color, fg, message in AQI_GUIDE:
        badge = (
            f'<span style="background:{color};color:{fg};padding:2px 10px;'
            f'border-radius:20px;font-size:12px;font-weight:600;">{label}</span>'
        )
        rows.append(f"| {lo}–{hi} | {badge} | {message} |")

    st.markdown(
        "| AQI Range | Category | Who is affected? |\n"
        "|-----------|----------|------------------|\n" + "\n".join(rows),
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title=f"{CITY} AQI", page_icon="🌿", layout="wide")

    # ── header ────────────────────────────────────────────────────────────────
    hcol, logo_col = st.columns([5, 1])
    with hcol:
        st.markdown(f"## 🌿 {CITY} &nbsp; Air Quality Forecast")
        st.caption(f"📍 {LAT}°N · {LON}°E &nbsp;|&nbsp; 3-day rolling average AQI · powered by Open-Meteo")
    with logo_col:
        st.markdown(
            "<div style='text-align:right;padding-top:12px;font-size:13px;"
            "font-weight:700;color:#7F8C8D;'>10 Pearls</div>",
            unsafe_allow_html=True,
        )
    st.divider()

    # ── fetch ─────────────────────────────────────────────────────────────────
    with st.spinner("Loading forecast…"):
        try:
            data = _fetch()
        except Exception as exc:
            st.error(f"Could not reach backend ({BACKEND_URL}): {exc}")
            st.stop()

    ts          = data["feature_timestamp"]
    current_aqi = data["current_aqi"]
    temp_c      = data["current_temperature_c"]
    pm25        = data["current_pm2_5"]
    day1        = data["day1_avg_aqi"]
    day2        = data["day2_avg_aqi"]
    day3        = data["day3_avg_aqi"]
    metrics     = data.get("model_metrics", [{"rmse": 0}, {"rmse": 0}, {"rmse": 0}])
    rmse1       = metrics[0].get("rmse", 0)
    rmse2       = metrics[1].get("rmse", 0)
    rmse3       = metrics[2].get("rmse", 0)

    # ── warning ───────────────────────────────────────────────────────────────
    if day1 > 150:
        st.warning("⚠️  Air quality is unhealthy. Limit outdoor exertion.")

    # ── current conditions ────────────────────────────────────────────────────
    st.markdown("#### Current Conditions")
    cc1, cc2, cc3, spacer = st.columns([1.4, 1, 1, 2.6])
    with cc1:
        st.metric("🌫️  AQI (current hour)", f"{current_aqi:.0f}",
                  help="US AQI from the latest feature-store row")
    with cc2:
        st.metric("🌡️  Temperature", f"{temp_c:.1f} °C")
    with cc3:
        st.metric("💨  PM 2.5", f"{pm25:.1f} µg/m³")
    with spacer:
        st.caption(
            f"Pipeline snapshot: {ts}",
            help=(
                "This timestamp reflects when data last flowed through the feature pipeline "
                "(Open-Meteo → Hopsworks). It updates each time the pipeline runs, "
                "independently of model retraining."
            ),
        )

    st.divider()

    # ── 3-day forecast cards ──────────────────────────────────────────────────
    st.markdown("#### 3-Day AQI Forecast")
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown(_forecast_card("Day 1 · next 24 h", day1, rmse1), unsafe_allow_html=True)
    with f2:
        st.markdown(_forecast_card("Day 2 · 24 – 48 h", day2, rmse2), unsafe_allow_html=True)
    with f3:
        st.markdown(_forecast_card("Day 3 · 48 – 72 h", day3, rmse3), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── trend chart ───────────────────────────────────────────────────────────
    chart_df = pd.DataFrame({
        "Period": ["Day 1", "Day 2", "Day 3"],
        "AQI": [day1, day2, day3],
        "Color": [_aqi_color(day1), _aqi_color(day2), _aqi_color(day3)],
    })

    y_max = max(max(day1, day2, day3) * 1.3, 200)

    thresholds = pd.DataFrame({
        "threshold": [50, 100, 150],
        "label": ["Good / Moderate", "Moderate / Sensitive", "Sensitive / Unhealthy"],
    })

    line = (
        alt.Chart(chart_df)
        .mark_line(strokeWidth=3, color="#2C3E50")
        .encode(
            x=alt.X("Period:N", sort=None, title=None,
                    axis=alt.Axis(labelFontSize=13, labelFontWeight="bold")),
            y=alt.Y("AQI:Q", scale=alt.Scale(domain=[0, y_max]),
                    title="AQI", axis=alt.Axis(grid=True, gridColor="#eee")),
        )
    )

    points = (
        alt.Chart(chart_df)
        .mark_point(size=140, filled=True, strokeWidth=0)
        .encode(
            x=alt.X("Period:N", sort=None),
            y=alt.Y("AQI:Q"),
            color=alt.Color("Color:N", scale=None),
            tooltip=[
                alt.Tooltip("Period:N", title="Period"),
                alt.Tooltip("AQI:Q", title="Avg AQI", format=".1f"),
            ],
        )
    )

    rules = (
        alt.Chart(thresholds)
        .mark_rule(strokeDash=[5, 4], opacity=0.45, color="#888")
        .encode(y="threshold:Q")
    )

    rule_labels = (
        alt.Chart(thresholds)
        .mark_text(align="right", dx=-4, dy=-6, fontSize=10, color="#888")
        .encode(y="threshold:Q", text="label:N", x=alt.value(280))
    )

    chart = (
        (rules + rule_labels + line + points)
        .properties(height=260)
        .configure_view(strokeWidth=0)
        .configure_axis(labelColor="#555", titleColor="#555")
    )
    st.altair_chart(chart, use_container_width=True)

    # ── model performance ─────────────────────────────────────────────────────
    with st.expander("📊  Model Performance", expanded=False):
        st.caption(
            "Metrics from the last training run on held-out test data (most recent 30 days). "
            "Negative R² at Day 2–3 is expected — AQI is harder to predict further out. "
            "All models beat the naïve persistence baseline (predicting current AQI unchanged)."
        )
        perf_df = pd.DataFrame([
            {
                "Day": f"Day {m['day']}",
                "RMSE": f"{m.get('rmse', '—'):.1f}" if isinstance(m.get("rmse"), float) else "—",
                "MAE":  f"{m.get('mae',  '—'):.1f}" if isinstance(m.get("mae"),  float) else "—",
                "R²":   f"{m.get('r2',   '—'):.3f}" if isinstance(m.get("r2"),   float) else "—",
                "Model": "XGBoost",
            }
            for m in metrics
        ])
        st.dataframe(perf_df, use_container_width=True, hide_index=True)

    # ── aqi health guide ──────────────────────────────────────────────────────
    with st.expander("💡  What does the AQI value mean?", expanded=False):
        st.caption("US Air Quality Index (AQI) — health guidance by category")
        _render_aqi_guide()

    # ── footer ────────────────────────────────────────────────────────────────
    st.divider()
    st.caption(
        "Pearls AQI Predictor · Built for **10 Pearls** · "
        "XGBoost model · Open-Meteo data · auto-refreshes every 30 min."
    )


if __name__ == "__main__":
    main()
