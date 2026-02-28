"""Streamlit web interface for Weather Model Scoring.

Run with:
    python3 -m streamlit run blur_weather/app.py

Tabs: Overview | Detail | Calibration
Theme: Light professional — iPad 10" optimised.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, List

# ── must be the very first Streamlit call ──────────────────────────────────────
st.set_page_config(
    page_title="Weather Model Scoring",
    page_icon="⛵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS theme injection ───────────────────────────────────────────────────────
BLUR_CSS = """
<style>
/* Main background */
[data-testid="stAppViewContainer"] {
    background-color: #FAFBFC;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #F0F2F5;
}

/* Typography */
h1, h2, h3, h4, h5, h6 {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    color: #1A1D23 !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E8EAED;
    border-radius: 8px;
    padding: 12px 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}

/* Tab buttons — Apple touch target */
button[data-baseweb="tab"] {
    min-height: 44px !important;
    font-size: 15px !important;
}

/* Sidebar primary button */
[data-testid="stSidebar"] button[kind="primary"] {
    min-height: 48px !important;
    font-size: 16px !important;
}

/* iPad responsive: wrap metric cards 2x2 */
@media (max-width: 1100px) {
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        min-width: 45% !important;
        margin-bottom: 8px;
    }
}
</style>
"""
st.markdown(BLUR_CSS, unsafe_allow_html=True)

from blur_weather.config import SMHI_STATIONS, COURSES, MODELS, DEFAULT_MODELS
from blur_weather.fetch import fetch_multi_model_forecast
from blur_weather.observe import fetch_smhi_wind_observations
from blur_weather.score import score_all_models, ModelScore
from blur_weather.fronts import score_front_timing, detect_frontal_events
import blur_weather.plot as plot


# ============================================================
# CACHED DATA FETCHERS
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)
def cached_fetch_forecasts(lat: float, lon: float, past_days: int = 2) -> dict:
    try:
        return fetch_multi_model_forecast(lat=lat, lon=lon,
                                          past_days=past_days, forecast_days=2)
    except Exception:
        return {}


@st.cache_data(ttl=900, show_spinner=False)
def cached_fetch_observations(station_key: str, period: str = "latest-day") -> pd.DataFrame:
    try:
        return fetch_smhi_wind_observations(station_key, period=period)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def cached_fetch_course_data(course_key: str) -> dict:
    course = COURSES[course_key]
    results = {}
    for station_key in course.nearby_stations:
        station = SMHI_STATIONS.get(station_key)
        if station is None:
            continue
        obs = fetch_smhi_wind_observations(station_key, period="latest-day")
        if obs.empty or len(obs) < 3:
            continue
        forecasts = fetch_multi_model_forecast(
            lat=station.lat, lon=station.lon, past_days=2, forecast_days=2
        )
        if forecasts:
            results[station.name] = {
                "station_key": station_key,
                "forecasts": forecasts,
                "observations": obs,
            }
    return results


# ============================================================
# HELPERS
# ============================================================

def filter_to_window(df: pd.DataFrame, hours: int) -> pd.DataFrame:
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(hours=hours)
    return df[df["datetime"] >= cutoff].copy()


def aggregate_course_scores(per_station: dict) -> list:
    from blur_weather.score import NudgeRecommendation

    model_stats = {}
    for station_name, data in per_station.items():
        for s in data["scores"]:
            if s.n_points < 3:
                continue
            if s.model_id not in model_stats:
                model_stats[s.model_id] = []
            model_stats[s.model_id].append(s)

    agg = []
    for model_id, score_list in model_stats.items():
        best_s = max(score_list, key=lambda s: s.n_points)
        agg.append(ModelScore(
            model_id=model_id,
            model_name=score_list[0].model_name,
            n_points=sum(s.n_points for s in score_list),
            tws_trend_correlation=np.mean([s.tws_trend_correlation for s in score_list]),
            tws_rmse=np.mean([s.tws_rmse for s in score_list]),
            tws_mae=np.mean([s.tws_mae for s in score_list]),
            tws_bias=np.mean([s.tws_bias for s in score_list]),
            tws_scale=np.mean([s.tws_scale for s in score_list]),
            twd_trend_correlation=np.mean([s.twd_trend_correlation for s in score_list]),
            twd_rmse=np.mean([s.twd_rmse for s in score_list]),
            twd_mae=np.mean([s.twd_mae for s in score_list]),
            twd_bias=np.mean([s.twd_bias for s in score_list]),
            time_lag_hours=np.mean([s.time_lag_hours for s in score_list]),
            composite_score=np.mean([s.composite_score for s in score_list]),
            ma_error=np.mean([s.ma_error for s in score_list]),
            nudge=best_s.nudge,
        ))

    agg.sort(key=lambda s: s.composite_score, reverse=True)
    return agg


def compute_front_results(forecasts: dict, observations: pd.DataFrame) -> tuple:
    """Run front detection for observations + all models.

    Returns (obs_events, front_results_list).
    """
    obs_events = []
    front_results = []

    has_obs_pressure = (
        "pressure_hpa" in observations.columns
        and not observations["pressure_hpa"].isna().all()
    )

    if has_obs_pressure:
        obs_events = detect_frontal_events(observations, source="obs")

        for model_id, fcst_df in forecasts.items():
            model_info = MODELS.get(model_id, {})
            model_name = model_info.get("name", model_id)
            result = score_front_timing(model_id, model_name, fcst_df, observations)
            if result is not None:
                front_results.append(result)

    return obs_events, front_results


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar() -> dict:
    st.sidebar.title("Weather Model Scoring")
    st.sidebar.markdown("*Offshore racing — forecast accuracy*")
    st.sidebar.divider()

    mode = st.sidebar.radio("Analysis Mode", ["Station", "Race Course"], horizontal=True)

    station_key = None
    course_key = None

    if mode == "Station":
        station_options = {k: v.name for k, v in SMHI_STATIONS.items()}
        station_key = st.sidebar.selectbox(
            "Station",
            options=list(station_options.keys()),
            format_func=lambda k: station_options[k],
            index=list(station_options.keys()).index("vinga"),
        )
        s = SMHI_STATIONS[station_key]
        st.sidebar.caption(f"{s.lat:.2f}°N, {s.lon:.2f}°E — {s.relevance}")

    else:
        course_options = {k: v.name for k, v in COURSES.items()}
        course_key = st.sidebar.selectbox(
            "Race Course",
            options=list(course_options.keys()),
            format_func=lambda k: course_options[k],
        )
        c = COURSES[course_key]
        st.sidebar.caption(f"{c.total_nm} NM — {len(c.nearby_stations)} stations")

    st.sidebar.divider()

    time_window = st.sidebar.radio(
        "Time Window",
        options=[6, 12, 24],
        format_func=lambda h: f"{h}h",
        index=2,
        horizontal=True,
    )

    st.sidebar.divider()
    run_clicked = st.sidebar.button("Run Analysis", type="primary", use_container_width=True)

    st.sidebar.divider()
    st.sidebar.caption(
        "Data: [Open-Meteo](https://open-meteo.com) · [SMHI](https://opendata.smhi.se)  \n"
        "Scores update every 15 min · UTC times throughout"
    )

    return {
        "mode": "station" if mode == "Station" else "course",
        "station_key": station_key,
        "course_key": course_key,
        "time_window_hours": time_window,
        "run_clicked": run_clicked,
    }


# ============================================================
# METRIC ROW
# ============================================================

def render_metric_row(scores: list, obs_events: Optional[list] = None,
                      front_results: Optional[list] = None):
    if not scores:
        return
    best = scores[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Model", best.model_name)
    c2.metric("Composite Score", f"{best.composite_score:.1f} / 100")
    c3.metric("TWS RMSE", f"{best.tws_rmse:.1f} kt",
              delta=f"{best.tws_bias:+.1f} kt bias", delta_color="inverse")
    c4.metric("TWD RMSE", f"{best.twd_rmse:.0f}°",
              delta=f"{best.twd_bias:+.0f}° bias", delta_color="inverse")

    # Front summary card
    if obs_events and front_results:
        best_fr = min(
            [fr for fr in front_results if fr.matched_pairs],
            key=lambda fr: fr.mean_timing_error_hours,
            default=None,
        )
        if best_fr:
            fr_alias = plot._model_alias(best_fr.model_id)
            err_sign = "+" if best_fr.mean_timing_error_hours >= 0 else ""
            st.info(
                f"{len(obs_events)} front(s) detected. "
                f"Best timing: **{fr_alias}** "
                f"({err_sign}{best_fr.mean_timing_error_hours:.1f}h mean error)"
            )
        else:
            st.info(f"{len(obs_events)} front(s) detected — no model matched timing.")


# ============================================================
# OVERVIEW TAB
# ============================================================

def render_overview_tab(scores: list):
    if not scores:
        st.info("No scoring results available.")
        return

    st.plotly_chart(plot.plot_model_ranking(scores), use_container_width=True)

    st.subheader("Full Rankings")
    rows = []
    for i, s in enumerate(scores, 1):
        rows.append({
            "Rank": i,
            "Model": s.model_name,
            "Composite": s.composite_score,
            "TWS RMSE (kt)": round(s.tws_rmse, 1),
            "TWS Bias (kt)": round(s.tws_bias, 1),
            "TWD RMSE (°)": round(s.twd_rmse, 0),
            "TWD Bias (°)": round(s.twd_bias, 0),
            "Lag (h)": round(s.time_lag_hours, 1),
            "Points": s.n_points,
        })
    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        column_config={
            "Composite": st.column_config.ProgressColumn(
                "Composite", min_value=0, max_value=100, format="%.1f"),
            "TWS RMSE (kt)": st.column_config.NumberColumn(format="%.1f kt"),
            "TWS Bias (kt)": st.column_config.NumberColumn(format="%+.1f kt"),
            "TWD RMSE (°)": st.column_config.NumberColumn(format="%.0f°"),
            "TWD Bias (°)": st.column_config.NumberColumn(format="%+.0f°"),
        },
        hide_index=True,
        use_container_width=True,
    )


# ============================================================
# DETAIL TAB — sub-navigation
# ============================================================

def render_detail_tab(
    forecasts: dict,
    observations: pd.DataFrame,
    scores: list,
    time_window_hours: int,
    obs_events: Optional[list] = None,
    front_results: Optional[list] = None,
):
    if observations.empty or not forecasts:
        st.warning("Not enough data to render detail views.")
        return

    sub_view = st.radio(
        "View",
        ["Time Series", "Rolling Error", "Lead Time", "Wind Regime"],
        horizontal=True,
        label_visibility="collapsed",
    )

    best_id = scores[0].model_id if scores else None

    if sub_view == "Time Series":
        _render_time_series(forecasts, observations, scores, time_window_hours,
                            obs_events, front_results)
    elif sub_view == "Rolling Error":
        _render_rolling_error(forecasts, observations, time_window_hours)
    elif sub_view == "Lead Time":
        _render_lead_time(scores)
    elif sub_view == "Wind Regime":
        _render_wind_regime(scores)


def _render_time_series(forecasts, observations, scores, time_window_hours,
                        obs_events, front_results):
    best_id = scores[0].model_id if scores else None
    obs_display = filter_to_window(observations, time_window_hours)
    fcsts_display = {mid: filter_to_window(df, time_window_hours)
                     for mid, df in forecasts.items()}

    if obs_display.empty:
        st.warning("No observations in the selected time window.")
        return

    fig = plot.plot_combined_timeseries(
        fcsts_display, obs_display,
        time_window_hours=time_window_hours,
        highlight_model=best_id,
    )

    # Overlay front annotations
    if obs_events or front_results:
        fig = plot.plot_front_annotations(fig, front_results or [], obs_events or [])

    st.plotly_chart(fig, use_container_width=True)

    # ── Front / Wind Shift Analysis ──────────────────────────────────────
    _render_front_analysis(forecasts, observations, obs_events, front_results,
                           time_window_hours)


def _render_front_analysis(forecasts, observations, obs_events, front_results,
                           time_window_hours):
    """Render the frontal passage analysis section below the timeseries."""
    has_pressure = (
        "pressure_hpa" in observations.columns
        and not observations["pressure_hpa"].isna().all()
    )

    if not has_pressure:
        st.caption(
            "No pressure data available for this station — "
            "front detection requires barometric pressure."
        )
        return

    st.divider()
    st.subheader("Wind Shift & Front Detection")

    # ── Narrative summary ──
    if obs_events:
        n = len(obs_events)
        front_word = "front" if n == 1 else "fronts"
        times_str = ", ".join(
            ev.datetime.strftime("%d %b %H:%M UTC") for ev in obs_events
        )
        st.markdown(
            f"**{n} frontal passage{'s' if n > 1 else ''} detected** "
            f"in observations: {times_str}"
        )

        # Per-event detail
        for i, ev in enumerate(obs_events):
            shift_dir = "veered" if ev.twd_shift_degrees > 0 else "backed"
            st.markdown(
                f"- **{ev.datetime.strftime('%H:%M UTC')}**: "
                f"Pressure tendency {ev.pressure_tendency:+.1f} hPa/3h, "
                f"wind {shift_dir} {abs(ev.twd_shift_degrees):.0f}°"
            )

        # Model comparison
        if front_results:
            st.markdown("**Model comparison:**")
            valid_fr = [fr for fr in front_results if fr is not None]
            for fr in sorted(valid_fr, key=lambda f: f.mean_timing_error_hours):
                alias = plot._model_alias(fr.model_id)
                if fr.matched_pairs:
                    errors = [f"{err:+.1f}h" for _, _, err in fr.matched_pairs]
                    err_str = ", ".join(errors)
                    mean_str = f"{fr.mean_timing_error_hours:.1f}h"
                    sign_word = ""
                    # Check if model is consistently early or late
                    signed_errors = [err for _, _, err in fr.matched_pairs]
                    mean_signed = sum(signed_errors) / len(signed_errors)
                    if abs(mean_signed) > 0.3:
                        sign_word = " late" if mean_signed > 0 else " early"
                    st.markdown(
                        f"- **{alias}**: {err_str} "
                        f"(mean {mean_str}{sign_word})"
                        + (f" — {fr.n_missed} missed" if fr.n_missed > 0 else "")
                        + (f", {fr.n_false_alarms} false alarm{'s' if fr.n_false_alarms > 1 else ''}"
                           if fr.n_false_alarms > 0 else "")
                    )
                else:
                    missed_str = f"{fr.n_missed} missed" if fr.n_missed > 0 else ""
                    fa_str = (f"{fr.n_false_alarms} false alarm{'s' if fr.n_false_alarms > 1 else ''}"
                              if fr.n_false_alarms > 0 else "")
                    parts = [p for p in [missed_str, fa_str] if p]
                    st.markdown(f"- **{alias}**: No match — {', '.join(parts)}")
    else:
        st.info(
            "No frontal passages detected in the observation window. "
            "This typically means stable high-pressure or gradual changes. "
            "The signature chart below still shows pressure tendency and wind "
            "shift — watch for signals building in the model forecasts."
        )

    # ── Signature chart ──
    st.caption(
        "Pressure tendency shows the rate of pressure change (falling = front approaching, "
        "trough = front passing). Wind shift shows the rate of direction change "
        "(sharp spike = frontal veer). Compare models against the observed (black) line "
        "to judge which model best captures the timing and intensity."
    )

    sig_fig = plot.plot_front_signature(
        forecasts, observations,
        obs_events=obs_events or [],
        time_window_hours=time_window_hours,
    )
    st.plotly_chart(sig_fig, use_container_width=True)

    # ── Timing table (if fronts detected) ──
    if front_results and obs_events:
        st.subheader("Timing Detail")
        st.plotly_chart(plot.plot_front_timing_table(front_results),
                        use_container_width=True)


def _render_rolling_error(forecasts, observations, time_window_hours):
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Rolling TWS RMSE")
        st.caption("Lower = better. Shows which model is most accurate right now.")
    with col2:
        window = st.slider("Window (hours)", min_value=1, max_value=6, value=3)

    obs_w = filter_to_window(observations, time_window_hours)
    fcsts_w = {mid: filter_to_window(df, time_window_hours)
               for mid, df in forecasts.items()}

    fig = plot.plot_rolling_rmse(fcsts_w, obs_w, window_hours=window,
                                  time_window_hours=time_window_hours)
    st.plotly_chart(fig, use_container_width=True)

    n_obs_hours = len(obs_w["datetime"].dt.floor("h").unique()) if not obs_w.empty else 0
    if n_obs_hours < window + 2:
        st.info(
            f"Only {n_obs_hours} hourly observation bins in the selected window. "
            f"Extend the time window or reduce the rolling window for a more meaningful chart."
        )


def _render_lead_time(scores):
    st.subheader("Lead-Time Breakdown")
    st.caption(
        "RMSE per lead-time bucket. Lead time approximated as hours before now "
        "(model init time not available from Open-Meteo)."
    )

    st.plotly_chart(plot.plot_lead_time_heatmap(scores), use_container_width=True)

    # Detail table
    models_with_data = [s for s in scores if s.lead_time_buckets]
    if models_with_data:
        st.subheader("Detail")
        rows = []
        for s in models_with_data:
            for b in s.lead_time_buckets:
                rows.append({
                    "Model": s.model_name,
                    "Bucket": b.label,
                    "TWS RMSE (kt)": round(b.tws_rmse, 1),
                    "TWS Bias (kt)": round(b.tws_bias, 1),
                    "TWD RMSE (°)": round(b.twd_rmse, 0),
                    "TWD Bias (°)": round(b.twd_bias, 0),
                    "Points": b.n_points,
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _render_wind_regime(scores):
    st.subheader("Wind Regime Analysis")
    st.caption("Model accuracy binned by observed wind speed. Nudge values are regime-specific.")

    st.plotly_chart(plot.plot_regime_bars(scores), use_container_width=True)

    # Regime-specific metric cards
    models_with_data = [s for s in scores if s.regime_buckets]
    if models_with_data:
        st.subheader("Regime-Specific Nudge")
        for s in models_with_data:
            alias = plot._model_alias(s.model_id)
            with st.expander(alias, expanded=(s == models_with_data[0])):
                cols = st.columns(min(len(s.regime_buckets), 4))
                for i, b in enumerate(s.regime_buckets):
                    with cols[i % len(cols)]:
                        sign = "+" if b.nudge_tws_offset >= 0 else ""
                        st.metric(
                            label=b.label,
                            value=f"{sign}{b.nudge_tws_offset:.1f} kt",
                            delta=f"RMSE {b.tws_rmse:.1f} kt",
                            delta_color="inverse",
                        )
                        st.caption(f"n={b.n_points} · TWD RMSE {b.twd_rmse:.0f}°")


# ============================================================
# CALIBRATION TAB
# ============================================================

def render_calibration_tab(scores: list):
    st.subheader("Expedition Calibration Panel")
    st.caption(
        "Enter these values manually in Expedition under **Boat → Calibration**. "
        "Nothing here is automated — all adjustments require your judgement."
    )
    st.warning(
        "Calibration recommendations are based on recent observations and statistical "
        "correlation only. Always sanity-check before applying to routing.",
        icon="⚠️",
    )

    if not scores:
        st.info("Run analysis to see nudge recommendations.")
        return

    st.plotly_chart(plot.plot_expedition_nudge_table(scores), use_container_width=True)

    # Model Accuracy-style reports
    st.subheader("Model Accuracy-style Reports")
    for s in scores:
        label = f"{s.model_name}  —  score {s.composite_score:.1f}/100"
        with st.expander(label, expanded=(s == scores[0])):
            st.code(s.summary(), language=None)

    best = scores[0]
    st.success(
        f"**Recommended: {best.model_name}**  \n"
        f"TWS: {best.nudge.tws_calibrate_str}  \n"
        f"TWD: {best.nudge.twd_calibrate_str}"
    )

    # Regime-specific calibration section
    models_with_regimes = [s for s in scores if s.regime_buckets]
    if models_with_regimes:
        st.divider()
        st.subheader("Regime-Specific Calibration")
        st.caption(
            "Wind-speed dependent nudge — more accurate than a single flat offset. "
            "Apply the regime that matches expected conditions."
        )
        best_regime = models_with_regimes[0]
        alias = plot._model_alias(best_regime.model_id)
        st.markdown(f"**{alias}** regime nudges:")
        for b in best_regime.regime_buckets:
            sign = "+" if b.nudge_tws_offset >= 0 else ""
            st.markdown(f"- **{b.label}**: {sign}{b.nudge_tws_offset:.1f} kt "
                        f"(RMSE {b.tws_rmse:.1f} kt, n={b.n_points})")


# ============================================================
# COURSE LEADERBOARD
# ============================================================

def render_course_leaderboard(per_station: dict, aggregate_scores: list):
    st.subheader("Course-Wide Model Ranking")
    st.caption("Composite scores averaged across all available stations.")
    st.plotly_chart(plot.plot_model_ranking(aggregate_scores), use_container_width=True)

    st.subheader("Best Model Per Station")
    station_names = list(per_station.keys())
    n_cols = min(len(station_names), 4)
    cols = st.columns(n_cols)
    for i, (station_name, data) in enumerate(per_station.items()):
        if data["scores"]:
            best = data["scores"][0]
            cols[i % n_cols].metric(
                label=station_name,
                value=best.model_name,
                delta=f"Score {best.composite_score:.0f}/100",
            )


# ============================================================
# STATION ANALYSIS
# ============================================================

def run_station_analysis(sidebar: dict) -> Optional[dict]:
    station_key = sidebar["station_key"]
    time_window_hours = sidebar["time_window_hours"]
    station = SMHI_STATIONS[station_key]

    with st.status("Fetching data...", expanded=True) as status:
        st.write(f"Fetching SMHI observations from **{station.name}**...")
        obs = cached_fetch_observations(station_key)
        if obs.empty:
            status.update(label="Failed", state="error")
            st.error(f"No observations from SMHI for {station.name}. Try again in a few minutes.")
            return None
        st.write(f"Got {len(obs)} observation records.")

        st.write("Fetching model forecasts from Open-Meteo...")
        forecasts = cached_fetch_forecasts(station.lat, station.lon)
        if not forecasts:
            status.update(label="Failed", state="error")
            st.error("Failed to fetch model forecasts. Check your network connection.")
            return None
        st.write(f"Got forecasts for {len(forecasts)} models.")

        st.write("Scoring models...")
        scores = score_all_models(forecasts, obs)

        if not scores or scores[0].n_points < 3:
            status.update(label="Insufficient data", state="error")
            st.warning(
                "Not enough overlapping data between forecasts and observations. "
                "Try a wider time window or check that the SMHI station is active."
            )
            return None

        st.write("Detecting frontal passages...")
        obs_events, front_results = compute_front_results(forecasts, obs)

        # Attach front results to scores
        front_map = {fr.model_id: fr for fr in front_results}
        for s in scores:
            s.front_timing = front_map.get(s.model_id)

        status.update(label="Analysis complete", state="complete")

    return {
        "mode": "station",
        "station": station,
        "station_key": station_key,
        "scores": scores,
        "forecasts": forecasts,
        "observations": obs,
        "time_window_hours": time_window_hours,
        "obs_events": obs_events,
        "front_results": front_results,
    }


# ============================================================
# COURSE ANALYSIS
# ============================================================

def run_course_analysis(sidebar: dict) -> Optional[dict]:
    course_key = sidebar["course_key"]
    time_window_hours = sidebar["time_window_hours"]
    course = COURSES[course_key]

    with st.status(f"Fetching data for {course.name}...", expanded=True) as status:
        st.write(f"Fetching data for {len(course.nearby_stations)} stations "
                 f"(this may take 20-40 seconds)...")
        raw = cached_fetch_course_data(course_key)

        if not raw:
            status.update(label="Failed", state="error")
            st.error("No data available for any station on this course.")
            return None

        missing = [k for k in course.nearby_stations
                   if SMHI_STATIONS.get(k) and SMHI_STATIONS[k].name not in raw]
        if missing:
            names = [SMHI_STATIONS[k].name for k in missing if k in SMHI_STATIONS]
            st.warning(f"Stations unavailable: {', '.join(names)}")

        st.write("Scoring models at each station...")
        per_station = {}
        for station_name, data in raw.items():
            scores = score_all_models(data["forecasts"], data["observations"])
            if scores and scores[0].n_points >= 3:
                per_station[station_name] = {
                    "scores": scores,
                    "forecasts": data["forecasts"],
                    "observations": data["observations"],
                }
                st.write(f"  {station_name} — best: {scores[0].model_name} "
                         f"({scores[0].composite_score:.0f}/100)")

        if not per_station:
            status.update(label="Insufficient data", state="error")
            st.warning("Not enough overlapping data at any station.")
            return None

        aggregate_scores = aggregate_course_scores(per_station)
        status.update(label="Analysis complete", state="complete")

    return {
        "mode": "course",
        "course": course,
        "course_key": course_key,
        "per_station": per_station,
        "aggregate_scores": aggregate_scores,
        "time_window_hours": time_window_hours,
    }


# ============================================================
# WELCOME PLACEHOLDER
# ============================================================

def render_welcome():
    st.markdown("""
    ## Welcome to Weather Model Scoring

    Select a station or race course in the sidebar, choose a time window,
    then click **Run Analysis** to score all weather models against live observations.

    ### What you'll get
    | Tab | Content |
    |---|---|
    | **Overview** | Model ranking, composite scores, front detection summary |
    | **Detail** | Time series, rolling error, lead-time breakdown, wind regime analysis |
    | **Calibration** | Expedition nudge values + regime-specific corrections |

    ### Data sources
    - **Observations**: SMHI coastal stations (10-min data, last 24h)
    - **Forecasts**: Open-Meteo (ECMWF, GFS, ICON, ICON-EU, Meteo-France, KNMI)
    - All data is free, no API keys required

    ---
    *Designed for J/99 BLUR offshore racing — Gotland Runt, Skagen, MBBR*
    """)


# ============================================================
# MAIN
# ============================================================

def main():
    sidebar = render_sidebar()

    if sidebar["run_clicked"]:
        if sidebar["mode"] == "station":
            result = run_station_analysis(sidebar)
        else:
            result = run_course_analysis(sidebar)

        if result is not None:
            st.session_state["results"] = result

    results = st.session_state.get("results")

    if results is None:
        render_welcome()
        return

    # ── Station mode ────────────────────────────────────────────────────────
    if results["mode"] == "station":
        station = results["station"]
        scores = results["scores"]
        forecasts = results["forecasts"]
        observations = results["observations"]
        hours = results["time_window_hours"]
        obs_events = results.get("obs_events", [])
        front_results = results.get("front_results", [])

        st.title(f"Model Scoring — {station.name}")
        st.caption(f"{station.lat:.2f}°N, {station.lon:.2f}°E · {station.relevance} · "
                   f"Last updated {datetime.utcnow().strftime('%H:%M UTC')}")

        render_metric_row(scores, obs_events, front_results)
        st.divider()

        tabs = st.tabs(["Overview", "Detail", "Calibration"])
        with tabs[0]:
            render_overview_tab(scores)
        with tabs[1]:
            render_detail_tab(forecasts, observations, scores, hours,
                              obs_events, front_results)
        with tabs[2]:
            render_calibration_tab(scores)

    # ── Course mode ─────────────────────────────────────────────────────────
    elif results["mode"] == "course":
        course = results["course"]
        per_station = results["per_station"]
        aggregate_scores = results["aggregate_scores"]
        hours = results["time_window_hours"]

        st.title(f"Model Scoring — {course.name}")
        st.caption(f"{course.total_nm} NM · {len(per_station)} stations active · "
                   f"Last updated {datetime.utcnow().strftime('%H:%M UTC')}")

        render_metric_row(aggregate_scores)
        st.divider()

        tabs = st.tabs(["Overview", "Detail", "Calibration"])

        with tabs[0]:
            render_course_leaderboard(per_station, aggregate_scores)

        with tabs[1]:
            # Course detail: show time series per station via nested tabs
            station_names = list(per_station.keys())
            if station_names:
                selected_station = st.radio(
                    "Station", station_names, horizontal=True,
                    label_visibility="collapsed",
                )
                data = per_station[selected_station]
                render_detail_tab(
                    data["forecasts"], data["observations"],
                    data["scores"], hours,
                )

        with tabs[2]:
            render_calibration_tab(aggregate_scores)


if __name__ == "__main__":
    main()
