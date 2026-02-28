"""Pure Plotly chart functions for BLUR Weather Intelligence GUI.

All functions accept pre-processed DataFrames and return plotly Figure objects.
No Streamlit imports — safe to test independently or use in notebooks.

Colour conventions:
  - Observations: thick black line (#000000)
  - Models: fixed 6-colour palette (matches DEFAULT_MODELS order for stability)
  - Best/highlighted model: gold (#FFD700), thicker line

Theme: Light professional look — #FAFBFC background, system fonts, subtle grid.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List

from .config import DEFAULT_MODELS, MODELS


# ============================================================
# COLOUR PALETTE
# ============================================================

_MODEL_COLOUR_LIST = [
    "#1f77b4",  # ECMWF IFS (blue)
    "#ff7f0e",  # GFS (orange)
    "#2ca02c",  # ICON global (green)
    "#d62728",  # ICON-EU (red)
    "#9467bd",  # Météo-France (purple)
    "#8c564b",  # KNMI Harmonie (brown)
    "#e377c2",  # UKMO (pink — 7th if added)
]
OBS_COLOUR = "#000000"
HIGHLIGHT_COLOUR = "#FFD700"


# ============================================================
# BLUR THEME
# ============================================================

BLUR_PLOTLY_LAYOUT = dict(
    paper_bgcolor="#FAFBFC",
    plot_bgcolor="#FFFFFF",
    font=dict(
        family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        size=13,
        color="#1A1D23",
    ),
)


def _apply_blur_theme(fig: go.Figure) -> go.Figure:
    """Apply the BLUR light theme to any Plotly figure."""
    fig.update_layout(**BLUR_PLOTLY_LAYOUT)
    fig.update_xaxes(gridcolor="#E8EAED", zeroline=False)
    fig.update_yaxes(gridcolor="#E8EAED", zeroline=False)
    return fig


# ============================================================
# HELPERS
# ============================================================

def _model_colour_map(model_ids: list) -> dict:
    """Return a stable dict mapping model_id -> hex colour."""
    colour_map = {}
    ordered = [m for m in DEFAULT_MODELS if m in model_ids]
    extras = [m for m in model_ids if m not in DEFAULT_MODELS]
    for i, mid in enumerate(ordered + extras):
        colour_map[mid] = _MODEL_COLOUR_LIST[i % len(_MODEL_COLOUR_LIST)]
    return colour_map


def _model_alias(model_id: str) -> str:
    """Return short alias for a model, e.g. 'ICON-EU'."""
    info = MODELS.get(model_id, {})
    return info.get("name", model_id)


def _filter_to_window(df: pd.DataFrame, hours: int) -> pd.DataFrame:
    """Keep rows within the last `hours` hours (UTC, tz-naive datetimes)."""
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(hours=hours)
    return df[df["datetime"] >= cutoff].copy()


# ============================================================
# COMBINED TIME SERIES (TWS + TWD + optional Pressure)
# ============================================================

def plot_combined_timeseries(
    forecasts: dict,
    observations: pd.DataFrame,
    time_window_hours: int = 24,
    highlight_model: Optional[str] = None,
) -> go.Figure:
    """Time series figure: TWS, TWD, and (if available) pressure — overview + 1h zoom."""
    colours = _model_colour_map(list(forecasts.keys()))
    obs = _filter_to_window(observations, time_window_hours)
    zoom_cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(hours=1)

    has_pressure = (
        "pressure_hpa" in obs.columns
        and obs["pressure_hpa"].notna().any()
        and any("pressure_hpa" in df.columns for df in forecasts.values())
    )

    n_rows = 3 if has_pressure else 2
    row_heights = [0.35, 0.35, 0.30] if has_pressure else [0.5, 0.5]

    titles = [
        f"TWS — last {time_window_hours}h", "TWS — last 1h",
        f"TWD — last {time_window_hours}h", "TWD — last 1h",
    ]
    if has_pressure:
        titles += [f"Pressure — last {time_window_hours}h", "Pressure — last 1h"]

    fig = make_subplots(
        rows=n_rows, cols=2,
        column_widths=[0.75, 0.25],
        row_heights=row_heights,
        subplot_titles=titles,
        vertical_spacing=0.10,
        horizontal_spacing=0.06,
    )

    def _line_style(model_id):
        if model_id == highlight_model:
            return dict(color=HIGHLIGHT_COLOUR, width=3)
        return dict(color=colours.get(model_id, "#888888"), width=1.5)

    def _add_obs_trace(row, y_col, mode="lines", marker_size=3, show_legend=False):
        for col, cutoff in [(1, obs["datetime"].min()), (2, zoom_cutoff)]:
            d = obs[obs["datetime"] >= cutoff]
            if d.empty or y_col not in d.columns:
                continue
            kwargs = dict(mode=mode, name="Observed",
                          line=dict(color=OBS_COLOUR, width=2.5),
                          showlegend=(show_legend and col == 1))
            if "markers" in mode:
                kwargs["marker"] = dict(size=marker_size, color=OBS_COLOUR)
            fig.add_trace(go.Scatter(x=d["datetime"], y=d[y_col], **kwargs),
                          row=row, col=col)

    def _add_model_traces(row, y_col, mode="lines", marker_size=2, show_legend=False):
        for model_id, fcst_df in forecasts.items():
            if y_col not in fcst_df.columns:
                continue
            fcst = _filter_to_window(fcst_df, time_window_hours)
            style = _line_style(model_id)
            alias = _model_alias(model_id)
            for col, cutoff in [(1, fcst["datetime"].min()), (2, zoom_cutoff)]:
                d = fcst[fcst["datetime"] >= cutoff]
                if d.empty:
                    continue
                kwargs = dict(mode=mode, name=alias, line=style,
                              showlegend=(show_legend and col == 1))
                if "markers" in mode:
                    kwargs["marker"] = dict(size=marker_size, color=style["color"])
                fig.add_trace(go.Scatter(x=d["datetime"], y=d[y_col], **kwargs),
                              row=row, col=col)

    def _add_zoom_shading(row):
        if not obs.empty:
            fig.add_vrect(
                x0=zoom_cutoff, x1=obs["datetime"].max(),
                fillcolor="rgba(100,100,100,0.08)", line_width=0,
                row=row, col=1,
            )

    # Row 1: TWS
    _add_obs_trace(1, "wind_speed_knots", show_legend=True)
    _add_model_traces(1, "wind_speed_knots", show_legend=True)
    _add_zoom_shading(1)
    fig.update_yaxes(title_text="knots", row=1, col=1)

    # Row 2: TWD
    _add_obs_trace(2, "wind_direction", mode="lines+markers", marker_size=3)
    _add_model_traces(2, "wind_direction", mode="lines+markers", marker_size=2)
    _add_zoom_shading(2)
    twd_tick_vals = [0, 90, 180, 270, 360]
    twd_tick_text = ["N", "E", "S", "W", "N"]
    for col in [1, 2]:
        fig.update_yaxes(range=[0, 360], tickvals=twd_tick_vals,
                         ticktext=twd_tick_text, row=2, col=col)
    fig.update_yaxes(title_text="direction", row=2, col=1)

    # Row 3: Pressure (optional)
    if has_pressure:
        _add_obs_trace(3, "pressure_hpa")
        _add_model_traces(3, "pressure_hpa")
        _add_zoom_shading(3)
        fig.update_yaxes(title_text="hPa", row=3, col=1)

    fig.update_xaxes(tickformat="%H:%M")
    fig.update_layout(
        height=680 if has_pressure else 520,
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0),
        margin=dict(t=80, b=40, l=60, r=20),
        hovermode="x unified",
    )
    return _apply_blur_theme(fig)


# ============================================================
# ROLLING RMSE
# ============================================================

def plot_rolling_rmse(
    forecasts: dict,
    observations: pd.DataFrame,
    window_hours: int = 3,
    time_window_hours: int = 24,
) -> go.Figure:
    """Rolling TWS RMSE over time — the 'who is best right now' chart."""
    colours = _model_colour_map(list(forecasts.keys()))
    obs = _filter_to_window(observations, time_window_hours)

    obs_h = obs.copy()
    obs_h["hour"] = obs_h["datetime"].dt.floor("h")
    obs_hourly = obs_h.groupby("hour")["wind_speed_knots"].mean().reset_index()
    obs_hourly.columns = ["hour", "obs_tws"]

    fig = go.Figure()
    final_rmse = {}

    for model_id, fcst_df in forecasts.items():
        fcst = _filter_to_window(fcst_df, time_window_hours)
        if fcst.empty:
            continue

        fcst_h = fcst.copy()
        fcst_h["hour"] = fcst_h["datetime"].dt.floor("h")
        fcst_hourly = fcst_h.groupby("hour")["wind_speed_knots"].mean().reset_index()
        fcst_hourly.columns = ["hour", "fcst_tws"]

        merged = pd.merge(fcst_hourly, obs_hourly, on="hour", how="inner")
        if len(merged) < 2:
            continue

        merged = merged.sort_values("hour")
        merged["sq_err"] = (merged["fcst_tws"] - merged["obs_tws"]) ** 2
        merged["rolling_rmse"] = np.sqrt(
            merged["sq_err"].rolling(window_hours, min_periods=2).mean()
        )

        n_valid = merged["rolling_rmse"].notna().sum()
        alias = _model_alias(model_id)
        if n_valid < window_hours:
            alias += " (insufficient data)"

        last_rmse = merged["rolling_rmse"].dropna().iloc[-1] if n_valid > 0 else np.inf
        final_rmse[model_id] = (last_rmse, alias, merged)

    sorted_models = sorted(final_rmse.items(), key=lambda x: x[1][0])

    for model_id, (last_rmse, alias, merged) in sorted_models:
        colour = colours.get(model_id, "#888888")
        fig.add_trace(go.Scatter(
            x=merged["hour"],
            y=merged["rolling_rmse"],
            mode="lines",
            name=f"{alias} ({last_rmse:.1f} kt)",
            line=dict(color=colour, width=2),
        ))

    if sorted_models:
        best_rmse = sorted_models[0][1][0]
        if np.isfinite(best_rmse):
            fig.add_hline(
                y=best_rmse,
                line_dash="dot",
                line_color="rgba(0,0,0,0.3)",
                annotation_text=f"Best: {best_rmse:.1f} kt",
                annotation_position="bottom right",
            )

    fig.update_layout(
        height=380,
        yaxis_title="Rolling TWS RMSE (knots) — lower is better",
        xaxis_tickformat="%H:%M",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=60, b=40, l=60, r=20),
        hovermode="x unified",
    )
    return _apply_blur_theme(fig)


# ============================================================
# MODEL RANKING BAR CHART
# ============================================================

def plot_model_ranking(scores: list) -> go.Figure:
    """Horizontal bar chart of composite scores (0-100), best model at top."""
    model_ids = [s.model_id for s in scores]
    colours = _model_colour_map(model_ids)

    names = [_model_alias(s.model_id) for s in scores]
    composites = [s.composite_score for s in scores]
    tws_rmse = [s.tws_rmse for s in scores]
    twd_rmse = [s.twd_rmse for s in scores]
    tws_bias = [s.tws_bias for s in scores]
    tws_corr = [s.tws_trend_correlation for s in scores]
    bar_colours = [colours.get(s.model_id, "#888888") for s in scores]

    names_r = names[::-1]
    composites_r = composites[::-1]
    custom_r = list(zip(tws_rmse[::-1], twd_rmse[::-1], tws_bias[::-1], tws_corr[::-1]))
    colours_r = bar_colours[::-1]

    fig = go.Figure(go.Bar(
        x=composites_r,
        y=names_r,
        orientation="h",
        marker_color=colours_r,
        text=[f"{c:.1f}" for c in composites_r],
        textposition="inside",
        customdata=custom_r,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Composite: %{x:.1f}/100<br>"
            "TWS RMSE: %{customdata[0]:.1f} kt<br>"
            "TWD RMSE: %{customdata[1]:.0f}°<br>"
            "TWS Bias: %{customdata[2]:+.1f} kt<br>"
            "Trend corr: %{customdata[3]:.0%}<extra></extra>"
        ),
    ))

    fig.update_layout(
        height=max(300, 50 * len(scores)),
        xaxis=dict(title="Composite Score (0-100)", range=[0, 100]),
        yaxis=dict(title=""),
        margin=dict(t=20, b=40, l=120, r=20),
    )
    return _apply_blur_theme(fig)


# ============================================================
# EXPEDITION NUDGE TABLE
# ============================================================

def plot_expedition_nudge_table(scores: list) -> go.Figure:
    """Colour-coded Plotly table of Expedition calibration values."""
    def _tws_colour(offset):
        a = abs(offset)
        if a < 1.0:
            return "#d4edda"
        if a < 2.0:
            return "#fff3cd"
        return "#f8d7da"

    def _twd_colour(offset):
        a = abs(offset)
        if a < 5.0:
            return "#d4edda"
        if a < 15.0:
            return "#fff3cd"
        return "#f8d7da"

    def _lag_colour(lag):
        a = abs(lag)
        if a < 0.5:
            return "#d4edda"
        if a < 2.0:
            return "#fff3cd"
        return "#f8d7da"

    header_vals = ["Rank", "Model", "TWS Offset (kt)", "TWS Scale (%)",
                   "TWD Offset (°)", "Time Lag (h)", "Calibration String"]

    rows = {k: [] for k in header_vals}
    cell_colours = {k: [] for k in header_vals}
    default_bg = "#ffffff"
    best_bg = "#FFF8DC"

    for i, s in enumerate(scores):
        bg = best_bg if i == 0 else default_bg
        n = s.nudge
        rank_str = f"#{i+1}" + (" ★" if i == 0 else "")
        tws_str = f"{n.tws_offset_knots:+.1f}"
        scale_str = f"{int(n.tws_scale * 100)}%"
        twd_str = f"{n.twd_offset_degrees:+.1f}"
        lag_str = f"{n.time_lag_hours:+.1f}h"
        cal_str = n.tws_calibrate_str

        rows["Rank"].append(rank_str)
        rows["Model"].append(_model_alias(s.model_id))
        rows["TWS Offset (kt)"].append(tws_str)
        rows["TWS Scale (%)"].append(scale_str)
        rows["TWD Offset (°)"].append(twd_str)
        rows["Time Lag (h)"].append(lag_str)
        rows["Calibration String"].append(cal_str)

        cell_colours["Rank"].append(bg)
        cell_colours["Model"].append(bg)
        cell_colours["TWS Offset (kt)"].append(_tws_colour(n.tws_offset_knots))
        cell_colours["TWS Scale (%)"].append(bg)
        cell_colours["TWD Offset (°)"].append(_twd_colour(n.twd_offset_degrees))
        cell_colours["Time Lag (h)"].append(_lag_colour(n.time_lag_hours))
        cell_colours["Calibration String"].append(bg)

    fig = go.Figure(go.Table(
        header=dict(
            values=[f"<b>{h}</b>" for h in header_vals],
            fill_color="#2c3e50",
            font=dict(color="white", size=13),
            align="left",
            height=36,
        ),
        cells=dict(
            values=[rows[h] for h in header_vals],
            fill_color=[cell_colours[h] for h in header_vals],
            font=dict(size=13),
            align="left",
            height=34,
        ),
    ))

    fig.update_layout(
        height=max(300, 50 + 40 * len(scores)),
        margin=dict(t=10, b=10, l=10, r=10),
    )
    return _apply_blur_theme(fig)


# ============================================================
# LEAD-TIME HEATMAP
# ============================================================

def plot_lead_time_heatmap(scores: list) -> go.Figure:
    """Heatmap: rows=models, cols=lead-time buckets, cells=TWS RMSE.

    Green (low RMSE) -> red (high RMSE) colour scale.
    Hover shows bias and point count.
    """
    # Collect models that have lead-time data
    models_with_data = [s for s in scores if s.lead_time_buckets]
    if not models_with_data:
        fig = go.Figure()
        fig.add_annotation(text="No lead-time data available", showarrow=False,
                           xref="paper", yref="paper", x=0.5, y=0.5, font=dict(size=16))
        fig.update_layout(height=200)
        return _apply_blur_theme(fig)

    # Build label sets
    all_labels = []
    for s in models_with_data:
        for b in s.lead_time_buckets:
            if b.label not in all_labels:
                all_labels.append(b.label)

    model_names = [_model_alias(s.model_id) for s in models_with_data]
    z = []
    hover_text = []

    for s in models_with_data:
        bucket_map = {b.label: b for b in s.lead_time_buckets}
        row_z = []
        row_hover = []
        for label in all_labels:
            b = bucket_map.get(label)
            if b and b.n_points >= 2:
                row_z.append(b.tws_rmse)
                row_hover.append(
                    f"RMSE: {b.tws_rmse:.1f} kt<br>"
                    f"Bias: {b.tws_bias:+.1f} kt<br>"
                    f"TWD RMSE: {b.twd_rmse:.0f}°<br>"
                    f"n={b.n_points}"
                )
            else:
                row_z.append(None)
                row_hover.append("No data")
        z.append(row_z)
        hover_text.append(row_hover)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=all_labels,
        y=model_names,
        text=hover_text,
        hovertemplate="<b>%{y}</b> — %{x}<br>%{text}<extra></extra>",
        texttemplate="%{z:.1f}",
        colorscale=[[0, "#2ca02c"], [0.5, "#fff3cd"], [1, "#d62728"]],
        zmin=0,
        zmax=max(v for row in z for v in row if v is not None) * 1.1 if any(v for row in z for v in row if v is not None) else 10,
        colorbar=dict(title="TWS RMSE (kt)"),
    ))

    fig.update_layout(
        height=max(250, 60 * len(model_names) + 80),
        xaxis_title="Lead Time (hours before now)",
        yaxis=dict(autorange="reversed"),
        margin=dict(t=30, b=50, l=120, r=20),
    )
    return _apply_blur_theme(fig)


# ============================================================
# WIND REGIME BARS
# ============================================================

def plot_regime_bars(scores: list) -> go.Figure:
    """Grouped bar chart: x=wind regimes, bars=models, y=TWS RMSE.

    Hover shows bias and regime-specific nudge.
    """
    models_with_data = [s for s in scores if s.regime_buckets]
    if not models_with_data:
        fig = go.Figure()
        fig.add_annotation(text="No wind regime data available", showarrow=False,
                           xref="paper", yref="paper", x=0.5, y=0.5, font=dict(size=16))
        fig.update_layout(height=200)
        return _apply_blur_theme(fig)

    colours = _model_colour_map([s.model_id for s in models_with_data])

    # Collect all regime labels
    all_labels = []
    for s in models_with_data:
        for b in s.regime_buckets:
            if b.label not in all_labels:
                all_labels.append(b.label)

    fig = go.Figure()

    for s in models_with_data:
        bucket_map = {b.label: b for b in s.regime_buckets}
        rmses = []
        custom = []
        for label in all_labels:
            b = bucket_map.get(label)
            if b and b.n_points >= 2:
                rmses.append(b.tws_rmse)
                custom.append((b.tws_bias, b.nudge_tws_offset, b.n_points, b.twd_rmse))
            else:
                rmses.append(0)
                custom.append((0, 0, 0, 0))

        alias = _model_alias(s.model_id)
        fig.add_trace(go.Bar(
            name=alias,
            x=all_labels,
            y=rmses,
            marker_color=colours.get(s.model_id, "#888888"),
            customdata=custom,
            hovertemplate=(
                f"<b>{alias}</b><br>"
                "%{x}<br>"
                "TWS RMSE: %{y:.1f} kt<br>"
                "Bias: %{customdata[0]:+.1f} kt<br>"
                "Nudge: %{customdata[1]:+.1f} kt<br>"
                "TWD RMSE: %{customdata[3]:.0f}°<br>"
                "n=%{customdata[2]}<extra></extra>"
            ),
        ))

    fig.update_layout(
        barmode="group",
        height=400,
        yaxis_title="TWS RMSE (knots)",
        xaxis_title="Wind Regime (observed)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=60, b=50, l=60, r=20),
    )
    return _apply_blur_theme(fig)


# ============================================================
# FRONT ANNOTATIONS (overlay on timeseries)
# ============================================================

def plot_front_annotations(
    fig: go.Figure,
    front_results: list,
    obs_events: list,
) -> go.Figure:
    """Add vertical lines for detected frontal passages to an existing figure.

    Black dashed = observed front. Coloured = model-predicted.
    """
    if not obs_events and not front_results:
        return fig

    colours = _model_colour_map([fr.model_id for fr in front_results if fr])

    # Observed fronts: black dashed
    for ev in obs_events:
        fig.add_vline(
            x=ev.datetime,
            line_dash="dash",
            line_color="#000000",
            line_width=2,
            annotation_text="Front (obs)",
            annotation_position="top left",
            annotation_font=dict(size=10, color="#000000"),
        )

    # Model fronts: coloured dotted
    for fr in front_results:
        if fr is None:
            continue
        colour = colours.get(fr.model_id, "#888888")
        alias = _model_alias(fr.model_id)
        for ev in fr.model_events:
            fig.add_vline(
                x=ev.datetime,
                line_dash="dot",
                line_color=colour,
                line_width=1.5,
                annotation_text=alias,
                annotation_position="bottom right",
                annotation_font=dict(size=9, color=colour),
            )

    return fig


# ============================================================
# FRONT TIMING TABLE
# ============================================================

def plot_front_timing_table(front_results: list) -> go.Figure:
    """Small Plotly table showing front timing comparison per model.

    Columns: Model | Obs Front 1 | Model Front 1 | Error 1 | ... | Mean Error
    """
    valid = [fr for fr in front_results if fr is not None]
    if not valid:
        fig = go.Figure()
        fig.add_annotation(text="No frontal passages detected", showarrow=False,
                           xref="paper", yref="paper", x=0.5, y=0.5, font=dict(size=16))
        fig.update_layout(height=150)
        return _apply_blur_theme(fig)

    # Determine max number of matched fronts
    max_fronts = max(len(fr.matched_pairs) for fr in valid) if valid else 0

    headers = ["Model"]
    for i in range(max_fronts):
        headers.extend([f"Obs Front {i+1}", f"Model Front {i+1}", f"Error {i+1} (h)"])
    headers.extend(["Missed", "False Alarms", "Mean Error (h)"])

    rows = {h: [] for h in headers}

    def _error_colour(err_hours):
        if abs(err_hours) < 1.0:
            return "#d4edda"
        if abs(err_hours) < 3.0:
            return "#fff3cd"
        return "#f8d7da"

    cell_colours = {h: [] for h in headers}

    for fr in valid:
        alias = _model_alias(fr.model_id)
        rows["Model"].append(alias)
        cell_colours["Model"].append("#ffffff")

        for i in range(max_fronts):
            obs_key = f"Obs Front {i+1}"
            mod_key = f"Model Front {i+1}"
            err_key = f"Error {i+1} (h)"

            if i < len(fr.matched_pairs):
                obs_ev, mod_ev, err = fr.matched_pairs[i]
                rows[obs_key].append(obs_ev.datetime.strftime("%d %H:%M"))
                rows[mod_key].append(mod_ev.datetime.strftime("%d %H:%M"))
                rows[err_key].append(f"{err:+.1f}")
                cell_colours[obs_key].append("#ffffff")
                cell_colours[mod_key].append("#ffffff")
                cell_colours[err_key].append(_error_colour(err))
            else:
                rows[obs_key].append("-")
                rows[mod_key].append("-")
                rows[err_key].append("-")
                cell_colours[obs_key].append("#ffffff")
                cell_colours[mod_key].append("#ffffff")
                cell_colours[err_key].append("#ffffff")

        rows["Missed"].append(str(fr.n_missed))
        rows["False Alarms"].append(str(fr.n_false_alarms))
        mean_str = f"{fr.mean_timing_error_hours:.1f}" if fr.matched_pairs else "N/A"
        rows["Mean Error (h)"].append(mean_str)

        cell_colours["Missed"].append("#f8d7da" if fr.n_missed > 0 else "#d4edda")
        cell_colours["False Alarms"].append("#fff3cd" if fr.n_false_alarms > 0 else "#d4edda")
        cell_colours["Mean Error (h)"].append(
            _error_colour(fr.mean_timing_error_hours) if fr.matched_pairs else "#ffffff"
        )

    fig = go.Figure(go.Table(
        header=dict(
            values=[f"<b>{h}</b>" for h in headers],
            fill_color="#2c3e50",
            font=dict(color="white", size=12),
            align="left",
            height=34,
        ),
        cells=dict(
            values=[rows[h] for h in headers],
            fill_color=[cell_colours[h] for h in headers],
            font=dict(size=12),
            align="left",
            height=32,
        ),
    ))

    fig.update_layout(
        height=max(180, 50 + 38 * len(valid)),
        margin=dict(t=10, b=10, l=10, r=10),
    )
    return _apply_blur_theme(fig)
