"""Core scoring engine — compares forecasts against observations.

Implements the Model Accuracy-style metrics plus extensions:
- TWS/TWD: Trend correlation, RMS error, bias, calibration
- Time lag detection via cross-correlation (new)
- Composite scoring with configurable weights
- Nudge recommendations

Output format matches Model Accuracy conventions for familiarity.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np
import pandas as pd
from scipy import signal, stats

from .config import ScoringWeights, DEFAULT_WEIGHTS, MODELS

logger = logging.getLogger(__name__)


@dataclass
class NudgeRecommendation:
    """Calibration recommendation for a model."""
    tws_offset_knots: float  # Add this to model TWS (positive = model underpredicts)
    tws_scale: float         # Multiply model TWS by this (e.g., 0.95 = reduce 5%)
    twd_offset_degrees: float  # Add this to model TWD (positive = rotate clockwise)
    time_lag_hours: float    # Shift model time by this (positive = model is late)
    
    @property
    def tws_calibrate_str(self) -> str:
        """Model Accuracy-style calibration string."""
        sign = "+" if self.tws_offset_knots >= 0 else ""
        pct = int(self.tws_scale * 100)
        return f"Calibrate {sign}{self.tws_offset_knots:.1f} knots ({pct}%)"
    
    @property
    def twd_calibrate_str(self) -> str:
        sign = "+" if self.twd_offset_degrees >= 0 else ""
        direction = "right" if self.twd_offset_degrees > 0 else "left"
        return f"Calibrate {sign}{self.twd_offset_degrees:.1f} degrees ({direction} of observed)"


@dataclass
class LeadTimeBucket:
    """Scoring metrics for a specific lead-time window."""
    label: str          # "T+0-6h"
    hours_start: int
    hours_end: int
    n_points: int
    tws_rmse: float
    tws_bias: float
    twd_rmse: float
    twd_bias: float


@dataclass
class RegimeBucket:
    """Scoring metrics binned by observed wind speed regime."""
    label: str            # "Light (0-8 kt)"
    tws_min: float
    tws_max: float
    n_points: int
    tws_rmse: float
    tws_bias: float
    twd_rmse: float
    twd_bias: float
    nudge_tws_offset: float  # regime-specific correction = -bias


LEAD_TIME_BUCKETS = [
    (0, 6, "T+0-6h"),
    (6, 12, "T+6-12h"),
    (12, 24, "T+12-24h"),
    (24, 48, "T+24-48h"),
]

WIND_REGIMES = [
    (0, 8, "Light (0-8 kt)"),
    (8, 15, "Moderate (8-15 kt)"),
    (15, 25, "Fresh (15-25 kt)"),
    (25, 999, "Strong (25+ kt)"),
]


@dataclass
class ModelScore:
    """Complete scoring result for a single model."""
    model_id: str
    model_name: str
    n_points: int
    
    # TWS metrics
    tws_trend_correlation: float  # -1 to 1 (higher = better)
    tws_rmse: float              # knots
    tws_mae: float               # knots
    tws_bias: float              # knots (positive = overpredicts)
    tws_scale: float             # ratio (predicted/observed mean)
    
    # TWD metrics
    twd_trend_correlation: float
    twd_rmse: float              # degrees
    twd_mae: float               # degrees
    twd_bias: float              # degrees (positive = clockwise bias)
    
    # Time lag
    time_lag_hours: float        # Cross-correlation detected offset
    
    # Composite
    composite_score: float       # 0-100 (higher = better)
    
    # Nudge
    nudge: NudgeRecommendation = None
    
    # Model Accuracy-style composite error (lower = better)
    ma_error: float = 0.0

    # Extended scoring (populated when compute_extras=True)
    lead_time_buckets: Optional[List[LeadTimeBucket]] = None
    regime_buckets: Optional[List[RegimeBucket]] = None
    front_timing: Optional[object] = None

    def summary(self) -> str:
        """Generate Model Accuracy-style text summary."""
        lines = [
            f"{self.model_name}:",
            f"  TWS: Trend correlation {self.tws_trend_correlation:.0%}, "
            f"RMS error {self.tws_rmse:.1f}, "
            f"Average error {abs(self.tws_bias):.1f} knots "
            f"{'above' if self.tws_bias > 0 else 'below'} observed.",
            f"  {self.nudge.tws_calibrate_str}",
            f"  TWD: Trend correlation {self.twd_trend_correlation:.0%}, "
            f"RMS error {self.twd_rmse:.1f}, "
            f"Average error {abs(self.twd_bias):.1f} degrees "
            f"{'right' if self.twd_bias > 0 else 'left'} of observed.",
            f"  {self.nudge.twd_calibrate_str}",
        ]
        if abs(self.time_lag_hours) > 0.5:
            lines.append(
                f"  Time lag: {self.time_lag_hours:+.1f} hours "
                f"({'model is late' if self.time_lag_hours > 0 else 'model is early'})"
            )
        lines.append(f"  Composite score: {self.composite_score:.1f}/100")
        lines.append(f"  MA error: {self.ma_error:.2f}")
        lines.append(f"  ({self.n_points} data points)")
        return "\n".join(lines)


# ============================================================
# CIRCULAR STATISTICS (for wind direction)
# ============================================================

def circular_diff(predicted: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Signed difference between two angles in degrees.
    
    Returns values in [-180, 180]. Positive = predicted is clockwise of observed.
    """
    diff = (predicted - observed + 180) % 360 - 180
    return diff


def circular_rmse(predicted: np.ndarray, observed: np.ndarray) -> float:
    """Root mean square of circular differences."""
    diffs = circular_diff(predicted, observed)
    return np.sqrt(np.mean(diffs ** 2))


def circular_mae(predicted: np.ndarray, observed: np.ndarray) -> float:
    """Mean absolute circular difference."""
    diffs = circular_diff(predicted, observed)
    return np.mean(np.abs(diffs))


def circular_bias(predicted: np.ndarray, observed: np.ndarray) -> float:
    """Mean signed circular difference (bias)."""
    diffs = circular_diff(predicted, observed)
    return np.mean(diffs)


def circular_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Circular correlation coefficient between two angle series."""
    a_rad = np.radians(a)
    b_rad = np.radians(b)
    
    sin_a = np.sin(a_rad - np.mean(a_rad))
    sin_b = np.sin(b_rad - np.mean(b_rad))
    
    numerator = np.sum(sin_a * sin_b)
    denominator = np.sqrt(np.sum(sin_a ** 2) * np.sum(sin_b ** 2))
    
    if denominator == 0:
        return 0.0
    return numerator / denominator


# ============================================================
# TIME LAG DETECTION
# ============================================================

def detect_time_lag(
    predicted: np.ndarray,
    observed: np.ndarray,
    max_lag_hours: int = 6,
    timestep_hours: float = 1.0,
) -> float:
    """Detect timing offset between forecast and observation using cross-correlation.
    
    Returns the lag in hours. Positive = forecast is late.
    
    Args:
        predicted: Forecast time series
        observed: Observation time series
        max_lag_hours: Maximum lag to search
        timestep_hours: Time step between samples
    
    Returns:
        Optimal lag in hours (positive = model is late, negative = early)
    """
    if len(predicted) < 4 or len(observed) < 4:
        return 0.0
    
    # Normalize both series
    pred_norm = (predicted - np.mean(predicted)) / (np.std(predicted) + 1e-8)
    obs_norm = (observed - np.mean(observed)) / (np.std(observed) + 1e-8)
    
    # Cross-correlate
    max_lag_samples = int(max_lag_hours / timestep_hours)
    correlations = []
    lags = range(-max_lag_samples, max_lag_samples + 1)
    
    for lag in lags:
        if lag >= 0:
            p = pred_norm[lag:]
            o = obs_norm[:len(p)]
        else:
            o = obs_norm[-lag:]
            p = pred_norm[:len(o)]
        
        n = min(len(p), len(o))
        if n < 3:
            correlations.append(0)
            continue
        
        corr = np.corrcoef(p[:n], o[:n])[0, 1]
        correlations.append(corr if not np.isnan(corr) else 0)
    
    # Find the lag with maximum correlation
    best_idx = np.argmax(correlations)
    best_lag = list(lags)[best_idx]
    
    return best_lag * timestep_hours


# ============================================================
# LEAD-TIME AND REGIME BREAKDOWN
# ============================================================

def compute_lead_time_buckets(
    merged: pd.DataFrame,
    now: Optional[pd.Timestamp] = None,
) -> List[LeadTimeBucket]:
    """Bin matched forecast/obs points by hours-before-now as a lead-time proxy.

    Since Open-Meteo doesn't expose model init time, we approximate lead time
    as how far in the past each observation hour is from 'now'.  Points from
    6 hours ago → T+0-6h bucket, 18 hours ago → T+12-24h, etc.

    Args:
        merged: DataFrame with columns [hour, wind_speed_knots_fcst,
                wind_speed_knots_obs, wind_direction_fcst, wind_direction_obs]
        now: Reference timestamp (default: current UTC time)

    Returns:
        List of LeadTimeBucket (only buckets with ≥ 2 points)
    """
    if now is None:
        now = pd.Timestamp.utcnow().tz_localize(None)

    merged = merged.copy()
    merged["hours_ago"] = (now - merged["hour"]).dt.total_seconds() / 3600.0

    buckets = []
    for h_start, h_end, label in LEAD_TIME_BUCKETS:
        mask = (merged["hours_ago"] >= h_start) & (merged["hours_ago"] < h_end)
        subset = merged[mask]
        if len(subset) < 2:
            continue

        tws_err = subset["wind_speed_knots_fcst"].values - subset["wind_speed_knots_obs"].values
        twd_diffs = circular_diff(
            subset["wind_direction_fcst"].values,
            subset["wind_direction_obs"].values,
        )

        buckets.append(LeadTimeBucket(
            label=label,
            hours_start=h_start,
            hours_end=h_end,
            n_points=len(subset),
            tws_rmse=float(np.sqrt(np.mean(tws_err ** 2))),
            tws_bias=float(np.mean(tws_err)),
            twd_rmse=float(np.sqrt(np.mean(twd_diffs ** 2))),
            twd_bias=float(np.mean(twd_diffs)),
        ))

    return buckets


def compute_regime_buckets(merged: pd.DataFrame) -> List[RegimeBucket]:
    """Bin matched points by *observed* wind speed regime.

    Regimes are defined in WIND_REGIMES.  The regime-specific nudge is
    simply -bias for that bucket.

    Args:
        merged: DataFrame with columns [wind_speed_knots_fcst,
                wind_speed_knots_obs, wind_direction_fcst, wind_direction_obs]

    Returns:
        List of RegimeBucket (only regimes with ≥ 2 points)
    """
    buckets = []
    for tws_min, tws_max, label in WIND_REGIMES:
        mask = (merged["wind_speed_knots_obs"] >= tws_min) & (merged["wind_speed_knots_obs"] < tws_max)
        subset = merged[mask]
        if len(subset) < 2:
            continue

        tws_err = subset["wind_speed_knots_fcst"].values - subset["wind_speed_knots_obs"].values
        twd_diffs = circular_diff(
            subset["wind_direction_fcst"].values,
            subset["wind_direction_obs"].values,
        )
        tws_bias = float(np.mean(tws_err))

        buckets.append(RegimeBucket(
            label=label,
            tws_min=tws_min,
            tws_max=tws_max,
            n_points=len(subset),
            tws_rmse=float(np.sqrt(np.mean(tws_err ** 2))),
            tws_bias=tws_bias,
            twd_rmse=float(np.sqrt(np.mean(twd_diffs ** 2))),
            twd_bias=float(np.mean(twd_diffs)),
            nudge_tws_offset=-tws_bias,
        ))

    return buckets


# ============================================================
# MAIN SCORING FUNCTION
# ============================================================

def score_model(
    model_id: str,
    forecast_df: pd.DataFrame,
    observation_df: pd.DataFrame,
    weights: ScoringWeights = None,
    compute_extras: bool = True,
) -> ModelScore:
    """Score a single model's forecast against observations.
    
    Both DataFrames must have columns: datetime, wind_speed_knots, wind_direction
    They will be aligned on datetime (nearest hour).
    
    Args:
        model_id: Model identifier from config.MODELS
        forecast_df: Forecast data
        observation_df: Observation data (truth)
        weights: Scoring weights (default: DEFAULT_WEIGHTS)
    
    Returns:
        ModelScore with all metrics and nudge recommendation
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    
    model_info = MODELS.get(model_id, {"name": model_id})
    model_name = model_info.get("name", model_id)
    
    # Align on datetime — round to nearest hour and merge
    fcst = forecast_df.copy()
    obs = observation_df.copy()
    
    fcst["hour"] = fcst["datetime"].dt.floor("h")
    obs["hour"] = obs["datetime"].dt.floor("h")
    
    # If observations are sub-hourly, aggregate to hourly
    if len(obs) > len(obs["hour"].unique()) * 1.5:
        # Group by hour, using circular mean for direction
        def _circ_mean(angles):
            rad = np.radians(angles.dropna().values)
            if len(rad) == 0:
                return np.nan
            return np.degrees(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360
        
        obs_hourly = obs.groupby("hour").agg({
            "wind_speed_knots": "mean",
            "wind_direction": _circ_mean,
        }).reset_index()
    else:
        obs_hourly = obs[["hour", "wind_speed_knots", "wind_direction"]].drop_duplicates("hour")
    
    fcst_hourly = fcst[["hour", "wind_speed_knots", "wind_direction"]].drop_duplicates("hour")
    
    # Merge
    merged = pd.merge(
        fcst_hourly, obs_hourly,
        on="hour", suffixes=("_fcst", "_obs"),
        how="inner",
    )
    
    if len(merged) < 3:
        logger.warning(f"Only {len(merged)} matched time points for {model_name}. Need at least 3.")
        return ModelScore(
            model_id=model_id, model_name=model_name, n_points=len(merged),
            tws_trend_correlation=0, tws_rmse=99, tws_mae=99, tws_bias=0, tws_scale=1,
            twd_trend_correlation=0, twd_rmse=180, twd_mae=180, twd_bias=0,
            time_lag_hours=0, composite_score=0, ma_error=99,
            nudge=NudgeRecommendation(0, 1, 0, 0),
        )
    
    pred_tws = merged["wind_speed_knots_fcst"].values
    obs_tws = merged["wind_speed_knots_obs"].values
    pred_twd = merged["wind_direction_fcst"].values
    obs_twd = merged["wind_direction_obs"].values
    
    # --- TWS metrics ---
    tws_errors = pred_tws - obs_tws
    tws_rmse = np.sqrt(np.mean(tws_errors ** 2))
    tws_mae = np.mean(np.abs(tws_errors))
    tws_bias = np.mean(tws_errors)
    tws_scale = np.mean(pred_tws) / (np.mean(obs_tws) + 1e-8)
    
    # Trend correlation (Pearson)
    if np.std(pred_tws) > 0 and np.std(obs_tws) > 0:
        tws_trend_corr, _ = stats.pearsonr(pred_tws, obs_tws)
    else:
        tws_trend_corr = 0.0
    
    # --- TWD metrics ---
    twd_rmse_val = circular_rmse(pred_twd, obs_twd)
    twd_mae_val = circular_mae(pred_twd, obs_twd)
    twd_bias_val = circular_bias(pred_twd, obs_twd)
    twd_trend_corr = circular_correlation(pred_twd, obs_twd)
    
    # --- Time lag ---
    time_lag = detect_time_lag(pred_tws, obs_tws, max_lag_hours=6, timestep_hours=1.0)
    
    # --- Nudge recommendation ---
    nudge = NudgeRecommendation(
        tws_offset_knots=-tws_bias,  # Negate: if model is +1.3 high, calibrate -1.3
        tws_scale=1.0 / (tws_scale + 1e-8),
        twd_offset_degrees=-twd_bias_val,
        time_lag_hours=time_lag,
    )
    
    # --- Model Accuracy-style composite error ---
    # MA uses a combined metric; lower is better. We approximate their formula:
    # Weighted combination of TWS RMSE and TWD RMSE
    ma_error = tws_rmse * 0.5 + (twd_rmse_val / 30.0) * 0.5 * tws_rmse
    
    # --- Composite score (0-100, higher = better) ---
    # Convert each metric to a 0-100 sub-score
    tws_rmse_score = max(0, 100 - tws_rmse * 12)   # 0 RMSE = 100, ~8 kt RMSE = 0
    twd_rmse_score = max(0, 100 - twd_rmse_val * 1.5)  # 0° RMSE = 100, ~67° RMSE = 0
    tws_trend_score = max(0, tws_trend_corr * 100)
    twd_trend_score = max(0, twd_trend_corr * 100)
    tws_bias_score = max(0, 100 - abs(tws_bias) * 15)
    twd_bias_score = max(0, 100 - abs(twd_bias_val) * 2)
    time_lag_score = max(0, 100 - abs(time_lag) * 20)
    
    composite = (
        tws_rmse_score * weights.tws_rmse +
        twd_rmse_score * weights.twd_rmse +
        tws_trend_score * weights.tws_trend +
        twd_trend_score * weights.twd_trend +
        tws_bias_score * weights.tws_bias_penalty +
        twd_bias_score * weights.twd_bias_penalty +
        time_lag_score * weights.time_lag_penalty
    )
    
    # --- Extended scoring ---
    lead_time = None
    regime = None
    front = None

    if compute_extras:
        lead_time = compute_lead_time_buckets(merged)
        regime = compute_regime_buckets(merged)
        # Front timing scored externally via fronts.score_front_timing()

    return ModelScore(
        model_id=model_id,
        model_name=model_name,
        n_points=len(merged),
        tws_trend_correlation=tws_trend_corr,
        tws_rmse=tws_rmse,
        tws_mae=tws_mae,
        tws_bias=tws_bias,
        tws_scale=tws_scale,
        twd_trend_correlation=twd_trend_corr,
        twd_rmse=twd_rmse_val,
        twd_mae=twd_mae_val,
        twd_bias=twd_bias_val,
        time_lag_hours=time_lag,
        composite_score=composite,
        ma_error=ma_error,
        nudge=nudge,
        lead_time_buckets=lead_time,
        regime_buckets=regime,
        front_timing=front,
    )


def score_all_models(
    forecasts: dict[str, pd.DataFrame],
    observations: pd.DataFrame,
    weights: ScoringWeights = None,
) -> list[ModelScore]:
    """Score all models and return ranked results.
    
    Args:
        forecasts: Dict of model_id -> forecast DataFrame
        observations: Observation DataFrame
        weights: Scoring weights
    
    Returns:
        List of ModelScore, sorted best to worst (highest composite first)
    """
    scores = []
    for model_id, fcst_df in forecasts.items():
        score = score_model(model_id, fcst_df, observations, weights)
        scores.append(score)
    
    # Sort by composite score (descending)
    scores.sort(key=lambda s: s.composite_score, reverse=True)
    
    return scores


def print_ranking(scores: list[ModelScore]) -> str:
    """Generate a full Model Accuracy-style ranking report."""
    lines = [
        "=" * 70,
        "WEATHER MODEL SCORING REPORT",
        "=" * 70,
        "",
    ]
    
    for i, score in enumerate(scores, 1):
        rank_marker = " *** RECOMMENDED ***" if i == 1 else ""
        lines.append(f"#{i}{rank_marker}")
        lines.append(score.summary())
        lines.append("")
    
    # Recommendation
    best = scores[0]
    lines.append("-" * 70)
    lines.append(
        f"ModelAccuracy recommends {best.model_name}, "
        f"with a composite score of {best.composite_score:.1f}/100 "
        f"(MA error: {best.ma_error:.2f})"
    )
    lines.append("")
    lines.append(f"Nudge for routing: {best.nudge.tws_calibrate_str}")
    lines.append(f"                   {best.nudge.twd_calibrate_str}")
    if abs(best.nudge.time_lag_hours) > 0.5:
        lines.append(
            f"                   Time shift: {best.nudge.time_lag_hours:+.1f} hours"
        )
    lines.append("-" * 70)
    
    report = "\n".join(lines)
    return report
