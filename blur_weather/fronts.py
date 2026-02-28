"""Point-based frontal passage detection and model timing comparison.

Detects fronts using co-occurrence of:
  - Pressure tendency sign change (falling → rising)
  - Wind direction shift > threshold within a time window

Works with hourly SMHI observations and hourly Open-Meteo forecasts.
No GRIB spatial analysis required — purely time-series based.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .score import circular_diff

logger = logging.getLogger(__name__)


@dataclass
class FrontalEvent:
    """A detected frontal passage at a single point."""
    datetime: object           # pd.Timestamp
    pressure_tendency: float   # hPa/3h (negative = falling, positive = rising)
    twd_shift_degrees: float   # signed wind shift over window
    source: str                # "obs" or model_id


@dataclass
class FrontTimingResult:
    """Comparison of front timing between a model and observations."""
    model_id: str
    model_name: str
    obs_events: List[FrontalEvent]
    model_events: List[FrontalEvent]
    matched_pairs: list    # [(obs_event, model_event, timing_error_hours), ...]
    mean_timing_error_hours: float
    n_missed: int          # obs fronts not matched by model
    n_false_alarms: int    # model fronts with no corresponding obs


def compute_pressure_tendency(
    df: pd.DataFrame,
    window_hours: int = 3,
) -> pd.Series:
    """Compute pressure tendency as dP/dt (hPa per window_hours).

    Args:
        df: DataFrame with columns [datetime, pressure_hpa], sorted by datetime
        window_hours: Differencing window in hours

    Returns:
        Series of pressure tendency values, same index as df
    """
    if "pressure_hpa" not in df.columns:
        return pd.Series(dtype=float)

    df = df.sort_values("datetime").copy()
    pressure = df["pressure_hpa"]

    tendency = pressure.diff(periods=window_hours)
    return tendency


def compute_wind_shift(
    df: pd.DataFrame,
    window_hours: int = 3,
) -> pd.Series:
    """Compute wind direction shift over a rolling window.

    Uses circular difference so 350° → 20° correctly yields +30°.

    Args:
        df: DataFrame with columns [datetime, wind_direction], sorted by datetime
        window_hours: Differencing window in hours

    Returns:
        Series of signed wind shifts (degrees), same index as df
    """
    df = df.sort_values("datetime").copy()
    twd = df["wind_direction"].values

    shifts = np.full(len(twd), np.nan)
    for i in range(window_hours, len(twd)):
        shifts[i] = circular_diff(
            np.array([twd[i]]),
            np.array([twd[i - window_hours]]),
        )[0]

    return pd.Series(shifts, index=df.index)


def detect_frontal_events(
    df: pd.DataFrame,
    source: str,
    pressure_threshold: float = 2.0,
    wind_shift_threshold: float = 30.0,
    window_hours: int = 3,
    min_gap_hours: int = 6,
) -> List[FrontalEvent]:
    """Detect frontal passages from a time series.

    A front is detected where:
      1. Pressure tendency changes sign (falling → rising), AND
      2. Absolute wind shift exceeds threshold within the same window

    Args:
        df: DataFrame with [datetime, pressure_hpa, wind_direction]
        source: "obs" or model_id
        pressure_threshold: Minimum |dP/dt| in hPa/window to count
        wind_shift_threshold: Minimum |wind shift| in degrees
        window_hours: Window for tendency/shift computation
        min_gap_hours: Minimum gap between detected fronts

    Returns:
        List of FrontalEvent, sorted chronologically
    """
    if "pressure_hpa" not in df.columns or df["pressure_hpa"].isna().all():
        return []
    if "wind_direction" not in df.columns:
        return []

    df = df.sort_values("datetime").reset_index(drop=True)
    tendency = compute_pressure_tendency(df, window_hours)
    wind_shift = compute_wind_shift(df, window_hours)

    events = []
    last_event_time = None

    for i in range(window_hours + 1, len(df)):
        prev_tendency = tendency.iloc[i - 1]
        curr_tendency = tendency.iloc[i]

        if pd.isna(prev_tendency) or pd.isna(curr_tendency):
            continue

        # Detect sign change: falling → rising (cold front passage)
        if prev_tendency < -pressure_threshold * 0.3 and curr_tendency > pressure_threshold * 0.3:
            ws = wind_shift.iloc[i]
            if pd.isna(ws):
                continue

            if abs(ws) >= wind_shift_threshold:
                event_time = df["datetime"].iloc[i]

                # Enforce minimum gap
                if last_event_time is not None:
                    gap = (event_time - last_event_time).total_seconds() / 3600
                    if gap < min_gap_hours:
                        continue

                events.append(FrontalEvent(
                    datetime=event_time,
                    pressure_tendency=float(curr_tendency),
                    twd_shift_degrees=float(ws),
                    source=source,
                ))
                last_event_time = event_time

    logger.info(f"  Detected {len(events)} frontal events in {source}")
    return events


def match_front_events(
    obs_events: List[FrontalEvent],
    model_events: List[FrontalEvent],
    max_match_hours: float = 6.0,
) -> Tuple[list, int, int]:
    """Match observed and modelled frontal events by nearest time.

    Greedy nearest-neighbour with consumed flag — each event matched at most once.

    Args:
        obs_events: Observed frontal events
        model_events: Model-predicted frontal events
        max_match_hours: Maximum time difference to count as a match

    Returns:
        (matched_pairs, n_missed, n_false_alarms)
        matched_pairs: [(obs_event, model_event, timing_error_hours), ...]
    """
    if not obs_events or not model_events:
        return [], len(obs_events), len(model_events)

    model_used = [False] * len(model_events)
    matched = []

    for obs_ev in obs_events:
        best_j = None
        best_dt = float("inf")

        for j, mod_ev in enumerate(model_events):
            if model_used[j]:
                continue
            dt_hours = (mod_ev.datetime - obs_ev.datetime).total_seconds() / 3600
            if abs(dt_hours) < abs(best_dt) and abs(dt_hours) <= max_match_hours:
                best_dt = dt_hours
                best_j = j

        if best_j is not None:
            model_used[best_j] = True
            matched.append((obs_ev, model_events[best_j], best_dt))

    n_missed = len(obs_events) - len(matched)
    n_false_alarms = sum(1 for used in model_used if not used)

    return matched, n_missed, n_false_alarms


def score_front_timing(
    model_id: str,
    model_name: str,
    forecast_df: pd.DataFrame,
    observation_df: pd.DataFrame,
) -> Optional[FrontTimingResult]:
    """Score a model's front timing against observations.

    Returns None if observations lack pressure data or no fronts are detected.

    Args:
        model_id: Model identifier
        model_name: Human-readable model name
        forecast_df: Model forecast with [datetime, pressure_hpa, wind_direction]
        observation_df: Observations with [datetime, pressure_hpa, wind_direction]

    Returns:
        FrontTimingResult or None
    """
    # Check prerequisites
    if "pressure_hpa" not in observation_df.columns:
        return None
    if observation_df["pressure_hpa"].isna().all():
        return None
    if "pressure_hpa" not in forecast_df.columns:
        return None

    obs_events = detect_frontal_events(observation_df, source="obs")
    model_events = detect_frontal_events(forecast_df, source=model_id)

    if not obs_events and not model_events:
        return None

    matched, n_missed, n_false_alarms = match_front_events(obs_events, model_events)

    mean_error = 0.0
    if matched:
        mean_error = float(np.mean([abs(dt) for _, _, dt in matched]))

    return FrontTimingResult(
        model_id=model_id,
        model_name=model_name,
        obs_events=obs_events,
        model_events=model_events,
        matched_pairs=matched,
        mean_timing_error_hours=mean_error,
        n_missed=n_missed,
        n_false_alarms=n_false_alarms,
    )
