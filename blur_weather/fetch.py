"""Fetch weather forecasts from Open-Meteo API for multiple models.

Open-Meteo provides free access to multiple weather models via a single API.
No API key needed. Returns hourly JSON data.

Usage:
    forecasts = fetch_multi_model_forecast(lat=57.7, lon=18.3, models=DEFAULT_MODELS)
    historical = fetch_historical_forecasts(lat=57.7, lon=18.3, date="2025-09-05")
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests
import pandas as pd

from .config import (
    OPEN_METEO_BASE_URL,
    OPEN_METEO_ARCHIVE_URL,
    WIND_VARIABLES,
    DEFAULT_MODELS,
    MODELS,
    ms_to_knots,
)

logger = logging.getLogger(__name__)


def fetch_multi_model_forecast(
    lat: float,
    lon: float,
    models: list[str] = None,
    forecast_days: int = 5,
    past_days: int = 0,
) -> dict[str, pd.DataFrame]:
    """Fetch forecasts from multiple models for a single location.
    
    Args:
        lat: Latitude
        lon: Longitude
        models: List of Open-Meteo model identifiers (from config.MODELS)
        forecast_days: Number of forecast days
        past_days: Include this many past days (for recent verification)
    
    Returns:
        Dict mapping model name to DataFrame with columns:
        [datetime, wind_speed_knots, wind_direction, wind_gusts_knots, pressure_hpa]
    """
    if models is None:
        models = DEFAULT_MODELS

    results = {}
    
    for model_id in models:
        model_info = MODELS.get(model_id, {"name": model_id})
        logger.info(f"Fetching {model_info.get('name', model_id)} for ({lat:.2f}, {lon:.2f})")
        
        try:
            params = {
                "latitude": lat,
                "longitude": lon,
                "hourly": ",".join(WIND_VARIABLES),
                "models": model_id,
                "forecast_days": forecast_days,
                "past_days": past_days,
                "wind_speed_unit": "ms",  # We'll convert to knots ourselves
                "timezone": "UTC",
            }
            
            response = requests.get(OPEN_METEO_BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if "hourly" not in data:
                logger.warning(f"No hourly data in response for {model_id}")
                continue
            
            hourly = data["hourly"]
            df = pd.DataFrame({
                "datetime": pd.to_datetime(hourly["time"]),
                "wind_speed_ms": hourly.get("wind_speed_10m", []),
                "wind_direction": hourly.get("wind_direction_10m", []),
                "wind_gusts_ms": hourly.get("wind_gusts_10m", []),
                "pressure_hpa": hourly.get("pressure_msl", []),
            })
            
            # Convert to knots
            df["wind_speed_knots"] = df["wind_speed_ms"].apply(
                lambda x: ms_to_knots(x) if pd.notna(x) else None
            )
            df["wind_gusts_knots"] = df["wind_gusts_ms"].apply(
                lambda x: ms_to_knots(x) if pd.notna(x) else None
            )
            
            df = df.dropna(subset=["wind_speed_knots", "wind_direction"])
            results[model_id] = df
            logger.info(f"  Got {len(df)} hourly records for {model_info.get('name', model_id)}")
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {model_id}: {e}")
        except (KeyError, ValueError) as e:
            logger.error(f"Failed to parse {model_id} response: {e}")
    
    return results


def fetch_historical_forecasts(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    models: list[str] = None,
) -> dict[str, pd.DataFrame]:
    """Fetch archived forecast data for a past period.
    
    Uses Open-Meteo Historical Forecast API to retrieve what each model
    predicted for a given date range. Essential for backtesting.
    
    Args:
        lat: Latitude
        lon: Longitude
        start_date: Start date as "YYYY-MM-DD"
        end_date: End date as "YYYY-MM-DD"
        models: List of model identifiers
    
    Returns:
        Dict mapping model name to DataFrame
    """
    if models is None:
        models = DEFAULT_MODELS
    
    results = {}
    
    for model_id in models:
        model_info = MODELS.get(model_id, {"name": model_id})
        logger.info(f"Fetching historical {model_info.get('name', model_id)} for {start_date} to {end_date}")
        
        try:
            params = {
                "latitude": lat,
                "longitude": lon,
                "hourly": ",".join(WIND_VARIABLES),
                "models": model_id,
                "start_date": start_date,
                "end_date": end_date,
                "wind_speed_unit": "ms",
                "timezone": "UTC",
            }
            
            response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            if "hourly" not in data:
                logger.warning(f"No historical data for {model_id}")
                continue
            
            hourly = data["hourly"]
            df = pd.DataFrame({
                "datetime": pd.to_datetime(hourly["time"]),
                "wind_speed_ms": hourly.get("wind_speed_10m", []),
                "wind_direction": hourly.get("wind_direction_10m", []),
                "wind_gusts_ms": hourly.get("wind_gusts_10m", []),
                "pressure_hpa": hourly.get("pressure_msl", []),
            })
            
            df["wind_speed_knots"] = df["wind_speed_ms"].apply(
                lambda x: ms_to_knots(x) if pd.notna(x) else None
            )
            df["wind_gusts_knots"] = df["wind_gusts_ms"].apply(
                lambda x: ms_to_knots(x) if pd.notna(x) else None
            )
            
            df = df.dropna(subset=["wind_speed_knots", "wind_direction"])
            results[model_id] = df
            logger.info(f"  Got {len(df)} records")
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch historical {model_id}: {e}")
    
    return results


def archive_forecasts(
    forecasts: dict[str, pd.DataFrame],
    lat: float,
    lon: float,
    data_dir: str = "data",
) -> Path:
    """Save fetched forecasts to disk for future analysis.
    
    Creates timestamped JSON files in the data directory.
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
    
    for model_id, df in forecasts.items():
        filename = f"forecast_{model_id}_{lat:.2f}_{lon:.2f}_{timestamp}.csv"
        filepath = data_path / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Archived {model_id} -> {filepath}")
    
    return data_path
