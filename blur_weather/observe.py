"""Fetch observations from SMHI Open Data API and parse Expedition log files.

SMHI provides free access to weather station data around the Swedish coast.
Expedition logs provide onboard instrument truth when sailing.

Both produce a common DataFrame format for comparison with forecasts.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import requests

from .config import (
    SMHI_BASE_URL,
    SMHI_WIND_SPEED_PARAM,
    SMHI_WIND_DIRECTION_PARAM,
    SMHI_WIND_GUST_PARAM,
    SMHI_PRESSURE_PARAM,
    SMHI_STATIONS,
    ms_to_knots,
)

logger = logging.getLogger(__name__)


# ============================================================
# SMHI Open Data API
# ============================================================

def fetch_smhi_observations(
    station_id: int,
    parameter: int = SMHI_WIND_SPEED_PARAM,
    period: str = "latest-day",
) -> pd.DataFrame:
    """Fetch observations from a single SMHI station.
    
    Args:
        station_id: SMHI station ID (see config.SMHI_STATIONS)
        parameter: SMHI parameter ID (4=wind speed, 3=wind dir, etc.)
        period: "latest-hour", "latest-day", "latest-months"
    
    Returns:
        DataFrame with columns [datetime, value]
    """
    url = f"{SMHI_BASE_URL}/version/1.0/parameter/{parameter}/station/{station_id}/period/{period}/data.json"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        records = []
        for entry in data.get("value", []):
            ts = entry.get("date", 0)  # Unix timestamp in milliseconds
            val = entry.get("value")
            if val is not None:
                records.append({
                    "datetime": pd.Timestamp(ts, unit="ms", tz="UTC").tz_localize(None),
                    "value": float(val),
                })
        
        return pd.DataFrame(records)
    
    except requests.RequestException as e:
        logger.error(f"SMHI API error for station {station_id}, param {parameter}: {e}")
        return pd.DataFrame()


def fetch_smhi_wind_observations(
    station_name: str,
    period: str = "latest-day",
) -> pd.DataFrame:
    """Fetch wind speed, direction, and pressure (if available) from a named SMHI station.

    Returns DataFrame with columns:
    [datetime, wind_speed_knots, wind_direction, station, lat, lon]
    Plus optional: [pressure_hpa]  — only if the station reports pressure.
    """
    station = SMHI_STATIONS.get(station_name)
    if station is None:
        logger.error(f"Unknown station: {station_name}. Available: {list(SMHI_STATIONS.keys())}")
        return pd.DataFrame()

    logger.info(f"Fetching SMHI observations from {station.name} (ID: {station.smhi_station_id})")

    # Fetch wind speed (m/s)
    speed_df = fetch_smhi_observations(station.smhi_station_id, SMHI_WIND_SPEED_PARAM, period)
    if speed_df.empty:
        logger.warning(f"No wind speed data from {station.name}")
        return pd.DataFrame()
    speed_df = speed_df.rename(columns={"value": "wind_speed_ms"})

    # Fetch wind direction
    dir_df = fetch_smhi_observations(station.smhi_station_id, SMHI_WIND_DIRECTION_PARAM, period)
    if dir_df.empty:
        logger.warning(f"No wind direction data from {station.name}")
        return pd.DataFrame()
    dir_df = dir_df.rename(columns={"value": "wind_direction"})

    # Merge on datetime (SMHI reports at same times for both params)
    merged = pd.merge(speed_df, dir_df, on="datetime", how="inner")
    merged["wind_speed_knots"] = merged["wind_speed_ms"].apply(ms_to_knots)
    merged["station"] = station.name
    merged["lat"] = station.lat
    merged["lon"] = station.lon

    # Fetch pressure (optional — not all stations report it)
    pres_df = fetch_smhi_observations(station.smhi_station_id, SMHI_PRESSURE_PARAM, period)
    if not pres_df.empty:
        pres_df = pres_df.rename(columns={"value": "pressure_hpa"})
        merged = pd.merge(merged, pres_df, on="datetime", how="left")
        logger.info(f"  Pressure data available from {station.name}")
    else:
        logger.info(f"  No pressure data from {station.name} (station may not report it)")

    cols = ["datetime", "wind_speed_knots", "wind_direction", "station", "lat", "lon"]
    if "pressure_hpa" in merged.columns:
        cols.append("pressure_hpa")

    logger.info(f"  Got {len(merged)} observation records from {station.name}")
    return merged[cols]


def fetch_course_observations(
    course_name: str,
    period: str = "latest-day",
) -> pd.DataFrame:
    """Fetch observations from all stations along a race course.
    
    Returns combined DataFrame from all nearby stations.
    """
    from .config import COURSES
    
    course = COURSES.get(course_name)
    if course is None:
        logger.error(f"Unknown course: {course_name}")
        return pd.DataFrame()
    
    all_obs = []
    for station_name in course.nearby_stations:
        obs = fetch_smhi_wind_observations(station_name, period)
        if not obs.empty:
            all_obs.append(obs)
    
    if all_obs:
        return pd.concat(all_obs, ignore_index=True)
    return pd.DataFrame()


# ============================================================
# EXPEDITION LOG PARSER
# ============================================================

def parse_expedition_log(filepath: str) -> pd.DataFrame:
    """Parse an Expedition CSV log file into a clean DataFrame.
    
    Expedition logs use a keyed format where each data line contains
    channel_number,value pairs. The first line maps channel names to numbers.
    
    Key channels:
        0  = UTC (Excel serial date)
        1  = BSP (Boat Speed)
        2  = AWA (Apparent Wind Angle)
        3  = AWS (Apparent Wind Speed)
        4  = TWA (True Wind Angle)
        5  = TWS (True Wind Speed)
        6  = TWD (True Wind Direction)
        13 = HDG (Heading)
        18 = Heel
        48 = Lat
        49 = Lon
        50 = COG
        51 = SOG
    
    Returns:
        DataFrame with columns:
        [datetime, bsp, twa, tws, twd, awa, aws, hdg, heel, lat, lon, cog, sog,
         wind_speed_knots, wind_direction]
        
        wind_speed_knots and wind_direction are aliases for tws and twd,
        matching the observation format for scoring.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Expedition log not found: {filepath}")
    
    logger.info(f"Parsing Expedition log: {filepath.name}")
    
    with open(filepath, "r") as f:
        lines = f.readlines()
    
    # Excel serial date epoch
    excel_epoch = datetime(1899, 12, 30)
    
    records = []
    for line in lines[3:]:  # Skip 3 header lines
        parts = line.strip().split(",")
        vals = {}
        i = 0
        while i < len(parts) - 1:
            try:
                key = int(parts[i])
                val = float(parts[i + 1])
                vals[key] = val
                i += 2
            except (ValueError, IndexError):
                i += 1
        
        # Must have at minimum: UTC, TWS, TWD, position
        if not all(k in vals for k in [0, 5, 6, 48]):
            continue
        
        utc_serial = vals[0]
        lat = vals.get(48, 0)
        bsp = vals.get(1, 0)
        
        # Filter: valid position and not clearly bad data
        if not (50 < lat < 70) or bsp > 25:
            continue
        
        dt = excel_epoch + timedelta(days=utc_serial)
        
        records.append({
            "datetime": dt,
            "bsp": bsp,
            "awa": vals.get(2, np.nan),
            "aws": vals.get(3, np.nan),
            "twa": vals.get(4, np.nan),
            "tws": vals.get(5, np.nan),
            "twd": vals.get(6, np.nan),
            "hdg": vals.get(13, np.nan),
            "heel": vals.get(18, np.nan),
            "lat": lat,
            "lon": vals.get(49, np.nan),
            "cog": vals.get(50, np.nan),
            "sog": vals.get(51, np.nan),
        })
    
    df = pd.DataFrame(records)
    
    # Add standard column names for scoring compatibility
    df["wind_speed_knots"] = df["tws"]
    df["wind_direction"] = df["twd"]
    
    logger.info(f"  Parsed {len(df)} valid records from {filepath.name}")
    logger.info(f"  Period: {df['datetime'].min()} to {df['datetime'].max()}")
    logger.info(f"  TWS range: {df['tws'].min():.1f} - {df['tws'].max():.1f} kt")
    
    return df


def resample_expedition_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample high-frequency Expedition data to hourly averages.
    
    Wind direction uses circular mean. Other fields use arithmetic mean.
    This matches the hourly resolution of forecast models.
    """
    df = df.set_index("datetime")
    
    # Circular mean for wind direction
    def circular_mean(angles):
        angles_rad = np.radians(angles.dropna())
        if len(angles_rad) == 0:
            return np.nan
        sin_mean = np.sin(angles_rad).mean()
        cos_mean = np.cos(angles_rad).mean()
        return np.degrees(np.arctan2(sin_mean, cos_mean)) % 360
    
    # Resample all numeric columns
    hourly = df.resample("1h").agg({
        "bsp": "mean",
        "tws": "mean",
        "twd": circular_mean,
        "twa": "mean",
        "aws": "mean",
        "awa": "mean",
        "heel": "mean",
        "lat": "mean",
        "lon": "mean",
        "sog": "mean",
    })
    
    hourly["wind_speed_knots"] = hourly["tws"]
    hourly["wind_direction"] = hourly["twd"]
    hourly["n_samples"] = df.resample("1h")["tws"].count()
    
    # Drop hours with too few samples
    hourly = hourly[hourly["n_samples"] >= 10]
    hourly = hourly.reset_index()
    
    logger.info(f"  Resampled to {len(hourly)} hourly bins")
    return hourly
