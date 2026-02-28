"""SMHI Open Data API adapter.

Fetches wind speed, wind direction, pressure, and temperature from
Swedish coastal weather stations.  No authentication required.

API docs: https://opendata.smhi.se/apidocs/metobs/
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List

from blur_weather.config import (
    SMHI_BASE_URL,
    SMHI_WIND_SPEED_PARAM,
    SMHI_WIND_DIRECTION_PARAM,
    SMHI_PRESSURE_PARAM,
)
from . import fetch_with_retry

logger = logging.getLogger(__name__)

SMHI_TEMP_PARAM = 1  # Air temperature (°C)

# Map SMHI parameter IDs to our standard field names
_PARAM_FIELD = {
    SMHI_WIND_SPEED_PARAM: "wind_speed_ms",
    SMHI_WIND_DIRECTION_PARAM: "wind_direction_deg",
    SMHI_PRESSURE_PARAM: "air_pressure_hpa",
    SMHI_TEMP_PARAM: "air_temperature_c",
}


def fetch_observations(station: dict) -> List[dict]:
    """Fetch latest-hour observations from a single SMHI station.

    Args:
        station: Dict with at least ``station_code`` (e.g. ``'smhi_71420'``).
                 The numeric SMHI station ID is extracted from the code.

    Returns:
        List of observation dicts in the standard collector format.
        Typically 1-2 observations (latest hour).
        Returns empty list on failure.
    """
    # Extract numeric station ID from station_code: "smhi_71420" -> 71420
    raw_code = station["station_code"].replace("smhi_", "")
    smhi_id = int(raw_code)

    # Fetch each parameter separately and collect {datetime -> {field: value}}
    all_data: Dict[datetime, dict] = {}

    for param_id, field_name in _PARAM_FIELD.items():
        param_data = _fetch_param(smhi_id, param_id)
        for ts, value in param_data.items():
            if ts not in all_data:
                all_data[ts] = {}
            all_data[ts][field_name] = value

    if not all_data:
        logger.warning("No data from SMHI station %s (%s)", station["name"], smhi_id)
        return []

    # Build standard observation dicts
    observations = []
    for ts in sorted(all_data.keys()):
        fields = all_data[ts]

        # Require at least wind speed or direction
        if "wind_speed_ms" not in fields and "wind_direction_deg" not in fields:
            continue

        observations.append({
            "station_code": station["station_code"],
            "observed_at": ts,
            "wind_speed_ms": fields.get("wind_speed_ms"),
            "wind_direction_deg": fields.get("wind_direction_deg"),
            "air_pressure_hpa": fields.get("air_pressure_hpa"),
            "air_temperature_c": fields.get("air_temperature_c"),
        })

    logger.debug("SMHI %s: %d observations", station["name"], len(observations))
    return observations


def _fetch_param(smhi_station_id: int, param_id: int) -> Dict[datetime, float]:
    """Fetch a single parameter from SMHI.

    Uses ``period=latest-hour`` to get the most recent observation(s).

    Returns:
        ``{datetime: value}`` mapping.  Empty dict on failure.
    """
    url = (
        f"{SMHI_BASE_URL}/version/1.0/parameter/{param_id}"
        f"/station/{smhi_station_id}/period/latest-hour/data.json"
    )

    try:
        resp = fetch_with_retry(url)
        data = resp.json()
    except Exception as exc:
        logger.warning(
            "SMHI param %d for station %d failed: %s",
            param_id, smhi_station_id, exc,
        )
        return {}

    result: Dict[datetime, float] = {}
    for entry in data.get("value", []):
        ts_ms = entry.get("date", 0)
        val = entry.get("value")
        if val is not None:
            try:
                ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                result[ts] = float(val)
            except (ValueError, OSError):
                continue

    return result
