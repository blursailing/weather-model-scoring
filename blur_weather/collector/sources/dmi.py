"""DMI Open Data API adapter.

Fetches observations from Danish meteorological stations.
No authentication required (API key requirement removed December 2025).

API docs: https://opendatadocs.dmi.govcloud.dk/

Notes:
    - DMI does not support comma-separated parameterId filtering.
      We fetch all parameters and filter client-side.
    - ``datetime=latest`` is not supported.  We use a 2-hour time range.
    - Pressure: we prefer ``pressure_at_sea`` (sea-level reduced) over
      raw station ``pressure``.
"""

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import List

from . import fetch_with_retry

logger = logging.getLogger(__name__)

DMI_BASE_URL = "https://dmigw.govcloud.dk/v2/metObs/collections/observation/items"

# Map DMI parameter IDs to standard field names.
# We prefer pressure_at_sea for consistency with other sources.
_PARAM_MAP = {
    "wind_speed": "wind_speed_ms",
    "wind_dir": "wind_direction_deg",
    "pressure_at_sea": "air_pressure_hpa",
    "temp_dry": "air_temperature_c",
}


def fetch_observations(station: dict) -> List[dict]:
    """Fetch latest observations from a single DMI station.

    Args:
        station: Dict with ``station_code`` like ``'dmi_06041'``.

    Returns:
        List of observation dicts in standard format.
    """
    # Extract DMI station ID: "dmi_06041" -> "06041"
    dmi_station_id = station["station_code"].replace("dmi_", "")

    # DMI doesn't support "latest" — use a 2-hour time range
    now = datetime.now(tz=timezone.utc)
    start = (now - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    end = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    params = {
        "stationId": dmi_station_id,
        "datetime": f"{start}/{end}",
        "limit": "100",
        # No parameterId filter — DMI doesn't support comma-separated lists.
        # We fetch all params and filter client-side.
    }

    try:
        resp = fetch_with_retry(DMI_BASE_URL, params=params)
        payload = resp.json()
    except Exception as exc:
        logger.warning(
            "DMI %s (%s) failed: %s",
            station["name"], dmi_station_id, exc,
        )
        return []

    # Parse GeoJSON response:
    # { "features": [ { "properties": { "parameterId": "wind_speed", "value": 8.3,
    #                                    "observed": "2026-02-28T12:00:00Z", ... } } ] }
    features = payload.get("features", [])
    if not features:
        logger.debug("DMI %s: empty response", station["name"])
        return []

    # Group by timestamp, keeping only parameters we care about
    by_time = defaultdict(dict)
    for feature in features:
        props = feature.get("properties", {})
        param_id = props.get("parameterId", "")
        value = props.get("value")
        observed_str = props.get("observed", "")

        field = _PARAM_MAP.get(param_id)
        if not field or value is None or not observed_str:
            continue

        try:
            ts = datetime.fromisoformat(observed_str.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except ValueError:
            continue

        by_time[ts][field] = float(value)

    # Build standard observation dicts
    observations = []
    for ts in sorted(by_time.keys()):
        fields = by_time[ts]
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

    logger.debug("DMI %s: %d observations", station["name"], len(observations))
    return observations
