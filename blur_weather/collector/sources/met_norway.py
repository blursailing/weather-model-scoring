"""MET Norway Frost API adapter.

Fetches observations from Norwegian coastal stations via the Frost API.
Requires HTTP Basic Auth with a client_id (free registration at frost.met.no).

API docs: https://frost.met.no/
"""

import logging
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import List

from . import fetch_with_retry

logger = logging.getLogger(__name__)

FROST_BASE_URL = "https://frost.met.no/observations/v0.jsonld"

ELEMENTS = (
    "wind_speed,"
    "wind_from_direction,"
    "air_pressure_at_sea_level,"
    "air_temperature"
)

# Map Frost element IDs to standard field names
_ELEMENT_MAP = {
    "wind_speed": "wind_speed_ms",
    "wind_from_direction": "wind_direction_deg",
    "air_pressure_at_sea_level": "air_pressure_hpa",
    "air_temperature": "air_temperature_c",
}


def fetch_observations(station: dict) -> List[dict]:
    """Fetch latest observations from a single MET Norway station.

    Args:
        station: Dict with ``station_code`` like ``'met_SN27500'``.

    Returns:
        List of observation dicts in standard format.
        Returns empty list if ``MET_NORWAY_CLIENT_ID`` is not set.
    """
    client_id = os.environ.get("MET_NORWAY_CLIENT_ID")
    if not client_id:
        logger.error("MET_NORWAY_CLIENT_ID not set — skipping MET Norway")
        return []

    # Extract Frost source ID: "met_SN27500" -> "SN27500"
    frost_source = station["station_code"].replace("met_", "")

    params = {
        "sources": frost_source,
        "referencetime": "latest",
        "elements": ELEMENTS,
    }

    try:
        resp = fetch_with_retry(
            FROST_BASE_URL,
            params=params,
            auth=(client_id, ""),
        )
        payload = resp.json()
    except Exception as exc:
        logger.warning(
            "MET Norway %s (%s) failed: %s",
            station["name"], frost_source, exc,
        )
        return []

    # Parse JSON-LD response:
    # { "data": [ { "referenceTime": "...", "observations": [ { "elementId": ..., "value": ... } ] } ] }
    data_items = payload.get("data", [])
    if not data_items:
        logger.debug("MET Norway %s: empty response", station["name"])
        return []

    # Group observations by referenceTime
    by_time = defaultdict(dict)
    for item in data_items:
        ref_time_str = item.get("referenceTime", "")
        if not ref_time_str:
            continue

        try:
            ts = datetime.fromisoformat(ref_time_str.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except ValueError:
            continue

        for obs in item.get("observations", []):
            element_id = obs.get("elementId", "")
            value = obs.get("value")
            field = _ELEMENT_MAP.get(element_id)
            if field and value is not None:
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

    logger.debug("MET Norway %s: %d observations", station["name"], len(observations))
    return observations
