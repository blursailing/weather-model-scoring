"""FMI Open Data WFS adapter.

Fetches observations from Finnish meteorological stations via the FMI WFS API.
No authentication required.  Response is XML (GML MultiPointCoverage), parsed
with lxml.

API docs: https://en.ilmatieteenlaitos.fi/open-data-manual
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional

from . import fetch_with_retry

logger = logging.getLogger(__name__)

FMI_WFS_URL = "https://opendata.fmi.fi/wfs"
FMI_STORED_QUERY = "fmi::observations::weather::multipointcoverage"
FMI_PARAMS = "WindSpeedMS,WindDirection,Pressure,Temperature"

# Map FMI parameter names to standard field names
_PARAM_MAP = {
    "WindSpeedMS": "wind_speed_ms",
    "WindDirection": "wind_direction_deg",
    "Pressure": "air_pressure_hpa",
    "Temperature": "air_temperature_c",
}

# GML / SWE namespaces
_NS = {
    "wfs": "http://www.opengis.net/wfs/2.0",
    "gml": "http://www.opengis.net/gml/3.2",
    "gmlcov": "http://www.opengis.net/gmlcov/1.0",
    "swe": "http://www.opengis.net/swe/2.0",
}


def fetch_observations(station: dict) -> List[dict]:
    """Fetch latest observations from a single FMI station.

    Args:
        station: Dict with ``station_code`` like ``'fmi_100908'``.

    Returns:
        List of observation dicts in standard format.
    """
    # Extract FMI station ID: "fmi_100908" -> "100908"
    fmi_station_id = station["station_code"].replace("fmi_", "")

    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "getFeature",
        "storedquery_id": FMI_STORED_QUERY,
        "fmisid": fmi_station_id,
        "parameters": FMI_PARAMS,
        "timestep": "60",  # hourly
    }

    try:
        resp = fetch_with_retry(FMI_WFS_URL, params=params)
        xml_bytes = resp.content
    except Exception as exc:
        logger.warning(
            "FMI %s (%s) failed: %s",
            station["name"], fmi_station_id, exc,
        )
        return []

    return _parse_multipointcoverage(xml_bytes, station)


def _parse_multipointcoverage(xml_bytes: bytes, station: dict) -> List[dict]:
    """Parse FMI's MultiPointCoverage XML response.

    The response contains:
      - ``gml:positions``: space-separated triples of (lat, lon, unix_epoch)
      - ``gml:doubleOrNilReasonTupleList``: rows of values in parameter order
      - ``swe:DataRecord/swe:field``: defines parameter order
    """
    try:
        from lxml import etree
    except ImportError:
        logger.error("lxml not installed — cannot parse FMI XML. pip install lxml")
        return []

    try:
        root = etree.fromstring(xml_bytes)
    except etree.XMLSyntaxError as exc:
        logger.warning("FMI %s: XML parse error: %s", station["name"], exc)
        return []

    # Determine parameter order from swe:DataRecord/swe:field elements
    field_elems = root.findall(".//swe:DataRecord/swe:field", _NS)
    if not field_elems:
        logger.debug("FMI %s: no swe:field elements found", station["name"])
        return []

    field_names = [el.get("name", "") for el in field_elems]
    logger.debug("FMI fields: %s", field_names)

    # Parse positions: "lat lon epoch lat lon epoch ..."
    # FMI uses gmlcov:positions (not gml:positions) in MultiPointCoverage
    positions_elem = root.find(".//gmlcov:positions", _NS)
    if positions_elem is None:
        # Fallback to gml:positions for forward compatibility
        positions_elem = root.find(".//gml:positions", _NS)
    if positions_elem is None or not positions_elem.text:
        logger.debug("FMI %s: no positions element found", station["name"])
        return []

    pos_tokens = positions_elem.text.strip().split()
    timestamps = []
    # Every 3rd token (index 2, 5, 8, ...) is a Unix epoch
    for i in range(2, len(pos_tokens), 3):
        try:
            epoch = float(pos_tokens[i])
            ts = datetime.fromtimestamp(epoch, tz=timezone.utc)
            timestamps.append(ts)
        except (ValueError, OSError):
            timestamps.append(None)

    # Parse values: each line = one observation, space-separated values
    values_elem = root.find(".//gml:doubleOrNilReasonTupleList", _NS)
    if values_elem is None or not values_elem.text:
        logger.debug("FMI %s: no value list element", station["name"])
        return []

    value_lines = values_elem.text.strip().split("\n")

    # Build observation dicts
    observations = []
    for idx, line in enumerate(value_lines):
        if idx >= len(timestamps) or timestamps[idx] is None:
            continue

        tokens = line.strip().split()
        if len(tokens) != len(field_names):
            continue

        fields = {}
        for fname, val_str in zip(field_names, tokens):
            std_field = _PARAM_MAP.get(fname)
            if std_field is None:
                continue
            value = _parse_fmi_value(val_str)
            if value is not None:
                fields[std_field] = value

        if "wind_speed_ms" not in fields and "wind_direction_deg" not in fields:
            continue

        observations.append({
            "station_code": station["station_code"],
            "observed_at": timestamps[idx],
            "wind_speed_ms": fields.get("wind_speed_ms"),
            "wind_direction_deg": fields.get("wind_direction_deg"),
            "air_pressure_hpa": fields.get("air_pressure_hpa"),
            "air_temperature_c": fields.get("air_temperature_c"),
        })

    logger.debug("FMI %s: %d observations", station["name"], len(observations))
    return observations


def _parse_fmi_value(val_str: str) -> Optional[float]:
    """Parse a single FMI value.  Returns None for NaN or invalid data."""
    val_str = val_str.strip()
    if val_str.lower() == "nan" or val_str == "":
        return None
    try:
        return float(val_str)
    except ValueError:
        return None
