"""Station registry — loads station definitions from YAML config.

The YAML file (``config/stations.yaml``) is the canonical source of truth.
Stations are synced into the MySQL/SQLite ``stations`` table at deploy time
via ``python -m blur_weather.collector.collect --sync``.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# Default path relative to project root
DEFAULT_STATIONS_YAML = Path(__file__).resolve().parent.parent.parent / "config" / "stations.yaml"

SOURCE_COUNTRY: Dict[str, str] = {
    "smhi": "SE",
    "met_no": "NO",
    "dmi": "DK",
    "fmi": "FI",
}


def make_station_code(source: str, raw_code) -> str:
    """Generate canonical station_code from source and raw code.

    Examples::

        make_station_code('smhi', 71420)      -> 'smhi_71420'
        make_station_code('met_no', 'SN27500') -> 'met_SN27500'
        make_station_code('dmi', '06041')      -> 'dmi_06041'
        make_station_code('fmi', '100908')     -> 'fmi_100908'
    """
    prefix = "met" if source == "met_no" else source
    return f"{prefix}_{raw_code}"


def load_stations(
    yaml_path: Optional[Path] = None,
    source: Optional[str] = None,
    race_area: Optional[str] = None,
) -> List[dict]:
    """Load station definitions from YAML.

    Args:
        yaml_path: Path to ``stations.yaml``.  Uses default if *None*.
        source: Filter by source (``'smhi'``, ``'met_no'``, ``'dmi'``,
                ``'fmi'``).  *None* returns all sources.
        race_area: Filter by race area (e.g. ``'gotland_runt'``).
                   *None* returns all areas.

    Returns:
        Flat list of station dicts, each with keys:

        - ``station_code`` — e.g. ``'smhi_71420'``
        - ``source`` — ``'smhi'``, ``'met_no'``, ``'dmi'``, ``'fmi'``
        - ``name`` — display name
        - ``lat`` — latitude
        - ``lon`` — longitude
        - ``country`` — ISO 2-letter code
        - ``race_area`` — grouping key
        - ``raw_code`` — original station ID from the met service
    """
    if yaml_path is None:
        yaml_path = DEFAULT_STATIONS_YAML

    with open(yaml_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    stations: List[dict] = []

    for src_name, areas in raw.items():
        if source is not None and src_name != source:
            continue
        for area_name, station_list in areas.items():
            if race_area is not None and area_name != race_area:
                continue
            for stn in station_list:
                stations.append({
                    "station_code": make_station_code(src_name, stn["code"]),
                    "source": src_name,
                    "name": stn["name"],
                    "lat": float(stn["lat"]),
                    "lon": float(stn["lon"]),
                    "country": SOURCE_COUNTRY[src_name],
                    "race_area": area_name,
                    "raw_code": stn["code"],
                })

    logger.info("Loaded %d stations from %s", len(stations), yaml_path.name)
    return stations
