"""BLUR Observation Collector — main entry point for cron execution.

Fetches observations from Nordic met services (SMHI, MET Norway, DMI, FMI)
and stores them in a MySQL or SQLite database.

Usage::

    # Collect from all sources
    python -m blur_weather.collector.collect

    # Collect from a single source
    python -m blur_weather.collector.collect --source smhi

    # Sync station definitions from YAML into the database
    python -m blur_weather.collector.collect --sync

    # Fetch observations but don't insert (for testing)
    python -m blur_weather.collector.collect --dry-run
"""

import argparse
import logging
import time
from typing import List, Optional

from .db import Database
from .stations import load_stations
from .sources import smhi, met_norway, dmi, fmi

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SOURCES = {
    "smhi": smhi.fetch_observations,
    "met_no": met_norway.fetch_observations,
    "dmi": dmi.fetch_observations,
    "fmi": fmi.fetch_observations,
}


def run(
    sources: Optional[List[str]] = None,
    dry_run: bool = False,
) -> None:
    """Run the observation collector.

    Args:
        sources: List of source names to collect.  *None* = all.
        dry_run: If True, fetch observations but don't insert into DB.
    """
    with Database.from_env() as db:
        sources_to_run = sources or list(SOURCES.keys())

        for source_name in sources_to_run:
            fetch_fn = SOURCES.get(source_name)
            if fetch_fn is None:
                logger.error("Unknown source: %s", source_name)
                continue

            stations = db.get_active_stations(source=source_name)
            if not stations:
                logger.warning(
                    "No active stations for %s — run --sync first?", source_name,
                )
                continue

            start = time.time()

            inserted = 0
            skipped = 0
            errors = 0
            error_details = []

            for station in stations:
                try:
                    obs_list = fetch_fn(station)
                    for obs in obs_list:
                        if dry_run:
                            logger.info(
                                "  [DRY-RUN] %s  %s  ws=%.1f m/s  wd=%.0f°",
                                obs["station_code"],
                                obs["observed_at"].strftime("%Y-%m-%d %H:%M"),
                                obs.get("wind_speed_ms") or 0,
                                obs.get("wind_direction_deg") or 0,
                            )
                            inserted += 1  # count for logging
                        else:
                            was_inserted = db.insert_observation(obs)
                            if was_inserted:
                                inserted += 1
                            else:
                                skipped += 1
                except Exception as exc:
                    errors += 1
                    error_details.append(f"{station['station_code']}: {exc}")
                    logger.warning(
                        "Error fetching %s: %s", station["station_code"], exc,
                    )

            duration = time.time() - start

            if not dry_run:
                db.log_collection(
                    source=source_name,
                    stations_queried=len(stations),
                    observations_inserted=inserted,
                    observations_skipped=skipped,
                    errors=errors,
                    error_detail="; ".join(error_details) if error_details else None,
                    duration_seconds=round(duration, 2),
                )

            logger.info(
                "%s: %d inserted, %d skipped, %d errors in %.1fs (%d stations)",
                source_name, inserted, skipped, errors, duration, len(stations),
            )


def sync_stations() -> None:
    """Sync station definitions from YAML into the database.

    Loads ``config/stations.yaml``, inserts or updates all stations.
    Run this once at setup, and again whenever the YAML changes.
    """
    stations = load_stations()
    logger.info("Loaded %d stations from YAML", len(stations))

    with Database.from_env() as db:
        n_inserted, n_updated = db.sync_stations(stations)
        logger.info(
            "Station sync complete: %d inserted, %d updated, %d total",
            n_inserted, n_updated, len(stations),
        )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="BLUR Observation Collector",
        prog="blur_weather.collector.collect",
    )
    parser.add_argument(
        "--source",
        choices=list(SOURCES.keys()),
        help="Collect from a single source only",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Sync stations from YAML into database",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch observations but don't insert into database",
    )
    args = parser.parse_args()

    if args.sync:
        sync_stations()
    else:
        run(
            sources=[args.source] if args.source else None,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
