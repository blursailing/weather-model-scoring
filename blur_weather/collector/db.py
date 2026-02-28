"""Database access layer for the observation collector.

Supports MySQL (production on Oderland) and SQLite (local development).
Backend is selected via the ``DB_BACKEND`` environment variable.

Usage::

    with Database.from_env() as db:
        db.insert_observation(obs)
"""

import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Try to load .env from project root
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass  # python-dotenv not installed; rely on real env vars


class Database:
    """Observation database with MySQL or SQLite backend."""

    def __init__(self, backend: str = "mysql", connection=None):
        self.backend = backend
        self._conn = connection
        self._closed = False

    # ------------------------------------------------------------------ #
    # Factory
    # ------------------------------------------------------------------ #

    @classmethod
    def from_env(cls) -> "Database":
        """Create a Database from environment variables.

        Env vars:
            DB_BACKEND  — ``'mysql'`` (default) or ``'sqlite'``
            DB_HOST, DB_NAME, DB_USER, DB_PASS — MySQL credentials
            DB_SQLITE_PATH — SQLite file path (default ``./data/observations.db``)
        """
        backend = os.environ.get("DB_BACKEND", "mysql").lower()
        db = cls(backend=backend)
        if backend == "sqlite":
            db._connect_sqlite()
        else:
            db._connect_mysql()
        return db

    # ------------------------------------------------------------------ #
    # Connections
    # ------------------------------------------------------------------ #

    def _connect_mysql(self) -> None:
        """Establish MySQL connection."""
        import mysql.connector  # lazy import

        self._conn = mysql.connector.connect(
            host=os.environ.get("DB_HOST", "localhost"),
            database=os.environ["DB_NAME"],
            user=os.environ["DB_USER"],
            password=os.environ["DB_PASS"],
            charset="utf8mb4",
            autocommit=True,
        )
        logger.info("Connected to MySQL: %s@%s/%s",
                     os.environ["DB_USER"],
                     os.environ.get("DB_HOST", "localhost"),
                     os.environ["DB_NAME"])

    def _connect_sqlite(self) -> None:
        """Establish SQLite connection with MySQL-compatible schema."""
        db_path = os.environ.get("DB_SQLITE_PATH", "./data/observations.db")
        db_dir = Path(db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

        # Create tables (SQLite-compatible equivalents)
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS stations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                station_code TEXT NOT NULL UNIQUE,
                source TEXT NOT NULL,
                name TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                country TEXT NOT NULL,
                race_area TEXT,
                active INTEGER DEFAULT 1,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                station_code TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                wind_speed_ms REAL,
                wind_direction_deg REAL,
                air_pressure_hpa REAL,
                air_temperature_c REAL,
                fetched_at TEXT DEFAULT (datetime('now')),
                UNIQUE(station_code, observed_at)
            );
            CREATE INDEX IF NOT EXISTS idx_observed_at
                ON observations(observed_at);
            CREATE INDEX IF NOT EXISTS idx_station_observed
                ON observations(station_code, observed_at);

            CREATE TABLE IF NOT EXISTS collection_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_at TEXT DEFAULT (datetime('now')),
                source TEXT NOT NULL,
                stations_queried INTEGER DEFAULT 0,
                observations_inserted INTEGER DEFAULT 0,
                observations_skipped INTEGER DEFAULT 0,
                errors INTEGER DEFAULT 0,
                error_detail TEXT,
                duration_seconds REAL
            );
        """)
        self._conn.commit()
        logger.info("Connected to SQLite: %s", db_path)

    # ------------------------------------------------------------------ #
    # Context manager
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        if self._conn and not self._closed:
            self._conn.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # ------------------------------------------------------------------ #
    # Station operations
    # ------------------------------------------------------------------ #

    def sync_stations(self, stations: List[dict]) -> Tuple[int, int]:
        """Sync station list from YAML into the stations table.

        Inserts new stations, updates existing ones.

        Args:
            stations: List of station dicts from ``stations.load_stations()``.

        Returns:
            (inserted, updated) counts.
        """
        inserted = 0
        updated = 0
        cur = self._conn.cursor()

        for stn in stations:
            if self.backend == "sqlite":
                # INSERT OR REPLACE for SQLite
                cur.execute(
                    """INSERT OR REPLACE INTO stations
                       (station_code, source, name, latitude, longitude, country, race_area, active)
                       VALUES (?, ?, ?, ?, ?, ?, ?, 1)""",
                    (stn["station_code"], stn["source"], stn["name"],
                     stn["lat"], stn["lon"], stn["country"], stn["race_area"]),
                )
            else:
                # MySQL: INSERT ... ON DUPLICATE KEY UPDATE
                cur.execute(
                    """INSERT INTO stations
                       (station_code, source, name, latitude, longitude, country, race_area, active)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, TRUE)
                       ON DUPLICATE KEY UPDATE
                           name=VALUES(name), latitude=VALUES(latitude),
                           longitude=VALUES(longitude), race_area=VALUES(race_area),
                           active=TRUE""",
                    (stn["station_code"], stn["source"], stn["name"],
                     stn["lat"], stn["lon"], stn["country"], stn["race_area"]),
                )

            if cur.rowcount == 1:
                inserted += 1
            elif cur.rowcount >= 2:
                # MySQL ON DUPLICATE KEY UPDATE reports rowcount=2 for updates
                updated += 1

        if self.backend == "sqlite":
            self._conn.commit()

        logger.info("Station sync: %d inserted, %d updated", inserted, updated)
        return inserted, updated

    def get_active_stations(self, source: Optional[str] = None) -> List[dict]:
        """Get all active stations, optionally filtered by source.

        Returns:
            List of dicts with keys: station_code, source, name,
            latitude, longitude, country, race_area.
        """
        cur = self._conn.cursor()
        ph = "?" if self.backend == "sqlite" else "%s"

        if source:
            cur.execute(
                f"""SELECT station_code, source, name, latitude, longitude,
                           country, race_area
                    FROM stations WHERE active = {'1' if self.backend == 'sqlite' else 'TRUE'}
                    AND source = {ph}""",
                (source,),
            )
        else:
            cur.execute(
                f"""SELECT station_code, source, name, latitude, longitude,
                           country, race_area
                    FROM stations WHERE active = {'1' if self.backend == 'sqlite' else 'TRUE'}""",
            )

        columns = ["station_code", "source", "name", "latitude", "longitude",
                    "country", "race_area"]
        return [dict(zip(columns, row)) for row in cur.fetchall()]

    # ------------------------------------------------------------------ #
    # Observation operations
    # ------------------------------------------------------------------ #

    def insert_observation(self, obs: dict) -> bool:
        """Insert a single observation.  Returns True if inserted, False if duplicate.

        Args:
            obs: Dict with keys: station_code, observed_at (datetime UTC),
                 wind_speed_ms, wind_direction_deg, air_pressure_hpa,
                 air_temperature_c.  Values may be None.
        """
        observed_at = obs["observed_at"]
        # Normalise to string for SQLite, datetime for MySQL
        if self.backend == "sqlite":
            if isinstance(observed_at, datetime):
                observed_at = observed_at.strftime("%Y-%m-%d %H:%M:%S")

        cur = self._conn.cursor()

        if self.backend == "sqlite":
            cur.execute(
                """INSERT OR IGNORE INTO observations
                   (station_code, observed_at, wind_speed_ms, wind_direction_deg,
                    air_pressure_hpa, air_temperature_c)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (obs["station_code"], observed_at,
                 obs.get("wind_speed_ms"), obs.get("wind_direction_deg"),
                 obs.get("air_pressure_hpa"), obs.get("air_temperature_c")),
            )
            self._conn.commit()
        else:
            cur.execute(
                """INSERT IGNORE INTO observations
                   (station_code, observed_at, wind_speed_ms, wind_direction_deg,
                    air_pressure_hpa, air_temperature_c)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (obs["station_code"], observed_at,
                 obs.get("wind_speed_ms"), obs.get("wind_direction_deg"),
                 obs.get("air_pressure_hpa"), obs.get("air_temperature_c")),
            )

        return cur.rowcount == 1

    def insert_observations_batch(self, observations: List[dict]) -> Tuple[int, int]:
        """Insert multiple observations.  Returns (inserted, skipped)."""
        inserted = 0
        skipped = 0
        for obs in observations:
            if self.insert_observation(obs):
                inserted += 1
            else:
                skipped += 1
        return inserted, skipped

    def get_observations(
        self,
        station_code: str,
        start: datetime,
        end: datetime,
    ) -> List[dict]:
        """Retrieve observations for a station in a time range.

        Args:
            station_code: e.g. ``'smhi_71420'``
            start: Start of range (UTC).
            end: End of range (UTC).

        Returns:
            List of dicts ordered by observed_at ascending.
        """
        cur = self._conn.cursor()
        ph = "?" if self.backend == "sqlite" else "%s"

        if self.backend == "sqlite":
            start_s = start.strftime("%Y-%m-%d %H:%M:%S")
            end_s = end.strftime("%Y-%m-%d %H:%M:%S")
        else:
            start_s, end_s = start, end

        cur.execute(
            f"""SELECT station_code, observed_at, wind_speed_ms,
                       wind_direction_deg, air_pressure_hpa, air_temperature_c
                FROM observations
                WHERE station_code = {ph}
                  AND observed_at >= {ph}
                  AND observed_at <= {ph}
                ORDER BY observed_at""",
            (station_code, start_s, end_s),
        )

        columns = ["station_code", "observed_at", "wind_speed_ms",
                    "wind_direction_deg", "air_pressure_hpa", "air_temperature_c"]
        rows = []
        for row in cur.fetchall():
            d = dict(zip(columns, row))
            # Parse datetime string if SQLite
            if isinstance(d["observed_at"], str):
                d["observed_at"] = datetime.strptime(
                    d["observed_at"], "%Y-%m-%d %H:%M:%S"
                ).replace(tzinfo=timezone.utc)
            rows.append(d)
        return rows

    def get_observations_df(
        self,
        station_code: str,
        start: datetime,
        end: datetime,
    ):
        """Retrieve observations as a pandas DataFrame.

        Returns DataFrame with columns matching ``score.py`` expectations::

            [datetime, wind_speed_knots, wind_direction, pressure_hpa]

        Wind speed is converted from m/s to knots via ``config.ms_to_knots()``.
        """
        import pandas as pd
        from blur_weather.config import ms_to_knots

        rows = self.get_observations(station_code, start, end)
        if not rows:
            return pd.DataFrame(columns=["datetime", "wind_speed_knots",
                                         "wind_direction", "pressure_hpa"])

        df = pd.DataFrame(rows)
        df = df.rename(columns={
            "observed_at": "datetime",
            "wind_direction_deg": "wind_direction",
            "air_pressure_hpa": "pressure_hpa",
        })

        # Convert m/s to knots
        df["wind_speed_knots"] = df["wind_speed_ms"].apply(
            lambda v: ms_to_knots(v) if v is not None else None
        )

        # Select and order columns to match score.py expectations
        result = df[["datetime", "wind_speed_knots", "wind_direction", "pressure_hpa"]].copy()
        result = result.dropna(subset=["wind_speed_knots", "wind_direction"])
        return result.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Collection log
    # ------------------------------------------------------------------ #

    def log_collection(
        self,
        source: str,
        stations_queried: int,
        observations_inserted: int,
        observations_skipped: int,
        errors: int,
        error_detail: Optional[str],
        duration_seconds: float,
    ) -> None:
        """Log a collection run to the collection_log table."""
        cur = self._conn.cursor()

        if self.backend == "sqlite":
            cur.execute(
                """INSERT INTO collection_log
                   (source, stations_queried, observations_inserted,
                    observations_skipped, errors, error_detail, duration_seconds)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (source, stations_queried, observations_inserted,
                 observations_skipped, errors, error_detail,
                 round(duration_seconds, 2)),
            )
            self._conn.commit()
        else:
            cur.execute(
                """INSERT INTO collection_log
                   (source, stations_queried, observations_inserted,
                    observations_skipped, errors, error_detail, duration_seconds)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (source, stations_queried, observations_inserted,
                 observations_skipped, errors, error_detail,
                 round(duration_seconds, 2)),
            )
