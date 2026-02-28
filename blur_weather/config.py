"""Configuration, constants, and model/course definitions."""

from dataclasses import dataclass, field
from typing import Optional

# ============================================================
# WEATHER MODELS available via Open-Meteo
# ============================================================
# These are the Open-Meteo model identifiers. Not all models cover the Baltic.
# See: https://open-meteo.com/en/docs

MODELS = {
    "ecmwf_ifs025": {
        "name": "ECMWF IFS",
        "alias": "Euro",
        "resolution_km": 25,
        "updates_per_day": 4,
        "forecast_days": 10,
        "notes": "Best overall global model. Overamplification bias known.",
    },
    "gfs_seamless": {
        "name": "GFS",
        "alias": "American",
        "resolution_km": 27,
        "updates_per_day": 4,
        "forecast_days": 16,
        "notes": "Free, frequent updates. Trails ECMWF by ~1 day skill.",
    },
    "icon_seamless": {
        "name": "ICON",
        "alias": "German (global)",
        "resolution_km": 13,
        "updates_per_day": 2,
        "forecast_days": 5,
        "notes": "Good European coverage. Triangular grid.",
    },
    "icon_eu": {
        "name": "ICON-EU",
        "alias": "German (Europe)",
        "resolution_km": 7,
        "updates_per_day": 2,  # 8x for D2 nest
        "forecast_days": 5,
        "notes": "Best European regional model. Excellent Baltic coverage.",
    },
    "meteofrance_seamless": {
        "name": "Météo-France",
        "alias": "French",
        "resolution_km": 10,  # Arpège global; AROME is 1.3km but limited domain
        "updates_per_day": 2,
        "forecast_days": 4,
        "notes": "Strong in your 2019 Kattegat test. Check Baltic domain.",
    },
    "knmi_seamless": {
        "name": "KNMI Harmonie",
        "alias": "Nordic/Dutch",
        "resolution_km": 5,
        "updates_per_day": 4,
        "forecast_days": 2,
        "notes": "Nordic mesoscale. Should cover Baltic well.",
    },
    "ukmo_seamless": {
        "name": "UKMO",
        "alias": "UK Met Office",
        "resolution_km": 10,
        "updates_per_day": 2,
        "forecast_days": 5,
        "notes": "Good offshore model. Similar accuracy to ECMWF.",
    },
    "ecmwf_aifs025": {
        "name": "ECMWF AIFS",
        "alias": "Euro AI",
        "resolution_km": 25,
        "updates_per_day": 4,
        "forecast_days": 10,
        "notes": "AI-based model from ECMWF. New and promising.",
    },
    "metno_seamless": {
        "name": "MET Nordic",
        "alias": "Nordic",
        "resolution_km": 2.5,
        "updates_per_day": 24,
        "forecast_days": 2.5,
        "notes": "Best resolution for Scandinavia. Only day+1 and day+2 lead times.",
    },
    "dmi_seamless": {
        "name": "DMI Harmonie",
        "alias": "Danish",
        "resolution_km": 2.5,
        "updates_per_day": 4,
        "forecast_days": 2.5,
        "notes": "Covers Denmark + southern Scandinavia.",
    },
    "gem_seamless": {
        "name": "GEM",
        "alias": "Canadian",
        "resolution_km": 25,
        "updates_per_day": 2,
        "forecast_days": 10,
        "notes": "Canadian model. Different physics, independent perspective.",
    },
    "jma_seamless": {
        "name": "JMA",
        "alias": "Japanese",
        "resolution_km": 20,
        "updates_per_day": 4,
        "forecast_days": 10,
        "notes": "Japanese Met Agency. Independent global perspective.",
    },
}

# Subset of models to use by default (the most relevant for Baltic racing)
DEFAULT_MODELS = [
    "ecmwf_ifs025",
    "gfs_seamless",
    "icon_seamless",
    "icon_eu",
    "meteofrance_seamless",
    "knmi_seamless",
]


# ============================================================
# SCORING WEIGHTS
# ============================================================

@dataclass
class ScoringWeights:
    """Weights for composite score calculation. Should sum to 1.0."""
    tws_rmse: float = 0.25
    tws_trend: float = 0.10
    twd_rmse: float = 0.25
    twd_trend: float = 0.10
    tws_bias_penalty: float = 0.10
    twd_bias_penalty: float = 0.10
    time_lag_penalty: float = 0.10

DEFAULT_WEIGHTS = ScoringWeights()


# ============================================================
# SMHI OBSERVATION STATIONS
# Relevant stations along common race courses
# Parameter IDs: 4 = wind speed, 3 = wind direction, 1 = air temp
# See: https://opendata.smhi.se/apidocs/metobs/index.html
# ============================================================

@dataclass
class ObservationStation:
    name: str
    smhi_station_id: int
    lat: float
    lon: float
    relevance: str  # Why this station matters for racing


# Key SMHI stations for Gotland Runt and Kattegat/Skagerrak racing
# Station IDs validated against SMHI Open Data API parameter 4 (wind speed), 2026-02-28
SMHI_STATIONS = {
    # Gotland Runt course
    "svenska_hogarna": ObservationStation("Svenska Högarna A", 99280, 59.44, 19.50, "Rounding mark — critical waypoint"),
    "landsort": ObservationStation("Landsort A", 87440, 58.74, 17.87, "Approach to start area"),
    "gotska_sandon": ObservationStation("Gotska Sandön A", 89230, 58.39, 19.19, "Northern approach to Gotland"),
    "farosund": ObservationStation("Fårösund Ar A", 78550, 57.92, 18.95, "Northern Gotland"),
    "visby_flygplats": ObservationStation("Visby Flygplats", 78400, 57.67, 18.35, "Western Gotland"),
    "hoburg": ObservationStation("Hoburg A", 68560, 56.92, 18.15, "Southern tip of Gotland"),
    "ostergarnsholm": ObservationStation("Östergarnsholm A", 78280, 57.44, 18.98, "Eastern Gotland"),
    "skarpo": ObservationStation("Skarpö A", 98160, 59.34, 18.74, "Return leg, outer archipelago"),

    # Kattegat / Skagerrak (for MBBR, Skagen etc.)
    "vinga": ObservationStation("Vinga A", 71380, 57.63, 11.60, "Gothenburg approach"),
    "nordkoster": ObservationStation("Nordkoster A", 81540, 58.89, 11.00, "Northern Kattegat"),
    "maseskar": ObservationStation("Måseskär A", 81050, 58.09, 11.33, "Mid-Kattegat"),
    "nidingen": ObservationStation("Nidingen A", 71190, 57.30, 11.90, "Southern approach"),
}


# ============================================================
# RACE COURSES
# ============================================================

@dataclass
class Waypoint:
    name: str
    lat: float
    lon: float
    leg_nm: float = 0  # Distance from previous waypoint
    notes: str = ""

@dataclass
class RaceCourse:
    name: str
    waypoints: list
    total_nm: float
    typical_month: int
    nearby_stations: list  # Keys into SMHI_STATIONS
    description: str = ""

COURSES = {
    "gotland_runt": RaceCourse(
        name="Gotland Runt",
        waypoints=[
            Waypoint("Start (Gråskärsfjärden)", 59.26, 18.92, 0, "South of Sandön"),
            Waypoint("Svenska Högarna", 59.44, 19.51, 25, "Rounding mark NE"),
            Waypoint("Gotska Sandön (pass)", 58.38, 19.20, 70, "Northern approach"),
            Waypoint("Fårö NE", 57.95, 19.50, 30, "NE tip of Gotland"),
            Waypoint("Östergarn", 57.42, 18.99, 35, "East coast"),
            Waypoint("Hoburg", 56.92, 18.15, 45, "Southern tip rounding"),
            Waypoint("Visby (pass)", 57.64, 18.28, 50, "West coast"),
            Waypoint("Gotska Sandön (pass)", 58.38, 19.00, 55, "Heading back NW"),
            Waypoint("Finish (Sandhamn)", 59.29, 18.92, 60, "Finish"),
        ],
        total_nm=350,
        typical_month=6,  # Late June
        nearby_stations=[
            "svenska_hogarna", "landsort", "gotska_sandon", "farosund",
            "visby_flygplats", "hoburg", "ostergarnsholm", "skarpo",
        ],
        description="350 NM around Gotland, start/finish Sandhamn",
    ),
    "skagen": RaceCourse(
        name="Skagen Race",
        waypoints=[
            Waypoint("Start (Gothenburg)", 57.69, 11.82, 0),
            Waypoint("Skagen", 57.75, 10.60, 60),
            Waypoint("Finish (Gothenburg)", 57.69, 11.82, 60),
        ],
        total_nm=120,
        typical_month=9,
        nearby_stations=["vinga", "maseskar", "nordkoster"],
        description="Gothenburg to Skagen and back",
    ),
}


# ============================================================
# OPEN-METEO API CONFIGURATION
# ============================================================

OPEN_METEO_BASE_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_HISTORICAL_URL = "https://api.open-meteo.com/v1/forecast"  # with past_days param
OPEN_METEO_ARCHIVE_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
OPEN_METEO_PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

# Models available in the Previous Runs API, relevant for Scandinavia/Baltic
PREVIOUS_RUNS_MODELS = [
    "ecmwf_ifs025",
    "ecmwf_aifs025",
    "gfs_seamless",
    "icon_seamless",
    "icon_eu",
    "meteofrance_seamless",
    "metno_seamless",
    "dmi_seamless",
    "knmi_seamless",
    "gem_seamless",
    "jma_seamless",
]

# Variables we need for wind scoring
WIND_VARIABLES = [
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "pressure_msl",
]

# SMHI API
SMHI_BASE_URL = "https://opendata-download-metobs.smhi.se/api"
SMHI_WIND_SPEED_PARAM = 4      # Mean wind speed (m/s)
SMHI_WIND_DIRECTION_PARAM = 3  # Wind direction (degrees)
SMHI_WIND_GUST_PARAM = 21      # Wind gust (m/s)
SMHI_PRESSURE_PARAM = 9        # Air pressure (hPa)


# ============================================================
# UNIT CONVERSIONS
# ============================================================

def ms_to_knots(ms: float) -> float:
    """Convert meters/second to knots."""
    return ms * 1.94384

def knots_to_ms(kts: float) -> float:
    """Convert knots to meters/second."""
    return kts / 1.94384
