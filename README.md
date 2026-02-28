# BLUR Weather Intelligence System

**J/99 BLUR (SWE-53435) — AI-assisted weather model scoring for offshore racing**

## What This Is

A Python toolkit that scores weather forecast models against observations to determine which model is most accurate for a given time and location. Inspired by [Model Accuracy](https://www.modelaccuracy.com/) sailing software, but automated, continuous, and Baltic Sea-focused.

The system answers the navigator's core question: **"Which GRIB should I route with, and how should I calibrate it?"**

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Score models against SMHI observations for a location
python -m blur_weather.score --lat 57.7 --lon 11.8 --hours 48

# Score models against an Expedition log file
python -m blur_weather.score --expedition-log path/to/log.csv

# Pre-race analysis for Gotland Runt course
python -m blur_weather.prerace --course gotland_runt --days 7

# Download and archive today's forecasts for the Baltic
python -m blur_weather.fetch --region baltic
```

## Architecture

```
blur_weather/
├── __init__.py
├── __main__.py          # CLI entry point
├── fetch.py             # Download forecasts from Open-Meteo API
├── observe.py           # Fetch observations from SMHI + ViVa
├── expedition.py        # Parse Expedition log files (your boat data)
├── score.py             # Core scoring engine (RMSE, bias, nudge calc)
├── report.py            # Generate human-readable reports
├── polar.py             # Parse and interpolate Expedition polars
├── config.py            # Configuration and constants
├── courses.py           # Race course definitions (waypoints, VIVA stations)
├── data/                # Archived forecasts and observations
└── reference/           # Polars, course files, static data
```

## Data Sources

| Source | Type | Access | Notes |
|--------|------|--------|-------|
| Open-Meteo Forecast API | Model forecasts (JSON) | Free, no key | GFS, ECMWF, ICON, ICON-EU, etc. |
| Open-Meteo Historical Forecast API | Archived forecasts | Free, no key | For backtesting |
| SMHI Open Data API | Weather station observations | Free, no key | Swedish coast stations |
| ViVa (Sjöfartsverket) | Coastal observations | Web scraping | Near-real-time |
| Expedition logs | Boat instrument data | Local CSV files | Your onboard truth |
| PredictWind GRIBs | GRIB forecast files | Your subscription | For actual routing (not scoring) |

## Scoring Methodology

Following Model Accuracy conventions with extensions:

**Per model, per analysis period:**
- **TWS Trend Correlation** (%) — Does the model follow the observed pattern?
- **TWS RMS Error** (knots) — How far off on average?
- **TWS Average Error / Bias** (knots) — Systematic over/under prediction
- **TWS Calibrate** — "Calibrate +1.3 knots" = add 1.3 kt to this model
- **TWD Trend Correlation** (%) — Direction pattern accuracy
- **TWD RMS Error** (degrees) — Direction accuracy
- **TWD Average Error / Bias** (degrees) — Systematic left/right offset
- **TWD Calibrate** — "Calibrate -6.9 degrees" = rotate 6.9° left
- **Time Lag** (hours) — Cross-correlation detected timing offset (NEW vs Model Accuracy)
- **Composite Score** — Weighted combination for ranking

**Nudge recommendation:**
```
Speed: multiply by X (e.g., 0.95 = reduce 5%)
Direction: rotate by Y degrees (e.g., -7° = shift counter-clockwise)
Timing: shift by Z hours (e.g., -2h = model is 2h late)
```

## Key Race Courses

### Gotland Runt
- Start: Sandhamn/Gråskärsfjärden (59.29°N, 18.92°E)
- Via: Svenska Högarna (59.44°N, 19.51°E)
- Round: Gotland (clockwise)
- Key points: Fårö (57.93°N, 19.51°E), Hoburg (56.92°N, 18.15°E), Visby (57.64°N, 18.28°E)
- Finish: Sandhamn
- Distance: ~350 NM
- Typical date: Late June

## BLUR Boat Data

- **Boat:** J/99
- **Polar:** Expedition format, v8 (2025), 4-24 kt TWS
- **Sails:** J1.5, J3.5 HWJ, C0, A3, S2, Storm Jib (North Sails 3Di RAW)
- **Crew config:** Doublehanded, 160 kg
- **Navigation software:** Expedition
- **Log format:** Expedition CSV with numbered channel keys
