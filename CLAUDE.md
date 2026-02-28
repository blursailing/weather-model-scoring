# CLAUDE.md — Instructions for Claude Code

## Project: BLUR Weather Intelligence System
This is a Python project for scoring weather forecast models against observations,
designed for offshore sailing racing (J/99 BLUR, Gotland Runt, etc.).

## Context
The owner (Peter Gustafsson) runs blur.se, an ambitious offshore racing project.
He uses Expedition navigation software, PredictWind for GRIBs, and is familiar
with Model Accuracy sailing software. The goal is to automate and extend what
Model Accuracy does — continuously score models, detect bias, and recommend
calibration ("nudging") for routing.

## Key Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Score models against an SMHI station
python -m blur_weather score --station vinga

# Score models against an Expedition log
python -m blur_weather score --expedition-log blur_weather/reference/log-2025Sep05-skagen01.csv

# Pre-race analysis for Gotland Runt
python -m blur_weather prerace --course gotland_runt
```

## Architecture
- `blur_weather/config.py` — All constants, model definitions, station coordinates, course waypoints
- `blur_weather/fetch.py` — Open-Meteo API client (forecasts from multiple models)
- `blur_weather/observe.py` — SMHI API client + Expedition log parser
- `blur_weather/score.py` — Core scoring engine (RMSE, bias, cross-correlation, nudge)
- `blur_weather/polar.py` — Expedition polar file parser
- `blur_weather/__main__.py` — CLI interface

## Data Sources (all free, no API keys)
- Open-Meteo: `api.open-meteo.com` — multi-model forecasts and historical archive
- SMHI: `opendata-download-metobs.smhi.se` — Swedish weather stations
- Expedition logs: local CSV files from Peter's boat

## Conventions
- Wind speed always in knots internally
- Wind direction in degrees true (0-360)
- Output format follows Model Accuracy conventions:
  "TWS: Trend correlation 75%, RMS error 3.2, Average error 1.3 knots below log data. Calibrate +1.3 knots."
- Circular statistics for all wind direction math
- Models ranked by composite score (0-100, higher = better)

## Current Status
- Core modules are implemented and ready to test
- Need network access to call Open-Meteo and SMHI APIs
- Reference data in `blur_weather/reference/`:
  - `J99_blur_2025_v8.txt` — BLUR's Expedition polar
  - `log-2025Sep05-skagen01.csv` — Skagen Race Sept 2025 (66K records)

## Next Steps (in priority order)
1. Test the full pipeline: fetch → observe → score → report
2. Validate scoring against the 2019 Kattegat Model Accuracy results (see blur.se blog)
3. Add ViVa scraping for real-time coastal observations
4. Build historical database of Baltic model performance
5. Add weather regime classification (frontal/high-pressure/thermal)
6. Create scheduled task for continuous monitoring
