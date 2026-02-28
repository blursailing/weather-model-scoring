"""CLI entry point for Weather Model Scoring.

Usage:
    # Score models against SMHI observations for last 48 hours
    python -m blur_weather score --station vinga --period latest-day
    
    # Score models against Expedition log
    python -m blur_weather score --expedition-log path/to/log.csv
    
    # Pre-race: score models along entire Gotland Runt course
    python -m blur_weather prerace --course gotland_runt
    
    # Fetch and archive current forecasts
    python -m blur_weather fetch --lat 57.7 --lon 18.3
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def cmd_score(args):
    """Score models against observations."""
    from .fetch import fetch_multi_model_forecast
    from .observe import (
        fetch_smhi_wind_observations,
        parse_expedition_log,
        resample_expedition_to_hourly,
    )
    from .score import score_all_models, print_ranking
    from .config import SMHI_STATIONS
    
    # Determine observation source
    if args.expedition_log:
        logger.info(f"Using Expedition log: {args.expedition_log}")
        raw_obs = parse_expedition_log(args.expedition_log)
        obs = resample_expedition_to_hourly(raw_obs)
        lat = obs["lat"].mean()
        lon = obs["lon"].mean()
        start_date = obs["datetime"].min().strftime("%Y-%m-%d")
        end_date = obs["datetime"].max().strftime("%Y-%m-%d")
        logger.info(f"Log covers {start_date} to {end_date} around ({lat:.2f}, {lon:.2f})")
    
    elif args.station:
        logger.info(f"Using SMHI station: {args.station}")
        # Always fetch at least latest-day so we have enough data to slice
        smhi_period = args.period or "latest-day"
        obs = fetch_smhi_wind_observations(args.station, period=smhi_period)
        if obs.empty:
            logger.error("No observations found. Check station name.")
            sys.exit(1)
        station = SMHI_STATIONS[args.station]
        lat, lon = station.lat, station.lon

        # Optionally restrict to the last N hours
        if args.hours:
            cutoff = datetime.utcnow() - timedelta(hours=args.hours)
            obs = obs[obs["datetime"] >= cutoff]
            logger.info(f"Filtered to last {args.hours} hours: {len(obs)} observations remaining")
            if obs.empty:
                logger.error("No observations in the requested time window.")
                sys.exit(1)

    else:
        logger.error("Specify either --expedition-log or --station")
        sys.exit(1)
    
    if obs.empty or len(obs) < 3:
        logger.error("Not enough observation data to score.")
        sys.exit(1)
    
    # Fetch forecasts for the observation location and time
    logger.info(f"Fetching model forecasts for ({lat:.2f}, {lon:.2f})...")
    
    if args.expedition_log:
        # For expedition logs, we need historical forecasts
        from .fetch import fetch_historical_forecasts
        forecasts = fetch_historical_forecasts(
            lat=lat, lon=lon,
            start_date=start_date, end_date=end_date,
        )
    else:
        # For SMHI stations, use current forecasts with past_days
        forecasts = fetch_multi_model_forecast(
            lat=lat, lon=lon,
            past_days=2, forecast_days=2,
        )
    
    if not forecasts:
        logger.error("Failed to fetch any model forecasts.")
        sys.exit(1)
    
    # Score
    logger.info("Scoring models...")
    scores = score_all_models(forecasts, obs)
    
    # Report
    report = print_ranking(scores)
    print()
    print(report)
    
    # Save report
    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        logger.info(f"Report saved to {args.output}")


def cmd_prerace(args):
    """Pre-race model scoring along an entire course."""
    from .observe import fetch_course_observations
    from .fetch import fetch_multi_model_forecast
    from .score import score_all_models, print_ranking
    from .config import COURSES, SMHI_STATIONS
    
    course = COURSES.get(args.course)
    if not course:
        logger.error(f"Unknown course: {args.course}. Available: {list(COURSES.keys())}")
        sys.exit(1)
    
    logger.info(f"Pre-race analysis for {course.name} ({course.total_nm} NM)")
    logger.info(f"Fetching observations from {len(course.nearby_stations)} stations...")
    
    # Fetch observations from all stations
    all_obs = fetch_course_observations(args.course, period=args.period or "latest-day")
    
    if all_obs.empty:
        logger.error("No observations available from course stations.")
        sys.exit(1)
    
    # Score per station and aggregate
    all_scores = {}
    for station_name in course.nearby_stations:
        station = SMHI_STATIONS.get(station_name)
        if station is None:
            continue
        
        station_obs = all_obs[all_obs["station"] == station.name]
        if len(station_obs) < 3:
            continue
        
        logger.info(f"\nScoring at {station.name} ({len(station_obs)} obs)...")
        
        forecasts = fetch_multi_model_forecast(
            lat=station.lat, lon=station.lon,
            past_days=2, forecast_days=2,
        )
        
        if forecasts:
            scores = score_all_models(forecasts, station_obs)
            all_scores[station.name] = scores
            
            # Quick summary
            if scores:
                best = scores[0]
                logger.info(f"  Best: {best.model_name} (score {best.composite_score:.0f})")
    
    # Aggregate results
    if all_scores:
        print()
        print("=" * 70)
        print(f"PRE-RACE REPORT: {course.name}")
        print("=" * 70)
        print()
        
        for station_name, scores in all_scores.items():
            print(f"--- {station_name} ---")
            for s in scores[:3]:  # Top 3
                print(f"  #{scores.index(s)+1} {s.model_name}: "
                      f"score {s.composite_score:.0f} | "
                      f"TWS RMSE {s.tws_rmse:.1f}kt | "
                      f"TWD RMSE {s.twd_rmse:.0f}° | "
                      f"{s.nudge.tws_calibrate_str}")
            print()


def cmd_fetch(args):
    """Fetch and archive current forecasts."""
    from .fetch import fetch_multi_model_forecast, archive_forecasts
    
    forecasts = fetch_multi_model_forecast(
        lat=args.lat, lon=args.lon,
        forecast_days=args.days or 5,
    )
    
    if forecasts:
        archive_forecasts(forecasts, args.lat, args.lon)
        logger.info(f"Archived {len(forecasts)} model forecasts.")
    else:
        logger.error("No forecasts fetched.")


def main():
    parser = argparse.ArgumentParser(
        description="Weather Model Scoring — Forecast accuracy for offshore racing",
        prog="blur_weather",
    )
    subparsers = parser.add_subparsers(dest="command")
    
    # Score command
    p_score = subparsers.add_parser("score", help="Score models against observations")
    p_score.add_argument("--expedition-log", help="Path to Expedition CSV log file")
    p_score.add_argument("--station", help="SMHI station name (e.g., 'vinga', 'hoburg')")
    p_score.add_argument("--period", default="latest-day", help="SMHI period: latest-hour, latest-day, latest-months")
    p_score.add_argument("--hours", type=int, default=None, help="Restrict scoring to last N hours of observations (e.g. --hours 12)")
    p_score.add_argument("--output", "-o", help="Save report to file")
    p_score.set_defaults(func=cmd_score)
    
    # Pre-race command
    p_prerace = subparsers.add_parser("prerace", help="Pre-race course analysis")
    p_prerace.add_argument("--course", required=True, help="Course name (gotland_runt, skagen)")
    p_prerace.add_argument("--period", default="latest-day")
    p_prerace.set_defaults(func=cmd_prerace)
    
    # Fetch command
    p_fetch = subparsers.add_parser("fetch", help="Fetch and archive forecasts")
    p_fetch.add_argument("--lat", type=float, required=True)
    p_fetch.add_argument("--lon", type=float, required=True)
    p_fetch.add_argument("--days", type=int, default=5)
    p_fetch.set_defaults(func=cmd_fetch)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()
