"""Source adapters for weather observation APIs (SMHI, MET Norway, DMI, FMI).

Each adapter implements:
    fetch_observations(station: dict) -> list[dict]

Returning standardised observation dicts with keys:
    station_code, observed_at, wind_speed_ms, wind_direction_deg,
    air_pressure_hpa, air_temperature_c
"""

import logging
import time

import requests

logger = logging.getLogger(__name__)


def fetch_with_retry(url, max_retries=2, backoff_base=1, timeout=30, **kwargs):
    """HTTP GET with exponential backoff retry.

    Retries on requests.RequestException.  Backoff schedule: 1 s, 4 s
    (base * 4^attempt).

    Args:
        url: Request URL.
        max_retries: Number of retries after the initial attempt.
        backoff_base: Base backoff in seconds.
        timeout: Request timeout in seconds.
        **kwargs: Forwarded to ``requests.get()``
                  (e.g. ``params``, ``auth``, ``headers``).

    Returns:
        requests.Response with status 2xx.

    Raises:
        requests.RequestException: After all retries are exhausted.
        requests.HTTPError: If the final response has a non-2xx status.
    """
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, timeout=timeout, **kwargs)
            resp.raise_for_status()
            return resp
        except requests.HTTPError as exc:
            # Don't retry on 4xx client errors (404, 403, etc.) — they won't resolve
            if exc.response is not None and 400 <= exc.response.status_code < 500:
                raise
            last_exc = exc
            if attempt < max_retries:
                wait = backoff_base * (4 ** attempt)
                logger.warning(
                    "Retry %d/%d for %s after %.1fs (%s)",
                    attempt + 1, max_retries, url, wait, exc,
                )
                time.sleep(wait)
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < max_retries:
                wait = backoff_base * (4 ** attempt)
                logger.warning(
                    "Retry %d/%d for %s after %.1fs (%s)",
                    attempt + 1, max_retries, url, wait, exc,
                )
                time.sleep(wait)
    raise last_exc
