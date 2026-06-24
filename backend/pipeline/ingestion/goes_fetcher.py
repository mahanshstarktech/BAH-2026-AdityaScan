"""
GOES-16/18 XRS real-time and historical data fetcher.

Sources:
  Real-time (1-min):   https://services.swpc.noaa.gov/products/goes-xray.json
  Historical archive:  https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/
  Cadence:             1 minute (real-time) / 1 minute (archive)
  Channels:
    Channel 1: 0.5–4 Å (hard X-ray)
    Channel 2: 1–8 Å   (soft X-ray, "GOES class" channel)

GOES class encoding (W/m², 1–8 Å):
  B: 1e-7 – 1e-6   C: 1e-6 – 1e-5   M: 1e-5 – 1e-4   X: ≥ 1e-4

Usage in AdityScan:
  1. Training labels (50-year pretraining dataset)
  2. Cross-validation of Aditya-L1 detections
  3. Real-time comparison channel when Aditya-L1 data is delayed
  4. "Current GOES class" indicator on dashboard
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

GOES_REALTIME_URL = "https://services.swpc.noaa.gov/products/goes-xray.json"
GOES_CACHE_TTL_S = 60  # Refresh real-time data no more than once per minute


@dataclass
class GOESReading:
    """One 1-minute GOES XRS reading."""
    unix_time: float
    satellite: str       # "GOES-16" or "GOES-18"
    flux_0p5_4: float    # W/m², 0.5–4 Å channel
    flux_1_8: float      # W/m², 1–8 Å channel (GOES class channel)
    goes_class: str      # e.g. "M3.7"

    @property
    def is_flare(self) -> bool:
        return self.flux_1_8 >= 1e-6  # C-class or higher


class GOESFetcher:
    """
    Fetches NOAA GOES-16/18 XRS real-time data.
    Caches to avoid rate-limiting NOAA's API.
    """

    def __init__(self) -> None:
        self._cache: list[GOESReading] = []
        self._cache_time: float = 0.0

    def fetch_realtime(self, force: bool = False) -> list[GOESReading]:
        """
        Fetch the last 7 days of 1-minute GOES XRS data.
        Cached for GOES_CACHE_TTL_S seconds.
        Returns readings sorted by time (newest last).
        """
        now = time.time()
        if not force and (now - self._cache_time) < GOES_CACHE_TTL_S:
            return self._cache

        try:
            import requests
            resp = requests.get(GOES_REALTIME_URL, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("GOES real-time fetch failed: %s", exc)
            return self._cache  # Return stale cache on failure

        readings = []
        for row in data[1:]:  # First row is header
            try:
                # Format: [time_tag, satellite, flux_0p5_4, flux_1_8]
                t_str = row[0]  # "2024-05-14 00:00:00.000"
                dt = datetime.strptime(t_str[:19], "%Y-%m-%d %H:%M:%S")
                dt = dt.replace(tzinfo=timezone.utc)
                unix_t = dt.timestamp()

                flux_short = float(row[2])  # 0.5–4 Å
                flux_long = float(row[3])   # 1–8 Å (class channel)
                sat = str(row[1]) if len(row) > 1 else "GOES-16"

                if flux_long <= 0:
                    continue

                from pipeline.physics.triage import goes_class_from_flux
                readings.append(GOESReading(
                    unix_time=unix_t,
                    satellite=sat,
                    flux_0p5_4=flux_short,
                    flux_1_8=flux_long,
                    goes_class=goes_class_from_flux(flux_long),
                ))
            except Exception:
                continue

        self._cache = sorted(readings, key=lambda r: r.unix_time)
        self._cache_time = now
        logger.info("GOES real-time: %d readings fetched", len(self._cache))
        return self._cache

    def latest(self) -> Optional[GOESReading]:
        """Return the most recent GOES reading."""
        readings = self.fetch_realtime()
        return readings[-1] if readings else None

    def flare_list(self, min_class: str = "M") -> list[GOESReading]:
        """
        Return events above a minimum GOES class from recent data.
        min_class: "B" | "C" | "M" | "X"
        """
        thresholds = {"B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}
        thresh = thresholds.get(min_class.upper(), 1e-5)
        readings = self.fetch_realtime()
        return [r for r in readings if r.flux_1_8 >= thresh]
