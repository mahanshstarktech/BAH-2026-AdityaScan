"""
Tier 1 X-ray Triage Engine — Always-On Light Curve Monitor.

This is the lowest-cost, highest-frequency module in the pipeline.
It runs on RAW LIGHT CURVES (no spectral fitting) at 1-second cadence
and drives the ActivityStateMachine via rolling z-score detection.

Input: Pre-binned SoLEXS/HEL1OS light curve counts (no XSPEC fitting)
Output: z-score + GOES flux proxy + trigger recommendation

Design constraints:
  - Must process one 1-s sample in < 1 ms (well under SoLEXS 1-s cadence)
  - Uses only numpy — no heavy dependencies in the hot path
  - Ring buffer size = 1 hour = 3600 samples (rolling background)

GOES proxy calibration:
  SoLEXS SDD2 1–8 keV channel count rate → GOES equivalent flux
  Calibration factor derived from cross-matching SoLEXS with GOES-16
  during simultaneous observations (July–Dec 2024 Aditya-L1 data).
  Provisional factor: 1 count/s ≈ 5e-10 W/m² (update from cross-cal data)

  Key insight: SoLEXS is a better calibrated soft X-ray detector than GOES
  (spectrally resolved vs. broadband), so we use SoLEXS as primary and
  GOES as cross-check, not the other way around.

SoLEXS bands from manual (Table in User Manual v1.0):
  SDD2 bands used: full 1–8 keV equivalent (the "GOES proxy" band)
  At solar max, use SDD2 (SDD1 may saturate for M+/X-class)
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Calibration constants ────────────────────────────────────────────────────
# PROVISIONAL — update after SoLEXS/GOES cross-calibration from early mission data
SOLEXS_SDD2_COUNTS_TO_GOES_WM2: float = 5.0e-10  # W/m² per count/s

# GOES class proxies from light curve (for log-scale display)
GOES_PROXY_B1 = 1e-7
GOES_PROXY_C1 = 1e-6
GOES_PROXY_M1 = 1e-5
GOES_PROXY_X1 = 1e-4

# Background window for z-score calculation
BG_WINDOW_S: int = 600          # 10-minute rolling background
BG_EXCLUDE_RECENT_S: int = 10   # exclude last 10 s from background (event in progress)
BG_MIN_SAMPLES: int = 30        # need at least 30 samples for reliable stats


def counts_to_goes_proxy(sdd2_counts_per_s: float) -> float:
    """
    Convert SoLEXS SDD2 count rate to approximate GOES 1–8 Å equivalent flux (W/m²).

    This is a provisional linear calibration. Replace with polynomial fit
    once cross-calibration dataset (Aditya-L1 vs GOES-16 simultaneous) is processed.
    """
    return float(sdd2_counts_per_s * SOLEXS_SDD2_COUNTS_TO_GOES_WM2)


def goes_class_from_flux(flux_wm2: float) -> str:
    """
    Return GOES class string from flux value.

    Parameters
    ----------
    flux_wm2 : float
        GOES 1–8 Å flux in W/m².

    Returns
    -------
    str
        e.g. "M3.7", "X1.2", "B4.2", "C6.1"
    """
    if flux_wm2 <= 0:
        return "A0.0"
    log = math.log10(flux_wm2)
    if flux_wm2 >= GOES_PROXY_X1:
        letter = "X"
        mantissa = flux_wm2 / GOES_PROXY_X1
    elif flux_wm2 >= GOES_PROXY_M1:
        letter = "M"
        mantissa = flux_wm2 / GOES_PROXY_M1
    elif flux_wm2 >= GOES_PROXY_C1:
        letter = "C"
        mantissa = flux_wm2 / GOES_PROXY_C1
    elif flux_wm2 >= GOES_PROXY_B1:
        letter = "B"
        mantissa = flux_wm2 / GOES_PROXY_B1
    else:
        letter = "A"
        mantissa = flux_wm2 / 1e-8
    return f"{letter}{mantissa:.1f}"


@dataclass
class TriageSample:
    """Output of one triage evaluation step."""
    unix_time: float
    solexs_sdd2_counts: float       # raw SoLEXS SDD2 count rate (cts/s)
    helios_cdte_30_40_counts: float # HEL1OS CdTe 30-40 keV (cts/s)
    goes_flux_proxy: float          # GOES-equivalent flux (W/m²)
    goes_class: str                 # e.g. "M3.7"
    z_score: float                  # rolling z-score (# of σ above background)
    bg_mean: float                  # background mean (counts/s)
    bg_std: float                   # background std (counts/s)
    alert_level: str                # "QUIET" | "WATCH" | "WARNING" | "ALERT"
    n_bg_samples: int               # how many samples in background estimate


@dataclass
class TriageEngine:
    """
    Rolling-window z-score detector for X-ray light curves.

    Always-on, extremely lightweight. No XSPEC, no ML, no fitting.
    Processes SoLEXS SDD2 + HEL1OS CdTe light curve counts in real time.

    Internal state: two ring buffers (SoLEXS, HEL1OS) + timestamp buffer.

    Usage
    -----
    engine = TriageEngine()
    for ts, cts_solexs, cts_hel1os in stream:
        sample = engine.evaluate(ts, cts_solexs, cts_hel1os)
        state_machine.update(z_score=sample.z_score, goes_flux=sample.goes_flux_proxy)
    """
    bg_window_s: int = BG_WINDOW_S
    bg_exclude_recent_s: int = BG_EXCLUDE_RECENT_S
    bg_min_samples: int = BG_MIN_SAMPLES

    # Ring buffers (1-hour capacity)
    _buf_times: deque = field(default_factory=lambda: deque(maxlen=3600))
    _buf_sdd2: deque = field(default_factory=lambda: deque(maxlen=3600))
    _buf_cdte: deque = field(default_factory=lambda: deque(maxlen=3600))

    # Alert thresholds
    _alert_thresholds: dict = field(default_factory=lambda: {
        "WATCH": 3.0,
        "WARNING": 5.0,
        "ALERT": 8.0,
    })

    def evaluate(
        self,
        unix_time: float,
        solexs_sdd2_counts: float,
        helios_cdte_30_40_counts: float = 0.0,
    ) -> TriageSample:
        """
        Process one 1-second sample and return triage assessment.

        Parameters
        ----------
        unix_time : float
            UNIX timestamp of this sample.
        solexs_sdd2_counts : float
            SoLEXS SDD2 count rate (counts/s) — primary signal.
        helios_cdte_30_40_counts : float
            HEL1OS CdTe 30–40 keV count rate (counts/s) — HXR context.

        Returns
        -------
        TriageSample with z_score, goes_proxy, and alert_level.
        """
        # Append to buffers
        self._buf_times.append(unix_time)
        self._buf_sdd2.append(float(solexs_sdd2_counts))
        self._buf_cdte.append(float(helios_cdte_30_40_counts))

        # Compute background stats (excluding recent 30 s)
        bg_mean, bg_std, n_bg = self._background_stats(unix_time)

        # Z-score of current sample
        if bg_std > 1e-10 and n_bg >= self.bg_min_samples:
            z_score = (solexs_sdd2_counts - bg_mean) / bg_std
        else:
            z_score = 0.0  # insufficient background — no detection

        # GOES proxy
        goes_flux = counts_to_goes_proxy(solexs_sdd2_counts)
        goes_class = goes_class_from_flux(goes_flux)

        # Alert level from z-score
        alert_level = self._classify_alert(z_score)

        return TriageSample(
            unix_time=unix_time,
            solexs_sdd2_counts=solexs_sdd2_counts,
            helios_cdte_30_40_counts=helios_cdte_30_40_counts,
            goes_flux_proxy=goes_flux,
            goes_class=goes_class,
            z_score=float(z_score),
            bg_mean=float(bg_mean),
            bg_std=float(bg_std),
            alert_level=alert_level,
            n_bg_samples=n_bg,
        )

    def recent_statistics(self, window_s: int = 300) -> dict:
        """
        Return statistics over the last window_s seconds.
        Useful for dashboard "last 5 minutes" panels.
        """
        now = self._buf_times[-1] if self._buf_times else 0.0
        start = now - window_s

        sdd2_recent = [
            c for t, c in zip(self._buf_times, self._buf_sdd2) if t >= start
        ]

        if not sdd2_recent:
            return {"mean": 0.0, "max": 0.0, "std": 0.0, "n": 0}

        arr = np.array(sdd2_recent)
        return {
            "mean_counts": float(np.mean(arr)),
            "max_counts": float(np.max(arr)),
            "std_counts": float(np.std(arr)),
            "n_samples": len(arr),
            "goes_proxy_mean": counts_to_goes_proxy(float(np.mean(arr))),
            "goes_class_mean": goes_class_from_flux(
                counts_to_goes_proxy(float(np.mean(arr)))
            ),
        }

    def get_light_curve(self, last_n_seconds: int = 300) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (times, counts) arrays for the last N seconds.
        Used by API endpoint for dashboard light curve plots.
        """
        times = np.array(self._buf_times)
        counts = np.array(self._buf_sdd2)
        if len(times) == 0:
            return np.array([]), np.array([])
        now = times[-1]
        mask = times >= (now - last_n_seconds)
        return times[mask], counts[mask]

    # ── Private helpers ──────────────────────────────────────────────────────

    def _background_stats(self, now: float) -> tuple[float, float, int]:
        """
        Compute rolling background (mean, std, n_samples).
        Excludes the most recent bg_exclude_recent_s seconds to avoid
        contaminating the background with the event under evaluation.
        """
        bg_start = now - self.bg_window_s
        bg_end = now - self.bg_exclude_recent_s

        bg_samples = [
            c for t, c in zip(self._buf_times, self._buf_sdd2)
            if bg_start <= t < bg_end
        ]

        if not bg_samples:
            return 0.0, 0.0, 0

        arr = np.array(bg_samples, dtype=np.float64)
        return float(np.mean(arr)), float(np.std(arr)), len(arr)

    def _classify_alert(self, z_score: float) -> str:
        if z_score >= self._alert_thresholds["ALERT"]:
            return "ALERT"
        elif z_score >= self._alert_thresholds["WARNING"]:
            return "WARNING"
        elif z_score >= self._alert_thresholds["WATCH"]:
            return "WATCH"
        else:
            return "QUIET"
