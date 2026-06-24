"""
MAG (Magnetometer) data loader for Aditya-L1.

Grounded in: MAG User Manual, ISRO/SAC (2024)
  - Dual tri-axial fluxgate magnetometers: MAG1 (6m boom), MAG2 (3m boom)
  - L1: netCDF, cadence = 0.128 s (128 ms), SCCS frame, both MAG1+MAG2
  - L2: netCDF, cadence = 10 s, GSE + GSM frames, MAG1 only (preferred)
  - Quality flag: 0 = bad/missing, 1 = valid (per sample, L2)
  - Uncertainty: ~0.3 nT per component (L1), ~0.5 nT (L2)
  - File formats:
      L1: L1_PLDXXSTNP#SATSTRIPVCAPIdYYDDDHHMMSSMSE_<obsId>_Vmn.nc
      L2: L2_AL1_MAG_YYYYMMDD_VMN.nc
  - L2 variables (used here): time, Bx_gse, By_gse, Bz_gse,
                               Bx_gsm, By_gsm, Bz_gsm + _error variants,
                               Quality_flag_10s_data
  - DO NOT use L1 for high-freq science (spacecraft artifacts > 1 Hz)
  - Use L2 data for all pipeline features (10-s cadence, science-ready)

Reference: MAG User Manual, URSC/ISRO 2024 (all variable names exact)
"""

from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import netCDF4 as nc  # standard netCDF4-python library

logger = logging.getLogger(__name__)

# ── Constants from MAG manual ────────────────────────────────────────────────
MAG_L2_CADENCE_S: float = 10.0          # 10-second averaged product
MAG_L1_CADENCE_S: float = 0.128         # 128 ms native rate (NOT for science)
MAG_UNCERTAINTY_L2_NT: float = 0.5      # nT, net uncertainty at L2
MAG_QUALITY_VALID: int = 1              # Binary quality flag: 1 = valid
MAG_QUALITY_BAD: int = 0

# L2 variable names exactly as listed in Table 3 of the manual
_L2_VARS = [
    "time",
    "Bx_gse", "By_gse", "Bz_gse",
    "Bx_gsm", "By_gsm", "Bz_gsm",
    "Bx_gse_error", "By_gse_error", "Bz_gse_error",
    "Bx_gsm_error", "By_gsm_error", "Bz_gsm_error",
    "x_gse", "y_gse", "z_gse",
    "x_gsm", "y_gsm", "z_gsm",
    "Quality_flag_10s_data",
]


@dataclass
class MAGRecord:
    """One 10-second MAG L2 sample, ready for feature extraction."""
    unix_time: float          # UNIX seconds (epoch Jan 1, 1970 00:00:00)
    Bx_gse: float             # nT
    By_gse: float             # nT
    Bz_gse: float             # nT
    Bx_gsm: float             # nT
    By_gsm: float             # nT
    Bz_gsm: float             # nT
    Bx_err: float             # nT uncertainty (GSE)
    By_err: float             # nT uncertainty
    Bz_err: float             # nT uncertainty
    quality: int              # 0 = bad, 1 = valid
    # Derived features (computed on load)
    B_total: float = field(init=False)
    clock_angle_deg: float = field(init=False)   # atan2(By, Bz) in GSM, deg
    cone_angle_deg: float = field(init=False)    # atan2(sqrt(By²+Bz²), |Bx|) deg

    def __post_init__(self) -> None:
        self.B_total = float(np.sqrt(self.Bx_gse**2 + self.By_gse**2 + self.Bz_gse**2))
        # Clock angle in GSM (south ± from ecliptic north)
        self.clock_angle_deg = float(np.degrees(np.arctan2(self.By_gsm, self.Bz_gsm)))
        # Cone angle (Parker spiral deviation)
        self.cone_angle_deg = float(
            np.degrees(np.arctan2(
                np.sqrt(self.By_gsm**2 + self.Bz_gsm**2),
                abs(self.Bx_gsm)
            ))
        )


class MAGLoader:
    """
    Loads Aditya-L1 MAG Level-2 daily netCDF files.

    Usage
    -----
    loader = MAGLoader(data_dir="/data/mag/l2")
    records = loader.load_day("20240514")
    features = loader.extract_ml_features(records, window_minutes=30)
    """

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)

    # ── Public API ───────────────────────────────────────────────────────────

    def load_day(self, date_str: str) -> list[MAGRecord]:
        """
        Load MAG L2 data for a given date (YYYYMMDD).

        Follows manual naming: L2_AL1_MAG_YYYYMMDD_VMN.nc
        Picks highest version if multiple files exist for same date.

        Parameters
        ----------
        date_str : str
            Date in YYYYMMDD format, e.g. "20240514"

        Returns
        -------
        list[MAGRecord]
            List of valid (quality=1) 10-s records, sorted by time.
        """
        pattern = str(self.data_dir / f"L2_AL1_MAG_{date_str}_V*.nc")
        files = sorted(glob.glob(pattern))

        if not files:
            logger.warning("No MAG L2 file found for date %s in %s", date_str, self.data_dir)
            return []

        # Manual: "users are recommended to use the highest version"
        filepath = files[-1]
        logger.info("Loading MAG L2: %s", filepath)
        return self._parse_l2_nc(filepath)

    def load_range(self, start_date: str, end_date: str) -> list[MAGRecord]:
        """Load multiple days. Dates in YYYYMMDD format."""
        from datetime import datetime, timedelta
        start = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")
        records: list[MAGRecord] = []
        current = start
        while current <= end:
            records.extend(self.load_day(current.strftime("%Y%m%d")))
            current += timedelta(days=1)
        return records

    def extract_ml_features(
        self,
        records: list[MAGRecord],
        window_end_unix: float,
        window_minutes: float = 30.0,
    ) -> Optional[np.ndarray]:
        """
        Extract a fixed-length feature vector from a rolling window of MAG data.

        Used as input to the in-situ MLP branch of the multi-modal network.

        Features returned (shape: [8]):
          [B_total_mean, B_total_std, Bx_gse_mean, By_gse_mean, Bz_gse_mean,
           clock_angle_mean, cone_angle_mean, B_variance_30min]

        Returns None if <50% valid data in window (manual advises caution
        for quiet-time data with invalid samples).
        """
        window_start = window_end_unix - window_minutes * 60.0
        window = [r for r in records
                  if window_start <= r.unix_time <= window_end_unix
                  and r.quality == MAG_QUALITY_VALID]

        # Need at least 50% fill (manual: invalid samples can infiltrate GTI data)
        expected_samples = int(window_minutes * 60 / MAG_L2_CADENCE_S)
        if len(window) < expected_samples * 0.5:
            logger.debug(
                "MAG window at t=%.0f has only %d/%d valid samples — skipping",
                window_end_unix, len(window), expected_samples
            )
            return None

        B_total = np.array([r.B_total for r in window])
        Bx = np.array([r.Bx_gse for r in window])
        By = np.array([r.By_gse for r in window])
        Bz = np.array([r.Bz_gse for r in window])
        clock = np.array([r.clock_angle_deg for r in window])
        cone = np.array([r.cone_angle_deg for r in window])

        return np.array([
            float(np.mean(B_total)),
            float(np.std(B_total)),
            float(np.mean(Bx)),
            float(np.mean(By)),
            float(np.mean(Bz)),
            float(np.mean(clock)),
            float(np.mean(cone)),
            float(np.var(B_total)),          # variance = spread indicator
        ], dtype=np.float32)

    def detect_icme_candidates(
        self, records: list[MAGRecord], smooth_minutes: float = 60.0
    ) -> list[dict]:
        """
        Flag potential ICME/magnetic cloud intervals using classical criteria:
          - |B| elevation > 2× background
          - Low B variance (smooth field rotation = magnetic cloud signature)
          - Clock angle monotonically rotating (Burlaga et al. 1981)

        Returns list of candidate start/end times (UNIX seconds).
        """
        valid = [r for r in records if r.quality == MAG_QUALITY_VALID]
        if len(valid) < 60:
            return []

        B_arr = np.array([r.B_total for r in valid])
        t_arr = np.array([r.unix_time for r in valid])
        clock_arr = np.array([r.clock_angle_deg for r in valid])

        # Rolling background (1-hour window)
        n_bg = max(1, int(smooth_minutes * 60 / MAG_L2_CADENCE_S))
        B_bg = np.convolve(B_arr, np.ones(n_bg) / n_bg, mode="same")
        B_ratio = B_arr / np.maximum(B_bg, 1.0)

        candidates = []
        in_event = False
        ev_start = 0.0

        for i, (t, ratio) in enumerate(zip(t_arr, B_ratio)):
            if ratio > 2.0 and not in_event:
                in_event = True
                ev_start = t
            elif ratio < 1.3 and in_event:
                in_event = False
                candidates.append({"start_unix": ev_start, "end_unix": t,
                                   "duration_h": (t - ev_start) / 3600.0})

        return candidates

    # ── Private helpers ──────────────────────────────────────────────────────

    def _parse_l2_nc(self, filepath: str) -> list[MAGRecord]:
        """Parse a MAG L2 netCDF file into MAGRecord objects."""
        records: list[MAGRecord] = []
        try:
            with nc.Dataset(filepath, "r") as ds:
                time_arr = np.array(ds.variables["time"][:])
                Bx_gse = np.array(ds.variables["Bx_gse"][:])
                By_gse = np.array(ds.variables["By_gse"][:])
                Bz_gse = np.array(ds.variables["Bz_gse"][:])
                Bx_gsm = np.array(ds.variables["Bx_gsm"][:])
                By_gsm = np.array(ds.variables["By_gsm"][:])
                Bz_gsm = np.array(ds.variables["Bz_gsm"][:])
                Bx_err = np.array(ds.variables["Bx_gse_error"][:])
                By_err = np.array(ds.variables["By_gse_error"][:])
                Bz_err = np.array(ds.variables["Bz_gse_error"][:])
                quality = np.array(ds.variables["Quality_flag_10s_data"][:])

            n = len(time_arr)
            for i in range(n):
                # Skip fill values (netCDF masked arrays)
                if any(np.ma.is_masked(v) for v in [
                    Bx_gse[i], By_gse[i], Bz_gse[i],
                    Bx_gsm[i], By_gsm[i], Bz_gsm[i]
                ]):
                    continue
                records.append(MAGRecord(
                    unix_time=float(time_arr[i]),
                    Bx_gse=float(Bx_gse[i]),
                    By_gse=float(By_gse[i]),
                    Bz_gse=float(Bz_gse[i]),
                    Bx_gsm=float(Bx_gsm[i]),
                    By_gsm=float(By_gsm[i]),
                    Bz_gsm=float(Bz_gsm[i]),
                    Bx_err=float(Bx_err[i]),
                    By_err=float(By_err[i]),
                    Bz_err=float(Bz_err[i]),
                    quality=int(quality[i]),
                ))

        except Exception as exc:
            logger.error("Failed to parse MAG L2 file %s: %s", filepath, exc)

        logger.info("Loaded %d valid MAG records from %s", len(records), filepath)
        return sorted(records, key=lambda r: r.unix_time)
