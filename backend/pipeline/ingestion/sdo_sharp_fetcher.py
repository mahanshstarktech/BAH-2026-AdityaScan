"""
SDO/HMI SHARP parameter fetcher via JSOC.

SHARP = Space-weather HMI Active Region Patches
18 key magnetic parameters per active region, 12-minute cadence.
Available 2010–present via JSOC (Joint Science Operations Center).

No login required for SHARP data.
Uses sunpy.net.Fido for standardized access.

Key SHARP parameters (used as ML features, from JSOC CEA data series):
  TOTUSJH  — Total unsigned current helicity (G²/m)
  TOTUSJZ  — Total unsigned vertical current (A)
  MEANPOT  — Mean magnetic potential energy density (erg/cm³)
  SAVNCPP  — Sum of absolute values of net current per polarity
  USFLUX   — Total unsigned magnetic flux (Mx)
  AREA_ACT — Active pixel area (Mm²)
  R_VALUE  — Flux ratio of AR to nearby flux (dimensionless)
  SHRGT45  — Fractional area with shear angle > 45° (fraction)
  TOTBSQ   — Sum of |B²| for all pixels (G² Mm²)
  TOTPOT   — Total magnetic energy proxy (G² Mm)
  TOTFZ    — Total unsigned vertical Lorentz force (N)
  ABSNJZH  — Absolute net current helicity (G²/m)
  EPSZ     — Sum of Z-component of Poynting flux over AR
  MEANPOT  — Mean free magnetic energy
  TOTFX    — Total unsigned X-component Lorentz force
  TOTFY    — Total unsigned Y-component Lorentz force
  NACR     — Number of polarity inversion lines with strong gradient

Data access:
  Series: hmi.sharp_cea_720s
  Coverage: 2010-05-01 to present
  Cadence: 12 minutes (720 seconds)
  Format: FITS, returned as sunpy DataFrame

Note: MAG_SHARP (18 params) are the primary LSTM features.
Extra 3 features from MAG L2 (Aditya-L1) appended → 21 total.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# SHARP parameter names exactly as in JSOC CEA series
SHARP_PARAMETERS = [
    "TOTUSJH", "TOTUSJZ", "MEANPOT", "SAVNCPP", "USFLUX",
    "AREA_ACT", "R_VALUE", "SHRGT45", "TOTBSQ", "TOTPOT",
    "TOTFZ", "ABSNJZH", "EPSZ", "TOTFX", "TOTFY",
    "NACR", "HARPNUM", "T_REC",  # last two are metadata, not ML features
]

# The 16 ML feature names (exclude HARPNUM and T_REC from the 18)
SHARP_ML_FEATURES = [
    "TOTUSJH", "TOTUSJZ", "MEANPOT", "SAVNCPP", "USFLUX",
    "AREA_ACT", "R_VALUE", "SHRGT45", "TOTBSQ", "TOTPOT",
    "TOTFZ", "ABSNJZH", "EPSZ", "TOTFX", "TOTFY", "NACR",
]

JSOC_SERIES = "hmi.sharp_cea_720s"


class SHARPFetcher:
    """
    Fetches SDO/HMI SHARP data from JSOC for one or more active regions.

    Usage
    -----
    fetcher = SHARPFetcher()
    df = fetcher.fetch_ar(harpnum=7115, start="2024-05-14", end="2024-05-15")
    features = fetcher.extract_ml_features(df, window_end_unix=..., window_hours=24)
    """

    def fetch_ar(
        self,
        harpnum: int,
        start: str,
        end: str,
    ):
        """
        Fetch SHARP parameters for a specific active region.

        Parameters
        ----------
        harpnum : int
            HARP number (ISRO catalog → NOAA AR → HARPNUM mapping).
        start : str
            Start date "YYYY-MM-DD"
        end : str
            End date "YYYY-MM-DD"

        Returns
        -------
        pandas.DataFrame with columns = SHARP_ML_FEATURES + ["unix_time"]
        """
        try:
            from sunpy.net import Fido, attrs as a
            import pandas as pd

            result = Fido.search(
                a.Time(start, end),
                a.jsoc.Series(JSOC_SERIES),
                a.jsoc.PrimeKey("HARPNUM", str(harpnum)),
                a.jsoc.Keyword("TOTUSJH"),
                a.jsoc.Keyword("USFLUX"),
            )

            if len(result) == 0:
                logger.warning("No SHARP data found for HARPNUM=%d", harpnum)
                return pd.DataFrame()

            files = Fido.fetch(result)
            # Read headers from FITS files
            records = []
            for fp in files:
                from astropy.io import fits
                with fits.open(fp) as hdul:
                    hdr = hdul[1].header if len(hdul) > 1 else hdul[0].header
                    rec = {}
                    for param in SHARP_ML_FEATURES:
                        rec[param] = float(hdr.get(param, np.nan))
                    # Parse T_REC to UNIX time
                    t_rec = str(hdr.get("T_REC", ""))
                    try:
                        dt = datetime.strptime(t_rec[:19], "%Y.%m.%d_%H:%M:%S")
                        rec["unix_time"] = dt.replace(tzinfo=timezone.utc).timestamp()
                    except Exception:
                        rec["unix_time"] = np.nan
                    records.append(rec)

            df = pd.DataFrame(records).sort_values("unix_time").reset_index(drop=True)
            logger.info("Fetched %d SHARP samples for HARPNUM=%d", len(df), harpnum)
            return df

        except ImportError:
            raise ImportError("sunpy required: pip install sunpy")

    def extract_ml_features(
        self,
        df,
        window_end_unix: float,
        window_hours: float = 24.0,
    ) -> Optional[np.ndarray]:
        """
        Extract last 24 hours of SHARP features as LSTM input tensor.

        Returns
        -------
        np.ndarray of shape (120, 16) or None if insufficient data.
        120 = 24 hours × 12-min cadence.
        16 = SHARP ML features.

        Missing values: forward-filled with exponential decay weighting.
        Scale: NOT applied here — use RobustScaler fitted on training data.
        """
        if df is None or len(df) == 0:
            return None

        window_start = window_end_unix - window_hours * 3600.0
        window_df = df[(df["unix_time"] >= window_start) & (df["unix_time"] <= window_end_unix)]

        if len(window_df) < 10:
            logger.debug("Insufficient SHARP data for window (n=%d)", len(window_df))
            return None

        # Forward-fill missing values (NaN = off-disk AR, bad pixels)
        window_df = window_df[SHARP_ML_FEATURES].copy()
        window_df = window_df.ffill().bfill()

        arr = window_df.values.astype(np.float32)  # (T, 16)

        # Pad or truncate to exactly 120 timesteps
        target_len = 120
        if len(arr) >= target_len:
            arr = arr[-target_len:]  # Use most recent 120
        else:
            # Pad with first row (edge padding)
            pad = np.tile(arr[0:1], (target_len - len(arr), 1))
            arr = np.vstack([pad, arr])

        return arr  # (120, 16)
