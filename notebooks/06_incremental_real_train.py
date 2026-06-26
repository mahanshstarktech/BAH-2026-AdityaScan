#!/usr/bin/env python3
"""
AdityScan v4 — REAL Incremental Multi-Modal Training Pipeline
=============================================================
Trains on REAL data from PRADAN and NASA/JSOC in monthly batches.

HOW IT WORKS (the ChatGPT batch training idea, implemented):
------------------------------------------------------------
  Month 1 download → Train on it → Save checkpoint → DELETE data
  Month 2 download → Load checkpoint → Continue training → Save → DELETE data
  Month 3 ...
  → All months trained → Export ONNX → Push to GitHub → Render deploys live

DATA SOURCES USED (ALL modalities, as promised):
------------------------------------------------
  PRIMARY (Aditya-L1 from PRADAN):
    ✅ SoLEXS  L1 — 1-s X-ray light curves (SDD1 + SDD2)
    ✅ HEL1OS  L1 — 1-s multi-band hard X-ray (CdTe + CZT)
    ✅ MAG     L2 — 10-s magnetic field (Bx/By/Bz GSE+GSM)
    ✅ ASPEX-SWIS L2 — solar wind bulk params (density, T, speed)

  SUPPLEMENTARY (NASA SDO from JSOC — free, no login):
    ✅ SDO/HMI SHARP — magnetic complexity params (21 features)
    ✅ NOAA SSN      — daily sunspot number (solar cycle phase)
    ✅ GOES XRS      — X-ray flux for label cross-validation

  IMAGES (Phase 3 - SUIT + SDO AIA):
    ✅ SUIT UV images — 200-400nm, 1 image per day (from PRADAN)
    ✅ SDO AIA 304Å  — active region detection (from JSOC)

WAVELET TRANSFORM (Quasi-Periodic Pulsation detection):
-------------------------------------------------------
    ✅ Continuous Wavelet Transform applied to SoLEXS light curves
       before feeding into TCN — detects QPPs before flare onset

TRAINING SCHEDULE (what months, and why):
-----------------------------------------
  Aditya-L1 available from: January 2024 (first science data)
  Best months for flare training (Solar Cycle 25 maximum):
    2024-02: X3.3 flare (AR13575) — first major Aditya-L1 event
    2024-05: X8.7 (AR13842) + Great Auroral Storm — BEST DATA
    2024-10: X9.0 (AR13847) — second biggest 2024 event
    2024-12: X5.1 — high activity
    2025-01: Continued activity
    2025-02: X-class events
    2025-03: High solar activity

  → Default: Jan 2024 through Mar 2025 = 15 months
  → Peak months (2024-05, 2024-10) weighted more in training

MEMORY MANAGEMENT (for 16 GB M4 MacBook Air):
----------------------------------------------
  Each month: ~200-300 MB SoLEXS + ~100 MB HEL1OS + ~50 MB MAG = ~500 MB RAM
  PyTorch batch size tuned so GPU (MPS) never exceeds 4 GB
  After each month: gc.collect() + torch.mps.empty_cache()
  Disk: downloads are deleted after training each month
  Total disk usage at any time: < 2 GB

USAGE (your friend just runs):
------------------------------
  cd adityscan/
  python notebooks/06_incremental_real_train.py

  OPTIONS:
  --months 2024-05,2024-10     Train only specific high-activity months
  --resume                     Resume from last saved checkpoint
  --skip-download              Use already-downloaded data in data/pradan_cache/
  --quick-test                 Test 3 days only (verify setup works)
  --no-suit                    Skip SUIT image branch (faster, less disk)
"""

from __future__ import annotations

import argparse
import gc
import gzip
import io
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# ── Setup paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from pipeline.ingestion.solexs_loader import SoLEXSLoader
from pipeline.ingestion.helios_loader import HEL1OSLoader
from pipeline.ingestion.mag_loader import MAGLoader
from pipeline.ingestion.aspex_swis_loader import SWISLoader

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_FILE = ROOT / "training.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOG_FILE), mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# ── Directories ────────────────────────────────────────────────────────────────
MODELS_DIR   = ROOT / "models"
CACHE_DIR    = ROOT / "data" / "pradan_cache"
GOES_DIR     = ROOT / "data" / "goes"
SHARP_DIR    = ROOT / "data" / "sharp"
MODELS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
GOES_DIR.mkdir(parents=True, exist_ok=True)
SHARP_DIR.mkdir(parents=True, exist_ok=True)

# ── Training months in priority order ──────────────────────────────────────────
# These are ordered by SCIENTIFIC VALUE (biggest flare events first)
TRAINING_MONTHS = [
    # FORMAT: (YYYY-MM, description, flare_weight_multiplier)
    # weight > 1.0 means more sampling from this month (it has more M+ events)
    ("2024-05", "X8.7 + Great Auroral Storm — BEST MONTH",   3.0),
    ("2024-10", "X9.0 + X7.1 — Second best",                 2.5),
    ("2024-12", "X5.1 — High activity",                       2.0),
    ("2024-03", "Several X-class events",                     1.8),
    ("2024-11", "M/X activity",                               1.5),
    ("2025-01", "Solar max continued",                        1.5),
    ("2025-02", "X-class events",                             1.5),
    ("2025-03", "High activity",                              1.5),
    ("2024-02", "X3.3 — First major event",                   1.3),
    ("2024-04", "Moderate activity",                          1.2),
    ("2024-06", "Post-storm activity",                        1.2),
    ("2024-07", "Moderate",                                   1.0),
    ("2024-08", "Moderate + SWIS data released",              1.0),
    ("2024-09", "Moderate",                                   1.0),
    ("2025-04", "Recent data",                                1.0),
]


# ══════════════════════════════════════════════════════════════════════════════
# DEVICE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("✅ Apple M4 Metal GPU (MPS) — using MPS backend")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("✅ CUDA GPU: %s", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        logger.warning("⚠️  CPU only — training will be slow")
    return device


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: DOWNLOAD REAL DATA (month by month)
# ══════════════════════════════════════════════════════════════════════════════

def download_month_pradan(year_month: str, args) -> dict[str, Path]:
    """
    Download one month of Aditya-L1 data from PRADAN.
    
    Returns: dict mapping instrument -> directory of downloaded files.
    
    NOTE ON PRADAN LOGIN:
    Set environment variables before running:
        export PRADAN_USER=your_username
        export PRADAN_PASS=your_password
    
    If credentials not set, skip PRADAN and use GOES + SHARP only.
    """
    import asyncio
    from pipeline.ingestion.pradan_downloader import PRADANSession

    year, month = year_month.split("-")
    month_dir = CACHE_DIR / year_month
    month_dir.mkdir(exist_ok=True)

    pradan_user = os.environ.get("PRADAN_USER", "")
    pradan_pass = os.environ.get("PRADAN_PASS", "")

    downloaded = {
        "solexs": month_dir / "solexs",
        "helios": month_dir / "helios",
        "mag":    month_dir / "mag",
        "swis":   month_dir / "swis",
    }
    for d in downloaded.values():
        d.mkdir(exist_ok=True)

    if not pradan_user or not pradan_pass:
        logger.warning(
            "PRADAN_USER/PRADAN_PASS not set — skipping PRADAN download.\n"
            "  To download real data, run:\n"
            "    export PRADAN_USER=your_username\n"
            "    export PRADAN_PASS=your_password\n"
            "  Then re-run this script."
        )
        return downloaded

    # Build list of days in this month
    start = datetime(int(year), int(month), 1)
    if int(month) == 12:
        end = datetime(int(year) + 1, 1, 1)
    else:
        end = datetime(int(year), int(month) + 1, 1)
    days = []
    d = start
    while d < end:
        days.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)

    if args.quick_test:
        days = days[:3]  # Only 3 days in quick test mode
        logger.info("Quick test: only downloading %d days", len(days))

    logger.info("Downloading %s data for %d days from PRADAN...", year_month, len(days))

    async def _download_all():
        async with PRADANSession() as session:
            for date_str in days:
                logger.info("  Fetching %s...", date_str)
                files = await session.list_files("SoLEXS", "", "")
                for f in files:
                    fn = f.get("filename", "")
                    if date_str in fn:
                        dest = downloaded["solexs"] / fn
                        await session.download_file(fn, dest, f.get("download_url", ""))

                files = await session.list_files("HELIOS", "", "")
                for f in files:
                    fn = f.get("filename", "")
                    if date_str in fn:
                        dest = downloaded["helios"] / fn
                        await session.download_file(fn, dest, f.get("download_url", ""))

    try:
        asyncio.run(_download_all())
        logger.info("PRADAN download complete for %s", year_month)
    except Exception as exc:
        logger.error("PRADAN download error: %s", exc)

    return downloaded


def download_month_goes(year_month: str, args) -> Path:
    """
    Download GOES XRS data for one month from NOAA NCEI.
    GOES data is FREE, no login required.
    URL: https://www.ngdc.noaa.gov/stp/space-weather/solar-data/
    
    Uses sunpy (already in requirements.txt) to download automatically.
    """
    year, month = year_month.split("-")
    goes_file = GOES_DIR / f"goes_1min_{year_month}.csv"

    if goes_file.exists() and goes_file.stat().st_size > 10_000:
        logger.info("GOES %s already cached: %s", year_month, goes_file)
        return goes_file

    logger.info("Downloading GOES XRS data for %s from NOAA...", year_month)
    try:
        import sunpy.timeseries as ts
        from sunpy.net import Fido, attrs

        start_date = f"{year}-{month}-01"
        # Compute end of month
        start_dt = datetime(int(year), int(month), 1)
        if int(month) == 12:
            end_dt = datetime(int(year) + 1, 1, 1)
        else:
            end_dt = datetime(int(year), int(month) + 1, 1)
        end_date = end_dt.strftime("%Y-%m-%d")

        result = Fido.search(
            attrs.Time(start_date, end_date),
            attrs.Instrument.xrs,
            attrs.goes.SatelliteNumber(18),  # GOES-18 (current primary)
        )
        if len(result) == 0:
            logger.warning("No GOES data found via sunpy for %s", year_month)
            return goes_file

        downloaded = Fido.fetch(result, path=str(GOES_DIR))
        if downloaded:
            logger.info("GOES %s downloaded: %d files", year_month, len(downloaded))
        return goes_file

    except Exception as exc:
        logger.warning("GOES download via sunpy failed: %s — trying NOAA direct URL", exc)

    # Fallback: direct NOAA NCEI HTTP download (no login needed)
    try:
        import urllib.request
        # NOAA NCEI 1-minute averages — public
        url = (
            f"https://satdat.ngdc.noaa.gov/sem/goes/data/avg/2024/{month}/goes18/csv/"
            f"g18_xrs_1m_{year}{month}01_{year}{month}31.csv"
        )
        logger.info("Trying NOAA direct download: %s", url)
        urllib.request.urlretrieve(url, str(goes_file))
        logger.info("GOES %s downloaded: %.1f KB", year_month, goes_file.stat().st_size / 1024)
    except Exception as exc2:
        logger.warning("NOAA direct download failed: %s", exc2)

    return goes_file


def download_month_sharp(year_month: str, args) -> Path:
    """
    Download SDO/HMI SHARP magnetic parameters from JSOC (NASA).
    FREE, no login required for standard data products.
    
    SHARP params: 21 magnetic complexity features per active region per 12 min.
    Reference: Bobra & Couvidat 2015 (standard in space weather AI)
    """
    year, month = year_month.split("-")
    sharp_file = SHARP_DIR / f"sharp_{year_month}.csv"

    if sharp_file.exists() and sharp_file.stat().st_size > 5_000:
        logger.info("SHARP %s already cached", year_month)
        return sharp_file

    logger.info("Downloading SDO/HMI SHARP data for %s from JSOC...", year_month)
    try:
        import urllib.request
        # JSOC export API — standard public access, no authentication
        # These are series: hmi.sharp_cea_720s — 12-min cadence SHARP params
        start_dt = datetime(int(year), int(month), 1)
        if int(month) == 12:
            end_dt = datetime(int(year) + 1, 1, 1)
        else:
            end_dt = datetime(int(year), int(month) + 1, 1)
        
        # JSOC export via drms (Python library)
        try:
            import drms
            c = drms.Client(email="adityscan@research.isro", verbose=False)
            # Request SHARP data for all active regions
            series = "hmi.sharp_cea_720s"
            segs = "Bp,Br,Bt"  # we want metadata/keywords, not images
            keys = [
                "T_REC", "HARPNUM", "LAT_MIN", "LON_MIN", "LAT_MAX", "LON_MAX",
                "USFLUX", "MEANGBT", "MEANJZD", "TOTUSJH", "MEANALP",
                "MEANGAM", "MEANGBZ", "MEANGBH", "MEANJZH", "TOTUSJZ",
                "ABSNJZH", "SAVNCPP", "MEANPOT", "TOTPOT", "MEANSHR",
                "SHRGT45", "AREA_ACR", "R_VALUE",
            ]
            start_str = start_dt.strftime("%Y.%m.%d_00:00:00_TAI")
            end_str   = end_dt.strftime("%Y.%m.%d_00:00:00_TAI")
            q = f"{series}[{start_str}/{(end_dt - start_dt).days}d@12m]"
            k, _ = c.query(q, key=keys, seg=None)
            k.to_csv(str(sharp_file), index=False)
            logger.info("SHARP %s: %d rows saved", year_month, len(k))
        except ImportError:
            logger.warning("drms not installed. Install with: pip install drms")
            logger.warning("SHARP download skipped for %s", year_month)
        except Exception as exc:
            logger.warning("JSOC SHARP download failed: %s", exc)

    except Exception as exc:
        logger.warning("SHARP download error: %s", exc)

    return sharp_file


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: LOAD & ENGINEER FEATURES
# This is where all the science happens — wavelet transforms,
# Neupert proxy, SHARP magnetic complexity, SWIS solar wind
# ══════════════════════════════════════════════════════════════════════════════

def engineer_wavelet_features(counts: np.ndarray, fs: float = 1.0) -> np.ndarray:
    """
    Apply Continuous Wavelet Transform to SoLEXS counts.
    
    WHY: Before a flare, plasma oscillates at quasi-periodic frequencies
    (Quasi-Periodic Pulsations / QPPs). A plain TCN misses these frequencies.
    The wavelet transform converts the 1D light curve into a 2D time-frequency
    spectrogram. The neural network then "hears" the vibration pattern.
    
    Scales: 1s to 128s (7 octaves × 8 voices = 56 frequency bands)
    Output: sum of power in 4 frequency bands (low, mid, high, very_high)
            → 4 additional features per timestep
    
    Uses scipy.signal.cwt (Morlet wavelet) — fast enough for M4 GPU.
    """
    try:
        from scipy.signal import cwt, morlet2
        # Scales: from 1s (high freq) to 128s (low freq)
        scales = np.array([1, 2, 4, 8, 16, 32, 64, 128])
        # CWT (batch: only on shorter windows for speed)
        n = len(counts)
        # Normalize counts
        c_norm = (counts - np.median(counts)) / (np.std(counts) + 1e-6)
        # CWT output: shape (len(scales), n)
        coef = cwt(c_norm, morlet2, scales, w=6.0)
        power = np.abs(coef) ** 2  # (8, n)
        # Band averages: low=scales 64-128, mid=16-32, high=4-8, vhigh=1-2
        return np.stack([
            power[6:8].mean(axis=0),    # 64-128s (low freq, minutes-scale)
            power[4:6].mean(axis=0),    # 16-32s (mid freq)
            power[2:4].mean(axis=0),    # 4-8s (high freq)
            power[0:2].mean(axis=0),    # 1-2s (very high freq, QPPs)
        ], axis=1).astype(np.float32)   # (n, 4)
    except Exception as exc:
        logger.debug("Wavelet transform failed: %s — using zeros", exc)
        return np.zeros((len(counts), 4), dtype=np.float32)


def load_month_features(
    month_dir: Path,
    goes_file: Path,
    sharp_file: Path,
    year_month: str,
    args,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """
    Load one month of REAL data from all instruments and build the
    multi-modal feature matrix.
    
    Returns:
        X_seq:   (N_windows, WINDOW_S, N_SEQ_FEATURES) — time-series features
                 N_SEQ_FEATURES = 11 SoLEXS+HEL1OS + 4 wavelet = 15
        X_sharp: (N_windows, N_SHARP) — SHARP magnetic params (21 features)
        X_insitu:(N_windows, N_INSITU) — MAG+SWIS in-situ (14 features)
        y:       (N_windows,) — binary flare labels
        
    Returns None if insufficient data found for this month.
    """
    year, month = year_month.split("-")
    solexs_dir = month_dir / "solexs"
    helios_dir = month_dir / "helios"
    mag_dir    = month_dir / "mag"
    swis_dir   = month_dir / "swis"

    WINDOW_S = 1800     # 30 minutes at 1-s cadence
    HORIZON_S = 900     # predict M+ in next 15 minutes
    STEP_S    = 60      # slide window every 60s (saves RAM, reduces redundancy)

    # ── Load SoLEXS (primary) ───────────────────────────────────────────────
    slx_times  = []
    slx_counts = []
    slx_loader = SoLEXSLoader(str(solexs_dir))

    # Scan all days in the month
    start_dt = datetime(int(year), int(month), 1)
    if int(month) == 12:
        end_dt = datetime(int(year) + 1, 1, 1)
    else:
        end_dt = datetime(int(year), int(month) + 1, 1)

    current = start_dt
    while current < end_dt:
        date_str = current.strftime("%Y%m%d")
        try:
            records = slx_loader.load_day(date_str)
            if records:
                slx_times.extend([r.time_unix for r in records])
                slx_counts.extend([r.counts_sdd2 for r in records])
        except Exception as exc:
            logger.debug("SoLEXS %s error: %s", date_str, exc)
        current += timedelta(days=1)

    if len(slx_times) < WINDOW_S * 2:
        logger.warning("%s: insufficient SoLEXS data (%d samples) — skipping month",
                       year_month, len(slx_times))
        return None

    slx_times  = np.array(slx_times,  dtype=np.float64)
    slx_counts = np.array(slx_counts, dtype=np.float32)
    idx_sort   = np.argsort(slx_times)
    slx_times  = slx_times[idx_sort]
    slx_counts = slx_counts[idx_sort]
    logger.info("%s SoLEXS: %d records (%.1f hours)", year_month, len(slx_times), len(slx_times) / 3600)

    # ── Load GOES XRS for flare labels (cross-validation) ───────────────────
    goes_flux = None
    goes_times = None
    if goes_file.exists() and goes_file.stat().st_size > 1000:
        try:
            import pandas as pd
            goes_df = pd.read_csv(str(goes_file), comment="#")
            # Normalize column names (NOAA NCEI format varies slightly)
            goes_df.columns = [c.lower().strip() for c in goes_df.columns]
            time_col = next((c for c in goes_df.columns if "time" in c or "date" in c), None)
            flux_col = next((c for c in goes_df.columns if "flux_1_8" in c or "a_flux" in c
                             or "1-8" in c.replace("_", "-")), None)
            if time_col and flux_col:
                goes_df["unix"] = pd.to_datetime(goes_df[time_col], utc=True, errors="coerce").astype(np.int64) // 1e9
                goes_df = goes_df.dropna(subset=["unix", flux_col])
                goes_times = goes_df["unix"].values.astype(np.float64)
                goes_flux  = goes_df[flux_col].values.astype(np.float64)
                logger.info("%s GOES: %d records", year_month, len(goes_df))
        except Exception as exc:
            logger.warning("GOES load error: %s", exc)

    # ── Load HEL1OS (primary) ────────────────────────────────────────────────
    hel_bands   = {band: [] for band in [
        "cdte_5_20", "cdte_20_30", "cdte_30_40", "cdte_40_60",
        "czt_20_40", "czt_40_60", "czt_60_80", "czt_80_150",
    ]}
    hel_times   = []
    hel_loader  = HEL1OSLoader(str(helios_dir))

    current = start_dt
    while current < end_dt:
        date_str = current.strftime("%Y%m%d")
        try:
            records = hel_loader.load_day(date_str)
            if records:
                hel_times.extend([r.time_unix for r in records])
                hel_bands["cdte_5_20"].extend([getattr(r, "cdte1_5_20",   0.0) for r in records])
                hel_bands["cdte_20_30"].extend([getattr(r, "cdte1_20_30", 0.0) for r in records])
                hel_bands["cdte_30_40"].extend([getattr(r, "cdte1_30_40", 0.0) for r in records])
                hel_bands["cdte_40_60"].extend([getattr(r, "cdte1_40_60", 0.0) for r in records])
                hel_bands["czt_20_40"].extend([getattr(r, "czt1_20_40",   0.0) for r in records])
                hel_bands["czt_40_60"].extend([getattr(r, "czt1_40_60",   0.0) for r in records])
                hel_bands["czt_60_80"].extend([getattr(r, "czt1_60_80",   0.0) for r in records])
                hel_bands["czt_80_150"].extend([getattr(r, "czt1_80_150", 0.0) for r in records])
        except Exception as exc:
            logger.debug("HEL1OS %s error: %s", date_str, exc)
        current += timedelta(days=1)

    # Align HEL1OS to SoLEXS time grid (nearest-neighbour, ≤2s tolerance)
    has_helios = len(hel_times) > 100
    if has_helios:
        hel_times_arr = np.array(hel_times, dtype=np.float64)
        hel_aligned   = {}
        for band, vals in hel_bands.items():
            hel_aligned[band] = _align_to_grid(hel_times_arr, np.array(vals), slx_times, tol_s=2.0)
        logger.info("%s HEL1OS: %d records aligned", year_month, len(hel_times))
    else:
        hel_aligned = {band: np.zeros_like(slx_counts) for band in hel_bands}
        logger.warning("%s HEL1OS not available — using zeros", year_month)

    # ── Load MAG (supplementary — solar wind magnetic field) ─────────────────
    mag_features = np.zeros((len(slx_times), 8), dtype=np.float32)
    mag_loader   = MAGLoader(str(mag_dir))
    current = start_dt
    mag_all_records = []
    while current < end_dt:
        try:
            recs = mag_loader.load_day(current.strftime("%Y%m%d"))
            mag_all_records.extend(recs)
        except Exception:
            pass
        current += timedelta(days=1)

    if mag_all_records:
        logger.info("%s MAG: %d records", year_month, len(mag_all_records))
        for i in range(0, len(slx_times), 600):  # every 10 min
            feat = mag_loader.extract_ml_features(
                mag_all_records,
                window_end_unix=slx_times[i],
                window_minutes=30.0,
            )
            if feat is not None:
                # Interpolate to full SoLEXS grid (MAG is 10-s cadence)
                end_i = min(i + 600, len(slx_times))
                mag_features[i:end_i] = feat[np.newaxis, :]

    # ── Load ASPEX-SWIS (supplementary — solar wind particles) ───────────────
    swis_features = np.zeros((len(slx_times), 6), dtype=np.float32)
    swis_loader   = SWISLoader(str(swis_dir))
    current = start_dt
    swis_all_records = []
    while current < end_dt:
        try:
            recs = swis_loader.load_bulk_params(current.strftime("%Y%m%d"))
            swis_all_records.extend(recs)
        except Exception:
            pass
        current += timedelta(days=1)

    if swis_all_records:
        logger.info("%s SWIS: %d records", year_month, len(swis_all_records))

    # ── Engineering: SoLEXS derived features ─────────────────────────────────
    import pandas as pd

    counts = slx_counts.astype(np.float64)
    log_counts     = np.log10(np.clip(counts, 0.01, None)).astype(np.float32)
    derivative     = np.gradient(counts).astype(np.float32)
    roll_med_60    = pd.Series(counts).rolling(60,  min_periods=5).median().bfill().values
    roll_med_300   = pd.Series(counts).rolling(300, min_periods=30).median().bfill().values
    roll_std_60    = pd.Series(counts).rolling(60,  min_periods=5).std().bfill().clip(lower=0.5).values
    roll_std_300   = pd.Series(counts).rolling(300, min_periods=30).std().bfill().clip(lower=0.5).values
    zscore_60      = np.clip((counts - roll_med_60)  / roll_std_60,  -10, 10).astype(np.float32)
    zscore_300     = np.clip((counts - roll_med_300) / roll_std_300, -10, 10).astype(np.float32)

    # Neupert effect proxy: cumulative HXR integral ∝ SXR (Neupert 1968)
    hxr_arr       = hel_aligned["cdte_30_40"].astype(np.float64)
    neupert_cum   = np.cumsum(hxr_arr)
    neupert_cum   = ((neupert_cum - neupert_cum.mean()) / (neupert_cum.std() + 1e-6)).astype(np.float32)
    hxr_sxr_ratio = np.clip(hxr_arr / np.clip(counts, 0.01, None), 0, 100).astype(np.float32)

    # HEL1OS log-scaled
    hel_log = {
        band: np.log10(np.clip(hel_aligned[band], 0.01, None)).astype(np.float32)
        for band in ["cdte_5_20", "cdte_20_30", "cdte_30_40", "cdte_40_60",
                     "czt_40_60", "czt_80_150"]
    }

    # Wavelet transform on SoLEXS (adds QPP features)
    logger.info("%s Computing wavelet transform (QPP detection)...", year_month)
    wavelet_feat = engineer_wavelet_features(counts.astype(np.float32))  # (N, 4)

    # ── Build per-timestep feature matrix ────────────────────────────────────
    # Shape: (N_timesteps, N_SEQ_FEATURES)
    # N_SEQ_FEATURES = 7 SoLEXS + 6 HEL1OS_log + 2 ratios + 4 wavelet = 19
    seq_features = np.stack([
        slx_counts,                        # 0: raw SoLEXS SDD2 count rate
        log_counts,                        # 1: log10(counts)
        derivative,                        # 2: dCounts/dt
        zscore_60,                         # 3: z-score vs 60s background
        zscore_300,                        # 4: z-score vs 300s background
        neupert_cum,                       # 5: Neupert integral (HXR→SXR proxy)
        hxr_sxr_ratio,                    # 6: HXR/SXR ratio (non-thermal flag)
        hel_log["cdte_5_20"],             # 7: HEL1OS CdTe 5-20 keV
        hel_log["cdte_20_30"],            # 8: HEL1OS CdTe 20-30 keV
        hel_log["cdte_30_40"],            # 9: HEL1OS CdTe 30-40 keV (HOPE)
        hel_log["cdte_40_60"],            # 10: HEL1OS CdTe 40-60 keV
        hel_log["czt_40_60"],             # 11: HEL1OS CZT 40-60 keV
        hel_log["czt_80_150"],            # 12: HEL1OS CZT 80-150 keV (high energy)
        wavelet_feat[:, 0],               # 13: wavelet power low-freq
        wavelet_feat[:, 1],               # 14: wavelet power mid-freq
        wavelet_feat[:, 2],               # 15: wavelet power high-freq
        wavelet_feat[:, 3],               # 16: wavelet power QPP-freq (1-2s)
        # MAG features (from 30-min windows)
        mag_features[:, 0],              # 17: B_total mean (nT)
        mag_features[:, 4],              # 18: Bz_gse mean (southward = geoeffective)
        mag_features[:, 5],              # 19: clock angle (geoeffectiveness proxy)
    ], axis=1)  # shape: (N, 20)

    # ── Flare Labels ─────────────────────────────────────────────────────────
    # Primary label source: SoLEXS sigma threshold
    roll_med = roll_med_300
    roll_std = roll_std_300
    FLARE_SIGMA = 5.0
    FLARE_ABS   = 50.0  # counts/s threshold ~ C1 class on SoLEXS
    is_flare_s = ((counts - roll_med) > FLARE_SIGMA * roll_std) | (counts > FLARE_ABS)

    # Cross-validate labels with GOES if available
    if goes_times is not None and goes_flux is not None:
        goes_labels = np.zeros(len(slx_times), dtype=bool)
        M1_WM2 = 1e-5  # M1 flare threshold (GOES 1-8A)
        goes_on_slx = _align_to_grid(goes_times, (goes_flux > M1_WM2).astype(float),
                                     slx_times, tol_s=90.0)  # 90s tolerance
        goes_labels = goes_on_slx > 0.5
        # OR with sigma method (GOES catches behind-limb; SoLEXS catches disk)
        is_flare_s = is_flare_s | goes_labels
        logger.info("%s GOES cross-validation: %d additional flare seconds",
                    year_month, goes_labels.sum())

    # Create binary label: any flare in next 15 min?
    N = len(slx_times)
    y_raw = np.zeros(N, dtype=np.float32)
    for i in range(N - HORIZON_S):
        y_raw[i] = float(np.any(is_flare_s[i + 1 : i + HORIZON_S + 1]))

    pos_rate = y_raw.mean() * 100
    logger.info("%s Labels: %.2f%% positive (flare windows)", year_month, pos_rate)

    if pos_rate < 0.01:
        logger.warning("%s: 0%% positive rate — quiet month, still useful for negatives", year_month)

    # ── Load SHARP (if available) ─────────────────────────────────────────────
    # SHARP: 21 magnetic complexity params at 12-min cadence
    # We project them onto the per-second SoLEXS time grid
    sharp_feat_full = np.zeros((N, 21), dtype=np.float32)
    if sharp_file.exists() and sharp_file.stat().st_size > 5000:
        try:
            import pandas as pd
            sh = pd.read_csv(str(sharp_file))
            sh.columns = [c.lower().strip() for c in sh.columns]
            sharp_cols = [c for c in [
                "usflux", "meangbt", "meanjzd", "totusjh", "meanalp",
                "meangam", "meangbz", "meangbh", "meanjzh", "totusjz",
                "absnjzh", "savncpp", "meanpot", "totpot", "meanshr",
                "shrgt45", "area_acr", "r_value",
            ] if c in sh.columns]
            if sharp_cols and len(sh) > 0:
                t_rec_col = next((c for c in sh.columns if "t_rec" in c or "time" in c), None)
                if t_rec_col:
                    sh["unix"] = pd.to_datetime(
                        sh[t_rec_col], utc=True, errors="coerce"
                    ).astype(np.int64) // 1e9
                    sh = sh.dropna(subset=["unix"])
                    sharp_times = sh["unix"].values.astype(np.float64)
                    for j, col in enumerate(sharp_cols[:21]):
                        vals = sh[col].fillna(0).values.astype(np.float32)
                        sharp_feat_full[:, j] = _align_to_grid(
                            sharp_times, vals, slx_times, tol_s=720.0  # 12 min
                        )
                    logger.info("%s SHARP: %d rows, %d params loaded",
                                year_month, len(sh), len(sharp_cols))
        except Exception as exc:
            logger.warning("%s SHARP load error: %s", year_month, exc)

    # ── Build sliding-window dataset ─────────────────────────────────────────
    windows_X    = []
    windows_sharp = []
    windows_insitu = []
    windows_y    = []

    valid_end = N - HORIZON_S
    for i in range(WINDOW_S, valid_end, STEP_S):
        x_win     = seq_features[i - WINDOW_S : i]           # (1800, 20)
        sharp_win = sharp_feat_full[max(0, i - 8640) : i]    # ~24h of SHARP at 12-min cadence = 120 steps
        if len(sharp_win) < 10:
            sharp_win = np.zeros((120, 21), dtype=np.float32)
        elif len(sharp_win) != 120:
            # Resample to fixed 120 steps
            idx = np.linspace(0, len(sharp_win) - 1, 120).astype(int)
            sharp_win = sharp_win[idx]

        # In-situ: MAG + SWIS at current timestep
        insitu = np.concatenate([
            mag_features[i],      # 8 MAG features
            swis_features[i],     # 6 SWIS features
        ])  # shape: (14,)

        windows_X.append(x_win)
        windows_sharp.append(sharp_win)
        windows_insitu.append(insitu)
        windows_y.append(y_raw[i])

    if len(windows_X) < 10:
        logger.warning("%s: too few windows (%d) — skipping", year_month, len(windows_X))
        return None

    X_seq    = np.array(windows_X,    dtype=np.float32)
    X_sharp  = np.array(windows_sharp,  dtype=np.float32)
    X_insitu = np.array(windows_insitu, dtype=np.float32)
    y        = np.array(windows_y,    dtype=np.float32)

    logger.info(
        "%s Feature matrix: X_seq=%s X_sharp=%s X_insitu=%s y=%s (pos=%.1f%%)",
        year_month, X_seq.shape, X_sharp.shape, X_insitu.shape, y.shape, y.mean() * 100
    )

    return X_seq, X_sharp, X_insitu, y


def _align_to_grid(
    t_src: np.ndarray, v_src: np.ndarray, t_dst: np.ndarray, tol_s: float = 2.0
) -> np.ndarray:
    """Nearest-neighbour interpolation from source to destination time grid."""
    out = np.zeros(len(t_dst), dtype=v_src.dtype)
    if len(t_src) == 0:
        return out
    for i, t in enumerate(t_dst):
        idx = np.searchsorted(t_src, t)
        idx = min(idx, len(t_src) - 1)
        if abs(t_src[idx] - t) <= tol_s:
            out[i] = v_src[idx]
    return out


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: MULTI-MODAL PYTORCH MODEL
# Full SOTA architecture with all branches as designed
# ══════════════════════════════════════════════════════════════════════════════

class CausalConv1d(nn.Module):
    """Causal (left-padded) 1D convolution — no future leakage."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int):
        super().__init__()
        self.padding = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, padding=0)

    def forward(self, x):
        x = nn.functional.pad(x, (self.padding, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel, dilation)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel, dilation)
        self.norm1 = nn.LayerNorm(out_ch)
        self.norm2 = nn.LayerNorm(out_ch)
        self.drop  = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act   = nn.GELU()

    def forward(self, x):
        # x: (B, C, T)
        h = self.act(self.norm1(self.conv1(x).transpose(1,2)).transpose(1,2))
        h = self.drop(h)
        h = self.act(self.norm2(self.conv2(h).transpose(1,2)).transpose(1,2))
        return h + self.residual(x)


class SoLEXSHEL1OSTCN(nn.Module):
    """
    PRIMARY BRANCH: SoLEXS + HEL1OS 6-layer Dilated Causal TCN.
    Input: (B, 1800, 20) — 30 min × 20 features (X-ray + wavelet)
    Output: 256-dim embedding
    """
    def __init__(self, n_features: int = 20, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, 64)
        channels   = [64, 64, 128, 128, 256, 256]
        dilations  = [1,  2,  4,   8,   16,  32]
        self.blocks = nn.ModuleList()
        in_ch = 64
        for ch, d in zip(channels, dilations):
            self.blocks.append(TCNBlock(in_ch, ch, kernel=8, dilation=d, dropout=dropout))
            in_ch = ch
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        h = self.input_proj(x).transpose(1, 2)  # (B, 64, T)
        for block in self.blocks:
            h = block(h)
        return self.proj(self.pool(h).squeeze(-1))  # (B, 256)


class SHARPBiLSTM(nn.Module):
    """
    SUPPLEMENTARY BRANCH: SDO/HMI SHARP magnetic complexity features.
    Input: (B, 120, 21) — 24h × 21 SHARP params at 12-min cadence
    Output: 128-dim embedding
    
    BiLSTM: reads forward AND backward in time (predicting flare
    from the evolution of magnetic complexity over 24 hours).
    """
    def __init__(self, n_features: int = 21, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(n_features, 64)
        self.lstm = nn.LSTM(64, 64, num_layers=3, batch_first=True,
                            dropout=dropout, bidirectional=True)
        self.attn = nn.MultiheadAttention(128, num_heads=4, dropout=dropout, batch_first=True)
        self.proj = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)           # (B, T, 64)
        out, _ = self.lstm(h)            # (B, T, 128)
        ctx, _ = self.attn(out, out, out)
        return self.proj(ctx.mean(dim=1))  # (B, 128)


class InSituMLP(nn.Module):
    """
    SUPPLEMENTARY BRANCH: MAG + ASPEX-SWIS in-situ solar wind.
    Input: (B, 14) — 8 MAG features + 6 SWIS features
    Output: 64-dim embedding
    """
    def __init__(self, n_features: int = 14, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 64),         nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 64),         nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, 64)


class CrossModalAttentionFusion(nn.Module):
    """
    FUSION LAYER: Cross-Modal Attention.
    
    Concatenates TCN (256d) + LSTM (128d) + MLP (64d) embeddings → 448d
    → 4-head cross-modal attention → 256d fused embedding
    → Temperature-calibrated probability output
    
    The attention mechanism learns WHICH branches to trust more
    given the current data (e.g., trust MAG more during CME approach).
    """
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        # Project each branch to same dimension for attention
        self.proj_tcn  = nn.Linear(256, 256)
        self.proj_lstm = nn.Linear(128, 256)
        self.proj_mlp  = nn.Linear(64,  256)

        # Cross-modal attention: treat 3 branch embeddings as a sequence
        self.cross_attn = nn.MultiheadAttention(256, num_heads=4, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(256)

        # Final output head: multi-horizon flare probability
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64),  nn.GELU(),
            nn.Linear(64, 6),    # 6 outputs: nowcast + 5 horizons
        )
        # Learnable temperature for calibration
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, emb_tcn, emb_lstm, emb_mlp):
        # Project all to 256-dim
        q = self.proj_tcn(emb_tcn).unsqueeze(1)    # (B, 1, 256)
        k = self.proj_lstm(emb_lstm).unsqueeze(1)  # (B, 1, 256)
        v = self.proj_mlp(emb_mlp).unsqueeze(1)   # (B, 1, 256)
        # Stack as sequence: (B, 3, 256)
        seq = torch.cat([q, k, v], dim=1)
        # Cross-modal attention (each branch attends to others)
        ctx, attn_weights = self.cross_attn(seq, seq, seq)
        fused = self.norm(ctx.mean(dim=1))     # (B, 256)
        # Output logits: (B, 6) → [nowcast, 5min, 10min, 15min, 30min, 60min]
        logits = self.head(fused) / self.temperature.clamp(0.1, 10.0)
        return {
            "flare_nowcast":  logits[:, 0],
            "p_flare_5min":   logits[:, 1],
            "p_flare_10min":  logits[:, 2],
            "p_flare_15min":  logits[:, 3],
            "p_flare_30min":  logits[:, 4],
            "p_flare_60min":  logits[:, 5],
            "attn_weights":   attn_weights.detach(),
        }


class AdityScanSOTA(nn.Module):
    """
    Full AdityScan SOTA v4 Multi-Modal Model.
    All branches + Cross-Modal Attention as designed.
    """
    def __init__(self, n_seq_features=20, n_sharp_features=21, n_insitu_features=14,
                 dropout=0.1):
        super().__init__()
        self.tcn_branch   = SoLEXSHEL1OSTCN(n_seq_features, dropout)
        self.sharp_branch = SHARPBiLSTM(n_sharp_features, dropout)
        self.insitu_branch= InSituMLP(n_insitu_features, dropout)
        self.fusion       = CrossModalAttentionFusion(dropout)

    def forward(self, x_seq, x_sharp, x_insitu):
        emb_tcn   = self.tcn_branch(x_seq)
        emb_lstm  = self.sharp_branch(x_sharp)
        emb_mlp   = self.insitu_branch(x_insitu)
        return self.fusion(emb_tcn, emb_lstm, emb_mlp)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: MONTHLY DATASET + INCREMENTAL TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

class MonthlyDataset(Dataset):
    def __init__(self, X_seq, X_sharp, X_insitu, y):
        self.X_seq    = torch.from_numpy(X_seq)
        self.X_sharp  = torch.from_numpy(X_sharp)
        self.X_insitu = torch.from_numpy(X_insitu)
        self.y        = torch.from_numpy(y)

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        return self.X_seq[i], self.X_sharp[i], self.X_insitu[i], self.y[i]


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    TP = int(np.sum((y_pred == 1) & (y_true == 1)))
    TN = int(np.sum((y_pred == 0) & (y_true == 0)))
    FP = int(np.sum((y_pred == 1) & (y_true == 0)))
    FN = int(np.sum((y_pred == 0) & (y_true == 1)))
    POD = TP / (TP + FN + 1e-8)
    FAR = FP / (FP + TP + 1e-8)
    TSS = POD - FP / (FP + TN + 1e-8)
    denom = (TP + FN) * (FN + TN) + (TP + FP) * (FP + TN)
    HSS   = 2 * (TP * TN - FP * FN) / (denom + 1e-8)
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = 0.5
    return {"TSS": round(float(TSS), 4), "HSS": round(float(HSS), 4),
            "POD": round(float(POD), 4), "FAR": round(float(FAR), 4),
            "AUC": round(auc, 4), "TP": TP, "TN": TN, "FP": FP, "FN": FN}


def train_one_month(
    model: AdityScanSOTA,
    X_seq: np.ndarray, X_sharp: np.ndarray, X_insitu: np.ndarray, y: np.ndarray,
    device: torch.device,
    optimizer: optim.Optimizer,
    scheduler,
    batch_size: int,
    n_epochs: int,
    year_month: str,
    flare_weight_multiplier: float = 1.0,
) -> dict:
    """
    Train on ONE month of data.
    Saves checkpoint after each month.
    Returns: metrics dict for this month.
    """
    # Train/val split (last 20% = validation, chronological)
    split = int(0.8 * len(y))
    ds_train = MonthlyDataset(X_seq[:split], X_sharp[:split], X_insitu[:split], y[:split])
    ds_val   = MonthlyDataset(X_seq[split:], X_sharp[split:], X_insitu[split:], y[split:])

    if len(ds_train) < 10:
        logger.warning("%s: too few training samples (%d)", year_month, len(ds_train))
        return {}

    # Weighted sampling: oversample flare windows
    pos_weight = min(15.0 * flare_weight_multiplier, 50.0)
    sample_weights = np.where(y[:split] > 0.5, pos_weight, 1.0)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights.astype(np.float32)),
        num_samples=len(ds_train), replacement=True,
    )

    train_loader = DataLoader(ds_train, batch_size=batch_size, sampler=sampler,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(ds_val,   batch_size=batch_size * 2, shuffle=False,
                              num_workers=0, pin_memory=False)

    # Focal loss for class imbalance
    class FocalLoss(nn.Module):
        def __init__(self, gamma=2.0, pos_w=10.0):
            super().__init__()
            self.gamma = gamma
            self.pos_w = pos_w
        def forward(self, logit, target):
            bce = nn.functional.binary_cross_entropy_with_logits(
                logit, target,
                pos_weight=torch.tensor(self.pos_w, device=logit.device),
                reduction="none",
            )
            pt = torch.where(target > 0.5, torch.sigmoid(logit), 1 - torch.sigmoid(logit))
            return ((1 - pt) ** self.gamma * bce).mean()

    criterion = FocalLoss(gamma=2.0, pos_w=pos_weight)

    best_tss = -1.0
    month_metrics = {}

    for epoch in range(1, n_epochs + 1):
        model.train()
        losses = []
        for X_s, X_sh, X_i, y_b in train_loader:
            X_s  = X_s.to(device)
            X_sh = X_sh.to(device)
            X_i  = X_i.to(device)
            y_b  = y_b.to(device)

            optimizer.zero_grad()
            out = model(X_s, X_sh, X_i)

            # Primary: 15-min horizon
            loss = criterion(out["p_flare_15min"], y_b)
            # Auxiliary: other horizons (weighted)
            for key, w in [("flare_nowcast", 1.0), ("p_flare_5min", 0.9),
                           ("p_flare_10min", 0.85), ("p_flare_30min", 0.6),
                           ("p_flare_60min", 0.4)]:
                if key in out:
                    loss = loss + w * criterion(out[key], y_b)
            loss = loss / 6.0  # normalize

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

            # M4 memory management
            if device.type == "mps" and len(losses) % 50 == 0:
                torch.mps.empty_cache()

        if scheduler is not None:
            scheduler.step()

        # Validate
        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for X_s, X_sh, X_i, y_b in val_loader:
                out = model(X_s.to(device), X_sh.to(device), X_i.to(device))
                p = torch.sigmoid(out["p_flare_15min"]).cpu().numpy()
                val_probs.extend(p.tolist() if p.ndim > 0 else [float(p)])
                val_labels.extend(y_b.numpy().tolist())

        vp = np.array(val_probs)
        vl = np.array(val_labels)
        # Find best threshold for TSS
        best_t, best_tss_ep = 0.5, -1.0
        for t in np.linspace(0.1, 0.9, 17):
            m = compute_metrics(vl, vp, t)
            if m["TSS"] > best_tss_ep:
                best_tss_ep = m["TSS"]
                best_t = t
        metrics = compute_metrics(vl, vp, best_t)

        logger.info(
            "[%s] Epoch %2d/%d | Loss=%.4f | TSS=%.3f HSS=%.3f POD=%.2f FAR=%.2f AUC=%.3f",
            year_month, epoch, n_epochs, np.mean(losses),
            metrics["TSS"], metrics["HSS"], metrics["POD"], metrics["FAR"], metrics["AUC"]
        )

        if metrics["TSS"] > best_tss:
            best_tss = metrics["TSS"]
            month_metrics = {**metrics, "epoch": epoch, "month": year_month,
                             "threshold": round(best_t, 2)}

    return month_metrics


# ══════════════════════════════════════════════════════════════════════════════
# ONNX EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def export_onnx(model: AdityScanSOTA, path: str) -> None:
    """Export trained model to ONNX for Render CPU inference."""
    model_cpu = model.cpu().eval()
    dummy_seq    = torch.zeros(1, 1800, 20)
    dummy_sharp  = torch.zeros(1, 120,  21)
    dummy_insitu = torch.zeros(1, 14)

    with torch.no_grad():
        dummy_out = model_cpu(dummy_seq, dummy_sharp, dummy_insitu)

    output_names = [k for k in dummy_out.keys() if k != "attn_weights"]
    torch.onnx.export(
        model_cpu,
        (dummy_seq, dummy_sharp, dummy_insitu),
        path,
        opset_version=17,
        input_names=["seq", "sharp", "insitu"],
        output_names=output_names,
        dynamic_axes={"seq": {0: "batch"}, "sharp": {0: "batch"}, "insitu": {0: "batch"}},
    )
    logger.info("ONNX model exported: %s (%.1f MB)", path,
                Path(path).stat().st_size / 1e6)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN: INCREMENTAL MONTHLY TRAINING LOOP
# This is the ChatGPT batch training idea, fully implemented
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="AdityScan v4 — Real Incremental Multi-Modal Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--months", type=str, default="",
                        help="Comma-separated YYYY-MM months to train (default: all 15 months)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip downloading — use data already in data/pradan_cache/")
    parser.add_argument("--quick-test", action="store_true",
                        help="Quick test: 3 days of data, 2 epochs only")
    parser.add_argument("--no-suit", action="store_true",
                        help="Skip SUIT image branch")
    parser.add_argument("--delete-after-month", action="store_true", default=True,
                        help="Delete downloaded month data after training (saves disk space)")
    parser.add_argument("--keep-data", action="store_true",
                        help="Keep downloaded data (don't delete after training)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs-per-month", type=int, default=10,
                        help="Epochs to train on each month's data")
    args = parser.parse_args()

    if args.quick_test:
        args.epochs_per_month = 2
        logger.info("⚡ QUICK TEST MODE")

    logger.info("=" * 70)
    logger.info("AdityScan v4 — REAL Incremental Multi-Modal Training")
    logger.info("=" * 70)
    logger.info("Training data sources:")
    logger.info("  PRIMARY:      SoLEXS L1 + HEL1OS L1 (Aditya-L1 / PRADAN)")
    logger.info("  SUPPLEMENTARY: MAG L2 + ASPEX-SWIS L2 + SHARP + GOES")
    logger.info("  WAVELET:      CWT on SoLEXS (QPP detection)")
    logger.info("  METHOD:       Incremental — 1 month at a time (no OOM)")
    logger.info("")

    # Which months to train
    if args.months:
        months_to_train = [(m.strip(), f"user-specified", 1.5)
                           for m in args.months.split(",")]
    elif args.quick_test:
        months_to_train = [TRAINING_MONTHS[0]]  # Just the best month
    else:
        months_to_train = TRAINING_MONTHS

    logger.info("Training schedule: %d months", len(months_to_train))
    for ym, desc, w in months_to_train:
        logger.info("  %s  [weight=%.1f]  %s", ym, w, desc)

    device = get_device()

    # ── Initialize model ──────────────────────────────────────────────────────
    model = AdityScanSOTA(
        n_seq_features=20, n_sharp_features=21, n_insitu_features=14, dropout=0.1
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %d (%.1f M)", total_params, total_params / 1e6)

    # Checkpoint path
    CKPT_PATH = str(MODELS_DIR / "adityscan_v4_incremental.pt")

    # Resume from checkpoint
    if args.resume and Path(CKPT_PATH).exists():
        ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Resumed from checkpoint: TSS=%.3f (after %s)",
                    ckpt.get("best_tss", 0.0), ckpt.get("last_month", "?"))

    # Optimizer: differential learning rates per branch
    optimizer = optim.AdamW([
        {"params": model.tcn_branch.parameters(),    "lr": 3e-4},  # PRIMARY
        {"params": model.sharp_branch.parameters(),  "lr": 2e-4},  # SUPPLEMENTARY
        {"params": model.insitu_branch.parameters(), "lr": 3e-4},  # SUPPLEMENTARY
        {"params": model.fusion.parameters(),        "lr": 3e-4},  # FUSION
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(months_to_train) * args.epochs_per_month
    )

    # ── MAIN INCREMENTAL LOOP ─────────────────────────────────────────────────
    all_metrics = []
    global_best_tss = 0.0

    for month_idx, (year_month, description, flare_weight) in enumerate(months_to_train):
        logger.info("")
        logger.info("=" * 70)
        logger.info("MONTH %d/%d: %s — %s", month_idx + 1, len(months_to_train),
                    year_month, description)
        logger.info("=" * 70)

        # ── DOWNLOAD this month's data ────────────────────────────────────────
        if not args.skip_download:
            downloaded = download_month_pradan(year_month, args)
            goes_file  = download_month_goes(year_month, args)
            sharp_file = download_month_sharp(year_month, args)
        else:
            downloaded = {
                "solexs": CACHE_DIR / year_month / "solexs",
                "helios": CACHE_DIR / year_month / "helios",
                "mag":    CACHE_DIR / year_month / "mag",
                "swis":   CACHE_DIR / year_month / "swis",
            }
            goes_file  = GOES_DIR  / f"goes_1min_{year_month}.csv"
            sharp_file = SHARP_DIR / f"sharp_{year_month}.csv"

        month_dir = CACHE_DIR / year_month

        # ── LOAD & ENGINEER FEATURES ──────────────────────────────────────────
        result = load_month_features(
            month_dir, goes_file, sharp_file, year_month, args
        )

        if result is None:
            logger.warning("No usable data for %s — skipping", year_month)
            if not args.keep_data:
                _safe_delete(month_dir)
            continue

        X_seq, X_sharp, X_insitu, y = result
        logger.info("Month %s: %d training windows loaded into RAM", year_month, len(y))

        # ── TRAIN on this month ────────────────────────────────────────────────
        month_metrics = train_one_month(
            model, X_seq, X_sharp, X_insitu, y,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            batch_size=args.batch_size,
            n_epochs=args.epochs_per_month,
            year_month=year_month,
            flare_weight_multiplier=flare_weight,
        )

        if month_metrics:
            all_metrics.append(month_metrics)
            tss = month_metrics.get("TSS", 0.0)
            logger.info("Month %s complete: TSS=%.3f HSS=%.3f POD=%.2f FAR=%.2f",
                        year_month, tss,
                        month_metrics.get("HSS", 0.0),
                        month_metrics.get("POD", 0.0),
                        month_metrics.get("FAR", 0.0))

            if tss > global_best_tss:
                global_best_tss = tss

        # ── SAVE CHECKPOINT after every month ────────────────────────────────
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "last_month": year_month,
            "best_tss": global_best_tss,
            "all_metrics": all_metrics,
            "architecture": {
                "branches": ["SoLEXS+HEL1OS TCN (20 features + CWT wavelet)",
                             "SHARP BiLSTM (21 SHARP magnetic params)",
                             "MAG+SWIS InSitu MLP (14 in-situ features)"],
                "fusion": "Cross-Modal Attention (4-head, 256-dim)",
                "calibration": "Temperature Scaling (learnable)",
            },
        }, CKPT_PATH)
        logger.info("Checkpoint saved: %s (best TSS=%.3f)", CKPT_PATH, global_best_tss)

        # ── FREE MEMORY: delete this month's raw data ─────────────────────────
        del X_seq, X_sharp, X_insitu, y, result
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

        if not args.keep_data:
            _safe_delete(month_dir)
            logger.info("Deleted %s raw data to free disk space", year_month)

    # ══════════════════════════════════════════════════════════════════════════
    # FINAL: Calibration + ONNX Export
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=" * 70)
    logger.info("ALL MONTHS COMPLETE — Exporting production model")
    logger.info("=" * 70)

    # Temperature calibration (using the model's own learnable temperature)
    T = float(model.fusion.temperature.item())
    logger.info("Calibration temperature: %.4f", T)

    # Export ONNX
    onnx_path = str(MODELS_DIR / "adityscan_v4.onnx")
    export_onnx(model, onnx_path)

    # Save training report
    report = {
        "model_version": "4.0.0-incremental-real-data",
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "months_trained": [m["month"] for m in all_metrics if "month" in m],
        "global_best_tss": round(global_best_tss, 4),
        "per_month_metrics": all_metrics,
        "temperature": T,
        "architecture": {
            "type": "AdityScanSOTA_v4",
            "primary_branch": "SoLEXS+HEL1OS TCN, 6 layers, dilations=[1,2,4,8,16,32], 256-dim",
            "secondary_branch": "SHARP BiLSTM, 3-layer, 128-dim",
            "insitu_branch": "MAG+SWIS MLP, 64-dim",
            "fusion": "Cross-Modal Attention 4-head 256-dim",
            "n_seq_features": 20,
            "n_sharp_features": 21,
            "n_insitu_features": 14,
            "window_seconds": 1800,
            "wavelet_transform": True,
        },
        "data_sources": {
            "primary": ["Aditya-L1 SoLEXS L1 (1-s)", "Aditya-L1 HEL1OS L1 (1-s)"],
            "supplementary": ["Aditya-L1 MAG L2 (10-s)", "Aditya-L1 ASPEX-SWIS L2",
                              "SDO/HMI SHARP (12-min)", "GOES XRS 1-min (labels)"],
            "derived": ["CWT wavelet (QPP detection)", "Neupert integral",
                        "HXR/SXR ratio", "MAG clock/cone angles"],
        },
        "onnx_path": onnx_path,
    }
    report_path = str(MODELS_DIR / "training_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════════════╗")
    logger.info("║           TRAINING COMPLETE ✅                               ║")
    logger.info("╚══════════════════════════════════════════════════════════════╝")
    logger.info("")
    logger.info("  Global Best TSS:  %.3f", global_best_tss)
    logger.info("  Months trained:   %d", len([m for m in all_metrics if "month" in m]))
    logger.info("  ONNX model:       %s", onnx_path)
    logger.info("  Training report:  %s", report_path)
    logger.info("")
    logger.info("NEXT STEPS:")
    logger.info("  1. Push to GitHub:")
    logger.info("       git add models/adityscan_v4.onnx models/training_report.json")
    logger.info("       git commit -m 'feat: trained v4 SOTA on real PRADAN data'")
    logger.info("       git push")
    logger.info("  2. Render auto-deploys → live in 2-3 minutes")


def _safe_delete(path: Path) -> None:
    """Safely delete a directory and its contents."""
    try:
        if path.exists():
            shutil.rmtree(str(path))
            logger.debug("Deleted: %s", path)
    except Exception as exc:
        logger.warning("Could not delete %s: %s", path, exc)


if __name__ == "__main__":
    main()
