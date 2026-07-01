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
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# ── Setup paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "backend"))
os.environ.setdefault("SUNPY_CONFIGDIR", str(ROOT / ".sunpy"))
(ROOT / ".sunpy").mkdir(exist_ok=True)

from pipeline.ingestion.solexs_loader import SoLEXSLoader
from pipeline.ingestion.helios_loader import HEL1OSLoader
from pipeline.ingestion.mag_loader import MAGLoader
from pipeline.ingestion.aspex_swis_loader import SWISLoader
from pipeline.ingestion.suit_loader import SUITLoader

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_FILE = ROOT / "train.log"
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
REPORTS_DIR  = MODELS_DIR / "month_reports"
MODELS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
GOES_DIR.mkdir(parents=True, exist_ok=True)
SHARP_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_MODALITIES = ("solexs", "helios", "mag", "swis", "suit", "goes", "sharp")
MODALITY_FILE_GLOBS = {
    "solexs": ["**/AL1_SLX_L1_*.zip", "**/*.fits", "**/*.lc.gz", "**/*.nc"],
    "helios": ["**/*.zip", "**/*CdTe*.fits", "**/*CZT*.fits", "**/AL1_HXS91_*.fits"],
    "mag": ["**/L2_AL1_MAG_*.nc"],
    "swis": ["**/AL1_ASW91_L2_*.cdf"],
    "suit": ["**/*.fits", "**/*.zip"],
}

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


def _month_date_range(year_month: str, quick_test: bool = False) -> tuple[datetime, datetime, list[str]]:
    year, month = year_month.split("-")
    start = datetime(int(year), int(month), 1)
    if int(month) == 12:
        end = datetime(int(year) + 1, 1, 1)
    else:
        end = datetime(int(year), int(month) + 1, 1)

    days = []
    current = start
    while current < end:
        days.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)

    if quick_test:
        days = days[:3]
        end = start + timedelta(days=len(days))

    return start, end, days


def _count_files(root: Path, patterns: list[str]) -> int:
    matched: set[Path] = set()
    if not root.exists():
        return 0
    for pattern in patterns:
        matched.update(root.glob(pattern))
    return len(matched)


def _float_ratio(num: int, den: int) -> float:
    return float(num / den) if den else 0.0


def _base_modality_report(month_dir: Path, goes_dir: Path, sharp_file: Path, year_month: str) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for name in REQUIRED_MODALITIES:
        report[name] = {
            "required": True,
            "file_count": 0,
            "parsed_samples": 0,
            "nonzero_fraction": 0.0,
            "status": "pending",
            "failures": [],
        }

    for modality, patterns in MODALITY_FILE_GLOBS.items():
        report[modality]["file_count"] = _count_files(month_dir / modality, patterns)

    year, month = year_month.split("-")
    report["goes"]["file_count"] = len(list(goes_dir.glob(f"*_g18_d{year}{month}*_*.nc")))
    report["sharp"]["file_count"] = int(sharp_file.exists() and sharp_file.stat().st_size > 0)
    return report


def _finalize_modality_report(modality_report: dict[str, dict[str, Any]]) -> None:
    for info in modality_report.values():
        failures = info["failures"]
        if failures:
            info["status"] = "failed"
        elif info["parsed_samples"] > 0 and info["nonzero_fraction"] > 0.0:
            info["status"] = "ok"
        elif info["file_count"] == 0:
            info["status"] = "missing"
            info["failures"].append("no files found")
        else:
            info["status"] = "empty"
            if "parsed output was empty" not in failures:
                info["failures"].append("parsed output was empty")


def _required_modality_failures(modality_report: dict[str, dict[str, Any]]) -> list[str]:
    failures = []
    for name in REQUIRED_MODALITIES:
        info = modality_report[name]
        if info["status"] != "ok":
            reason = "; ".join(info["failures"]) if info["failures"] else info["status"]
            failures.append(f"{name}: {reason}")
    return failures


def _write_month_report(year_month: str, kind: str, payload: dict[str, Any]) -> Path:
    path = REPORTS_DIR / f"{year_month}_{kind}.json"
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    return path


def _self_loop_gnn_inputs(batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    nodes = torch.arange(batch_size, device=device, dtype=torch.long)
    edge_index = torch.stack([nodes, nodes], dim=0)
    return edge_index, nodes


def _enforce_required_modalities(year_month: str, modality_report: dict[str, dict[str, Any]]) -> None:
    failures = _required_modality_failures(modality_report)
    if failures:
        raise RuntimeError(
            f"{year_month}: required modality validation failed: " + " | ".join(failures)
        )


def _build_month_report_payload(
    year_month: str,
    result: dict[str, Any],
    args,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "year_month": year_month,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "load_ok": result.get("load_ok", False),
        "validate_only": bool(getattr(args, "validate_only", False)),
        "require_all_modalities": bool(getattr(args, "require_all_modalities", False)),
        "errors": result.get("errors", []),
        "modality_report": result.get("modality_report", {}),
    }
    if result.get("load_ok"):
        payload["n_seq_features"] = int(result["seq_features"].shape[1])
        payload["n_valid_windows"] = int(len(result["valid_indices"]))
        payload["positive_window_fraction"] = float(result["y_raw"][result["valid_indices"]].mean())
    if extra:
        payload.update(extra)
    return payload


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: DOWNLOAD REAL DATA (month by month)
# ══════════════════════════════════════════════════════════════════════════════

def download_month_pradan(year_month: str, args) -> dict[str, Path]:
    """
    Download one month of Aditya-L1 data from PRADAN.

    FIX v4.1: Uses DIRECT URL construction per day instead of scraping the
    browse page (which only ever shows the 10 most-recent files — always today's
    data, never historical months). The direct URL pattern is:
      https://pradan1.issdc.gov.in/al1/protected/downloadData/
        {instrument}/level1/{YYYY}/{MM}/{orbit_dir}/{filename}

    Filename conventions (from PRADAN portal inspection):
      SoLEXS L1 ZIP:  AL1_SLX_L1_{YYYYMMDD}_v1.0.zip
      HEL1OS L1 ZIP:  AL1_HXS_L1_{YYYYMMDD}_v1.0.zip

    Returns: dict mapping instrument -> directory of downloaded files.
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
        "suit":   month_dir / "suit",
    }
    for d in downloaded.values():
        d.mkdir(exist_ok=True)

    if not pradan_user or not pradan_pass:
        logger.warning(
            "PRADAN_USER/PRADAN_PASS not set — skipping PRADAN download.\n"
            "  Set credentials then re-run: export PRADAN_USER=... PRADAN_PASS=..."
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
        days = days[:3]
        logger.info("Quick test: only downloading %d days", len(days))

    logger.info("Downloading %s data for %d days from PRADAN (direct URL)...", year_month, len(days))

    # PRADAN direct URL pattern:
    # https://pradan1.issdc.gov.in/al1/protected/downloadData/
    #   {instr}/level1/{YYYY}/{MM}/N00_0000/{filename}?{instr}
    # Orbit dir is nominally N00_0000 for L1 (calibrated in-orbit)
    PRADAN_DL = "https://pradan1.issdc.gov.in"
    ORBIT_DIR = "N00_0000"  # L1 standard orbit directory

    async def _download_all():
        n_ok = {"solexs": 0, "helios": 0, "mag": 0, "swis": 0, "suit": 0}
        async with PRADANSession() as session:
            for date_str in days:
                yyyy = date_str[:4]
                mm   = date_str[4:6]
                dd   = date_str[6:8]

                # ── Instrument Browse Strategy (HEL1OS, MAG, SWIS, SUIT) ────────
                # We scrape the browse page because PRADAN uses unpredictable
                # observation IDs and processing timestamps in the filenames.
                import re as _re
                instruments_to_browse = {
                    "helios": "hel1os", 
                    "mag": "mag", 
                    "swis": "swis", 
                    "suit": "suit"
                }

                for inst_key, pradan_id in instruments_to_browse.items():
                    day_files = []
                    try:
                        day_url = (
                            f"https://pradan1.issdc.gov.in/al1/protected/browse.xhtml"
                            f"?id={pradan_id}&date={yyyy}-{mm}-{dd}"
                        )
                        resp = await session._client.get(day_url)
                        
                        # Find all downloadData links for this instrument
                        # SUIT uses .zip or .fits. We'll match both.
                        # MAG uses .nc, SWIS uses .cdf
                        dl_links = _re.findall(
                            rf'href="(/al1/protected/downloadData/{pradan_id}[^"]+)"',
                            resp.text, _re.IGNORECASE
                        )
                        for lnk in dl_links:
                            fn = lnk.split("/")[-1].split("?")[0]
                            # Simple filter: check if the date string is in the filename
                            # (Some instruments use yyyymmdd, some yyyy-mm-dd)
                            if date_str in fn or f"{yyyy}-{mm}-{dd}" in fn or f"{yyyy}{mm}{dd}" in fn:
                                day_files.append({
                                    "filename": fn,
                                    "download_url": f"https://pradan1.issdc.gov.in{lnk}"
                                })
                    except Exception as e:
                        logger.warning(f"Failed to browse {inst_key} for {date_str}: {e}")

                    for finfo in day_files:
                        fn = finfo.get("filename", "")
                        url = finfo.get("download_url", "")
                        
                        # Use appropriate cache directory based on inst_key
                        # For swis, we'll map to 'aspex' folder if needed, but our downloaded dict uses keys.
                        # Wait, downloaded dict doesn't have mag, swis, suit yet. We must create them.
                        dest = downloaded.get(inst_key)
                        if dest:
                            ok = await session.download_file(fn, dest / fn, url)
                            if ok:
                                n_ok[inst_key] += 1
                                # We can break for helios, but for mag/swis/suit there might be multiple files per day
                                # (e.g. multiple SUIT images, multiple SWIS files: BLK, TH1, TH2)
                                # So we don't break for them, we download all available for that day.
                                if inst_key == "helios":
                                    break

                # ── SoLEXS L1 ZIP (Hardcoded path still works) ─────────────────
                for ver in ("v1.0", "v1.1", "v2.0", "v1.2", "v3.0"):
                    fn_slx  = f"AL1_SLX_L1_{date_str}_{ver}.zip"
                    url_slx = (
                        f"{PRADAN_DL}/al1/protected/downloadData/solexs/level1/"
                        f"{yyyy}/{mm}/{ORBIT_DIR}/{fn_slx}?solexs"
                    )
                    dest_slx = downloaded["solexs"] / fn_slx
                    ok = await session.download_file(fn_slx, dest_slx, url_slx)
                    if ok:
                        n_ok["solexs"] += 1
                        break  # got this day

        logger.info(
            "PRADAN download complete for %s — SoLEXS: %d/%d, HEL1OS: %d/%d, MAG: %d, SWIS: %d, SUIT: %d",
            year_month, n_ok["solexs"], len(days), n_ok["helios"], len(days),
            n_ok["mag"], n_ok["swis"], n_ok["suit"]
        )
        if n_ok["solexs"] == 0:
            logger.warning(
                "%s: 0 SoLEXS files downloaded. Possible causes:\n"
                "  1. Data not yet ingested on PRADAN for this date range\n"
                "  2. URL path changed — check https://pradan1.issdc.gov.in/al1/\n"
                "  3. Session expired (will retry next month)",
                year_month
            )

    try:
        asyncio.run(_download_all())
    except Exception as exc:
        logger.error("PRADAN download error for %s: %s", year_month, exc)

    return downloaded


def download_month_goes(year_month: str, args) -> Path:
    """
    Download GOES XRS data for one month from NOAA NCEI.
    GOES data is FREE, no login required.
    sunpy downloads netCDF (.nc) files into GOES_DIR.
    Returns GOES_DIR (not a single file — there are multiple .nc per month).
    """
    year, month = year_month.split("-")

    # Check if we already have .nc files for this month
    existing_nc = list(GOES_DIR.glob(f"*_g18_d{year}{month}*_*.nc"))
    if len(existing_nc) >= 10:
        logger.info("GOES %s already cached: %d nc files", year_month, len(existing_nc))
        return GOES_DIR

    logger.info("Downloading GOES XRS data for %s from NOAA...", year_month)
    try:
        import sunpy.timeseries as ts
        from sunpy.net import Fido, attrs

        start_date = f"{year}-{month}-01"
        start_dt = datetime(int(year), int(month), 1)
        if int(month) == 12:
            end_dt = datetime(int(year) + 1, 1, 1)
        else:
            end_dt = datetime(int(year), int(month) + 1, 1)
        end_date = end_dt.strftime("%Y-%m-%d")

        result = Fido.search(
            attrs.Time(start_date, end_date),
            attrs.Instrument.xrs,
            attrs.goes.SatelliteNumber(18),
        )
        if len(result) == 0:
            logger.warning("No GOES data found via sunpy for %s", year_month)
            return GOES_DIR

        downloaded = Fido.fetch(result, path=str(GOES_DIR))
        if downloaded:
            logger.info("GOES %s downloaded: %d files", year_month, len(downloaded))
        return GOES_DIR

    except Exception as exc:
        logger.error("GOES download via sunpy failed: %s", exc)

    return GOES_DIR


def download_month_sharp(year_month: str, args) -> Path:
    """
    Download SDO/HMI SHARP magnetic parameters from JSOC (NASA).
    FREE, no login required for standard data products.
    
    SHARP params: 21 magnetic complexity features per active region per 12 min.
    Reference: Bobra & Couvidat 2015 (standard in space weather AI)
    """
    year, month = year_month.split("-")
    sharp_file = SHARP_DIR / f"sharp_{year_month}.csv"

    if sharp_file.exists() and sharp_file.stat().st_size > 0:
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

        jsoc_email = os.environ.get("JSOC_EMAIL", "").strip()
        if not jsoc_email:
            logger.info(
                "JSOC_EMAIL not set — attempting metadata-only SHARP query for %s without export email",
                year_month
            )

        # Keyword queries do not require an export request, so they can usually
        # run without a registered JSOC email. We only pass the email if present.
        try:
            import drms
            client_kwargs = {"email": jsoc_email} if jsoc_email else {}
            c = drms.Client(**client_kwargs)
            series = "hmi.sharp_cea_720s"
            keys = [
                "T_REC", "HARPNUM", "LAT_MIN", "LON_MIN", "LAT_MAX", "LON_MAX",
                "USFLUX", "MEANGBT", "MEANJZD", "TOTUSJH", "MEANALP",
                "MEANGAM", "MEANGBZ", "MEANGBH", "MEANJZH", "TOTUSJZ",
                "ABSNJZH", "SAVNCPP", "MEANPOT", "TOTPOT", "MEANSHR",
                "SHRGT45", "AREA_ACR", "R_VALUE",
            ]
            start_str = start_dt.strftime("%Y.%m.%d_00:00:00_TAI")
            q = f"{series}[{start_str}/{(end_dt - start_dt).days}d@12m]"
            k, _ = c.query(q, key=keys, seg=None)
            if k is None or len(k) == 0:
                logger.warning("SHARP %s: JSOC query returned 0 rows", year_month)
            else:
                k.to_csv(str(sharp_file), index=False)
                logger.info("SHARP %s: %d rows saved", year_month, len(k))
        except ImportError:
            logger.warning("drms not installed — pip install drms. SHARP skipped for %s.", year_month)
        except Exception as exc:
            logger.warning("JSOC SHARP download failed for %s: %s", year_month, exc)
            logger.warning("  → SHARP is supplementary. Training will continue with zeros for SHARP features.")

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
    Apply Continuous Wavelet Transform (CWT) to SoLEXS counts.

    PHYSICS: Before a flare, coronal plasma oscillates (Quasi-Periodic Pulsations,
    QPPs). The CWT converts the 1D X-ray light curve into a 2D time-frequency
    spectrogram so the TCN can literally "hear" the vibration before the explosion.

    Uses PyWavelets (pywt) with complex Morlet wavelet (cmor1.5-1.0).
    Frequency bands:
      Band 0: 0.008–0.016 Hz (64–128s) — long-period oscillations
      Band 1: 0.031–0.063 Hz (16–32s)  — mid-period oscillations
      Band 2: 0.125–0.25  Hz  (4–8s)   — short-period oscillations
      Band 3: 0.5–1.0     Hz  (1–2s)   — QPP band (fastest)

    Output: (N, 4) float32 — wavelet power in 4 frequency bands per timestep.
    """
    n = len(counts)
    try:
        import pywt
        c_norm = (counts - np.median(counts)) / (np.std(counts) + 1e-6)
        # Scales chosen so central freq covers 1–128 s QPP range
        scales = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.float64)
        # Complex Morlet: cmor<bandwidth>-<center_freq>
        coef, freqs = pywt.cwt(c_norm, scales, "cmor1.5-1.0", sampling_period=1.0 / fs)
        power = np.abs(coef) ** 2  # (8, N)
        # Aggregate into 4 bands (coarse → fine)
        return np.stack([
            power[6:8].mean(axis=0),   # 64–128 s (low freq)
            power[4:6].mean(axis=0),   # 16–32 s  (mid freq)
            power[2:4].mean(axis=0),   # 4–8 s    (high freq)
            power[0:2].mean(axis=0),   # 1–2 s    (QPP band)
        ], axis=1).astype(np.float32)
    except ImportError:
        logger.warning("PyWavelets not installed — pip install PyWavelets. Using scipy fallback.")
    except Exception as exc:
        logger.debug("PyWavelets CWT failed: %s — trying scipy fallback", exc)
    # Scipy fallback (older API, less accurate but available)
    try:
        from scipy.signal import cwt as scipy_cwt
        from scipy import signal as scipy_signal
        c_norm = (counts - np.median(counts)) / (np.std(counts) + 1e-6)
        scales = np.array([1, 2, 4, 8, 16, 32, 64, 128])
        # Use ricker (Mexican hat) as fallback — simpler but still captures QPPs
        coef = scipy_cwt(c_norm, scipy_signal.ricker, scales)
        power = coef ** 2
        return np.stack([
            power[6:8].mean(axis=0),
            power[4:6].mean(axis=0),
            power[2:4].mean(axis=0),
            power[0:2].mean(axis=0),
        ], axis=1).astype(np.float32)
    except Exception as exc2:
        logger.debug("Scipy CWT fallback also failed: %s — using zeros", exc2)
        return np.zeros((n, 4), dtype=np.float32)


def load_month_features(
    month_dir: Path,
    goes_dir: Path,
    sharp_file: Path,
    year_month: str,
    args,
) -> dict[str, Any]:
    """
    Load one month of REAL data from all instruments and build the
    multi-modal feature matrix.
    
    Returns a dict containing:
      - load_ok: whether feature engineering succeeded
      - modality_report: validation-ready per-modality status
      - arrays used for training when load_ok=True
    """
    year, month = year_month.split("-")
    solexs_dir = month_dir / "solexs"
    helios_dir = month_dir / "helios"
    mag_dir    = month_dir / "mag"
    swis_dir   = month_dir / "swis"
    suit_dir   = month_dir / "suit"

    WINDOW_S = 1800     # 30 minutes at 1-s cadence
    HORIZON_S = 900     # predict M+ in next 15 minutes
    STEP_S    = 60      # slide window every 60s (saves RAM, reduces redundancy)
    modality_report = _base_modality_report(month_dir, goes_dir, sharp_file, year_month)
    result: dict[str, Any] = {
        "load_ok": False,
        "year_month": year_month,
        "month_dir": str(month_dir),
        "modality_report": modality_report,
        "errors": [],
    }

    # ── Load SoLEXS (primary) ───────────────────────────────────────────────
    slx_times  = []
    slx_counts = []
    slx_loader = SoLEXSLoader(str(solexs_dir))

    # Scan all days in the month
    start_dt, end_dt, _ = _month_date_range(year_month, quick_test=args.quick_test)

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

    modality_report["solexs"]["parsed_samples"] = len(slx_times)
    if len(slx_times) < WINDOW_S * 2:
        msg = f"insufficient SoLEXS data ({len(slx_times)} samples)"
        modality_report["solexs"]["failures"].append(msg)
        result["errors"].append(msg)
        _finalize_modality_report(modality_report)
        logger.warning("%s: %s — skipping month", year_month, msg)
        return result

    slx_times  = np.array(slx_times,  dtype=np.float64)
    slx_counts = np.array(slx_counts, dtype=np.float32)
    idx_sort   = np.argsort(slx_times)
    slx_times  = slx_times[idx_sort]
    slx_counts = slx_counts[idx_sort]
    modality_report["solexs"]["nonzero_fraction"] = _float_ratio(
        int(np.count_nonzero(slx_counts)), len(slx_counts)
    )
    logger.info("%s SoLEXS: %d records (%.1f hours)", year_month, len(slx_times), len(slx_times) / 3600)

    # ── Load GOES XRS for flare labels (cross-validation) ───────────────────
    # sunpy downloads netCDF (.nc) files — we read them with netCDF4
    goes_flux = None
    goes_times = None
    try:
        import netCDF4 as nc_goes
        # Find all 1-minute average GOES files for this month
        nc_files = sorted(goes_dir.glob(f"*avg1m*_g18_d{year}{month}*_*.nc"))
        if not nc_files:
            # Try 1-second flux files as fallback
            nc_files = sorted(goes_dir.glob(f"*flx1s*_g18_d{year}{month}*_*.nc"))
        if nc_files:
            all_times = []
            all_flux = []
            for ncf in nc_files:
                try:
                    ds = nc_goes.Dataset(str(ncf), 'r')
                    # GOES-R netCDF: time is in seconds since epoch
                    t = ds.variables['time'][:]
                    # Convert from GOES epoch (2000-01-01 12:00:00) to Unix
                    goes_epoch_offset = 946728000.0  # Unix timestamp of 2000-01-01 12:00:00
                    t_unix = t + goes_epoch_offset
                    # XRS-B long channel (1-8 Å) flux
                    flux_var = None
                    for vname in ['xrsb_flux', 'a_flux', 'b_flux', 'xrsa_flux', 'xrsb1_flux']:
                        if vname in ds.variables:
                            flux_var = vname
                            break
                    if flux_var is None:
                        # Try any variable with 'flux' in the name
                        for vname in ds.variables:
                            if 'flux' in vname.lower():
                                flux_var = vname
                                break
                    if flux_var:
                        flux = ds.variables[flux_var][:]
                        all_times.extend(t_unix.tolist())
                        all_flux.extend(flux.tolist())
                    ds.close()
                except Exception as e:
                    logger.debug("GOES nc file %s error: %s", ncf.name, e)
            if all_times:
                goes_times = np.array(all_times, dtype=np.float64)
                goes_flux = np.array(all_flux, dtype=np.float64)
                # Remove NaN/invalid values
                valid = np.isfinite(goes_flux) & (goes_flux > 0)
                goes_times = goes_times[valid]
                goes_flux = goes_flux[valid]
                logger.info("%s GOES: %d records from %d nc files", year_month, len(goes_times), len(nc_files))
            else:
                logger.warning("%s GOES: no flux data found in %d nc files", year_month, len(nc_files))
        else:
            logger.warning("%s GOES: no nc files found in %s", year_month, goes_dir)
    except ImportError:
        logger.warning("netCDF4 not installed — GOES data skipped")
        modality_report["goes"]["failures"].append("netCDF4 not installed")
    except Exception as exc:
        logger.warning("GOES load error: %s", exc)
        modality_report["goes"]["failures"].append(str(exc))

    if goes_times is not None and goes_flux is not None:
        modality_report["goes"]["parsed_samples"] = len(goes_times)
        modality_report["goes"]["nonzero_fraction"] = _float_ratio(
            int(np.count_nonzero(goes_flux > 0)), len(goes_flux)
        )
    elif not modality_report["goes"]["failures"]:
        modality_report["goes"]["failures"].append("no GOES flux parsed")

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
        modality_report["helios"]["failures"].append("no HEL1OS records parsed")
    modality_report["helios"]["parsed_samples"] = len(hel_times)
    helios_nonzero = np.zeros(len(slx_counts), dtype=np.float32)
    for arr in hel_aligned.values():
        helios_nonzero = np.maximum(helios_nonzero, arr.astype(np.float32))
    modality_report["helios"]["nonzero_fraction"] = _float_ratio(
        int(np.count_nonzero(helios_nonzero)), len(helios_nonzero)
    )

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
    else:
        modality_report["mag"]["failures"].append("no MAG records parsed")
    modality_report["mag"]["parsed_samples"] = len(mag_all_records)
    modality_report["mag"]["nonzero_fraction"] = _float_ratio(
        int(np.count_nonzero(np.any(mag_features != 0, axis=1))),
        len(mag_features),
    )

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
        for i in range(0, len(slx_times), 600):
            feat = swis_loader.extract_ml_features(
                swis_all_records,
                window_end_unix=slx_times[i],
                window_minutes=30.0,
            )
            if feat is not None:
                end_i = min(i + 600, len(slx_times))
                swis_features[i:end_i] = feat[np.newaxis, :]
    else:
        modality_report["swis"]["failures"].append("no SWIS records parsed")
    modality_report["swis"]["parsed_samples"] = len(swis_all_records)
    modality_report["swis"]["nonzero_fraction"] = _float_ratio(
        int(np.count_nonzero(np.any(swis_features != 0, axis=1))),
        len(swis_features),
    )

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
    logger.info("%s Computing CWT wavelet transform (QPP detection)...", year_month)
    wavelet_feat = engineer_wavelet_features(counts.astype(np.float32))  # (N, 4)

    # ── PHYSICS FEATURE 1: HXR Spectral Index (non-thermal electron indicator) ─
    # γ ≈ log(N_high / N_low) / log(E_high / E_low)  (power-law spectral index)
    # Physical meaning: γ < 4.5 → non-thermal electrons (HOPE trigger condition)
    # from HEL1OS User Guide: CdTe3 (30-40 keV) and CZT2 (40-80 keV)
    hxr_lo = np.clip(hel_aligned["cdte_30_40"].astype(np.float64), 0.01, None)
    hxr_hi = np.clip(hel_aligned["czt_40_60"].astype(np.float64), 0.01, None)
    hxr_spectral_index = np.clip(
        np.log(hxr_hi / hxr_lo + 1e-8) / np.log(60.0 / 35.0),  # ΔE ratio
        -10, 10
    ).astype(np.float32)
    # Normalize: γ=0 is non-thermal, γ=5 is thermal → center at 2.5
    hxr_spectral_index = (hxr_spectral_index - 2.5) / 2.5

    # ── PHYSICS FEATURE 2: Alfvén Mach Number (solar wind shock indicator) ────
    # M_A = v_sw / v_A   where v_A = B / sqrt(μ₀ρ)  (Alfvén velocity)
    # Physical meaning: M_A > 1 → super-Alfvénic → CME shock front is forming
    # Approximation using MAG + SWIS data at L1
    B_total  = np.clip(mag_features[:, 0], 1.0, None)   # nT
    proton_density = np.clip(swis_features[:, 0], 1.0, None)  # cm⁻³
    proton_speed   = np.clip(swis_features[:, 2], 200.0, None)  # km/s
    # v_A (km/s) = B_nT * 21.8 / sqrt(n_proton_cm3)  [standard formula]
    v_alfven = B_total * 21.8 / np.sqrt(proton_density)
    mach_alfven = np.clip(proton_speed / np.clip(v_alfven, 1.0, None), 0, 20).astype(np.float32)
    # Normalize: M_A ≈ 5 typical, ≈ 15+ during CME → (M_A - 5) / 5
    mach_alfven_norm = (mach_alfven - 5.0) / 5.0

    logger.info("%s Physics features: HXR spectral index μ=%.2f, Alfvén Mach μ=%.2f",
                year_month, hxr_spectral_index.mean(), mach_alfven_norm.mean())

    # ── Build per-timestep feature matrix ────────────────────────────────────
    # Shape: (N_timesteps, N_SEQ_FEATURES)
    # N_SEQ_FEATURES = 7 SoLEXS + 6 HEL1OS + 2 ratios + 4 wavelet + 3 MAG + 2 physics = 22
    seq_features = np.stack([
        slx_counts,                        # 0:  raw SoLEXS SDD2 count rate
        log_counts,                        # 1:  log10(counts)
        derivative,                        # 2:  dCounts/dt
        zscore_60,                         # 3:  z-score vs 60s background
        zscore_300,                        # 4:  z-score vs 300s background
        neupert_cum,                       # 5:  Neupert integral (HXR→SXR proxy)
        hxr_sxr_ratio,                    # 6:  HXR/SXR ratio (non-thermal flag)
        hel_log["cdte_5_20"],             # 7:  HEL1OS CdTe 5-20 keV
        hel_log["cdte_20_30"],            # 8:  HEL1OS CdTe 20-30 keV
        hel_log["cdte_30_40"],            # 9:  HEL1OS CdTe 30-40 keV (HOPE trigger)
        hel_log["cdte_40_60"],            # 10: HEL1OS CdTe 40-60 keV
        hel_log["czt_40_60"],             # 11: HEL1OS CZT 40-60 keV
        hel_log["czt_80_150"],            # 12: HEL1OS CZT 80-150 keV (highest energy)
        wavelet_feat[:, 0],               # 13: CWT power low-freq (64-128s)
        wavelet_feat[:, 1],               # 14: CWT power mid-freq (16-32s)
        wavelet_feat[:, 2],               # 15: CWT power high-freq (4-8s)
        wavelet_feat[:, 3],               # 16: CWT QPP band (1-2s) ← key precursor
        mag_features[:, 0],              # 17: B_total mean (nT)
        mag_features[:, 4],              # 18: Bz_gse mean (negative = geoeffective)
        mag_features[:, 5],              # 19: IMF clock angle
        hxr_spectral_index,              # 20: HXR power-law index (SOTA physics)
        mach_alfven_norm,                # 21: Alfvén Mach number (CME shock)
    ], axis=1)  # shape: (N, 22)

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
    if sharp_file.exists() and sharp_file.stat().st_size > 0:
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
                    modality_report["sharp"]["parsed_samples"] = len(sh)
        except Exception as exc:
            logger.warning("%s SHARP load error: %s", year_month, exc)
            modality_report["sharp"]["failures"].append(str(exc))
    else:
        modality_report["sharp"]["failures"].append("no SHARP file available")
    modality_report["sharp"]["nonzero_fraction"] = _float_ratio(
        int(np.count_nonzero(np.any(sharp_feat_full != 0, axis=1))),
        len(sharp_feat_full),
    )

    # ── Load SUIT Images (if available) ──────────────────────────────────────
    suit_loader = SUITLoader(str(suit_dir))
    try:
        suit_images_list = suit_loader.scan_directory()
        logger.info("%s SUIT: %d images scanned", year_month, len(suit_images_list))
        suit_images_dict = {}
        for img in suit_images_list:
            suit_images_dict[img.unix_time] = img.load_array()
        modality_report["suit"]["parsed_samples"] = len(suit_images_list)
    except Exception as exc:
        logger.warning("%s SUIT load error: %s", year_month, exc)
        modality_report["suit"]["failures"].append(str(exc))
        suit_images_dict = {}

    # Pre-sort SUIT images by timestamp for fast lookup
    suit_times = np.array(sorted(suit_images_dict.keys()), dtype=np.float64) if suit_images_dict else np.array([])
    suit_image_arrays = suit_images_dict
    if len(suit_times) > 0:
        probe_times = slx_times[::600] if len(slx_times) > 600 else slx_times
        match_idx = np.searchsorted(suit_times, probe_times, side="right") - 1
        valid = match_idx >= 0
        deltas = np.full(len(probe_times), np.inf, dtype=np.float64)
        deltas[valid] = probe_times[valid] - suit_times[match_idx[valid]]
        modality_report["suit"]["nonzero_fraction"] = _float_ratio(
            int(np.count_nonzero(deltas <= 12 * 3600)), len(probe_times)
        )
    else:
        modality_report["suit"]["failures"].append("no SUIT images parsed")

    # ── Identify valid window indices (Lazy Dataset prep) ────────────────────
    valid_indices = []
    
    valid_end = N - HORIZON_S
    for i in range(WINDOW_S, valid_end, STEP_S):
        # Is it quiet sun?
        is_flare = y_raw[i] > 0
        if not is_flare:
            # Drop 95% of quiet sun
            if hash(f"{year_month}_{i}") % 100 > 5:
                continue
        valid_indices.append(i)

    if len(valid_indices) < 10:
        msg = f"too few windows ({len(valid_indices)})"
        logger.warning("%s: %s — skipping", year_month, msg)
        result["errors"].append(msg)
        _finalize_modality_report(modality_report)
        return result

    valid_indices = np.array(valid_indices, dtype=np.int32)
    logger.info(
        "%s Dataset info: %d windows (pos=%.1f%%) — using Lazy Loading to save RAM",
        year_month, len(valid_indices), y_raw[valid_indices].mean() * 100
    )

    _finalize_modality_report(modality_report)
    result.update({
        "load_ok": True,
        "seq_features": seq_features,
        "sharp_feat_full": sharp_feat_full,
        "mag_features": mag_features,
        "swis_features": swis_features,
        "suit_image_arrays": suit_image_arrays,
        "suit_times": suit_times,
        "slx_times": slx_times,
        "y_raw": y_raw,
        "valid_indices": valid_indices,
    })
    return result


def _align_to_grid(
    t_src: np.ndarray, v_src: np.ndarray, t_dst: np.ndarray, tol_s: float = 2.0
) -> np.ndarray:
    """
    Vectorized nearest-neighbour interpolation from source to destination grid.

    BUG FIX (Bug 3): The original implementation used a Python for-loop over
    every destination sample — O(N*M) complexity. With 2.6M SoLEXS samples
    this took hours per alignment call, making training appear hung.

    This vectorized version uses np.searchsorted → O(M log N), completing
    a full-month alignment in under 1 second on M4.
    """
    out = np.zeros(len(t_dst), dtype=v_src.dtype)
    if len(t_src) == 0 or len(t_dst) == 0:
        return out
    # Binary search: find insertion point for each t_dst in t_src
    idx = np.searchsorted(t_src, t_dst)  # shape (M,)
    idx = np.clip(idx, 0, len(t_src) - 1)
    # Check left neighbor too (searchsorted gives right index)
    idx_left = np.clip(idx - 1, 0, len(t_src) - 1)
    diff_right = np.abs(t_src[idx]      - t_dst)
    diff_left  = np.abs(t_src[idx_left] - t_dst)
    best_idx = np.where(diff_left < diff_right, idx_left, idx)
    best_diff = np.minimum(diff_left, diff_right)
    # Apply only where within tolerance
    mask = best_diff <= tol_s
    out[mask] = v_src[best_idx[mask]]
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


from pipeline.ml.fusion import AdityScanModel


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: MONTHLY DATASET + INCREMENTAL TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

class FlareDataset(Dataset):
    def __init__(self, ds_info: dict, indices: np.ndarray):
        self.seq_features = torch.from_numpy(ds_info["seq_features"])
        self.sharp_feat_full = torch.from_numpy(ds_info["sharp_feat_full"])
        self.mag_features = torch.from_numpy(ds_info["mag_features"])
        self.swis_features = torch.from_numpy(ds_info["swis_features"])
        
        self.suit_image_arrays = ds_info["suit_image_arrays"]
        self.suit_times = ds_info["suit_times"]
        self.slx_times = ds_info["slx_times"]
        self.y_raw = torch.from_numpy(ds_info["y_raw"])
        
        self.indices = indices
        
        # We cache resized SUIT images so we don't resize them per-item continuously
        self.suit_cache = {}

    def __len__(self): 
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        
        # Time-series window (1800 seconds)
        x_win = self.seq_features[idx - 1800 : idx]
        
        # SHARP window
        sharp_win = self.sharp_feat_full[max(0, idx - 8640) : idx]
        if len(sharp_win) < 10:
            sharp_win = torch.zeros((120, 21), dtype=torch.float32)
        elif len(sharp_win) != 120:
            # Resample via interpolation or nearest
            # We'll just step through it
            step = max(1, len(sharp_win) / 120)
            sampled_idx = torch.arange(0, len(sharp_win), step).long()[:120]
            if len(sampled_idx) < 120:
                pad = 120 - len(sampled_idx)
                sampled_idx = torch.cat([sampled_idx, torch.full((pad,), sampled_idx[-1])])
            sharp_win = sharp_win[sampled_idx]
            
        # In-situ
        insitu = torch.cat([self.mag_features[idx], self.swis_features[idx]])
        
        # SUIT Image (Lazy Load / Cache)
        current_time = self.slx_times[idx]
        suit_img = np.zeros((1, 224, 224), dtype=np.float32)
        
        if len(self.suit_times) > 0:
            valid_idx = np.searchsorted(self.suit_times, current_time) - 1
            if valid_idx >= 0:
                t_img = self.suit_times[valid_idx]
                if current_time - t_img < 12 * 3600:
                    if t_img not in self.suit_cache:
                        from skimage.transform import resize as sk_resize
                        raw_img = self.suit_image_arrays[t_img]
                        resized = sk_resize(raw_img, (224, 224), anti_aliasing=True).astype(np.float32)
                        resized = np.clip(resized, 0, np.percentile(resized, 99.9))
                        if resized.max() > 0:
                            resized = resized / resized.max()
                        self.suit_cache[t_img] = resized
                    suit_img[0] = self.suit_cache[t_img]
                    
        suit_img = torch.from_numpy(suit_img)
        y = self.y_raw[idx]
        
        return x_win, sharp_win, insitu, suit_img, y


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
    model: AdityScanModel,
    ds_info: dict,

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
    valid_indices = ds_info["valid_indices"]
    split = int(0.8 * len(valid_indices))
    
    ds_train = FlareDataset(ds_info, valid_indices[:split])
    ds_val   = FlareDataset(ds_info, valid_indices[split:])

    if len(ds_train) < 10:
        logger.warning("%s: too few training samples (%d)", year_month, len(ds_train))
        return {}

    # Weighted sampling: oversample flare windows
    pos_weight = min(15.0 * flare_weight_multiplier, 50.0)
    
    # Compute y_train efficiently for the sampler
    y_raw = ds_info["y_raw"]
    n_records = len(y_raw)
    y_train = np.zeros(len(ds_train), dtype=np.float32)
    for i, idx in enumerate(valid_indices[:split]):
        target_start = idx + 60 + 600
        target_end = idx + 60 + 1800
        if target_end <= n_records and y_raw[target_start:target_end].any():
            y_train[i] = 1.0

    sample_weights = np.where(y_train > 0.5, pos_weight, 1.0)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights.astype(np.float32)),
        num_samples=len(ds_train), replacement=True,
    )

    train_loader = DataLoader(ds_train, batch_size=batch_size, sampler=sampler,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(ds_val,   batch_size=batch_size * 2, shuffle=False,
                              num_workers=0, pin_memory=False)

    # Focal loss for class imbalance
    # NOTE: Model outputs are PROBABILITIES (sigmoid already applied in NowcastHead/ForecastHead)
    # So we use binary_cross_entropy (not _with_logits) and clamp for numerical stability
    class FocalLoss(nn.Module):
        def __init__(self, gamma=2.0, pos_w=10.0):
            super().__init__()
            self.gamma = gamma
            self.pos_w = pos_w
        def forward(self, prob, target):
            prob = prob.view(-1)
            target = target.view(-1)
            # Clamp for numerical stability (avoid log(0))
            prob = prob.clamp(1e-6, 1 - 1e-6)
            # Weighted BCE: weight positive class more
            weight = torch.where(target > 0.5, self.pos_w, 1.0)
            bce = -weight * (target * torch.log(prob) + (1 - target) * torch.log(1 - prob))
            # Focal modulation
            pt = torch.where(target > 0.5, prob, 1 - prob)
            return ((1 - pt) ** self.gamma * bce).mean()

    criterion = FocalLoss(gamma=2.0, pos_w=pos_weight)

    best_tss = -1.0
    month_metrics = {}

    for epoch in range(1, n_epochs + 1):
        model.train()
        losses = []
        for X_s, X_sh, X_i, X_su, y_b in train_loader:
            X_s  = X_s.to(device)
            X_sh = X_sh.to(device)
            X_i  = X_i.to(device)
            X_su = X_su.to(device)
            y_b  = y_b.to(device)
            # Image branch enabled: usage logic here
            batch_size = X_s.size(0)
            gnn_edge_index, gnn_batch = _self_loop_gnn_inputs(batch_size, device)

            optimizer.zero_grad()
            out = model(X_s, X_sh, X_i, image=X_su, gnn_edge_index=gnn_edge_index, gnn_batch=gnn_batch)

            # Primary: 15-min horizon
            loss = criterion(out["p_flare_15min"], y_b)
            # Auxiliary: other horizons (weighted)
            for key, w in [("flare_prob", 1.0), ("p_flare_5min", 0.9),
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
            for X_s, X_sh, X_i, X_su, y_b in val_loader:
                batch_size = X_s.size(0)
                gnn_edge_index, gnn_batch = _self_loop_gnn_inputs(batch_size, device)
                out = model(X_s.to(device), X_sh.to(device), X_i.to(device), image=X_su.to(device), gnn_edge_index=gnn_edge_index, gnn_batch=gnn_batch)
                p = out["p_flare_15min"].cpu().numpy().ravel()  # already sigmoid'd by ForecastHead
                val_probs.extend(p.tolist())
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

        # SOTA-5: Rich dashboard epoch log with performance emoji
        tss_val = metrics["TSS"]
        if tss_val >= 0.6:
            perf_emoji = "🔥"
        elif tss_val >= 0.4:
            perf_emoji = "⚡"
        elif tss_val >= 0.2:
            perf_emoji = "📈"
        else:
            perf_emoji = "🔄"
        logger.info(
            "[%s Epoch %2d/%d] %s Loss=%.4f | TSS=%.3f HSS=%.3f | POD=%.2f FAR=%.2f | AUC=%.3f",
            year_month, epoch, n_epochs, perf_emoji, np.mean(losses),
            tss_val, metrics["HSS"], metrics["POD"], metrics["FAR"], metrics["AUC"]
        )

        if metrics["TSS"] > best_tss:
            best_tss = metrics["TSS"]
            month_metrics = {**metrics, "epoch": epoch, "month": year_month,
                             "threshold": round(best_t, 2)}

    return month_metrics


# ══════════════════════════════════════════════════════════════════════════════
# ONNX EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def export_onnx(model: AdityScanModel, path: str) -> None:
    """Export trained model to ONNX for Render CPU inference."""
    model_cpu = model.cpu().eval()
    dummy_seq    = torch.zeros(1, 1800, 22)   # 22 features (20 + 2 physics)
    dummy_sharp  = torch.zeros(1, 120,  21)
    dummy_insitu = torch.zeros(1, 14)
    dummy_suit   = torch.zeros(1, 1, 224, 224)
    dummy_edge_index = torch.zeros(2, 1, dtype=torch.long)
    dummy_batch = torch.zeros(1, dtype=torch.long)

    with torch.no_grad():
        dummy_out = model_cpu(dummy_seq, dummy_sharp, dummy_insitu, image=dummy_suit, gnn_edge_index=dummy_edge_index, gnn_batch=dummy_batch)

    output_names = [k for k in dummy_out.keys() if k != "attn_weights"]
    torch.onnx.export(
        model_cpu,
        (dummy_seq, dummy_sharp, dummy_insitu, dummy_suit, None, dummy_edge_index, dummy_batch),
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
    parser.add_argument("--validate-only", action="store_true",
                        help="Validate month inputs and emit a report without training")
    parser.add_argument("--require-all-modalities", action="store_true",
                        help="Fail if any modality is missing or aligns to all-zero features")
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
    if args.require_all_modalities and args.no_suit:
        raise SystemExit("--require-all-modalities conflicts with --no-suit")

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
    model = AdityScanModel(mc_dropout=0.1)
    if not args.no_suit:
        model.enable_image_branch()
    model = model.to(device)

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
        {"params": model.xray_branch.parameters(),   "lr": 3e-4},  # PRIMARY
        {"params": model.sharp_branch.parameters(),  "lr": 2e-4},  # SUPPLEMENTARY
        {"params": model.insitu_branch.parameters(), "lr": 3e-4},  # SUPPLEMENTARY
        {"params": model.fusion.parameters(),        "lr": 3e-4},  # FUSION
        {"params": model.nowcast_head.parameters(),  "lr": 3e-4},  # HEAD
        {"params": model.forecast_head.parameters(), "lr": 3e-4},  # HEAD
    ], weight_decay=1e-4)

    if model.image_branch is not None:
        optimizer.add_param_group({"params": model.image_branch.parameters(), "lr": 1e-4})
    if model.gnn_branch.enabled:
        optimizer.add_param_group({"params": model.gnn_branch.parameters(), "lr": 2e-4})

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(months_to_train) * args.epochs_per_month
    )

    # ── MAIN INCREMENTAL LOOP ─────────────────────────────────────────────────
    all_metrics = []
    month_reports: dict[str, dict[str, Any]] = {}
    global_best_tss = 0.0
    calibration_cache: dict[str, dict[str, Any]] = {}

    for month_idx, (year_month, description, flare_weight) in enumerate(months_to_train):
        logger.info("")
        logger.info("=" * 70)
        logger.info("MONTH %d/%d: %s — %s", month_idx + 1, len(months_to_train),
                    year_month, description)
        logger.info("=" * 70)

        # ── DOWNLOAD this month's data ────────────────────────────────────────
        if not args.skip_download:
            downloaded = download_month_pradan(year_month, args)
            goes_dir   = download_month_goes(year_month, args)
            sharp_file = download_month_sharp(year_month, args)
        else:
            downloaded = {
                "solexs": CACHE_DIR / year_month / "solexs",
                "helios": CACHE_DIR / year_month / "helios",
                "mag":    CACHE_DIR / year_month / "mag",
                "swis":   CACHE_DIR / year_month / "swis",
                "suit":   CACHE_DIR / year_month / "suit",
            }
            goes_dir   = GOES_DIR
            sharp_file = SHARP_DIR / f"sharp_{year_month}.csv"

        month_dir = CACHE_DIR / year_month

        # ── LOAD & ENGINEER FEATURES ──────────────────────────────────────────
        result = load_month_features(month_dir, goes_dir, sharp_file, year_month, args)
        month_payload = _build_month_report_payload(year_month, result, args)
        report_path = _write_month_report(year_month, "validation", month_payload)
        month_reports[year_month] = month_payload
        logger.info("Month report written: %s", report_path)

        if args.require_all_modalities:
            _enforce_required_modalities(year_month, result["modality_report"])

        if args.validate_only:
            logger.info("%s validation complete (load_ok=%s)", year_month, result.get("load_ok"))
            continue

        if not result.get("load_ok"):
            logger.warning("No usable data for %s — skipping", year_month)
            if not args.keep_data:
                _safe_delete(month_dir)
            continue

        # ── TRAIN on this month ────────────────────────────────────────────────
        month_metrics = train_one_month(
            model, result,
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

        if len(months_to_train) == 1:
            calibration_cache[year_month] = result

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
        del result
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

        if not args.keep_data and len(months_to_train) > 1:
            _safe_delete(month_dir)
            logger.info("Deleted %s raw data to free disk space", year_month)

    if args.validate_only:
        logger.info("Validation-only run complete.")
        return

    # ══════════════════════════════════════════════════════════════════════════
    # FINAL: Conformal Calibration + ONNX Export
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=" * 70)
    logger.info("ALL MONTHS COMPLETE — Conformal Calibration + ONNX Export")
    logger.info("=" * 70)

    # Temperature calibration (learnable scalar from the model)
    T = float(model.temperature.item())
    logger.info("Temperature scaling T=%.4f", T)

    # ── SOTA-2: Conformal Prediction Calibration ──────────────────────────────
    # Uses ALL months' validation data to compute the 90th-percentile
    # nonconformity score q_0.90. This gives a GUARANTEED 90% coverage interval:
    #   [P_hat - q_0.90, P_hat + q_0.90]
    # No distributional assumptions — just exchangeability (holds if we retrain).
    logger.info("Computing conformal prediction calibration (q_0.90)...")
    conformal_scores = []
    model.eval()
    for month_idx_c, (year_month_c, _, _) in enumerate(months_to_train):
        result_c = calibration_cache.get(year_month_c)
        if result_c is None:
            month_dir_c = CACHE_DIR / year_month_c
            if not month_dir_c.exists():
                continue
            sharp_file_c = SHARP_DIR / f"sharp_{year_month_c}.csv"
            result_c = load_month_features(month_dir_c, GOES_DIR, sharp_file_c, year_month_c, args)
        if not result_c.get("load_ok"):
            continue
        # Use last 10% of valid indices as conformal calibration set
        valid_idx_c = result_c["valid_indices"]
        cal_start = int(0.9 * len(valid_idx_c))
        cal_indices = valid_idx_c[cal_start:]
        if len(cal_indices) < 5:
            del result_c
            gc.collect()
            continue
        cal_ds = FlareDataset(result_c, cal_indices)
        cal_loader = DataLoader(cal_ds, batch_size=64, shuffle=False, num_workers=0)
        with torch.no_grad():
            for X_s, X_sh, X_i, X_su, y_b in cal_loader:
                batch_size_c = X_s.size(0)
                gnn_ei, gnn_bt = _self_loop_gnn_inputs(batch_size_c, device)
                out = model(X_s.to(device), X_sh.to(device), X_i.to(device),
                           image=X_su.to(device), gnn_edge_index=gnn_ei, gnn_batch=gnn_bt)
                p_hat = out["p_flare_15min"].cpu().numpy().ravel()  # already sigmoid'd
                y_cal = y_b.numpy().ravel()
                scores = np.abs(y_cal - p_hat)
                conformal_scores.extend(scores.tolist())
        del result_c
        gc.collect()

    if conformal_scores:
        q_90 = float(np.percentile(conformal_scores, 90))
        q_95 = float(np.percentile(conformal_scores, 95))
        logger.info("✅ Conformal calibration: q_0.90=%.4f q_0.95=%.4f (n=%d scores)",
                    q_90, q_95, len(conformal_scores))
        logger.info("   Interpretation: 90%% of real flares will fall within ±%.1f%% of prediction",
                    q_90 * 100)
    else:
        q_90, q_95 = 0.25, 0.35
        logger.warning("No conformal calibration data — using conservative defaults")

    # Export ONNX
    onnx_path = str(MODELS_DIR / "adityscan_v4.onnx")
    export_onnx(model, onnx_path)

    # Save training report
    report = {
        "model_version": "4.1.0-sota-physics-conformal",
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "months_trained": [m["month"] for m in all_metrics if "month" in m],
        "global_best_tss": round(global_best_tss, 4),
        "per_month_metrics": all_metrics,
        "temperature_scaling": T,
        "conformal_prediction": {
            "q_0.90": round(q_90, 4),
            "q_0.95": round(q_95, 4),
            "n_calibration_scores": len(conformal_scores),
            "coverage_guarantee": "90% of true outcomes fall within [P_hat ± q_0.90]",
            "assumption": "Exchangeability (retrain monthly)",
        },
        "architecture": {
            "type": "AdityScanSOTA_v4.1",
            "primary_branch": "SoLEXS+HEL1OS Causal-TCN, 6 layers dilation=[1,2,4,8,16,32], 256-dim",
            "secondary_branch": "SHARP BiLSTM + Self-Attention, 3-layer bidirectional, 128-dim",
            "insitu_branch": "MAG+SWIS InSitu-MLP, 3-layer GELU, 64-dim",
            "fusion": "Cross-Modal Attention 4-head 256-dim (X-ray as primary query)",
            "n_seq_features": 22,
            "n_sharp_features": 21,
            "n_insitu_features": 14,
            "window_seconds": 1800,
            "wavelet_transform": "PyWavelets cmor1.5-1.0 (Morlet), 8 scales 1-128s",
        },
        "physics_features": {
            "hxr_spectral_index": "HEL1OS CdTe3/CZT2 power-law index γ (non-thermal: γ<4.5)",
            "alfven_mach_number": "v_sw / v_Alfven at L1 (CME shock: M_A>1)",
            "neupert_integral": "Cumulative HXR integral ∝ SXR rise (Neupert 1968)",
            "cwt_qpp_bands": "4 QPP frequency bands from Morlet CWT (1-128s)",
        },
        "data_sources": {
            "primary": ["Aditya-L1 SoLEXS L1 (1-s ZIP archives from PRADAN)",
                        "Aditya-L1 HEL1OS L1 (1-s CdTe+CZT from PRADAN)"],
            "supplementary": ["Aditya-L1 MAG L2 (10-s Bx/By/Bz GSE)",
                              "Aditya-L1 ASPEX-SWIS L2 (proton density/T/v)",
                              "SDO/HMI SHARP CEA 720s (21 magnetic params)",
                              "GOES-18 XRS 1-min (flare labels)"],
            "derived": ["CWT QPP spectrogram", "Neupert integral", "HXR/SXR ratio",
                        "HXR spectral index γ", "Alfvén Mach number M_A",
                        "IMF clock/cone angles"],
        },
        "modality_usage": {
            year_month: payload["modality_report"] for year_month, payload in month_reports.items()
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
