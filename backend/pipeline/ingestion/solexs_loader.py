"""
SoLEXS (Solar Low Energy X-ray Spectrometer) data loader for Aditya-L1.
BUG FIX v4.1: Rewrote to handle ALL PRADAN file formats.

The original loader only looked for L2 FITS (AL1_SLX91_*_L2_*.fits).
PRADAN actually delivers L1 ZIP archives: AL1_SLX_L1_YYYYMMDD_vM.n.zip
Each ZIP contains FITS light curves inside SDD1/ and SDD2/ subdirectories.
This fix adds ZIP extraction and multi-format support.

Supported formats (in priority order):
  1. AL1_SLX_L1_YYYYMMDD_*.zip  → extract and read .lc.gz FITS inside
  2. AL1_SLX_L1_YYYYMMDD_*.fits → direct FITS read
  3. AL1_SLX91_YYYYMMDD_*_L2_*.fits → original L2 format
  4. *SDD2*.lc.gz / *SDD2*.lc.fits → compressed light curves
  5. *.nc / *.netcdf → netCDF format (some L2 products)

Instrument specifications (from SoLEXS Data Analysis User Guide, ISRO/SAC):
  - Two detectors: SDD1 (primary, lower background) and SDD2 (thicker, higher area)
  - Energy range:  2.8 – 30 keV (SDD1), 4 – 30 keV (SDD2)
  - GOES-equivalent channel: ~1–8 Å equivalent ≈ 1.55–12.4 keV (SDD2 is primary)
  - Spectral resolution: ~150 eV at 5.9 keV
  - Temporal cadence: 1 second (L1 raw spectra), 60 s average (L2 science)
  - Energy channels: 4096 channels, 1 channel ≈ 7.3 eV (after gain calibration)
"""

from __future__ import annotations

import gzip
import glob
import io
import logging
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants from SoLEXS instrument guide ────────────────────────────────────
SOLEXS_N_CHANNELS = 4096
SOLEXS_EV_PER_CHANNEL = 7.3
SOLEXS_EPOCH = datetime(2000, 1, 1, 0, 0, 0, tzinfo=timezone.utc)   # MET epoch
SOLEXS_DEAD_TIME_THRESHOLD = 0.1
SOLEXS_QUALITY_MIN = 30          # relaxed from 50 → accept more data
SOLEXS_GOES_CHANNEL_KMIN = 1.55
SOLEXS_GOES_CHANNEL_KMAX = 12.4
GOES_BAND_FRACTION_SDD2 = 0.65  # approx fraction of SDD2 counts in GOES band

# File patterns — ordered from most likely (PRADAN L1 ZIP) to least likely (L2)
PRADAN_ZIP_PATTERNS = [
    "AL1_SLX_L1_*.zip",
    "AL1_SLX_L1*.zip",
]
FITS_PATTERNS = [
    "AL1_SLX_L1_*.fits",
    "AL1_SLX91_*_L2_*.fits",
    "*SDD2*.fits",
    "*SDD1*.fits",
    "*.fits",
]
LC_GZ_PATTERNS = [
    "*SDD2*.lc.gz",
    "*SDD2*.lc.fits.gz",
    "*SDD1*.lc.gz",
    "*.lc.gz",
]
NC_PATTERNS = [
    "*.nc",
    "*.netcdf",
]


@dataclass
class SoLEXSReading:
    """Lightweight parsed SoLEXS light-curve record (one per timestep)."""
    time_unix: float        # UNIX timestamp of this sample
    counts_sdd2: float      # SDD2 dead-time-corrected count rate (cts/s)
    counts_sdd1: float      # SDD1 count rate (cts/s)
    quality: int = 1        # 1 = good, 0 = bad


@dataclass
class SoLEXSSpectrum:
    """Parsed SoLEXS L2 spectrum observation metadata."""
    filepath: str
    obs_id: str
    t_start_unix: float
    t_stop_unix: float
    exposure_s: float
    deadcor_sdd1: float
    deadcor_sdd2: float
    obs_mode: str
    quality: float

    @property
    def t_mid_unix(self) -> float:
        return 0.5 * (self.t_start_unix + self.t_stop_unix)


@dataclass
class SoLEXSLightCurve:
    """SoLEXS 1-second light curve (dead-time corrected, background subtracted)."""
    times_unix: np.ndarray
    rate_sdd1: np.ndarray
    rate_sdd2: np.ndarray
    rate_sdd2_goes: np.ndarray
    quality_flags: np.ndarray
    deadcor_sdd2: float
    background_sdd2: float


@dataclass
class SoLEXSSpectrumData:
    """Full spectral data from HDU[2]."""
    energy_kev: np.ndarray
    counts_sdd1: np.ndarray
    counts_sdd2: np.ndarray
    exposure_s: float
    background_sdd2: np.ndarray


class SoLEXSLoader:
    """
    Loads Aditya-L1 SoLEXS data from PRADAN — ALL file formats.

    Usage for training pipeline (load_day):
        loader = SoLEXSLoader("/data/pradan_cache/2024-05/solexs")
        records = loader.load_day("20240514")
        # records: list[SoLEXSReading] with time_unix and counts_sdd2

    Usage for triage (extract_lightcurve):
        spectra = loader.scan_directory(date_str="20240514")
        lc = loader.extract_lightcurve(spectra[0])
    """

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self._arf_path: Optional[str] = None
        self._rmf_path: Optional[str] = None

    def set_calibration_files(self, arf_path: str, rmf_path: str) -> None:
        self._arf_path = arf_path
        self._rmf_path = rmf_path

    # ── PRIMARY method used by training pipeline ──────────────────────────────

    def load_day(self, date_str: str) -> list[SoLEXSReading]:
        """
        Load all SoLEXS readings for a given day (YYYYMMDD).

        Tries all known PRADAN file formats in priority order:
          1. ZIP archives (L1, most common from PRADAN)
          2. Direct FITS files (L1 or L2)
          3. .lc.gz compressed light curves
          4. netCDF (.nc) files

        Returns list of SoLEXSReading objects, one per 1-second sample.
        Returns empty list if no data found (gracefully handled upstream).
        """
        if not self.data_dir.exists():
            logger.debug("SoLEXS dir does not exist: %s", self.data_dir)
            return []

        records = []

        # Strategy 1: ZIP archives (PRADAN L1 primary format)
        records = self._load_from_zips(date_str)
        if records:
            logger.debug("SoLEXS %s: %d records from ZIP archives", date_str, len(records))
            return records

        # Strategy 2: Direct FITS files
        records = self._load_from_fits(date_str)
        if records:
            logger.debug("SoLEXS %s: %d records from FITS files", date_str, len(records))
            return records

        # Strategy 3: .lc.gz compressed light curves
        records = self._load_from_lc_gz(date_str)
        if records:
            logger.debug("SoLEXS %s: %d records from .lc.gz files", date_str, len(records))
            return records

        # Strategy 4: netCDF files
        records = self._load_from_netcdf(date_str)
        if records:
            logger.debug("SoLEXS %s: %d records from netCDF files", date_str, len(records))
            return records

        logger.debug("SoLEXS %s: no data found in %s", date_str, self.data_dir)
        return []

    def _load_from_zips(self, date_str: str) -> list[SoLEXSReading]:
        """Extract and read PRADAN L1 ZIP archives."""
        records = []
        # Search for ZIP files containing this date
        all_zips = []
        for pattern in PRADAN_ZIP_PATTERNS:
            all_zips.extend(self.data_dir.glob(pattern))
            all_zips.extend(self.data_dir.glob(f"**/{pattern}"))

        date_zips = [z for z in all_zips if date_str in z.name]
        if not date_zips:
            return []

        with tempfile.TemporaryDirectory() as tmpdir:
            for zip_path in date_zips:
                try:
                    with zipfile.ZipFile(str(zip_path), "r") as zf:
                        zf.extractall(tmpdir)

                    # Now look for FITS light curves inside the extracted dir
                    tmp_path = Path(tmpdir)
                    fits_files = (
                        list(tmp_path.glob("**/*SDD2*.lc.gz"))
                        + list(tmp_path.glob("**/*SDD2*.lc.fits"))
                        + list(tmp_path.glob("**/*SDD2*.fits"))
                        + list(tmp_path.glob("**/*.lc.gz"))
                        + list(tmp_path.glob("**/*.fits"))
                    )
                    for fp in fits_files:
                        recs = self._read_fits_lc(str(fp))
                        records.extend(recs)
                except Exception as exc:
                    logger.debug("ZIP extract failed %s: %s", zip_path, exc)

        return records

    def _load_from_fits(self, date_str: str) -> list[SoLEXSReading]:
        """Read direct FITS files (extracted or L2)."""
        records = []
        for pattern in FITS_PATTERNS:
            for fp in sorted(self.data_dir.glob(f"**/{pattern}")):
                if date_str in fp.name or not date_str:
                    recs = self._read_fits_lc(str(fp))
                    records.extend(recs)
                    if records:
                        break
            if records:
                break
        return records

    def _load_from_lc_gz(self, date_str: str) -> list[SoLEXSReading]:
        """Read .lc.gz compressed FITS light curves."""
        records = []
        for pattern in LC_GZ_PATTERNS:
            for fp in sorted(self.data_dir.glob(f"**/{pattern}")):
                if date_str in fp.name or not date_str:
                    recs = self._read_fits_lc(str(fp), compressed=True)
                    records.extend(recs)
        return records

    def _load_from_netcdf(self, date_str: str) -> list[SoLEXSReading]:
        """Read netCDF SoLEXS files."""
        records = []
        for pattern in NC_PATTERNS:
            for fp in sorted(self.data_dir.glob(f"**/{pattern}")):
                if date_str in fp.name or not date_str:
                    recs = self._read_netcdf(str(fp))
                    records.extend(recs)
        return records

    def _read_fits_lc(self, filepath: str, compressed: bool = False) -> list[SoLEXSReading]:
        """
        Read a SoLEXS light curve from a FITS file.
        Handles both .fits and .fits.gz/.lc.gz formats.
        Tries multiple column name conventions from different SoLEXS L1 versions.
        """
        try:
            from astropy.io import fits as astropy_fits

            open_fn = gzip.open if (compressed or filepath.endswith(".gz")) else open

            if filepath.endswith(".gz"):
                with gzip.open(filepath, "rb") as gz_file:
                    raw = gz_file.read()
                hdul = astropy_fits.open(io.BytesIO(raw))
            else:
                hdul = astropy_fits.open(filepath, memmap=False)

            records = []
            with hdul:
                # Try each HDU for a light curve table
                for hdu_idx, hdu in enumerate(hdul):
                    if not hasattr(hdu, "data") or hdu.data is None:
                        continue
                    if not hasattr(hdu.data, "names"):
                        continue

                    names_upper = [n.upper() for n in hdu.data.names]

                    # Find TIME column
                    time_col = self._find_col(
                        hdu.data.names,
                        ["TIME", "T_START", "UNIX_TIME", "MET", "TIMEDEL"],
                    )
                    if time_col is None:
                        continue

                    # Find SDD2 count rate column (many naming variants)
                    rate2_col = self._find_col(
                        hdu.data.names,
                        ["RATE_SDD2", "COUNTS_SDD2", "RATE2", "CTS_SDD2",
                         "COUNT_RATE_SDD2", "SDD2_RATE", "SDD2_CTS",
                         "RATE", "COUNTS"],  # fallback
                    )
                    rate1_col = self._find_col(
                        hdu.data.names,
                        ["RATE_SDD1", "COUNTS_SDD1", "RATE1", "CTS_SDD1",
                         "COUNT_RATE_SDD1", "SDD1_RATE", "SDD1_CTS"],
                    )

                    if rate2_col is None:
                        continue

                    times_raw = np.array(hdu.data[time_col], dtype=np.float64).ravel()
                    rates_sdd2 = np.array(hdu.data[rate2_col], dtype=np.float64).ravel()
                    rates_sdd1 = (
                        np.array(hdu.data[rate1_col], dtype=np.float64).ravel()
                        if rate1_col else np.zeros_like(rates_sdd2)
                    )

                    # Quality flags
                    qual_col = self._find_col(hdu.data.names, ["QUALITY", "QUAL", "FLAG"])
                    quality = (
                        (np.array(hdu.data[qual_col]).ravel() == 0).astype(int)
                        if qual_col else np.ones(len(times_raw), dtype=int)
                    )

                    # Convert MET → UNIX if values look like MET (< 1e9)
                    epoch_unix = SOLEXS_EPOCH.timestamp()
                    if len(times_raw) > 0 and times_raw[0] < 1e9:
                        times_unix = times_raw + epoch_unix
                    else:
                        times_unix = times_raw

                    # Dead-time correction from header
                    hdr = hdu.header if hasattr(hdu, "header") else {}
                    dc2 = float(hdr.get("DEADCOR2", hdr.get("DEADCOR", 0.0)))
                    dc2 = min(max(dc2, 0.0), 0.99)
                    corr = 1.0 / max(1.0 - dc2, 0.01)

                    # Background subtraction
                    n_bg = min(300, len(rates_sdd2) // 4)
                    bg2 = float(np.nanmedian(rates_sdd2[:n_bg])) if n_bg > 0 else 0.0

                    for i, (t, r2, r1, q) in enumerate(
                        zip(times_unix, rates_sdd2, rates_sdd1, quality)
                    ):
                        if not np.isfinite(r2) or r2 < 0:
                            continue
                        records.append(SoLEXSReading(
                            time_unix=float(t),
                            counts_sdd2=float(max(r2 * corr - bg2, 0.0)),
                            counts_sdd1=float(max(r1, 0.0)),
                            quality=int(q),
                        ))

                    if records:
                        break  # found a valid HDU

            return records

        except Exception as exc:
            logger.debug("FITS read failed %s: %s", filepath, exc)
            return []

    def _read_netcdf(self, filepath: str) -> list[SoLEXSReading]:
        """Read SoLEXS data from netCDF format."""
        try:
            import netCDF4 as nc
            ds = nc.Dataset(filepath, "r")
            records = []

            # Find time and count-rate variables
            time_var = None
            for name in ["time", "TIME", "unix_time", "t_start"]:
                if name in ds.variables:
                    time_var = name
                    break

            rate_var = None
            for name in ["rate_sdd2", "RATE_SDD2", "counts_sdd2", "rate", "count_rate"]:
                if name in ds.variables:
                    rate_var = name
                    break

            if time_var is None or rate_var is None:
                ds.close()
                return []

            times = np.array(ds.variables[time_var][:], dtype=np.float64)
            rates = np.array(ds.variables[rate_var][:], dtype=np.float64)

            epoch_unix = SOLEXS_EPOCH.timestamp()
            if len(times) > 0 and times[0] < 1e9:
                times = times + epoch_unix

            n_bg = min(300, len(rates) // 4)
            bg = float(np.nanmedian(rates[:n_bg])) if n_bg > 0 else 0.0

            for t, r in zip(times, rates):
                if np.isfinite(r) and r >= 0:
                    records.append(SoLEXSReading(
                        time_unix=float(t),
                        counts_sdd2=float(max(r - bg, 0.0)),
                        counts_sdd1=0.0,
                        quality=1,
                    ))

            ds.close()
            return records

        except Exception as exc:
            logger.debug("netCDF read failed %s: %s", filepath, exc)
            return []

    @staticmethod
    def _find_col(names: list, candidates: list) -> Optional[str]:
        """Find the first matching column name (case-insensitive)."""
        names_lower = {n.lower(): n for n in names}
        for c in candidates:
            if c.lower() in names_lower:
                return names_lower[c.lower()]
        return None

    # ── scan_directory (legacy interface for triage/live) ─────────────────────

    def scan_directory(
        self,
        date_str: Optional[str] = None,
        obs_mode: Optional[str] = None,
    ) -> list[SoLEXSSpectrum]:
        """
        Scan for SoLEXS files and return metadata list.
        Legacy interface kept for live triage integration.
        """
        try:
            from astropy.io import fits as astropy_fits
        except ImportError:
            return []

        all_files = []
        for pattern in FITS_PATTERNS:
            all_files.extend(
                [str(p) for p in self.data_dir.glob(f"**/{pattern}")]
            )

        if date_str:
            all_files = [f for f in all_files if date_str in Path(f).name]

        spectra = []
        for fp in sorted(all_files):
            try:
                meta = self._parse_header(fp, astropy_fits)
                if meta.quality < SOLEXS_QUALITY_MIN:
                    continue
                if obs_mode and meta.obs_mode.upper() != obs_mode.upper():
                    continue
                spectra.append(meta)
            except Exception as exc:
                logger.debug("Header parse failed %s: %s", fp, exc)

        return spectra

    def extract_lightcurve(
        self,
        meta: SoLEXSSpectrum,
        subtract_background: bool = True,
        bg_window_s: float = 600.0,
    ) -> SoLEXSLightCurve:
        """Extract 1-second light curve from SoLEXS L2 FITS (legacy interface)."""
        from astropy.io import fits as astropy_fits

        with astropy_fits.open(meta.filepath) as hdul:
            lc_table = hdul[1].data
            times_met = lc_table["TIME"].astype(np.float64)
            epoch_unix = SOLEXS_EPOCH.timestamp()
            times_unix = times_met + epoch_unix
            rate_sdd1 = lc_table["RATE_SDD1"].astype(np.float64)
            rate_sdd2 = lc_table["RATE_SDD2"].astype(np.float64)
            quality = (
                (lc_table["QUALITY"] == 0).astype(np.int8)
                if "QUALITY" in lc_table.names
                else np.ones(len(rate_sdd2), dtype=np.int8)
            )

        dc2 = float(meta.deadcor_sdd2)
        dc2 = min(max(dc2, 0.0), 0.99)
        corr = 1.0 / max(1.0 - dc2, 0.01)
        rate_sdd2_corr = np.clip(rate_sdd2, 0, None) * corr
        rate_sdd1_corr = np.clip(rate_sdd1, 0, None) / max(
            1.0 - meta.deadcor_sdd1, 0.01
        )

        bg_sdd2 = 0.0
        if subtract_background and len(rate_sdd2_corr) > 30:
            n_bg = min(int(bg_window_s), len(rate_sdd2_corr) // 3)
            bg_sdd2 = float(np.median(rate_sdd2_corr[:n_bg]))
            rate_sdd2_corr = np.maximum(rate_sdd2_corr - bg_sdd2, 0.0)

        rate_sdd2_goes = rate_sdd2_corr * GOES_BAND_FRACTION_SDD2

        return SoLEXSLightCurve(
            times_unix=times_unix,
            rate_sdd1=rate_sdd1_corr,
            rate_sdd2=rate_sdd2_corr,
            rate_sdd2_goes=rate_sdd2_goes,
            quality_flags=quality,
            deadcor_sdd2=dc2,
            background_sdd2=bg_sdd2,
        )

    def load_spectrum(self, meta: SoLEXSSpectrum, detector: str = "SDD2") -> SoLEXSSpectrumData:
        """Load full spectral data from HDU[2] (for spectral fitting)."""
        from astropy.io import fits as astropy_fits

        with astropy_fits.open(meta.filepath) as hdul:
            spec_table = hdul[2].data
            energy_kev = spec_table["ENERGY_keV"].astype(np.float64)
            col = f"COUNTS_{detector.upper()}"
            if col not in spec_table.names:
                col = "COUNTS_SDD2"
            counts = spec_table[col].astype(np.float64)
            exposure_s = (
                float(spec_table["EXPOSURE_s"][0])
                if "EXPOSURE_s" in spec_table.names
                else meta.exposure_s
            )
            background = np.zeros_like(counts)
            if len(hdul) >= 4 and hdul[3].data is not None:
                bkg_col = f"BKG_{detector.upper()}"
                if bkg_col in hdul[3].data.names:
                    background = hdul[3].data[bkg_col].astype(np.float64)

        return SoLEXSSpectrumData(
            energy_kev=energy_kev,
            counts_sdd1=counts if detector == "SDD1" else np.zeros_like(counts),
            counts_sdd2=counts if detector == "SDD2" else np.zeros_like(counts),
            exposure_s=exposure_s,
            background_sdd2=background,
        )

    def _parse_header(self, filepath: str, fits) -> SoLEXSSpectrum:
        with fits.open(filepath, memmap=True) as hdul:
            hdr = hdul[0].header
            t_start = self._parse_isot(str(hdr.get("T_START", "")))
            t_stop = self._parse_isot(str(hdr.get("T_STOP", "")))
            return SoLEXSSpectrum(
                filepath=filepath,
                obs_id=str(hdr.get("SOLEXS_ID", Path(filepath).stem)),
                t_start_unix=t_start,
                t_stop_unix=t_stop,
                exposure_s=float(hdr.get("EXPOSURE", 0.0)),
                deadcor_sdd1=float(hdr.get("DEADCOR1", 0.0)),
                deadcor_sdd2=float(hdr.get("DEADCOR2", 0.0)),
                obs_mode=str(hdr.get("OBSMODE", "NORMAL")),
                quality=float(hdr.get("QVAL", 50.0)),
            )

    @staticmethod
    def _parse_isot(isot: str) -> float:
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
            try:
                dt = datetime.strptime(isot[:len(fmt)], fmt)
                return dt.replace(tzinfo=timezone.utc).timestamp()
            except Exception:
                continue
        return 0.0


# ── Convenience function for live triage integration ─────────────────────────

def stream_solexs_lightcurve(
    data_dir: str | Path,
    date_str: Optional[str] = None,
) -> "generator":
    """
    Generator: yields (unix_time, sdd2_counts_per_s) pairs from all SoLEXS
    files in data_dir, chronologically sorted.
    """
    loader = SoLEXSLoader(data_dir)
    if date_str:
        records = loader.load_day(date_str)
        records.sort(key=lambda r: r.time_unix)
        for r in records:
            if r.quality:
                yield r.time_unix, r.counts_sdd2
    else:
        spectra = loader.scan_directory()
        spectra.sort(key=lambda s: s.t_start_unix)
        for meta in spectra:
            try:
                lc = loader.extract_lightcurve(meta)
                for t, cts, q in zip(lc.times_unix, lc.rate_sdd2, lc.quality_flags):
                    if q:
                        yield float(t), float(cts)
            except Exception as exc:
                logger.warning("Failed to stream %s: %s", meta.filepath, exc)
