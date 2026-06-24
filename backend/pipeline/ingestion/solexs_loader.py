"""
SoLEXS (Solar Low Energy X-ray Spectrometer) data loader for Aditya-L1.

Instrument specifications (from SoLEXS Data Analysis User Guide, ISRO/SAC):
  - Two detectors: SDD1 (primary, lower background) and SDD2 (thicker, higher area)
  - Energy range:  2.8 – 30 keV (SDD1), 4 – 30 keV (SDD2)
  - GOES-equivalent channel: ~1–8 Å equivalent ≈ 1.55–12.4 keV (SDD2 is primary)
  - Spectral resolution: ~150 eV at 5.9 keV (Mn Kα from onboard Fe-55 source)
  - Temporal cadence: 1 second (L1 raw spectra), 60 s average (L2 science)
  - Dead time correction: required at >M1 class (count rates saturate SDD1)
  - Energy channels: 4096 channels, 1 channel ≈ 7.3 eV (after gain calibration)

PRADAN file naming convention (L2 science):
  AL1_SLX91_<YYYYMMDD>_<HHMMSS>_L2_V<version>.fits
  AL1_SLX91_<YYYYMMDD>_<HHMMSS>_L2_V<version>.nc   (netCDF variant)

FITS structure (from analysis guide Table 2.1):
  HDU[0]  — Primary (empty, header only)
  HDU[1]  — BINARY TABLE: TIME (s since epoch), RATE_SDD1 (cts/s), RATE_SDD2 (cts/s)
  HDU[2]  — SPECTRUM TABLE: ENERGY_keV, COUNTS_SDD1, COUNTS_SDD2, EXPOSURE_s
  HDU[3]  — BACKGROUND TABLE: BKG_SDD1, BKG_SDD2 (modeled background)

Key FITS header keywords (HDU[0]):
  T_START   : Observation start (ISOT UTC string)
  T_STOP    : Observation end
  EXPOSURE  : Total exposure time (s)
  DEADCOR1  : Dead time correction fraction SDD1
  DEADCOR2  : Dead time correction fraction SDD2
  FILTER    : Filter setting
  OBSMODE   : Observation mode ('NORMAL' | 'FLARE' — cadence changes)
  QVAL      : Quality value (0=bad, 100=perfect)
  SOLEXS_ID : PRADAN observation ID

Background subtraction strategy:
  - Pre-flare window: 10 minutes before trigger (GOES proxy > B5)
  - Fit linear trend to background counts → subtract from flare spectrum
  - Use HDU[3] BKG arrays if available (modeled background from quiescent period)

Spectral fitting (called by spectral_fitter.py, not here):
  - Model: VAPEC (variable abundance) + power-law (for non-thermal tail)
  - Tool: xspec.py wrapper (PyXSPEC) or sherpa
  - Parameters fit: T_MK (plasma temperature), EM (emission measure), χ²_red
  - Requires: ARF (effective area response), RMF (redistribution matrix function)
    → Available from PRADAN calibration files: AL1_SLX91_ARF_V01.fits, *_RMF_V01.fits

Light curve extraction (primary output for Tier 1 triage):
  - Use RATE_SDD2 column from HDU[1] (total count rate, 1-s cadence)
  - Apply dead-time correction: corrected = rate / (1 - DEADCOR2)
  - This feeds directly into TriageEngine.evaluate()
"""

from __future__ import annotations

import glob
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants from SoLEXS instrument guide ────────────────────────────────────
SOLEXS_N_CHANNELS = 4096
SOLEXS_EV_PER_CHANNEL = 7.3         # eV/channel (approximate, use gain from header)
SOLEXS_EPOCH = datetime(2000, 1, 1, 0, 0, 0, tzinfo=timezone.utc)  # MET epoch
SOLEXS_DEAD_TIME_THRESHOLD = 0.1    # warn if dead-time correction > 10%
SOLEXS_QUALITY_MIN = 50             # minimum QVAL to accept (0–100 scale)
SOLEXS_GOES_CHANNEL_KMIN = 1.55     # keV lower bound for GOES-equivalent band
SOLEXS_GOES_CHANNEL_KMAX = 12.4     # keV upper bound for GOES-equivalent band

# PRADAN filename pattern (adjust to actual PRADAN glob once data arrives)
SOLEXS_FILE_PATTERN = "AL1_SLX91_*_L2_*.fits"


@dataclass
class SoLEXSSpectrum:
    """
    One SoLEXS L2 spectrum observation.
    Minimal parsed representation — actual pixel arrays loaded on demand.
    """
    filepath: str
    obs_id: str             # SOLEXS_ID from header
    t_start_unix: float     # observation start (UNIX)
    t_stop_unix: float      # observation stop (UNIX)
    exposure_s: float       # exposure time in seconds
    deadcor_sdd1: float     # dead-time correction fraction for SDD1
    deadcor_sdd2: float     # dead-time correction fraction for SDD2
    obs_mode: str           # 'NORMAL' | 'FLARE'
    quality: float          # QVAL (0–100)

    @property
    def t_mid_unix(self) -> float:
        return 0.5 * (self.t_start_unix + self.t_stop_unix)

    @property
    def is_flare_mode(self) -> bool:
        return "FLARE" in self.obs_mode.upper()

    @property
    def dead_time_ok(self) -> bool:
        return self.deadcor_sdd2 < SOLEXS_DEAD_TIME_THRESHOLD


@dataclass
class SoLEXSLightCurve:
    """
    SoLEXS 1-second light curve extracted from L2 FITS.
    Corresponds to RATE_SDD1, RATE_SDD2 columns in HDU[1].
    Dead-time corrected and background-subtracted.
    """
    times_unix: np.ndarray       # shape (N,), UNIX timestamps
    rate_sdd1: np.ndarray        # shape (N,), cts/s dead-time corrected
    rate_sdd2: np.ndarray        # shape (N,), cts/s dead-time corrected (primary)
    rate_sdd2_goes: np.ndarray   # shape (N,), count rate in GOES-equiv band
    quality_flags: np.ndarray    # shape (N,), 1=good, 0=bad
    deadcor_sdd2: float          # scalar dead-time correction applied
    background_sdd2: float       # mean background (cts/s) subtracted


@dataclass
class SoLEXSSpectrumData:
    """Full spectral data loaded from HDU[2]."""
    energy_kev: np.ndarray       # shape (N_chan,)
    counts_sdd1: np.ndarray      # shape (N_chan,)
    counts_sdd2: np.ndarray      # shape (N_chan,)
    exposure_s: float
    background_sdd2: np.ndarray  # shape (N_chan,), from HDU[3] or pre-flare estimate


class SoLEXSLoader:
    """
    Loads Aditya-L1 SoLEXS L2 FITS data from PRADAN.

    Two main use cases:
    1. Light curve extraction → feeds TriageEngine (always-on Tier 1)
    2. Spectrum loading → feeds spectral_fitter.py (ELEVATED/ACTIVE mode only)

    Usage
    -----
    loader = SoLEXSLoader(data_dir="/data/solexs/l2")
    spectra = loader.scan_directory(date_str="20240514")
    lc = loader.extract_lightcurve(spectra[0])
    spec_data = loader.load_spectrum(spectra[0])
    """

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self._arf_path: Optional[str] = None
        self._rmf_path: Optional[str] = None

    def set_calibration_files(self, arf_path: str, rmf_path: str) -> None:
        """
        Set ARF and RMF calibration file paths.
        Required for spectral fitting (not needed for light curve only).

        PRADAN calibration files:
          ARF: AL1_SLX91_ARF_SDD2_V01.fits  (effective area response)
          RMF: AL1_SLX91_RMF_SDD2_V01.fits  (redistribution matrix)
        """
        self._arf_path = arf_path
        self._rmf_path = rmf_path

    # ── Scanning ─────────────────────────────────────────────────────────────

    def scan_directory(
        self,
        date_str: Optional[str] = None,
        obs_mode: Optional[str] = None,
    ) -> list[SoLEXSSpectrum]:
        """
        Scan data directory for SoLEXS L2 FITS files, parse headers.
        Returns list of SoLEXSSpectrum metadata (no pixel data loaded).

        Parameters
        ----------
        date_str : str, optional
            Filter by date (YYYYMMDD) from filename.
        obs_mode : str, optional
            Filter by OBSMODE header value: 'NORMAL' or 'FLARE'.
        """
        from astropy.io import fits

        pattern = str(self.data_dir / "**" / SOLEXS_FILE_PATTERN)
        all_files = glob.glob(pattern, recursive=True)

        if date_str:
            all_files = [f for f in all_files if date_str in Path(f).name]

        spectra: list[SoLEXSSpectrum] = []
        for fp in sorted(all_files):
            try:
                meta = self._parse_header(fp, fits)
                if meta.quality < SOLEXS_QUALITY_MIN:
                    logger.debug("Skipping low-quality SoLEXS file: %s (QVAL=%.0f)", fp, meta.quality)
                    continue
                if obs_mode and meta.obs_mode.upper() != obs_mode.upper():
                    continue
                spectra.append(meta)
            except Exception as exc:
                logger.debug("Failed to parse %s: %s", fp, exc)

        logger.info("Found %d SoLEXS L2 files (date=%s, mode=%s)", len(spectra), date_str, obs_mode)
        return spectra

    # ── Light curve extraction ────────────────────────────────────────────────

    def extract_lightcurve(
        self,
        meta: SoLEXSSpectrum,
        subtract_background: bool = True,
        bg_window_s: float = 600.0,
    ) -> SoLEXSLightCurve:
        """
        Extract 1-second light curve from SoLEXS L2 FITS.

        Processing steps:
          1. Read HDU[1] TIME + RATE_SDD1 + RATE_SDD2 columns
          2. Apply dead-time correction: rate_corrected = rate / (1 - deadcor)
          3. Estimate pre-flare background (first bg_window_s seconds if quiet)
          4. Subtract background from corrected rate
          5. Extract GOES-equivalent band (1.55–12.4 keV) count rate

        Parameters
        ----------
        subtract_background : bool
            If True, subtract estimated pre-flare background.
        bg_window_s : float
            Duration (seconds) of pre-flare baseline used for background.
        """
        from astropy.io import fits

        with fits.open(meta.filepath) as hdul:
            # HDU[1]: time series table
            lc_table = hdul[1].data
            times_met = lc_table["TIME"].astype(np.float64)   # seconds since EPOCH

            # Convert MET to UNIX
            epoch_unix = SOLEXS_EPOCH.timestamp()
            times_unix = times_met + epoch_unix

            # Raw count rates
            rate_sdd1 = lc_table["RATE_SDD1"].astype(np.float64)
            rate_sdd2 = lc_table["RATE_SDD2"].astype(np.float64)

            # Quality flags (if present)
            if "QUALITY" in lc_table.names:
                quality = (lc_table["QUALITY"] == 0).astype(np.int8)  # 0 = good in OGIP
            else:
                quality = np.ones(len(rate_sdd2), dtype=np.int8)

        # Dead-time correction (scalar per observation, from header)
        # corrected = raw / (1 - dead_time_fraction)
        dc2 = float(meta.deadcor_sdd2)
        if dc2 >= 1.0:
            logger.warning("Dead-time fraction %.3f >= 1.0 — clamping to 0.99", dc2)
            dc2 = 0.99
        correction_factor = 1.0 / (1.0 - dc2)
        rate_sdd2_corr = np.clip(rate_sdd2, 0, None) * correction_factor
        rate_sdd1_corr = np.clip(rate_sdd1, 0, None) / max(1.0 - meta.deadcor_sdd1, 0.01)

        if meta.deadcor_sdd2 > SOLEXS_DEAD_TIME_THRESHOLD:
            logger.warning(
                "High dead-time correction: SDD2=%.1f%% — "
                "consider using SDD1 for this observation (may be saturated)",
                meta.deadcor_sdd2 * 100
            )

        # Background estimation (pre-flare quiet window)
        bg_sdd2 = 0.0
        if subtract_background and len(rate_sdd2_corr) > 30:
            n_bg = min(int(bg_window_s), len(rate_sdd2_corr) // 3)
            bg_sdd2 = float(np.median(rate_sdd2_corr[:n_bg]))
            rate_sdd2_corr = np.maximum(rate_sdd2_corr - bg_sdd2, 0.0)

        # GOES-equivalent band count rate
        # Without spectral info: approximate as fraction of total SDD2 counts
        # Using known SoLEXS SDD2 channel sensitivity (energy-weighted fraction)
        # Provisional: GOES-equiv band ≈ 65% of total SDD2 count rate
        # (Update this with actual ARF integration once calibration files arrive)
        GOES_BAND_FRACTION_SDD2 = 0.65
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

    # ── Spectral data loading ─────────────────────────────────────────────────

    def load_spectrum(
        self,
        meta: SoLEXSSpectrum,
        detector: str = "SDD2",
    ) -> SoLEXSSpectrumData:
        """
        Load full spectral data from HDU[2] for spectral fitting.

        Returns SoLEXSSpectrumData ready for PyXSPEC / Sherpa.
        detector: 'SDD1' or 'SDD2' (SDD2 preferred for most science)

        Note: For X-class flares, SDD1 may be less saturated than SDD2.
        Use detector='SDD1' if SDD2 deadcor > 30%.
        """
        from astropy.io import fits

        with fits.open(meta.filepath) as hdul:
            spec_table = hdul[2].data
            energy_kev = spec_table["ENERGY_keV"].astype(np.float64)

            col = f"COUNTS_{detector.upper()}"
            if col not in spec_table.names:
                col = "COUNTS_SDD2"  # fallback
            counts = spec_table[col].astype(np.float64)
            exposure_s = float(spec_table["EXPOSURE_s"][0]) if "EXPOSURE_s" in spec_table.names else meta.exposure_s

            # Background from HDU[3] if available
            background = np.zeros_like(counts)
            if len(hdul) >= 4 and hdul[3].data is not None:
                bkg_table = hdul[3].data
                bkg_col = f"BKG_{detector.upper()}"
                if bkg_col in bkg_table.names:
                    background = bkg_table[bkg_col].astype(np.float64)

        return SoLEXSSpectrumData(
            energy_kev=energy_kev,
            counts_sdd1=counts if detector == "SDD1" else np.zeros_like(counts),
            counts_sdd2=counts if detector == "SDD2" else np.zeros_like(counts),
            exposure_s=exposure_s,
            background_sdd2=background,
        )

    def get_xspec_pha_args(self, meta: SoLEXSSpectrum) -> dict:
        """
        Return keyword arguments for loading this spectrum into PyXSPEC.

        Usage in spectral_fitter.py:
            import xspec
            s = xspec.Spectrum(**loader.get_xspec_pha_args(meta))
            s.response = arf_path
            s.response.rmf = rmf_path
        """
        return {
            "pha": meta.filepath,
            "arf": self._arf_path,
            "rmf": self._rmf_path,
            "backFile": None,  # background subtracted inline
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _parse_header(self, filepath: str, fits) -> SoLEXSSpectrum:
        """Parse SoLEXS L2 FITS header into SoLEXSSpectrum metadata."""
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
        """Parse an ISO 8601 UTC string to UNIX timestamp. Returns 0.0 on failure."""
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
            try:
                dt = datetime.strptime(isot[:len(fmt)].replace(".", "."), fmt)
                return dt.replace(tzinfo=timezone.utc).timestamp()
            except Exception:
                continue
        return 0.0


# ── Convenience function for Tier 1 triage integration ───────────────────────

def stream_solexs_lightcurve(
    data_dir: str | Path,
    date_str: Optional[str] = None,
) -> "generator":
    """
    Generator: yields (unix_time, sdd2_counts_per_s) pairs from all SoLEXS
    L2 files in data_dir, chronologically sorted. Designed for streaming
    into TriageEngine.evaluate() in the triage worker.

    Usage
    -----
    engine = TriageEngine()
    for t, cts in stream_solexs_lightcurve("/data/solexs/l2", "20240514"):
        sample = engine.evaluate(t, cts)
        state_machine.update(z_score=sample.z_score, goes_flux=sample.goes_flux_proxy)
    """
    loader = SoLEXSLoader(data_dir)
    spectra = loader.scan_directory(date_str=date_str)
    spectra.sort(key=lambda s: s.t_start_unix)

    for meta in spectra:
        try:
            lc = loader.extract_lightcurve(meta)
            for t, cts, q in zip(lc.times_unix, lc.rate_sdd2, lc.quality_flags):
                if q:
                    yield float(t), float(cts)
        except Exception as exc:
            logger.warning("Failed to extract light curve from %s: %s", meta.filepath, exc)
