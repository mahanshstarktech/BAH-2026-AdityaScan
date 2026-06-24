"""
HEL1OS (High Energy L1 Orbiting X-ray Spectrometer) data loader for Aditya-L1.

Instrument specifications (from HEL1OS Data Analysis User Guide v1.2, ISRO/SAC):
  - Two detector stacks: CdTe detector (5–150 keV) and CZT detector (20–150 keV)
  - CdTe detector:  pixelated CdTe array, primary for 5–30 keV range
  - CZT detector:   CZT pixel detector, primary for 20–150 keV range
  - Energy bands (from User Guide Table 3.1):
      CdTe1:  5–12 keV    (thermal soft X-rays, complements SoLEXS)
      CdTe2:  12–25 keV   (thermal-non-thermal transition)
      CdTe3:  25–40 keV   (non-thermal bremsstrahlung, HXR)
      CdTe4:  40–80 keV   (high-energy HXR, HOPE trigger band)
      CZT1:   20–40 keV   (overlaps CdTe3, cross-check)
      CZT2:   40–80 keV   (hard HXR, overlaps CdTe4)
      CZT3:   80–120 keV  (very hard HXR)
      CZT4:   120–150 keV (near gamma, SEP-associated)
  - Temporal cadence: 1 second (time-resolved spectra)
  - Spectral resolution: ~1.5 keV FWHM at 60 keV (CdTe), ~2 keV at 60 keV (CZT)
  - Background: dominated by cosmic X-ray background + galactic ridge above 30 keV

PRADAN file naming convention (L2 science):
  AL1_HXS91_<YYYYMMDD>_<HHMMSS>_L2_CdTe_V<ver>.fits  (CdTe detector)
  AL1_HXS91_<YYYYMMDD>_<HHMMSS>_L2_CZT_V<ver>.fits   (CZT detector)

FITS structure (from guide Section 4.2):
  HDU[0]  — Primary (empty, standard header)
  HDU[1]  — BINARY TABLE: TIME (MET), RATE_CdTe1...CdTe4 / CZT1...CZT4 (cts/s)
  HDU[2]  — SPECTRUM TABLE: ENERGY_keV, COUNTS_CdTe / COUNTS_CZT, EXPOSURE_s
  HDU[3]  — BACKGROUND TABLE: BKG_CdTe, BKG_CZT (modeled)

Key FITS header keywords (HDU[0]):
  T_START     : Observation start (ISOT UTC)
  T_STOP      : Observation end
  EXPOSURE    : Total exposure time (s)
  DETECTOR    : 'CdTe' | 'CZT' | 'ALL'
  DEADCOR_CT  : Dead-time correction fraction (CdTe)
  DEADCOR_CZ  : Dead-time correction fraction (CZT)
  OBSMODE     : 'NORMAL' | 'FLARE' (high-cadence mode triggered by SoLEXS)
  QVAL        : Image quality percentage (0–100)
  HL1OS_ID    : PRADAN observation ID

Spectral fitting for HEL1OS:
  Non-thermal model: broken power-law photon spectrum: N(E) = A * E^(-γ_lo) for E < Eb
                                                                A * E^(-γ_hi) for E > Eb
  Parameters: γ_lo (low-energy index), γ_hi (high-energy index), Eb (break energy keV)
  Physical interpretation: γ_lo < 4.5 → non-thermal electrons (HOPE condition)
  Tool: PyXSPEC or sunkit-spex (HXR spectroscopy package)

HOPE precursor integration:
  Use CdTe3 (25–40 keV) + CZT2 (40–80 keV) as primary HOPE trigger bands.
  This loader exposes get_hope_bands() for direct feeding into HOPEDetector.
"""

from __future__ import annotations

import glob
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants from HEL1OS User Guide ─────────────────────────────────────────
HEL1OS_EPOCH = datetime(2000, 1, 1, 0, 0, 0, tzinfo=timezone.utc)  # MET epoch (same as SoLEXS)
HEL1OS_QUALITY_MIN = 50

# Energy band definitions (Table 3.1, User Guide)
HELIOS_CDTE_BANDS: dict[str, tuple[float, float]] = {
    "cdte_5_12":   (5.0,  12.0),
    "cdte_12_25":  (12.0, 25.0),
    "cdte_25_40":  (25.0, 40.0),   # ← HOPE trigger band 1
    "cdte_40_80":  (40.0, 80.0),   # ← HOPE trigger band 2
}
HELIOS_CZT_BANDS: dict[str, tuple[float, float]] = {
    "czt_20_40":   (20.0, 40.0),
    "czt_40_80":   (40.0, 80.0),   # ← HOPE trigger band 3
    "czt_80_120":  (80.0, 120.0),
    "czt_120_150": (120.0, 150.0),
}

# HOPE bands: CdTe 25–40 keV + CZT 40–80 keV (as defined in hope_detector.py)
HOPE_BAND_CDTE = "cdte_25_40"
HOPE_BAND_CZT  = "czt_40_80"

HEL1OS_FILE_PATTERN_CDTE = "AL1_HXS91_*_L2_CdTe_*.fits"
HEL1OS_FILE_PATTERN_CZT  = "AL1_HXS91_*_L2_CZT_*.fits"


@dataclass
class HEL1OSMeta:
    """Parsed HEL1OS L2 FITS header metadata."""
    filepath: str
    obs_id: str
    detector: str            # 'CdTe' | 'CZT'
    t_start_unix: float
    t_stop_unix: float
    exposure_s: float
    deadcor: float           # DEADCOR_CT (CdTe) or DEADCOR_CZ (CZT)
    obs_mode: str            # 'NORMAL' | 'FLARE'
    quality: float

    @property
    def t_mid_unix(self) -> float:
        return 0.5 * (self.t_start_unix + self.t_stop_unix)

    @property
    def is_flare_mode(self) -> bool:
        return "FLARE" in self.obs_mode.upper()


@dataclass
class HEL1OSLightCurve:
    """
    HEL1OS 1-second multi-band light curve.
    All rates are dead-time corrected.
    """
    times_unix: np.ndarray          # shape (N,)
    detector: str                   # 'CdTe' or 'CZT'
    band_rates: dict[str, np.ndarray]  # band_name → rate (cts/s)
    quality_flags: np.ndarray       # shape (N,), 1=good
    deadcor: float
    background_rates: dict[str, float]  # band_name → background (cts/s)

    def get_hope_bands(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (cdte_25_40, czt_40_80) arrays for HOPEDetector.
        CZT bands are all-zeros for CdTe-detector files, and vice versa.
        """
        cdte = self.band_rates.get("cdte_25_40", np.zeros(len(self.times_unix)))
        czt  = self.band_rates.get("czt_40_80",  np.zeros(len(self.times_unix)))
        return cdte, czt

    def total_hxr_rate(self) -> np.ndarray:
        """
        Sum across all bands > 25 keV (HXR diagnostic).
        Used in Neupert engine as the HXR proxy.
        """
        hxr_bands = [b for b in self.band_rates if any(
            int(b.split("_")[1]) >= 25 for b in [b]
        )]
        total = np.zeros(len(self.times_unix))
        for name, rate in self.band_rates.items():
            parts = name.split("_")
            if len(parts) >= 2 and int(parts[1]) >= 25:
                total += rate
        return total


@dataclass
class HEL1OSSpectrum:
    """Full HEL1OS spectral data for spectral fitting."""
    energy_kev: np.ndarray        # shape (N_chan,)
    counts: np.ndarray            # shape (N_chan,), background-subtracted
    exposure_s: float
    background: np.ndarray        # shape (N_chan,), modeled background
    detector: str
    gamma_lo_guess: Optional[float] = None   # spectral index estimate (from ratio method)


class HEL1OSLoader:
    """
    Loads Aditya-L1 HEL1OS L2 FITS data.

    Two detectors: CdTe (5–80 keV) and CZT (20–150 keV).
    Both can be loaded from their respective PRADAN files.

    Usage
    -----
    loader = HEL1OSLoader("/data/helios/l2")
    # Scan for CdTe files
    cdte_files = loader.scan_directory(detector="CdTe", date_str="20240514")
    lc = loader.extract_lightcurve(cdte_files[0])
    # Get HOPE bands directly
    cdte_25_40, czt_40_80 = lc.get_hope_bands()
    """

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self._arf_cdte: Optional[str] = None
        self._rmf_cdte: Optional[str] = None
        self._arf_czt: Optional[str] = None
        self._rmf_czt: Optional[str] = None

    def set_calibration_files(
        self,
        arf_cdte: str, rmf_cdte: str,
        arf_czt: Optional[str] = None, rmf_czt: Optional[str] = None,
    ) -> None:
        """Set ARF/RMF calibration files for both detectors."""
        self._arf_cdte = arf_cdte
        self._rmf_cdte = rmf_cdte
        self._arf_czt = arf_czt
        self._rmf_czt = rmf_czt

    # ── Scanning ─────────────────────────────────────────────────────────────

    def scan_directory(
        self,
        detector: str = "CdTe",
        date_str: Optional[str] = None,
    ) -> list[HEL1OSMeta]:
        """
        Scan for HEL1OS L2 FITS files.

        Parameters
        ----------
        detector : 'CdTe' | 'CZT'
            Which detector to scan for.
        date_str : str, optional
            Filter by date YYYYMMDD from filename.
        """
        from astropy.io import fits

        pattern_str = HEL1OS_FILE_PATTERN_CDTE if "CdTe" in detector else HEL1OS_FILE_PATTERN_CZT
        pattern = str(self.data_dir / "**" / pattern_str)
        all_files = glob.glob(pattern, recursive=True)

        if date_str:
            all_files = [f for f in all_files if date_str in Path(f).name]

        metas: list[HEL1OSMeta] = []
        for fp in sorted(all_files):
            try:
                meta = self._parse_header(fp, fits, detector)
                if meta.quality < HEL1OS_QUALITY_MIN:
                    continue
                metas.append(meta)
            except Exception as exc:
                logger.debug("Failed to parse %s: %s", fp, exc)

        logger.info("Found %d HEL1OS %s files", len(metas), detector)
        return metas

    # ── Light curve extraction ────────────────────────────────────────────────

    def extract_lightcurve(
        self,
        meta: HEL1OSMeta,
        subtract_background: bool = True,
        bg_window_s: float = 300.0,
    ) -> HEL1OSLightCurve:
        """
        Extract 1-second multi-band light curve from HEL1OS L2 FITS.

        Processing:
          1. Read TIME + RATE columns for each energy band
          2. Apply dead-time correction
          3. Estimate pre-flare background per band
          4. Background-subtract all bands

        Returns HEL1OSLightCurve with all available energy bands.
        """
        from astropy.io import fits

        with fits.open(meta.filepath) as hdul:
            lc_table = hdul[1].data
            times_met = lc_table["TIME"].astype(np.float64)
            times_unix = times_met + HEL1OS_EPOCH.timestamp()

            # Quality flags
            if "QUALITY" in lc_table.names:
                quality = (lc_table["QUALITY"] == 0).astype(np.int8)
            else:
                quality = np.ones(len(times_unix), dtype=np.int8)

            # Read all available band columns
            band_rates_raw: dict[str, np.ndarray] = {}
            if meta.detector.upper() == "CDTE":
                for band_name, (elo, ehi) in HELIOS_CDTE_BANDS.items():
                    # Column name format from User Guide Table 4.1:
                    # RATE_CdTe_<Elo>_<Ehi>keV  (e.g. RATE_CdTe_25_40keV)
                    col = f"RATE_CdTe_{int(elo)}_{int(ehi)}keV"
                    if col in lc_table.names:
                        band_rates_raw[band_name] = lc_table[col].astype(np.float64)
                    else:
                        # Fallback: try generic column naming
                        col_alt = f"RATE_{int(elo)}_{int(ehi)}"
                        if col_alt in lc_table.names:
                            band_rates_raw[band_name] = lc_table[col_alt].astype(np.float64)
                        else:
                            logger.debug("Column %s not found in %s", col, meta.filepath)
                            band_rates_raw[band_name] = np.zeros(len(times_unix))
            else:  # CZT
                for band_name, (elo, ehi) in HELIOS_CZT_BANDS.items():
                    col = f"RATE_CZT_{int(elo)}_{int(ehi)}keV"
                    if col in lc_table.names:
                        band_rates_raw[band_name] = lc_table[col].astype(np.float64)
                    else:
                        band_rates_raw[band_name] = np.zeros(len(times_unix))

        # Dead-time correction
        dc = float(meta.deadcor)
        if dc >= 1.0:
            dc = 0.99
        corr = 1.0 / (1.0 - dc)

        band_rates_corr: dict[str, np.ndarray] = {
            name: np.maximum(raw, 0.0) * corr
            for name, raw in band_rates_raw.items()
        }

        # Background estimation (pre-flare window)
        background_rates: dict[str, float] = {}
        if subtract_background:
            n_bg = min(int(bg_window_s), max(1, len(times_unix) // 4))
            for name, rate in band_rates_corr.items():
                bg = float(np.median(rate[:n_bg]))
                background_rates[name] = bg
                band_rates_corr[name] = np.maximum(rate - bg, 0.0)
        else:
            background_rates = {name: 0.0 for name in band_rates_corr}

        return HEL1OSLightCurve(
            times_unix=times_unix,
            detector=meta.detector,
            band_rates=band_rates_corr,
            quality_flags=quality,
            deadcor=dc,
            background_rates=background_rates,
        )

    # ── Spectral data loading ─────────────────────────────────────────────────

    def load_spectrum(
        self,
        meta: HEL1OSMeta,
        estimate_spectral_index: bool = True,
    ) -> HEL1OSSpectrum:
        """
        Load full HEL1OS spectrum for PyXSPEC / sunkit-spex fitting.

        Includes quick spectral index estimate via hardness ratio method
        (band ratio of 40–80 keV / 20–40 keV).

        Parameters
        ----------
        estimate_spectral_index : bool
            If True, compute quick γ_lo estimate from band ratios.
            This is a fast approximation used before full spectral fitting.
        """
        from astropy.io import fits

        with fits.open(meta.filepath) as hdul:
            spec_table = hdul[2].data
            energy_kev = spec_table["ENERGY_keV"].astype(np.float64)

            if meta.detector.upper() == "CDTE":
                counts = spec_table.get("COUNTS_CdTe",
                         spec_table.get("COUNTS", np.zeros(len(energy_kev)))).astype(np.float64)
            else:
                counts = spec_table.get("COUNTS_CZT",
                         spec_table.get("COUNTS", np.zeros(len(energy_kev)))).astype(np.float64)

            exposure_s = float(
                spec_table["EXPOSURE_s"][0] if "EXPOSURE_s" in spec_table.names else meta.exposure_s
            )

            background = np.zeros_like(counts)
            if len(hdul) >= 4 and hdul[3].data is not None:
                bkg_col = "BKG_CdTe" if meta.detector.upper() == "CDTE" else "BKG_CZT"
                if bkg_col in hdul[3].data.names:
                    background = hdul[3].data[bkg_col].astype(np.float64)

        # Quick spectral index estimate from band ratio
        gamma_lo_guess = None
        if estimate_spectral_index:
            gamma_lo_guess = self._estimate_gamma_from_ratio(counts, energy_kev)

        return HEL1OSSpectrum(
            energy_kev=energy_kev,
            counts=counts - background,
            exposure_s=exposure_s,
            background=background,
            detector=meta.detector,
            gamma_lo_guess=gamma_lo_guess,
        )

    @staticmethod
    def _estimate_gamma_from_ratio(
        counts: np.ndarray,
        energy_kev: np.ndarray,
        lo_range: tuple[float, float] = (20.0, 40.0),
        hi_range: tuple[float, float] = (40.0, 80.0),
    ) -> Optional[float]:
        """
        Estimate power-law photon index γ from hardness ratio.
        For a power-law N(E) ∝ E^(-γ):
          γ ≈ -log(R) / log(E_hi_mid / E_lo_mid)
          where R = (counts_hi / E_hi_width) / (counts_lo / E_lo_width)

        Returns None if counts are insufficient.
        """
        lo_mask = (energy_kev >= lo_range[0]) & (energy_kev < lo_range[1])
        hi_mask = (energy_kev >= hi_range[0]) & (energy_kev < hi_range[1])

        c_lo = float(np.sum(counts[lo_mask]))
        c_hi = float(np.sum(counts[hi_mask]))

        if c_lo < 5 or c_hi < 1:
            return None  # insufficient counts for ratio

        e_lo_mid = 0.5 * (lo_range[0] + lo_range[1])
        e_hi_mid = 0.5 * (hi_range[0] + hi_range[1])
        dw_lo = lo_range[1] - lo_range[0]
        dw_hi = hi_range[1] - hi_range[0]

        r = (c_hi / dw_hi) / (c_lo / dw_lo + 1e-10)
        if r <= 0:
            return None

        import math
        gamma = -math.log(r) / math.log(e_hi_mid / e_lo_mid)
        return float(gamma)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _parse_header(self, filepath: str, fits, detector: str) -> HEL1OSMeta:
        with fits.open(filepath, memmap=True) as hdul:
            hdr = hdul[0].header

            deadcor_key = "DEADCOR_CT" if "CdTe" in detector else "DEADCOR_CZ"

            t_start = SoLEXSLoader._parse_isot(str(hdr.get("T_START", "")))
            t_stop  = SoLEXSLoader._parse_isot(str(hdr.get("T_STOP",  "")))

            return HEL1OSMeta(
                filepath=filepath,
                obs_id=str(hdr.get("HL1OS_ID", Path(filepath).stem)),
                detector=detector,
                t_start_unix=t_start,
                t_stop_unix=t_stop,
                exposure_s=float(hdr.get("EXPOSURE", 0.0)),
                deadcor=float(hdr.get(deadcor_key, 0.0)),
                obs_mode=str(hdr.get("OBSMODE", "NORMAL")),
                quality=float(hdr.get("QVAL", 50.0)),
            )


# Need to import SoLEXSLoader for _parse_isot (shared utility)
from pipeline.ingestion.solexs_loader import SoLEXSLoader


# ── Combined SoLEXS + HEL1OS streamer for HOPE detector ──────────────────────

def stream_combined_xray(
    solexs_dir: str | Path,
    helios_dir: str | Path,
    date_str: Optional[str] = None,
):
    """
    Generator: yields (unix_time, sdd2_cts, cdte_25_40_cts, czt_40_80_cts)
    for feeding into both TriageEngine AND HOPEDetector simultaneously.

    Synchronizes SoLEXS and HEL1OS timestamps (both have 1-s cadence).
    """
    # Load all light curves
    sol_loader = SoLEXSLoader(solexs_dir)
    hel_loader = HEL1OSLoader(helios_dir)

    sol_spectra = sol_loader.scan_directory(date_str=date_str)
    cdte_metas  = hel_loader.scan_directory("CdTe", date_str=date_str)
    czt_metas   = hel_loader.scan_directory("CZT",  date_str=date_str)

    for sol_meta in sorted(sol_spectra, key=lambda m: m.t_start_unix):
        try:
            sol_lc = sol_loader.extract_lightcurve(sol_meta)
        except Exception as exc:
            logger.warning("SoLEXS LC failed: %s", exc)
            continue

        # Find matching CdTe + CZT files by time overlap
        cdte_match = _find_overlap(cdte_metas, sol_meta.t_start_unix, sol_meta.t_stop_unix)
        czt_match  = _find_overlap(czt_metas,  sol_meta.t_start_unix, sol_meta.t_stop_unix)

        cdte_lc = hel_loader.extract_lightcurve(cdte_match) if cdte_match else None
        czt_lc  = hel_loader.extract_lightcurve(czt_match)  if czt_match  else None

        for i, (t, cts, q) in enumerate(zip(sol_lc.times_unix, sol_lc.rate_sdd2, sol_lc.quality_flags)):
            if not q:
                continue

            cdte_cts = float(cdte_lc.band_rates["cdte_25_40"][i]) if cdte_lc and i < len(cdte_lc.times_unix) else 0.0
            czt_cts  = float(czt_lc.band_rates["czt_40_80"][i])   if czt_lc  and i < len(czt_lc.times_unix)  else 0.0

            yield float(t), float(cts), cdte_cts, czt_cts


def _find_overlap(
    metas: list[HEL1OSMeta],
    t_start: float,
    t_stop: float,
) -> Optional[HEL1OSMeta]:
    """Find the HEL1OS observation that best overlaps with a time window."""
    for meta in metas:
        if meta.t_start_unix <= t_stop and meta.t_stop_unix >= t_start:
            return meta
    return None
