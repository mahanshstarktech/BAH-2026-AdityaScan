"""
ASPEX-SWIS data loader for Aditya-L1.

Grounded in: ASPEX-SWIS User Manual, PRL/ISRO (2024)
  - ASPEX = Aditya Solar Wind Particle EXperiment
  - SWIS = Solar Wind Ion Spectrometer (sub-instrument of ASPEX)
  - Two sensors: THA-1 (head 1) and THA-2 (head 2)
  - Measures solar wind ions, 50 energy channels each
  - L2 data products (science-ready, CDF v3.9):
      THA-1 flux:  AL1_ASW91_L2_TH1_YYYYMMDD_<obsId>_Vmn.cdf
      THA-2 flux:  AL1_ASW91_L2_TH2_YYYYMMDD_<obsId>_Vmn.cdf
      Bulk params: AL1_ASW91_L2_BLK_YYYYMMDD_<obsId>_Vmn.cdf
  - Bulk parameters (from non-linear fitting of core solar wind flux):
      proton number density, proton temperature, solar wind bulk speed
  - CDF files compressed with CDF built-in Gzip level 2
  - Read using cdflib (Python) — all metadata in file itself
  - Cadence: bulk params file has variable cadence (multiple cycles merged)
  - Spatial: spacecraft position in GSE written in all L2 files
  - Status (as of manual): Aug-Sep 2024 data released; bulk params Aug 2024 only
  - PAPA (particle analyser) is a separate instrument — do not confuse with ASPEX

Reference: ASPEX-SWIS User Manual, PRL, ISRO 2024
"""

from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
# SWIS measures solar wind ions. Key physics: Alfvén speed, proton plasma beta,
# charge-to-mass selection for solar energetic particles (SEPs)
SWIS_N_ENERGY_CHANNELS = 50          # 50 energy channels per THA head
SWIS_ENERGY_RANGE_EV = (14.63, 3000.0)  # eV — from PAPA manual Table 8 (SWIS compatible)


@dataclass
class SWISBulkParams:
    """
    Bulk parameters from ASPEX-SWIS L2 BLK file.
    Derived from non-linear fitting of ion flux spectra (core component).
    """
    unix_time: float        # observation time, UNIX seconds (UTC)
    proton_density: float   # cm⁻³  (number density)
    proton_temperature: float  # Kelvin (core temperature from fitting)
    proton_speed: float     # km/s  (bulk velocity)
    density_err: float      # uncertainty in density
    temperature_err: float  # uncertainty in temperature
    speed_err: float        # uncertainty in speed
    # Spacecraft position in GSE (km)
    x_gse_km: float = 0.0
    y_gse_km: float = 0.0
    z_gse_km: float = 0.0

    @property
    def dynamic_pressure_nPa(self) -> float:
        """Solar wind dynamic pressure P_dyn = 0.5 * m_p * n * v² in nPa."""
        m_p_kg = 1.6726e-27  # proton mass kg
        n_si = self.proton_density * 1e6      # cm⁻³ → m⁻³
        v_si = self.proton_speed * 1e3        # km/s → m/s
        return float(0.5 * m_p_kg * n_si * v_si**2 * 1e9)  # Pa → nPa

    @property
    def alfven_speed_kms(self) -> float:
        """
        Nominal Alfvén speed (needs |B| from MAG).
        Returns np.nan — requires MAG coupling.
        Computed in fusion layer using MAG + SWIS simultaneously.
        """
        return float("nan")


@dataclass
class SWISFluxSpectrum:
    """
    Differential energy flux spectrum from one THA head (TH1 or TH2).
    Used for SEP (Solar Energetic Particle) detection when flux
    in high-energy channels (>1 keV) rises anomalously.
    """
    unix_time: float            # observation start time (UTC UNIX)
    energy_bins_eV: np.ndarray  # shape (50,) central energies per channel
    flux: np.ndarray            # shape (50,) differential energy flux (counts/cm²/s/eV/sr or similar)
    flux_err: np.ndarray        # shape (50,) flux uncertainty
    head: str                   # "THA1" or "THA2"

    def sep_flag(
        self,
        energy_threshold_eV: float = 1000.0,
        flux_sigma: float = 3.0,
        background_flux: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Simple SEP (Solar Energetic Particle) onset flag.
        Returns True if flux in channels above energy_threshold_eV
        exceeds background by flux_sigma standard deviations.

        Parameters
        ----------
        energy_threshold_eV : float
            Only consider channels above this energy (default 1 keV).
        flux_sigma : float
            Sigma threshold for anomaly detection.
        background_flux : np.ndarray, optional
            Background flux array (same shape as self.flux).
            If None, uses a flat baseline — caller should pass rolling mean.
        """
        high_energy_mask = self.energy_bins_eV >= energy_threshold_eV
        if not np.any(high_energy_mask):
            return False

        flux_he = self.flux[high_energy_mask]
        err_he = self.flux_err[high_energy_mask]

        if background_flux is not None:
            bg_he = background_flux[high_energy_mask]
        else:
            # Fallback: quiet-Sun estimate. Flag if any channel 5× average
            bg_he = np.full_like(flux_he, np.median(flux_he) + 1e-20)

        z_scores = (flux_he - bg_he) / np.maximum(err_he, 1e-20)
        return bool(np.any(z_scores > flux_sigma))


class SWISLoader:
    """
    Loads Aditya-L1 ASPEX-SWIS Level-2 CDF data files.

    Primary science use: solar wind bulk parameters for pre-conditioning
    features fed into the in-situ MLP branch of the AdityScan model.

    Usage
    -----
    loader = SWISLoader(data_dir="/data/aspex/l2")
    bulk = loader.load_bulk_params("20240814")
    features = loader.extract_ml_features(bulk, window_end_unix=..., window_minutes=30)
    """

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)

    # ── Public API ───────────────────────────────────────────────────────────

    def load_bulk_params(self, date_str: str) -> list[SWISBulkParams]:
        """
        Load SWIS bulk parameter (BLK) files for a given date.

        File naming (from manual, section 5.3):
            AL1_ASW91_L2_BLK_YYYYMMDD_<obsId>_Vmn.cdf

        Multiple observation IDs may exist per day. All are loaded and merged.
        Highest version is selected per observation ID.
        """
        try:
            import cdflib
        except ImportError:
            raise ImportError(
                "cdflib is required to read SWIS CDF files. "
                "Install with: pip install cdflib"
            )

        pattern = str(self.data_dir / f"AL1_ASW91_L2_BLK_{date_str}_*.cdf")
        files = self._select_highest_versions(glob.glob(pattern))

        all_records: list[SWISBulkParams] = []
        for fp in files:
            logger.info("Loading SWIS BLK: %s", fp)
            all_records.extend(self._parse_blk_cdf(fp, cdflib))

        return sorted(all_records, key=lambda r: r.unix_time)

    def load_flux_spectra(self, date_str: str, head: str = "TH1") -> list[SWISFluxSpectrum]:
        """
        Load flux spectrum files (TH1 or TH2) for a given date.

        File naming:
            AL1_ASW91_L2_TH1_YYYYMMDD_<obsId>_Vmn.cdf  (head="TH1")
            AL1_ASW91_L2_TH2_YYYYMMDD_<obsId>_Vmn.cdf  (head="TH2")
        """
        try:
            import cdflib
        except ImportError:
            raise ImportError("cdflib required: pip install cdflib")

        head_upper = head.upper()
        if head_upper not in ("TH1", "TH2"):
            raise ValueError(f"head must be 'TH1' or 'TH2', got '{head}'")

        pattern = str(self.data_dir / f"AL1_ASW91_L2_{head_upper}_{date_str}_*.cdf")
        files = self._select_highest_versions(glob.glob(pattern))

        all_spectra: list[SWISFluxSpectrum] = []
        for fp in files:
            logger.info("Loading SWIS %s spectra: %s", head_upper, fp)
            all_spectra.extend(self._parse_flux_cdf(fp, cdflib, head_upper))

        return sorted(all_spectra, key=lambda s: s.unix_time)

    def extract_ml_features(
        self,
        bulk_records: list[SWISBulkParams],
        window_end_unix: float,
        window_minutes: float = 30.0,
    ) -> Optional[np.ndarray]:
        """
        Extract a 6-element solar wind feature vector for the in-situ MLP branch.

        Features (shape [6]):
          [mean_density, mean_temperature, mean_speed,
           density_std, speed_std, mean_dynamic_pressure_nPa]

        Returns None if insufficient data in window.
        """
        t0 = window_end_unix - window_minutes * 60.0
        window = [r for r in bulk_records if t0 <= r.unix_time <= window_end_unix]

        if len(window) < 2:
            logger.debug("SWIS window empty at t=%.0f", window_end_unix)
            return None

        n_arr = np.array([r.proton_density for r in window])
        T_arr = np.array([r.proton_temperature for r in window])
        v_arr = np.array([r.proton_speed for r in window])
        pdyn_arr = np.array([r.dynamic_pressure_nPa for r in window])

        return np.array([
            float(np.mean(n_arr)),
            float(np.mean(T_arr)),
            float(np.mean(v_arr)),
            float(np.std(n_arr)),
            float(np.std(v_arr)),
            float(np.mean(pdyn_arr)),
        ], dtype=np.float32)

    # ── Private helpers ──────────────────────────────────────────────────────

    def _select_highest_versions(self, files: list[str]) -> list[str]:
        """
        For files with same observation ID, keep highest version number.
        Manual: "use the highest version if multiple files for a given duration."
        Version encoded as 'Vmn' at end of filename before .cdf
        """
        from collections import defaultdict
        grouped: dict[str, list[str]] = defaultdict(list)
        for f in files:
            base = os.path.basename(f)
            # Key = everything except version suffix
            # AL1_ASW91_L2_BLK_20240814_<obsId>_V01.cdf → strip _V*.cdf
            key = "_".join(base.split("_")[:-1])
            grouped[key].append(f)
        result = []
        for key, candidates in grouped.items():
            # Sort by filename (version is at end) → highest version last
            result.append(sorted(candidates)[-1])
        return sorted(result)

    def _parse_blk_cdf(self, filepath: str, cdflib) -> list[SWISBulkParams]:
        """
        Parse a SWIS L2 BLK CDF file.

        From manual section 5.3:
          Variables (10 total, metadata in CDF):
          - epoch_for_cdf: observation time
          - proton_number_density (cm⁻³)
          - proton_temperature (K)
          - bulk_speed (km/s)
          - density_error, temperature_error, speed_error
          - spacecraft position x/y/z in GSE (km)
        """
        records: list[SWISBulkParams] = []
        try:
            cdf = cdflib.CDF(filepath)
            info = cdf.cdf_info()

            # Manual: "epoch_for_cdf" is the UTC observation time (CDF Epoch)
            # cdflib converts CDF Epoch to UNIX ms: divide by 1000 for seconds
            epoch_ms = np.array(cdf.varget("epoch_for_cdf"))
            unix_times = epoch_ms / 1000.0  # ms → s (CDF Epoch is ms since 1970)

            n_density = np.array(cdf.varget("proton_number_density"))
            temperature = np.array(cdf.varget("proton_temperature"))
            speed = np.array(cdf.varget("bulk_speed"))

            # Error variables (may not exist in early data releases)
            try:
                d_err = np.array(cdf.varget("density_error"))
                t_err = np.array(cdf.varget("temperature_error"))
                s_err = np.array(cdf.varget("speed_error"))
            except Exception:
                d_err = np.zeros_like(n_density)
                t_err = np.zeros_like(temperature)
                s_err = np.zeros_like(speed)

            # Spacecraft position (may be optional in early releases)
            try:
                x_gse = np.array(cdf.varget("x_gse"))
                y_gse = np.array(cdf.varget("y_gse"))
                z_gse = np.array(cdf.varget("z_gse"))
            except Exception:
                x_gse = np.zeros(len(unix_times))
                y_gse = np.zeros(len(unix_times))
                z_gse = np.zeros(len(unix_times))

            for i in range(len(unix_times)):
                # Skip fill/bad values
                if n_density[i] < 0 or speed[i] < 0 or temperature[i] < 0:
                    continue
                records.append(SWISBulkParams(
                    unix_time=float(unix_times[i]),
                    proton_density=float(n_density[i]),
                    proton_temperature=float(temperature[i]),
                    proton_speed=float(speed[i]),
                    density_err=float(d_err[i]),
                    temperature_err=float(t_err[i]),
                    speed_err=float(s_err[i]),
                    x_gse_km=float(x_gse[i]),
                    y_gse_km=float(y_gse[i]),
                    z_gse_km=float(z_gse[i]),
                ))
        except Exception as exc:
            logger.error("Failed to parse SWIS BLK %s: %s", filepath, exc)

        logger.info("Loaded %d SWIS bulk records from %s", len(records), filepath)
        return records

    def _parse_flux_cdf(self, filepath: str, cdflib, head: str) -> list[SWISFluxSpectrum]:
        """
        Parse a SWIS L2 TH1/TH2 flux CDF file.

        From manual (sections 5.1, 5.2):
          11 variables per file:
          - epoch_for_cdf: UTC time
          - energies: 50-channel central energy bins
          - differential_energy_flux: flux values
          - flux_error: per-channel uncertainty
        """
        spectra: list[SWISFluxSpectrum] = []
        try:
            cdf = cdflib.CDF(filepath)
            epoch_ms = np.array(cdf.varget("epoch_for_cdf"))
            unix_times = epoch_ms / 1000.0
            energies = np.array(cdf.varget("energies"))          # shape (50,) or (N, 50)
            flux_all = np.array(cdf.varget("differential_energy_flux"))  # shape (N, 50)
            flux_err_all = np.array(cdf.varget("flux_error"))

            # energies may be 1D (same for all times) or 2D
            for i in range(len(unix_times)):
                en = energies[i] if energies.ndim == 2 else energies
                spectra.append(SWISFluxSpectrum(
                    unix_time=float(unix_times[i]),
                    energy_bins_eV=en.astype(np.float32),
                    flux=flux_all[i].astype(np.float32),
                    flux_err=flux_err_all[i].astype(np.float32),
                    head=head,
                ))
        except Exception as exc:
            logger.error("Failed to parse SWIS flux %s: %s", filepath, exc)

        return spectra
