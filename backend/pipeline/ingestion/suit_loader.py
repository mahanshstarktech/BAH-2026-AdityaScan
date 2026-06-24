"""
SUIT (Solar UV Imaging Telescope) data loader for Aditya-L1.

Grounded in: SUIT User Manual, IUCAA (2024)
  - Full-disk solar imager, wavelength range: ~200–400 nm (NUV)
  - 11 science filters (NB01–NB08 narrowband, BB01–BB03 broadband)
  - Key flare-relevant bands:
      NB02 = 214 nm (chromosphere UV continuum, flare ribbon marker)
      NB05 = 280 nm (Mg II h&k, chromospheric emission)
      BB03 = 388.5 nm (broadband, flare brightening)
  - Image format: FITS
  - Image size: 4096×4096 pixels (full disk, ~0.74 arcsec/pixel)
  - FITS header key fields (from manual section 3.2/3.3/3.4):
      T_OBS         : Observation time (UTC string)
      DATE-OBS      : Observation date
      FTR_NAME      : Filter combination name (e.g. "NB02")
      CRPIX1/2      : Sun center in pixels
      CDELT1/2      : Pixel scale (arcsec/pixel)
      R_SUN         : Solar radius in pixels
      RSUN_OBS      : Solar radius in arcsec
      DSUN_OBS      : Distance Aditya → Sun center (meters)
      HGLT_OBS      : Observer Stoneyhurst lat (degrees)
      HL1OSFLG      : HEL1OS flare flag (from health params section 3.4)
      SX2FLG        : SoLEXS-2 flare flag (linked to flare trigger)
      NRMFLG        : Normal flare flag
      QVAL          : Image quality percentage (0-100)
      QDESC         : Quality description string
  - Flare trigger: SUIT has onboard "Find Flare" mode triggered by HEL1OS
    (FFEXTTO flag: external flare enabled) — cadence increases on flare
  - File naming convention (L2 science-ready FITS):
      SUIT_<filter>_<YYYYMMDD>_<HHMMSS>_L2_Vmn.fits  (approximate)
      Actual naming from PRADAN — use FTR_NAME header to identify filter

  Computing note: 4096×4096 = 16.7M pixels per image.
    Downsample to 512×512 before CNN (32× area reduction).
    Full-disk crop around active region (±200 arcsec → ~270×270 px @ 0.74"/px).

Reference: SUIT User Manual, IUCAA/ISRO 2024
"""

from __future__ import annotations

import glob
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants from SUIT manual ───────────────────────────────────────────────
SUIT_PIXEL_SCALE_ARCSEC: float = 0.74   # arcsec/pixel (approx, verify from CDELT1)
SUIT_IMAGE_SIZE_PX: int = 4096          # native image size (square)
SUIT_CNN_SIZE_PX: int = 512             # downsampled size for CNN input
SUIT_AR_CUTOUT_ARCSEC: float = 400.0   # total AR cutout width in arcsec

# Filter names exactly as in FTR_NAME FITS header
SUIT_FLARE_FILTERS = {
    "NB02": {"wavelength_nm": 214, "description": "UV continuum, flare ribbon"},
    "NB05": {"wavelength_nm": 280, "description": "Mg II chromospheric emission"},
    "BB03": {"wavelength_nm": 388, "description": "Broadband, flare brightening"},
    "NB04": {"wavelength_nm": 256, "description": "C IV, transition region"},
}


@dataclass
class SUITImage:
    """
    One SUIT science image, partially parsed from FITS header.
    The full pixel data is not stored in memory — load on demand via load_array().
    """
    filepath: str
    filter_name: str          # FTR_NAME header value, e.g. "NB02"
    t_obs: str                # T_OBS header (UTC string)
    unix_time: float          # Derived from T_OBS
    sun_cx_px: float          # CRPIX1: Sun center X in pixels
    sun_cy_px: float          # CRPIX2: Sun center Y in pixels
    r_sun_px: float           # R_SUN: solar radius in pixels
    r_sun_arcsec: float       # RSUN_OBS: solar radius in arcsec
    cdelt1: float             # arcsec/pixel in x
    cdelt2: float             # arcsec/pixel in y
    quality_pct: float        # QVAL: image quality (0–100)
    hel1os_flare_flag: int    # HL1OSFLG: HEL1OS triggered this exposure
    solexs2_flare_flag: int   # SX2FLG: SoLEXS-2 triggered this exposure
    normal_flare_flag: int    # NRMFLG: normal flare flag
    dsun_obs_m: float         # DSUN_OBS: distance Aditya–Sun (meters)

    def load_array(self) -> np.ndarray:
        """Load image data from FITS file as float32 array, shape (4096, 4096)."""
        from astropy.io import fits
        with fits.open(self.filepath) as hdul:
            # Primary extension or first extension with data
            for ext in hdul:
                if ext.data is not None and ext.data.ndim == 2:
                    return ext.data.astype(np.float32)
        raise ValueError(f"No 2D image data found in {self.filepath}")

    def load_ar_cutout(
        self,
        ar_x_arcsec: float,
        ar_y_arcsec: float,
        half_width_arcsec: float = SUIT_AR_CUTOUT_ARCSEC / 2,
    ) -> np.ndarray:
        """
        Extract a cutout around an active region in solar coordinates.

        Parameters
        ----------
        ar_x_arcsec : float
            Active region X coordinate in arcsec from Sun center (HPC X).
        ar_y_arcsec : float
            Active region Y coordinate in arcsec (HPC Y).
        half_width_arcsec : float
            Half-width of the cutout box in arcsec.

        Returns
        -------
        np.ndarray of shape (N, N) where N = 2 * half_width_arcsec / cdelt1
        """
        full = self.load_array()
        # Convert arcsec coordinates to pixel coordinates
        # HPC: origin at Sun center, +X toward solar west, +Y toward north
        # In pixel space: x_px = CRPIX1 + (arcsec / cdelt1)
        cx_px = int(self.sun_cx_px + ar_x_arcsec / self.cdelt1)
        cy_px = int(self.sun_cy_px - ar_y_arcsec / self.cdelt2)  # y flipped

        half_px = int(half_width_arcsec / self.cdelt1)
        x0 = max(0, cx_px - half_px)
        x1 = min(SUIT_IMAGE_SIZE_PX, cx_px + half_px)
        y0 = max(0, cy_px - half_px)
        y1 = min(SUIT_IMAGE_SIZE_PX, cy_px + half_px)

        return full[y0:y1, x0:x1]


class SUITLoader:
    """
    Loads Aditya-L1 SUIT science FITS images.

    Pipeline integration:
    - NOT always-on. Only triggered when activity_mode >= ELEVATED.
    - Primary use: CNN feature extraction on active region UV brightening.
    - Secondary use: Detect flare ribbon onset (NB02 band, UV continuum).

    Usage
    -----
    loader = SUITLoader(data_dir="/data/suit/l2")
    images = loader.scan_directory(date_str="20240514", filter_name="NB02")
    features = loader.extract_cnn_input(images[0], target_size=512)
    """

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)

    def scan_directory(
        self,
        date_str: Optional[str] = None,
        filter_name: Optional[str] = None,
    ) -> list[SUITImage]:
        """
        Scan data directory for SUIT FITS files and parse headers.
        Returns metadata list (images NOT loaded into memory).

        Parameters
        ----------
        date_str : str, optional
            Filter by date (YYYYMMDD) from filename or T_OBS header.
        filter_name : str, optional
            Filter by FTR_NAME header value (e.g. "NB02").
        """
        from astropy.io import fits

        pattern = str(self.data_dir / "**" / "*.fits")
        all_files = glob.glob(pattern, recursive=True)

        images: list[SUITImage] = []
        for fp in sorted(all_files):
            try:
                img = self._parse_fits_header(fp, fits)
                if date_str and date_str not in img.t_obs.replace("-", ""):
                    continue
                if filter_name and img.filter_name != filter_name:
                    continue
                if img.quality_pct < 80.0:
                    logger.debug("Skipping low-quality SUIT image: %s (QVAL=%.1f)", fp, img.quality_pct)
                    continue
                images.append(img)
            except Exception as exc:
                logger.debug("Skipping %s: %s", fp, exc)

        logger.info("Found %d SUIT images (filter=%s, date=%s)", len(images), filter_name, date_str)
        return images

    def extract_cnn_input(
        self,
        image: SUITImage,
        target_size: int = SUIT_CNN_SIZE_PX,
        ar_cutout: Optional[tuple[float, float]] = None,
    ) -> np.ndarray:
        """
        Load and preprocess a SUIT image for CNN input.

        Processing steps:
          1. Load full FITS array (4096×4096)
          2. Optional: crop to AR region (ar_cutout = (x_arcsec, y_arcsec))
          3. Downsample to target_size × target_size
          4. Log-normalize (solar images are log-distributed)
          5. Standardize to [0, 1] using percentile clipping

        Returns
        -------
        np.ndarray, shape (1, target_size, target_size), float32
            Single-channel image ready for EfficientNet-B0 input.
        """
        try:
            from skimage.transform import resize as sk_resize
        except ImportError:
            raise ImportError("scikit-image required: pip install scikit-image")

        if ar_cutout is not None:
            arr = image.load_ar_cutout(ar_cutout[0], ar_cutout[1])
        else:
            arr = image.load_array()

        # Handle bad pixels / NaN
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.clip(arr, 0, None)  # ensure non-negative

        # Log normalization (standard in solar physics imaging)
        arr = np.log1p(arr)

        # Resize to CNN input size
        arr_resized = sk_resize(arr, (target_size, target_size), anti_aliasing=True).astype(np.float32)

        # Percentile normalization (2%–98% clip → [0, 1])
        p2, p98 = np.percentile(arr_resized, [2, 98])
        if p98 > p2:
            arr_norm = np.clip((arr_resized - p2) / (p98 - p2), 0.0, 1.0)
        else:
            arr_norm = np.zeros_like(arr_resized)

        return arr_norm[np.newaxis, :, :]  # (1, H, W)

    def compute_uv_brightness_index(
        self,
        image: SUITImage,
        ar_cutout: tuple[float, float],
        percentile: float = 95.0,
    ) -> float:
        """
        Compute a scalar UV brightness index for an active region.

        This is the key scalar feature fed into the fusion model when
        full CNN inference is not warranted (quiet mode).
        Returns the Nth percentile of the AR cutout pixel values (log-scaled).

        Parameters
        ----------
        ar_cutout : (x_arcsec, y_arcsec)
            Active region center in helioprojective Cartesian coordinates.
        percentile : float
            Percentile for brightness (default 95th = bright core).
        """
        arr = image.load_ar_cutout(ar_cutout[0], ar_cutout[1])
        arr = np.clip(arr, 0, None)
        arr = np.log1p(arr)
        if arr.size == 0:
            return 0.0
        return float(np.percentile(arr, percentile))

    # ── Private helpers ──────────────────────────────────────────────────────

    def _parse_fits_header(self, filepath: str, fits) -> SUITImage:
        """
        Parse SUIT FITS header into a SUITImage dataclass.
        Uses exact header keyword names from SUIT manual section 3.2–3.4.
        """
        from datetime import datetime, timezone

        with fits.open(filepath, memmap=True) as hdul:
            hdr = hdul[0].header

            t_obs = str(hdr.get("T_OBS", ""))
            # Parse to UNIX time
            try:
                dt = datetime.strptime(t_obs[:19], "%Y-%m-%dT%H:%M:%S")
                dt = dt.replace(tzinfo=timezone.utc)
                unix_time = dt.timestamp()
            except Exception:
                unix_time = 0.0

            return SUITImage(
                filepath=filepath,
                filter_name=str(hdr.get("FTR_NAME", "UNKNOWN")),
                t_obs=t_obs,
                unix_time=unix_time,
                sun_cx_px=float(hdr.get("CRPIX1", SUIT_IMAGE_SIZE_PX / 2)),
                sun_cy_px=float(hdr.get("CRPIX2", SUIT_IMAGE_SIZE_PX / 2)),
                r_sun_px=float(hdr.get("R_SUN", 1600.0)),
                r_sun_arcsec=float(hdr.get("RSUN_OBS", 960.0)),
                cdelt1=float(hdr.get("CDELT1", SUIT_PIXEL_SCALE_ARCSEC)),
                cdelt2=float(hdr.get("CDELT2", SUIT_PIXEL_SCALE_ARCSEC)),
                quality_pct=float(hdr.get("QVAL", 0.0)),
                hel1os_flare_flag=int(hdr.get("HL1OSFLG", 0)),
                solexs2_flare_flag=int(hdr.get("SX2FLG", 0)),
                normal_flare_flag=int(hdr.get("NRMFLG", 0)),
                dsun_obs_m=float(hdr.get("DSUN_OBS", 1.496e11)),
            )
