"""
PRADAN (ISRO Space Science Data Centre) Authenticated Downloader
================================================================
Downloads Aditya-L1 SoLEXS and HEL1OS Level-1 data from the ISRO PRADAN portal.

File structure (from SoLEXS User Manual v1.0 and HEL1OS User Manual):

SoLEXS:
  ZIP: AL1_SLX_L1_YYYYMMDD_vM.n.zip
  Contents:
    AL1_SLX_L1_YYYYMMDD_vM.n/
      SDD1/
        AL1_SOLEXS_YYYYMMDD_SDD1_L1.gti.gz
        AL1_SOLEXS_YYYYMMDD_SDD1_L1.lc.gz   ← light curve, FITS
        AL1_SOLEXS_YYYYMMDD_SDD1_L1.pi.gz   ← spectra, FITS
      SDD2/
        AL1_SOLEXS_YYYYMMDD_SDD2_L1.lc.gz   ← light curve SDD2 (primary)
  LC FITS columns (RATE extension):
    TIME:   Unix time (float64)
    COUNTS: counts per second, 2–22 keV

HEL1OS:
  ZIP: HLS_YYYYMMDD_HHMMSS_XXXXXsec_lev1_VXXX.zip
  Contents:
    YYYY/MM/DD/HLS_.../
      cdte/
        lightcurve_cdte1.fits   ← CdTe1 1-s LC, bands: 5-20, 20-30, 30-40, 40-60 keV
        lightcurve_cdte2.fits   ← CdTe2
      czt/
        lightcurve_czt1.fits    ← CZT1 1-s LC, bands: 20-40, 40-60, 60-80, 80-150 keV
        lightcurve_czt2.fits    ← CZT2
      events/evt.fits
      aux/hk.fits, gti*.fits

Credentials: read from PRADAN_USER and PRADAN_PASS environment variables.
Data cached in: data/pradan_cache/

Download URL pattern (from portal network inspection):
  https://pradan.issdc.gov.in/al1/protected/getData?fileName=<filename>
"""

from __future__ import annotations

import asyncio
import gzip
import io
import logging
import os
import re
import time
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import httpx
import numpy as np

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

PRADAN_BASE      = "https://pradan1.issdc.gov.in/al1"
PRADAN_LIST_URL  = "https://pradan1.issdc.gov.in/al1/protected/getMetaData"
PRADAN_DATA_URL  = "https://pradan1.issdc.gov.in/al1/protected/getData"
PRADAN_BROWSE_URL = "https://pradan1.issdc.gov.in/al1/protected/browse.xhtml"

# Instrument IDs as used in the portal
INSTRUMENT_BROWSE_IDS = {
    "SoLEXS": "solexs",
    "HELIOS":  "hel1os",
}

CACHE_DIR = Path("data/pradan_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PRADAN_USER = os.environ.get("PRADAN_USER", "")
PRADAN_PASS = os.environ.get("PRADAN_PASS", "")

DOWNLOAD_TIMEOUT = httpx.Timeout(300.0, connect=15.0)   # large files need time


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class SoLEXSReading:
    """Parsed SoLEXS light curve — last N seconds of 1-s cadence data."""
    times_unix: np.ndarray      # shape (N,)
    counts_sdd2: np.ndarray     # shape (N,) cts/s, 2–22 keV
    counts_sdd1: np.ndarray     # shape (N,) cts/s (smaller aperture)
    date_str: str               # YYYYMMDD
    fetched_at: float           # time.time()


@dataclass
class HEL1OSReading:
    """Parsed HEL1OS light curve — last available observation."""
    times_unix: np.ndarray          # shape (N,)
    # CdTe bands (counts/s)
    cdte1_5_20:  np.ndarray        # 5-20 keV
    cdte1_20_30: np.ndarray        # 20-30 keV
    cdte1_30_40: np.ndarray        # 30-40 keV  ← HOPE trigger low
    cdte1_40_60: np.ndarray        # 40-60 keV  ← HOPE trigger high
    # CZT bands (counts/s)
    czt1_20_40:  np.ndarray        # 20-40 keV
    czt1_40_60:  np.ndarray        # 40-60 keV
    czt1_60_80:  np.ndarray        # 60-80 keV
    czt1_80_150: np.ndarray        # 80-150 keV
    obs_start: str                  # ISO timestamp
    fetched_at: float


# ── Session manager ───────────────────────────────────────────────────────────

class PRADANSession:
    """
    Manages authenticated session with ISRO PRADAN portal.
    Handles login, session refresh, and file downloads.
    """

    def __init__(self, username: str = PRADAN_USER, password: str = PRADAN_PASS):
        if not username or not password:
            raise EnvironmentError(
                "PRADAN_USER and PRADAN_PASS environment variables must be set. "
                "Run: export PRADAN_USER=... && export PRADAN_PASS=..."
            )
        self._username = username
        self._password = password
        self._client: Optional[httpx.AsyncClient] = None
        self._logged_in = False
        self._session_expiry = 0.0

    async def __aenter__(self) -> "PRADANSession":
        self._client = httpx.AsyncClient(
            headers={
                "User-Agent": "AdityScan/3.0 (research; ISRO/BAH 2026)",
                "Accept": "text/html,application/xhtml+xml,application/json,*/*",
                "Referer": PRADAN_BASE,
            },
            follow_redirects=True,
            timeout=DOWNLOAD_TIMEOUT,
        )
        await self._login()
        return self

    async def __aexit__(self, *_) -> None:
        if self._client:
            await self._client.aclose()

    async def _login(self) -> None:
        """
        Authenticate via ISRO Keycloak SSO (idp.issdc.gov.in).

        Flow:
          1. GET /al1/protected/payload.xhtml  →  302  →  Keycloak login page
          2. Parse Keycloak form action URL from the HTML
          3. POST username + password to the action URL
          4. Follow 302s back to pradan1 with auth code → session cookie
        """
        logger.info("Logging in to PRADAN/Keycloak as '%s'", self._username)
        try:
            # ── Step 1: hit the protected page to trigger Keycloak redirect ────────
            resp = await self._client.get(
                f"{PRADAN_BASE}/protected/payload.xhtml",
            )

            final_url = str(resp.url)

            # If already on PRADAN (not Keycloak) we're logged in
            if "idp.issdc.gov.in" not in final_url:
                self._logged_in = True
                self._session_expiry = time.time() + 3300
                logger.info("PRADAN session already valid (url=%s)", final_url[:60])
                return

            keycloak_page_url = final_url
            html = resp.text

            # ── Step 2: parse the Keycloak form action URL ─────────────────────────
            # Keycloak renders:  <form … action="https://idp.issdc.gov.in/auth/…">
            action_match = re.search(
                r'<form[^>]+action=["\']([^"\']+)["\']', html, re.IGNORECASE
            )
            if not action_match:
                raise PermissionError(
                    "Could not find Keycloak login form. "
                    f"Page snippet: {html[:400]}"
                )

            action_url = action_match.group(1).replace("&amp;", "&")
            logger.debug("Keycloak action URL: %s", action_url[:120])

            # ── Step 3: POST credentials to Keycloak ───────────────────────────────
            resp = await self._client.post(
                action_url,
                data={
                    "username": self._username,
                    "password": self._password,
                    "credentialId": "",
                },
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Origin": "https://idp.issdc.gov.in",
                    "Referer": keycloak_page_url,
                },
            )

            # ── Step 4: verify we landed back on PRADAN ────────────────────────────
            final_url = str(resp.url)

            if "idp.issdc.gov.in" in final_url and (
                "login" in final_url.lower() or "error" in resp.text.lower()
            ):
                raise PermissionError(
                    "Keycloak rejected credentials — check PRADAN_USER / PRADAN_PASS."
                    f" Final URL: {final_url}"
                )

            # Success — redirected away from Keycloak
            self._logged_in = True
            self._session_expiry = time.time() + 3300   # Keycloak default: 55 min
            logger.info(
                "PRADAN/Keycloak login OK — landed on %s — cookies: %s",
                final_url[:80],
                list(self._client.cookies.keys()),
            )

        except httpx.HTTPError as exc:
            logger.error("PRADAN Keycloak login HTTP error: %s", exc)
            raise

    async def _ensure_logged_in(self) -> None:
        """Re-login if session may have expired."""
        if time.time() > self._session_expiry - 60:
            logger.info("PRADAN session expired, re-logging in")
            await self._login()

    async def list_files(self, instrument: str, from_date: str, to_date: str) -> list[dict]:
        """
        List available files for an instrument by scraping the browse.xhtml page.
        Falls back to JSON REST API if available.

        Parameters
        ----------
        instrument : "SoLEXS" | "HELIOS"
        from_date  : "YYYY-MM-DDTHH:MM:SS.000Z"  (used for REST API fallback)
        to_date    : "YYYY-MM-DDTHH:MM:SS.000Z"

        Returns list of dicts with keys: filename, startTime, endTime, fileSizeInKB
        """
        await self._ensure_logged_in()

        browse_id = INSTRUMENT_BROWSE_IDS.get(instrument, instrument.lower())
        browse_url = f"{PRADAN_BROWSE_URL}?id={browse_id}"

        try:
            resp = await self._client.get(browse_url)
            resp.raise_for_status()

            # If redirected to Keycloak, session expired
            if "idp.issdc.gov.in" in str(resp.url):
                self._session_expiry = 0.0  # force re-login
                logger.warning("Session expired during list — will re-login next cycle")
                return []

            # Parse JSF table HTML to extract filenames
            return _parse_browse_html(resp.text, instrument)

        except Exception as exc:
            logger.warning("PRADAN browse failed for %s: %s", instrument, exc)
            return []

    async def download_file(self, filename: str, dest: Path, download_url: str = "") -> bool:
        """
        Download a data file from PRADAN to dest path.
        Uses local cache — skips download if file already cached.
        A valid file must be >50 KB (HTML error pages are <10 KB).

        Parameters
        ----------
        filename     : just the basename, used for cache key
        dest         : where to save the file
        download_url : full URL to use (from browse page href).
                       If empty, falls back to getData?fileName= endpoint.

        Returns True on success.
        """
        if dest.exists() and dest.stat().st_size > 50_000:
            logger.debug("Cache hit: %s (%.1f KB)", dest.name, dest.stat().st_size / 1024)
            return True
        elif dest.exists():
            # Stale/corrupt small file — delete and re-download
            dest.unlink()

        await self._ensure_logged_in()
        logger.info("Downloading %s from PRADAN...", filename)

        # Use full URL from browse page if provided, else legacy getData endpoint
        url = download_url if download_url else f"{PRADAN_DATA_URL}?fileName={filename}"

        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            async with self._client.stream("GET", url) as resp:
                resp.raise_for_status()

                # Detect HTML redirect/error pages by Content-Type before saving
                content_type = resp.headers.get("content-type", "")
                if "text/html" in content_type:
                    body = await resp.aread()
                    logger.warning(
                        "PRADAN returned HTML for %s (%d bytes) — session may have expired.",
                        filename, len(body),
                    )
                    self._session_expiry = 0.0
                    return False

                downloaded = 0
                with open(dest, "wb") as f:
                    async for chunk in resp.aiter_bytes(chunk_size=65536):
                        f.write(chunk)
                        downloaded += len(chunk)

            if downloaded < 50_000:
                dest.unlink(missing_ok=True)
                logger.warning(
                    "Downloaded %s is too small (%.1f KB) — likely an error page",
                    filename, downloaded / 1024,
                )
                return False

            logger.info("Downloaded %s → %.1f KB ✓", filename, downloaded / 1024)
            return True

        except Exception as exc:
            logger.error("Download failed for %s: %s", filename, exc)
            dest.unlink(missing_ok=True)
            return False



# ── SoLEXS fetcher ────────────────────────────────────────────────────────────

async def fetch_solexs_latest(
    session: PRADANSession,
    days_back: int = 3,
) -> Optional[SoLEXSReading]:
    """
    Download and parse the most recent SoLEXS L1 daily ZIP file.

    Strategy:
      1. Scrape browse.xhtml?id=solexs to get the newest file listing
      2. Download the first file (newest, as sorted by portal)
      3. Fall back to guessing by date if browse fails

    LC FITS (AL1_SOLEXS_YYYYMMDD_SDDn_L1.lc.gz) RATE extension columns:
      TIME:   Unix time (float64 seconds)
      COUNTS: count rate in 2–22 keV (cts/s)
    """
    from astropy.io import fits as af

    today = date.today()

    # ── Step 1: get file list from browse page ────────────────────────────────
    files = await session.list_files("SoLEXS", "", "")
    if files:
        # Portal sorts newest-first — take the top file
        newest = files[0]
        zip_name = newest.get("filename", newest.get("fileName", ""))
        download_url = newest.get("download_url", "")
        if zip_name:
            cache_zip = CACHE_DIR / zip_name
            ok = await session.download_file(zip_name, cache_zip, download_url)
            if ok:
                # Extract date from filename  AL1_SLX_L1_YYYYMMDD_vM.n.zip
                date_match = re.search(r"(\d{8})", zip_name)
                date_str = date_match.group(1) if date_match else today.strftime("%Y%m%d")
                try:
                    reading = _parse_solexs_zip(cache_zip, date_str, af)
                    if reading is not None:
                        logger.info("SoLEXS loaded from browse: %s, %d samples", zip_name, len(reading.times_unix))
                        return reading
                except Exception as exc:
                    logger.error("SoLEXS parse error: %s", exc)
                    cache_zip.unlink(missing_ok=True)

    # ── Step 2: fallback — try recent dates with known naming convention ───────
    logger.info("Browse listing empty — falling back to date-guessing for SoLEXS")
    for delta in range(days_back + 1):
        target_date = today - timedelta(days=delta)
        date_str = target_date.strftime("%Y%m%d")
        zip_name = f"AL1_SLX_L1_{date_str}_v1.0.zip"
        cache_zip = CACHE_DIR / zip_name

        ok = await session.download_file(zip_name, cache_zip)
        if not ok:
            continue

        try:
            reading = _parse_solexs_zip(cache_zip, date_str, af)
            if reading is not None:
                logger.info("SoLEXS loaded (fallback): %s, %d samples", date_str, len(reading.times_unix))
                return reading
        except Exception as exc:
            logger.error("SoLEXS parse error for %s: %s", zip_name, exc)
            cache_zip.unlink(missing_ok=True)

    return None



def _parse_solexs_zip(zip_path: Path, date_str: str, af) -> Optional[SoLEXSReading]:
    """
    Unzip AL1_SLX_L1_YYYYMMDD_vM.n.zip and parse LC files.
    Structure: AL1_SLX_L1_YYYYMMDD_vM.n/SDD2/AL1_SOLEXS_YYYYMMDD_SDD2_L1.lc.gz
    LC FITS: RATE extension → TIME (Unix), COUNTS (cts/s)
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()

        # Find SDD2 LC file (primary aperture, larger area)
        sdd2_lc = _find_in_zip(names, "SDD2", ".lc.gz")
        sdd1_lc = _find_in_zip(names, "SDD1", ".lc.gz")

        if not sdd2_lc:
            logger.warning("SDD2 LC file not found in %s", zip_path.name)
            return None

        def read_lc(member: str) -> tuple[np.ndarray, np.ndarray]:
            """Read TIME and COUNTS from a gzipped FITS LC file inside a ZIP."""
            raw = zf.read(member)
            with gzip.open(io.BytesIO(raw)) as gz_file:
                fits_bytes = gz_file.read()
            with af.open(io.BytesIO(fits_bytes), memmap=False) as hdul:
                # Find RATE extension (may be HDU 1 or named RATE)
                rate_hdu = None
                for hdu in hdul:
                    if hasattr(hdu, 'columns') and "COUNTS" in (hdu.columns.names if hdu.columns else []):
                        rate_hdu = hdu
                        break
                if rate_hdu is None:
                    rate_hdu = hdul[1]  # fallback
                data = rate_hdu.data
                times  = data["TIME"].astype(np.float64)
                counts = data["COUNTS"].astype(np.float64)
            return times, counts

        times_sdd2, counts_sdd2 = read_lc(sdd2_lc)
        times_sdd1 = np.array([])
        counts_sdd1 = np.zeros_like(counts_sdd2)
        if sdd1_lc:
            try:
                times_sdd1, counts_sdd1 = read_lc(sdd1_lc)
                # Align to SDD2 time grid
                counts_sdd1 = _align_to_grid(times_sdd1, counts_sdd1, times_sdd2)
            except Exception as exc:
                logger.warning("SDD1 LC read failed: %s", exc)

    return SoLEXSReading(
        times_unix=times_sdd2,
        counts_sdd2=counts_sdd2,
        counts_sdd1=counts_sdd1,
        date_str=date_str,
        fetched_at=time.time(),
    )


# ── HEL1OS fetcher ────────────────────────────────────────────────────────────

async def fetch_helios_latest(
    session: PRADANSession,
    days_back: int = 7,
) -> Optional[HEL1OSReading]:
    """
    Download and parse the most recent HEL1OS L1 ZIP file.
    Extracts CdTe1 and CZT1 light curves at 1-s cadence.

    LC file bands (from HEL1OS User Manual Section 2.5):
      CdTe: 5-20, 20-30, 30-40, 40-60 keV  (+ full range 1.8-90 keV)
      CZT:  20-40, 40-60, 60-80, 80-150 keV (+ full range 18-160 keV)
    """
    from astropy.io import fits as af

    # ── Step 1: get file list from browse page ────────────────────────────────
    files = await session.list_files("HELIOS", "", "")
    if files:
        # Portal sorts newest-first — take the top file
        newest = files[0]
        fn = newest.get("filename", newest.get("fileName", ""))
        download_url = newest.get("download_url", "")
        if fn:
            cache_zip = CACHE_DIR / fn
            ok = await session.download_file(fn, cache_zip, download_url)
            if ok:
                try:
                    reading = _parse_helios_zip(cache_zip, af)
                    if reading is not None:
                        logger.info("HEL1OS loaded from browse: %s, %d samples", fn, len(reading.times_unix))
                        return reading
                except Exception as exc:
                    logger.error("HEL1OS parse error for %s: %s", fn, exc)
                    cache_zip.unlink(missing_ok=True)

    logger.warning("HEL1OS: browse listing empty or download failed")
    return None



def _parse_helios_zip(zip_path: Path, af) -> Optional[HEL1OSReading]:
    """
    Unzip HEL1OS ZIP and parse CdTe1 and CZT1 light curves.

    ACTUAL FITS structure (confirmed from V111 data):
      Each energy band is a SEPARATE HDU, named like:
        CZT1_LC_BAND_20.00KEV_TO_40.00KEV
        CDTE1_LC_BAND_5.00KEV_TO_20.00KEV
      Columns per HDU:
        MJD      - Modified Julian Date (float64)
        ISOT     - ISO timestamp string
        CTR      - count rate (cts/s, float64)
        STAT_ERR - 1-sigma error on CTR

    We convert MJD → Unix time: unix = (MJD - 40587) * 86400
    """
    MJD_TO_UNIX = 40587.0 * 86400.0  # MJD epoch offset

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()

        cdte1_path = _find_in_zip(names, "cdte", "lightcurve_cdte1.fits")
        czt1_path  = _find_in_zip(names, "czt",  "lightcurve_czt1.fits")

        if not cdte1_path and not czt1_path:
            logger.warning("No lightcurve FITS files found in %s", zip_path.name)
            return None

        def read_hel1os_fits(member: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
            """
            Read all energy band HDUs from a HEL1OS LC FITS file.
            Returns: dict mapping HDU name -> (times_unix, ctr)
            """
            raw = zf.read(member)
            bands = {}
            with af.open(io.BytesIO(raw), memmap=False) as hdul:
                for hdu in hdul[1:]:  # skip PRIMARY
                    if not hasattr(hdu, "data") or hdu.data is None:
                        continue
                    cols = [c.name for c in hdu.columns]
                    if "MJD" not in cols or "CTR" not in cols:
                        continue
                    mjd = hdu.data["MJD"].astype(np.float64)
                    unix_t = mjd * 86400.0 - MJD_TO_UNIX
                    ctr = hdu.data["CTR"].astype(np.float64)
                    bands[hdu.name] = (unix_t, ctr)
            return bands

        # Parse CdTe1
        bands_cdte: dict[str, tuple] = {}
        if cdte1_path:
            try:
                bands_cdte = read_hel1os_fits(cdte1_path)
                logger.debug("CdTe1 HDUs: %s", list(bands_cdte.keys()))
            except Exception as exc:
                logger.warning("CdTe1 read failed: %s", exc)

        # Parse CZT1
        bands_czt: dict[str, tuple] = {}
        if czt1_path:
            try:
                bands_czt = read_hel1os_fits(czt1_path)
                logger.debug("CZT1 HDUs: %s", list(bands_czt.keys()))
            except Exception as exc:
                logger.warning("CZT1 read failed: %s", exc)

        if not bands_cdte and not bands_czt:
            return None

        def find_band(bands: dict, lo: str, hi: str) -> tuple[np.ndarray, np.ndarray]:
            """Find HDU matching energy range (lo-hi keV) by name substring."""
            for name, (t, ctr) in bands.items():
                name_u = name.upper()
                if lo.upper() in name_u and hi.upper() in name_u:
                    return t, ctr
            return np.array([]), np.array([])

        # CdTe1 bands: 5-20, 20-30, 30-40, 40-60 keV
        t_5_20,  c_5_20  = find_band(bands_cdte, "5.00", "20.00")
        t_20_30, c_20_30 = find_band(bands_cdte, "20.00", "30.00")
        t_30_40, c_30_40 = find_band(bands_cdte, "30.00", "40.00")
        t_40_60, c_40_60 = find_band(bands_cdte, "40.00", "60.00")

        # CZT1 bands: 20-40, 40-60, 60-80, 80-150 keV
        t_20_40, c_20_40 = find_band(bands_czt, "20.00", "40.00")
        t_40_60z,c_40_60z= find_band(bands_czt, "40.00", "60.00")
        t_60_80, c_60_80 = find_band(bands_czt, "60.00", "80.00")
        t_80_150,c_80_150= find_band(bands_czt, "80.00", "150.00")

        # Use the densest time grid as reference
        all_times = [t for t in (t_5_20, t_20_40) if len(t) > 0]
        if not all_times:
            return None
        times = max(all_times, key=len)

        # Ensure all arrays have the same length (zero-pad if needed)
        def pad_to(arr, n):
            if len(arr) >= n:
                return arr[:n]
            return np.pad(arr, (0, n - len(arr)))

        n = len(times)

    return HEL1OSReading(
        times_unix=times,
        cdte1_5_20  = pad_to(c_5_20,  n),
        cdte1_20_30 = pad_to(c_20_30, n),
        cdte1_30_40 = pad_to(c_30_40, n),
        cdte1_40_60 = pad_to(c_40_60, n),
        czt1_20_40  = pad_to(c_20_40,  n),
        czt1_40_60  = pad_to(c_40_60z, n),
        czt1_60_80  = pad_to(c_60_80,  n),
        czt1_80_150 = pad_to(c_80_150, n),
        obs_start=zip_path.stem,
        fetched_at=time.time(),
    )


# ── Helper utilities ──────────────────────────────────────────────────────────

def _parse_browse_html(html: str, instrument: str) -> list[dict]:
    """
    Scrape the PRADAN browse.xhtml JSF page to extract file listing.

    The browse page has direct download href links of the form:
      /al1/protected/downloadData/solexs/level1/YYYY/MM/N00_0000/FILENAME.zip?solexs

    We extract these hrefs directly — much more reliable than parsing table rows.
    Returns list of dicts with keys: filename, download_url, fileSizeInKB
    Sorted newest-first (as displayed on the portal).
    """
    results = []
    browse_id = INSTRUMENT_BROWSE_IDS.get(instrument, instrument.lower())

    # Match: href="/al1/protected/downloadData/{id}/level1/.../FILENAME.zip?{id}"
    dl_pattern = re.compile(
        r'href="(/al1/protected/downloadData/' + re.escape(browse_id) +
        r'/[^"]+\.zip[^"]*)"',
        re.IGNORECASE,
    )

    seen = set()
    for m in dl_pattern.finditer(html):
        path = m.group(1)
        filename = path.split("/")[-1].split("?")[0]  # strip query string
        if filename in seen:
            continue
        seen.add(filename)
        full_url = f"https://pradan1.issdc.gov.in{path}"
        results.append({
            "filename": filename,
            "download_url": full_url,
            "fileSizeInKB": 0.0,  # not easily parsed from hrefs alone
        })

    logger.info("_parse_browse_html: found %d files for %s", len(results), instrument)
    return results





    logger.debug("_parse_browse_html: found %d files for %s", len(results), instrument)
    return results



def _find_in_zip(names: list[str], *substrings: str) -> Optional[str]:
    """Find first ZIP entry containing all given substrings (case-insensitive)."""
    for name in names:
        name_lower = name.lower()
        if all(s.lower() in name_lower for s in substrings):
            return name
    return None


def _align_to_grid(
    t_src: np.ndarray, v_src: np.ndarray, t_dst: np.ndarray
) -> np.ndarray:
    """Resample v_src (on t_src) to t_dst grid using nearest-neighbour."""
    if len(t_src) == 0:
        return np.zeros(len(t_dst))
    out = np.zeros(len(t_dst))
    for i, t in enumerate(t_dst):
        idx = np.argmin(np.abs(t_src - t))
        if np.abs(t_src[idx] - t) < 5:  # within 5 s
            out[i] = v_src[idx]
    return out


# ── Convenience: compute flux proxy from SoLEXS count rate ───────────────────

def solexs_counts_to_flux_proxy(counts_sdd2: float) -> float:
    """
    Convert SoLEXS SDD2 count rate (cts/s) to approximate GOES 1–8 Å flux (W/m²).

    Empirical cross-calibration from SoLEXS User Manual Table 1 + GOES XRS:
      SDD2 large aperture (0.1 mm²): ~1 cts/s ≈ B1 class
      ~10 cts/s ≈ C1 class
      ~100 cts/s ≈ M1 class
      ~1000 cts/s ≈ X1 class

    This is approximate until formal cross-calibration is published.
    """
    if counts_sdd2 <= 0:
        return 0.0
    import math
    # Log-linear mapping anchored to: 1 ct/s → 1e-7 W/m² (B1)
    log_flux = math.log10(max(counts_sdd2, 0.01)) - 7.0 + (-7.0)
    return max(10 ** log_flux, 1e-10)


# ── Top-level: fetch all Aditya-L1 data ──────────────────────────────────────

async def fetch_all_aditya_data() -> dict:
    """
    Fetch latest SoLEXS + HEL1OS data in one call.
    Returns dict with keys: 'solexs', 'helios', 'error'
    
    Called from main.py background poller when credentials are available.
    """
    if not PRADAN_USER or not PRADAN_PASS:
        return {
            "solexs": None, "helios": None,
            "error": "PRADAN_USER/PRADAN_PASS not set",
        }

    try:
        async with PRADANSession() as session:
            solexs, helios = await asyncio.gather(
                fetch_solexs_latest(session, days_back=3),
                fetch_helios_latest(session, days_back=7),
                return_exceptions=True,
            )

        return {
            "solexs": solexs if not isinstance(solexs, Exception) else None,
            "helios": helios if not isinstance(helios, Exception) else None,
            "error": None,
        }
    except Exception as exc:
        logger.error("Aditya-L1 data fetch failed: %s", exc)
        return {"solexs": None, "helios": None, "error": str(exc)}
