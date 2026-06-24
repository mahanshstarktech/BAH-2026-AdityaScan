"""
AdityScan v3 — NOAA SWPC Live Data Ingestion
=============================================
Polls free, public NOAA SWPC REST endpoints (no auth required) every 60s
and returns structured data for the FastAPI AppState.

Endpoints used:
  GOES XRS 6-hour:     https://services.swpc.noaa.gov/json/goes/primary/xrays-6-hour.json
  Solar wind MAG 6hr:  https://services.swpc.noaa.gov/products/solar-wind/mag-6-hour.json
  Solar wind plasma:   https://services.swpc.noaa.gov/products/solar-wind/plasma-6-hour.json
  Flare events 7-day:  https://services.swpc.noaa.gov/json/goes/primary/xray-flares-7-day.json
  3-day probabilities: https://services.swpc.noaa.gov/text/3-day-forecast.txt
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ── NOAA SWPC base URL ─────────────────────────────────────────────────────────
SWPC = "https://services.swpc.noaa.gov"

TIMEOUT = httpx.Timeout(15.0, connect=5.0)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class GOESSnapshot:
    """Current GOES XRS reading."""
    timestamp_utc: datetime
    flux_1_8: float          # W/m² (1–8 Å channel — GOES class denominator)
    flux_0p5_4: float        # W/m² (0.5–4 Å channel)
    goes_class: str          # e.g. "M2.3", "C7.1", "X1.5"
    z_score: float           # vs 1-hour background (rolling)
    satellite: str           # "GOES-16" or "GOES-18"


@dataclass
class SolarWindSnapshot:
    """Current L1 in-situ solar wind conditions (ACE/DSCOVR proxy via NOAA)."""
    timestamp_utc: datetime
    bt: float                # |B| total field, nT
    bx: float                # Bx GSE, nT
    by: float                # By GSE, nT
    bz: float                # Bz GSE, nT — negative = southward = geoeffective
    speed: float             # Solar wind proton speed, km/s
    density: float           # Proton number density, cm⁻³
    temperature: float       # Proton temperature, K
    clock_angle_deg: float   # atan2(By, Bz), degrees
    cone_angle_deg: float    # angle B makes with Sun-Earth line
    dyn_pressure_npa: float  # nPa = 1.67e-6 * n * v²
    alfven_mach: float       # Ma = v / vA


@dataclass
class FlareEvent:
    """Single flare event from NOAA GOES event list."""
    start_time_utc: datetime
    peak_time_utc: Optional[datetime]
    end_time_utc: Optional[datetime]
    goes_class: str
    location: str            # "N23W45" heliographic
    region: str              # "3724" AR number


@dataclass
class SWPCProbabilities:
    """NOAA 3-day M+/X+ flare probabilities (best available public forecast)."""
    issued_utc: datetime
    m_class_pct: float       # P(≥M class in next 24h) from NOAA forecasters
    x_class_pct: float       # P(≥X class in next 24h)
    proton_pct: float        # P(≥10 MeV proton event in next 24h)


@dataclass
class LiveSnapshot:
    """Combined snapshot of all real-time data."""
    goes: Optional[GOESSnapshot] = None
    wind: Optional[SolarWindSnapshot] = None
    flares: list[FlareEvent] = field(default_factory=list)
    probs: Optional[SWPCProbabilities] = None
    fetch_error: Optional[str] = None


# ── Unit helpers ──────────────────────────────────────────────────────────────

def flux_to_goes_class(flux_wm2: float) -> str:
    """Convert GOES 1-8Å flux (W/m²) to standard letter class string."""
    if flux_wm2 < 1e-8:
        return "A0.0"
    if flux_wm2 < 1e-7:
        m = flux_wm2 / 1e-8
        return f"A{m:.1f}"
    if flux_wm2 < 1e-6:
        m = flux_wm2 / 1e-7
        return f"B{m:.1f}"
    if flux_wm2 < 1e-5:
        m = flux_wm2 / 1e-6
        return f"C{m:.1f}"
    if flux_wm2 < 1e-4:
        m = flux_wm2 / 1e-5
        return f"M{m:.1f}"
    m = flux_wm2 / 1e-4
    return f"X{m:.1f}"


def compute_clock_angle(by: float, bz: float) -> float:
    """Geocentric solar ecliptic clock angle in degrees (0=northward, 180=southward)."""
    return math.degrees(math.atan2(by, bz)) % 360


def compute_cone_angle(bx: float, bt: float) -> float:
    """Parker spiral cone angle (angle between B and Sun-Earth line)."""
    if bt == 0:
        return 0.0
    return math.degrees(math.acos(min(1.0, abs(bx) / bt)))


def compute_dynamic_pressure(density_cm3: float, speed_kms: float) -> float:
    """Solar wind dynamic pressure in nPa. P = 1.673e-6 * n * v²"""
    return 1.673e-6 * density_cm3 * speed_kms ** 2


def compute_alfven_mach(speed_kms: float, bt_nt: float, density_cm3: float) -> float:
    """Alfvén Mach number. Ma = v / vA, vA = B / sqrt(μ₀ρ)."""
    if bt_nt <= 0 or density_cm3 <= 0:
        return 0.0
    # vA in km/s: 21.8 * B(nT) / sqrt(n(cm⁻³))
    va = 21.8 * bt_nt / math.sqrt(density_cm3)
    return speed_kms / va if va > 0 else 0.0


# ── Fetchers ──────────────────────────────────────────────────────────────────

async def fetch_goes_xrs(client: httpx.AsyncClient) -> Optional[GOESSnapshot]:
    """
    Fetch the latest GOES XRS 1-minute reading.
    Returns the most recent valid (non-null) record.
    """
    try:
        resp = await client.get(
            f"{SWPC}/json/goes/primary/xrays-6-hour.json",
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        # NOAA sometimes sends truncated JSON (mid-update). Handle gracefully.
        try:
            records = resp.json()
        except Exception:
            # Try to parse only the valid prefix by stripping the truncated tail
            text = resp.text.strip()
            # Find the last valid complete JSON object
            for end in range(len(text), 0, -1):
                try:
                    records = __import__("json").loads(text[:end] + "]" if not text[:end].endswith("]") else text[:end])
                    break
                except Exception:
                    continue
            else:
                logger.warning("GOES XRS JSON completely unparseable — skipping")
                return None

        # Records are 1-min cadence; latest is last in list
        # Filter out nulls and pick most recent valid
        valid = [
            r for r in reversed(records)
            if r.get("flux") is not None and r.get("energy") == "0.1-0.8nm"
        ]
        if not valid:
            logger.warning("No valid GOES XRS records found")
            return None

        # Also need the short channel (0.05-0.4nm) for class ratio
        valid_short = [
            r for r in reversed(records)
            if r.get("flux") is not None and r.get("energy") == "0.05-0.4nm"
        ]

        latest_long = valid[0]
        flux_1_8 = float(latest_long["flux"])
        flux_0p5_4 = float(valid_short[0]["flux"]) if valid_short else flux_1_8 * 0.3

        # Parse timestamp
        ts_str = latest_long.get("time_tag", "")
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except ValueError:
            ts = datetime.now(timezone.utc)

        # Z-score vs 1-hour background (last 60 records in 1-min cadence)
        long_records = [
            r for r in records
            if r.get("flux") is not None and r.get("energy") == "0.1-0.8nm"
        ]
        bg_fluxes = [float(r["flux"]) for r in long_records[:-10]]  # exclude last 10 min
        if len(bg_fluxes) >= 20:
            bg_log = [math.log10(max(f, 1e-9)) for f in bg_fluxes[-60:]]
            bg_mean = sum(bg_log) / len(bg_log)
            bg_std = max(math.sqrt(sum((x - bg_mean)**2 for x in bg_log) / len(bg_log)), 0.01)
            current_log = math.log10(max(flux_1_8, 1e-9))
            z_score = (current_log - bg_mean) / bg_std
        else:
            z_score = 0.0

        goes_class = flux_to_goes_class(flux_1_8)
        satellite = latest_long.get("satellite", 16)
        sat_name = f"GOES-{satellite}"

        logger.debug("GOES XRS: %s (%.2e W/m²) z=%.2f", goes_class, flux_1_8, z_score)
        return GOESSnapshot(
            timestamp_utc=ts,
            flux_1_8=flux_1_8,
            flux_0p5_4=flux_0p5_4,
            goes_class=goes_class,
            z_score=round(z_score, 2),
            satellite=sat_name,
        )

    except Exception as exc:
        logger.error("GOES XRS fetch failed: %s", exc)
        return None


async def fetch_solar_wind(client: httpx.AsyncClient) -> Optional[SolarWindSnapshot]:
    """
    Fetch latest solar wind magnetic field and plasma from NOAA SWPC.
    Combines MAG (Bt/Bx/By/Bz) and plasma (speed/density/temp) endpoints.
    """
    try:
        mag_resp, plasma_resp = await asyncio.gather(
            client.get(f"{SWPC}/products/solar-wind/mag-6-hour.json", timeout=TIMEOUT),
            client.get(f"{SWPC}/products/solar-wind/plasma-6-hour.json", timeout=TIMEOUT),
            return_exceptions=True,
        )

        if isinstance(mag_resp, Exception) or isinstance(plasma_resp, Exception):
            raise Exception("Solar wind fetch failed")

        mag_resp.raise_for_status()
        plasma_resp.raise_for_status()

        # Format: [[time, Bx, By, Bz, Bt, lat, lon], ...]
        # First row is header
        mag_data = mag_resp.json()
        plasma_data = plasma_resp.json()

        # Get last valid MAG row (skip header at index 0)
        mag_rows = [r for r in mag_data[1:] if r[1] not in (None, "null", "-9999.9")]
        plasma_rows = [r for r in plasma_data[1:] if r[1] not in (None, "null", "-9999.9")]

        if not mag_rows or not plasma_rows:
            logger.warning("No valid solar wind data")
            return None

        # Latest MAG: [time_tag, Bx, By, Bz, Bt, lat, lon]
        mag = mag_rows[-1]
        plasma = plasma_rows[-1]

        ts_str = str(mag[0])
        try:
            ts = datetime.fromisoformat(ts_str.replace(" ", "T").replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except ValueError:
            ts = datetime.now(timezone.utc)

        bx = float(mag[1])
        by = float(mag[2])
        bz = float(mag[3])
        bt = float(mag[4])

        # Plasma: [time_tag, density, speed, temperature]
        density = float(plasma[1])
        speed = float(plasma[2])
        temp = float(plasma[3])

        clock = compute_clock_angle(by, bz)
        cone = compute_cone_angle(bx, bt)
        dyn_p = compute_dynamic_pressure(density, speed)
        alfven = compute_alfven_mach(speed, bt, density)

        logger.debug("Solar wind: Bz=%.1f nT, speed=%.0f km/s, density=%.1f cm⁻³",
                     bz, speed, density)

        return SolarWindSnapshot(
            timestamp_utc=ts,
            bt=round(bt, 2), bx=round(bx, 2), by=round(by, 2), bz=round(bz, 2),
            speed=round(speed, 1), density=round(density, 2), temperature=round(temp, 0),
            clock_angle_deg=round(clock, 1), cone_angle_deg=round(cone, 1),
            dyn_pressure_npa=round(dyn_p, 3), alfven_mach=round(alfven, 2),
        )

    except Exception as exc:
        logger.error("Solar wind fetch failed: %s", exc)
        return None


async def fetch_flare_events(client: httpx.AsyncClient) -> list[FlareEvent]:
    """Fetch last 7 days of GOES-detected flare events from NOAA."""
    try:
        resp = await client.get(
            f"{SWPC}/json/goes/primary/xray-flares-7-day.json",
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json()

        events = []
        for r in reversed(raw):  # most recent first
            try:
                cls = r.get("classType", "?")
                if not cls or cls == "?":
                    continue

                def parse_ts(s: Optional[str]) -> Optional[datetime]:
                    if not s:
                        return None
                    try:
                        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
                        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
                    except Exception:
                        return None

                events.append(FlareEvent(
                    start_time_utc=parse_ts(r.get("beginTime")) or datetime.now(timezone.utc),
                    peak_time_utc=parse_ts(r.get("peakTime")),
                    end_time_utc=parse_ts(r.get("endTime")),
                    goes_class=cls,
                    location=r.get("location", "—"),
                    region=str(r.get("activeRegionNum", "—")),
                ))
            except Exception:
                continue

        logger.debug("Flare events: %d loaded", len(events))
        return events[:20]  # keep only most recent 20

    except Exception as exc:
        logger.error("Flare events fetch failed: %s", exc)
        return []


async def fetch_3day_probabilities(client: httpx.AsyncClient) -> Optional[SWPCProbabilities]:
    """
    Parse NOAA SWPC 3-day forecast text for M+/X+ probabilities.
    Returns Day 1 (today) probabilities.
    """
    try:
        resp = await client.get(
            f"{SWPC}/text/3-day-forecast.txt",
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        text = resp.text

        # Parse issued time
        issued_ts = datetime.now(timezone.utc)
        issued_match = re.search(r"Issued:\s*(\d{4}\s+\w+\s+\d+\s+\d{4}\s+UTC)", text)
        if issued_match:
            try:
                issued_ts = datetime.strptime(
                    issued_match.group(1), "%Y %B %d %H%M UTC"
                ).replace(tzinfo=timezone.utc)
            except Exception:
                pass

        # Extract M-class probability for Day 1
        # NOAA format examples:
        #   "M-class          30%  25%  20%"
        #   "M class          30%  25%  20%"
        m_match = re.search(r"M[- ]class[\s:]+?(\d+)%", text, re.IGNORECASE)
        x_match = re.search(r"X[- ]class[\s:]+?(\d+)%", text, re.IGNORECASE)
        proton_match = re.search(r"Proton[\s:]+?(\d+)%", text, re.IGNORECASE)
        # Alternative: look for "Flares: ... M30 X05" format
        if not m_match:
            m_match2 = re.search(r"M(\d+)", text)
            m_match = m_match2
        if not x_match:
            x_match2 = re.search(r"X(\d+)", text)
            x_match = x_match2

        m_pct = float(m_match.group(1)) if m_match else 0.0
        x_pct = float(x_match.group(1)) if x_match else 0.0
        proton_pct = float(proton_match.group(1)) if proton_match else 0.0

        logger.debug("SWPC 3-day: M=%d%%, X=%d%%, Proton=%d%%", m_pct, x_pct, proton_pct)
        return SWPCProbabilities(
            issued_utc=issued_ts,
            m_class_pct=m_pct,
            x_class_pct=x_pct,
            proton_pct=proton_pct,
        )

    except Exception as exc:
        logger.error("3-day forecast fetch failed: %s", exc)
        return None


# ── Main poller ───────────────────────────────────────────────────────────────

async def poll_once() -> LiveSnapshot:
    """
    Fetch all NOAA data sources in parallel and return a combined snapshot.
    This is called every 60 seconds by the FastAPI background task.
    """
    async with httpx.AsyncClient(
        headers={"User-Agent": "AdityScan/3.0 (research; contact: adityscan@example.com)"},
        follow_redirects=True,
    ) as client:
        goes, wind, flares, probs = await asyncio.gather(
            fetch_goes_xrs(client),
            fetch_solar_wind(client),
            fetch_flare_events(client),
            fetch_3day_probabilities(client),
            return_exceptions=False,
        )

    return LiveSnapshot(goes=goes, wind=wind, flares=flares, probs=probs)


# ── Derived nowcast (rule-based, until ML model is trained) ───────────────────

def build_nowcast_result(snapshot: LiveSnapshot) -> dict:
    """
    Convert a LiveSnapshot into the `nowcast_result` dict the API serves.
    Uses NOAA published probabilities + GOES flux as the primary signal.
    Once the ML model is trained, this function is replaced by model inference.
    """
    if not snapshot.goes:
        return {}

    goes = snapshot.goes
    wind = snapshot.wind
    probs = snapshot.probs

    flux = goes.flux_1_8
    z = goes.z_score
    cls = goes.goes_class[0]  # B / C / M / X

    # Base flare probability from GOES class + z-score
    base_prob = {
        "A": 2.0, "B": 5.0, "C": 15.0, "M": 55.0, "X": 90.0,
    }.get(cls, 5.0)

    # Boost by z-score
    z_boost = min(30.0, max(0.0, (z - 2.0) * 5.0))

    # Use NOAA published probability as ceiling reference
    noaa_m = probs.m_class_pct if probs else base_prob
    noaa_x = probs.x_class_pct if probs else (base_prob * 0.3)

    # Combined estimate
    flare_prob = min(99.0, max(0.0, base_prob + z_boost))
    if cls in ("M", "X"):
        flare_prob = max(flare_prob, noaa_m)

    # Uncertainty: larger for lower-class (less signal)
    uncertainty = {"A": 8.0, "B": 7.0, "C": 6.0, "M": 5.0, "X": 3.0}.get(cls, 6.0)

    # Class probabilities (distribution)
    class_probs = _class_probability_distribution(flux, z)

    # CME risk: elevated if M+, Bz southward, high solar wind pressure
    cme_risk = _estimate_cme_risk(cls, wind)

    # Multi-horizon forecast using NOAA probabilities as 24h anchor
    # Then interpolate for shorter horizons
    p_24h = noaa_m + noaa_x
    forecast = {
        "5min":  {"mean": round(flare_prob * 0.85, 1), "lower": round(flare_prob * 0.85 - 12, 1), "upper": round(flare_prob * 0.85 + 12, 1)},
        "10min": {"mean": round(flare_prob * 0.92, 1), "lower": round(flare_prob * 0.92 - 10, 1), "upper": round(flare_prob * 0.92 + 10, 1)},
        "15min": {"mean": round(flare_prob, 1),        "lower": round(flare_prob - 13, 1),         "upper": round(flare_prob + 13, 1)},
        "30min": {"mean": round(flare_prob * 0.88, 1), "lower": round(flare_prob * 0.88 - 15, 1),  "upper": round(flare_prob * 0.88 + 15, 1)},
        "60min": {"mean": round(p_24h * 0.8, 1),       "lower": round(p_24h * 0.8 - 18, 1),        "upper": round(p_24h * 0.8 + 18, 1)},
    }
    # Clamp all to [0, 99]
    for h in forecast.values():
        h["mean"] = max(0, min(99, h["mean"]))
        h["lower"] = max(0, min(99, h["lower"]))
        h["upper"] = max(0, min(99, h["upper"]))

    active_modalities = ["goes"]
    if wind:
        active_modalities.extend(["mag", "swis"])

    return {
        "flare_probability": round(flare_prob, 1),
        "flare_prob_uncertainty": uncertainty,
        "class_probs": class_probs,
        "cme_risk": round(cme_risk, 1),
        "active_modalities": active_modalities,
        "forecast": forecast,
        "noaa_published": {
            "m_class_pct": noaa_m,
            "x_class_pct": noaa_x,
            "proton_pct": probs.proton_pct if probs else 0.0,
        },
        "source": "noaa_swpc_live",
    }


def _class_probability_distribution(flux: float, z: float) -> dict:
    """Rough Bayesian-style class distribution given current flux."""
    log_flux = math.log10(max(flux, 1e-9))

    # Gaussian centres for each class in log-space
    centres = {"B": -7.0, "C": -6.0, "M": -5.0, "X": -4.0, "X+": -3.5}
    raw = {k: math.exp(-0.5 * ((log_flux - c) / 0.8) ** 2) for k, c in centres.items()}
    total = sum(raw.values()) or 1.0
    pct = {k: round(v / total * 100, 1) for k, v in raw.items()}

    # Inject z-score signal: boost M/X if elevated
    if z > 4:
        excess = min(20.0, (z - 4) * 3)
        pct["M"] = min(95, pct["M"] + excess * 0.6)
        pct["X"] = min(95, pct["X"] + excess * 0.4)

    return pct


def _estimate_cme_risk(cls: str, wind: Optional[SolarWindSnapshot]) -> float:
    """Heuristic CME risk percentage."""
    base = {"A": 1.0, "B": 3.0, "C": 12.0, "M": 45.0, "X": 80.0}.get(cls, 5.0)

    if wind is None:
        return base

    # Southward Bz boosts geomagnetic effectiveness
    if wind.bz < -5:
        base = min(95, base + 15)
    elif wind.bz < -10:
        base = min(95, base + 25)

    # High dynamic pressure boosts
    if wind.dyn_pressure_npa > 4.0:
        base = min(95, base + 10)

    return base
