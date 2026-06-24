"""
FastAPI application — AdityScan v3 Backend API.
================================================
LIVE DATA MODE — polls NOAA SWPC every 60 seconds.

Endpoints:
  GET  /api/nowcast              → current flare probability + uncertainty
  GET  /api/forecast             → multi-horizon forecast
  GET  /api/lightcurves          → recent GOES XRS light curve (6-hour)
  GET  /api/catalog              → real NOAA 7-day flare catalog
  GET  /api/status               → system status
  GET  /api/solar-wind           → in-situ solar wind time series
  WS   /ws/live                  → WebSocket real-time dashboard push
  GET  /health                   → health check
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pipeline.ingestion.noaa_swpc_live import (
    poll_once,
    build_nowcast_result,
    LiveSnapshot,
    GOESSnapshot,
    SolarWindSnapshot,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Application state ─────────────────────────────────────────────────────────

class AppState:
    """Global live state, filled every ~60 s by the NOAA poller."""
    # Current snapshot
    latest: Optional[LiveSnapshot] = None
    # Derived
    activity_mode: str = "QUIET"
    goes_class: str = "A0.0"
    goes_flux: float = 0.0
    z_score: float = 0.0
    last_update_unix: float = 0.0
    # Ring buffer: last 6 h of GOES readings for light curve (1-min cadence = 360 points)
    goes_ring: list[dict] = []
    wind_ring: list[dict] = []
    # Derived nowcast
    nowcast_result: Optional[dict] = None
    # WebSocket connections
    active_websockets: list[WebSocket] = []
    # Aditya-L1 availability (updated from PRADAN or set manually)
    solexs_available: bool = False
    hel1os_available: bool = False
    mag_available: bool = False
    swis_available: bool = False
    suit_available: bool = False
    goes_rt_available: bool = True

state = AppState()

POLL_INTERVAL_S = 60  # NOAA SWPC updates at 1-min cadence
RING_MAX = 360        # 6 hours at 1-min cadence


def _flux_to_mode(flux: float, z: float) -> str:
    """Determine ActivityMode from GOES flux + z-score."""
    if flux >= 1e-4:            # X-class
        return "EXTREME"
    if flux >= 1e-5:            # M-class
        return "ACTIVE"
    if flux >= 1e-6 or z >= 3: # C-class or significant rise
        return "ELEVATED"
    return "QUIET"


# ── Background tasks ──────────────────────────────────────────────────────────

async def _noaa_poller() -> None:
    """
    Background asyncio task: polls NOAA SWPC every POLL_INTERVAL_S seconds.
    Updates global AppState and pushes to all WebSocket clients.
    """
    logger.info("NOAA SWPC poller starting (interval=%ds)", POLL_INTERVAL_S)
    while True:
        try:
            snapshot = await poll_once()
            state.latest = snapshot

            if snapshot.goes:
                g = snapshot.goes
                state.goes_class = g.goes_class
                state.goes_flux = g.flux_1_8
                state.z_score = g.z_score
                state.last_update_unix = g.timestamp_utc.timestamp()
                state.activity_mode = _flux_to_mode(g.flux_1_8, g.z_score)
                state.goes_rt_available = True

                # Append to ring buffer
                state.goes_ring.append({
                    "t": int(g.timestamp_utc.timestamp() * 1000),
                    "flux_1_8": g.flux_1_8,
                    "flux_0p5_4": g.flux_0p5_4,
                    "goes_class": g.goes_class,
                    "z": g.z_score,
                })
                if len(state.goes_ring) > RING_MAX:
                    state.goes_ring = state.goes_ring[-RING_MAX:]

            if snapshot.wind:
                w = snapshot.wind
                state.wind_ring.append({
                    "t": int(w.timestamp_utc.timestamp() * 1000),
                    "bt": w.bt, "bx": w.bx, "by": w.by, "bz": w.bz,
                    "speed": w.speed, "density": w.density,
                    "dyn_pressure": w.dyn_pressure_npa,
                    "alfven_mach": w.alfven_mach,
                    "clock_angle": w.clock_angle_deg,
                    "cone_angle": w.cone_angle_deg,
                })
                if len(state.wind_ring) > RING_MAX:
                    state.wind_ring = state.wind_ring[-RING_MAX:]

                state.mag_available = True
                state.swis_available = True

            # Build nowcast
            state.nowcast_result = build_nowcast_result(snapshot)

            logger.info(
                "Live: GOES %s | z=%.2f | mode=%s | M%%=%.0f",
                state.goes_class, state.z_score, state.activity_mode,
                state.nowcast_result.get("flare_probability", 0) if state.nowcast_result else 0,
            )

            # Push to all connected WebSocket clients
            await _ws_broadcast()

        except Exception as exc:
            logger.error("NOAA poller error: %s", exc, exc_info=True)
            state.goes_rt_available = False

        await asyncio.sleep(POLL_INTERVAL_S)


async def _ws_broadcast() -> None:
    """Push current state to all connected WebSocket clients."""
    if not state.active_websockets or not state.nowcast_result:
        return

    if not state.latest or not state.latest.goes:
        return

    g = state.latest.goes
    w = state.latest.wind

    payload = {
        "type": "update",
        "timestamp": state.last_update_unix,
        "activity_mode": state.activity_mode,
        "goes_class": state.goes_class,
        "goes_flux": state.goes_flux,
        "z_score": round(state.z_score, 2),
        "imf_bz": round(w.bz, 2) if w else None,
        "imf_bt": round(w.bt, 2) if w else None,
        "sw_speed": round(w.speed, 1) if w else None,
        "sw_density": round(w.density, 2) if w else None,
        "dyn_pressure": round(w.dyn_pressure_npa, 3) if w else None,
        "clock_angle": round(w.clock_angle_deg, 1) if w else None,
        "cone_angle": round(w.cone_angle_deg, 1) if w else None,
        "alfven_mach": round(w.alfven_mach, 2) if w else None,
        "flare_probability": state.nowcast_result.get("flare_probability", 0),
        "flare_prob_uncertainty": state.nowcast_result.get("flare_prob_uncertainty", 0),
        "class_probabilities": state.nowcast_result.get("class_probs", {}),
        "cme_risk": state.nowcast_result.get("cme_risk", 0),
        "forecast": state.nowcast_result.get("forecast", {}),
        "noaa_published": state.nowcast_result.get("noaa_published", {}),
        "active_modalities": state.nowcast_result.get("active_modalities", []),
        "hope_fired": state.activity_mode == "EXTREME",
        "satellite": g.satellite,
    }

    msg = json.dumps(payload)
    dead = []
    for ws in state.active_websockets:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    for d in dead:
        if d in state.active_websockets:
            state.active_websockets.remove(d)


# ── App lifespan ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AdityScan v3 API starting up — Live NOAA SWPC mode")
    # Fire first poll immediately so data is ready before first request
    try:
        snapshot = await poll_once()
        state.latest = snapshot
        if snapshot.goes:
            g = snapshot.goes
            state.goes_class = g.goes_class
            state.goes_flux = g.flux_1_8
            state.z_score = g.z_score
            state.last_update_unix = g.timestamp_utc.timestamp()
            state.activity_mode = _flux_to_mode(g.flux_1_8, g.z_score)
            state.goes_rt_available = True
            state.goes_ring.append({
                "t": int(g.timestamp_utc.timestamp() * 1000),
                "flux_1_8": g.flux_1_8, "flux_0p5_4": g.flux_0p5_4,
                "goes_class": g.goes_class, "z": g.z_score,
            })
        if snapshot.wind:
            state.mag_available = True
            state.swis_available = True
        state.nowcast_result = build_nowcast_result(snapshot)
        logger.info("Initial GOES reading: %s", state.goes_class)
    except Exception as exc:
        logger.warning("Initial poll failed: %s — serving empty state", exc)

    poller_task = asyncio.create_task(_noaa_poller())
    yield
    poller_task.cancel()
    logger.info("AdityScan v3 API shutdown.")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="AdityScan v3 API",
    description="Real-time solar flare nowcasting. Live data from GOES/NOAA SWPC.",
    version="3.0.0",
    lifespan=lifespan,
)

ALLOWED_ORIGINS = os.environ.get(
    "CORS_ORIGINS",
    "https://adityscan.pages.dev,http://localhost:5173,http://localhost:8080,http://localhost:4173",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Pydantic models ───────────────────────────────────────────────────────────

class NowcastResponse(BaseModel):
    timestamp: float
    activity_mode: str
    goes_class: str
    goes_flux_wm2: float
    z_score: float
    flare_probability_pct: float
    flare_prob_uncertainty_pct: float
    class_probabilities: dict
    cme_risk_pct: float
    active_modalities: list[str]
    noaa_published: dict
    data_source: str = "noaa_swpc_live"

class ForecastResponse(BaseModel):
    timestamp: float
    horizons: dict
    model_version: str = "3.0.0-rule-based"
    noaa_published: dict

class SatelliteStatus(BaseModel):
    name: str
    instrument: str
    agency: str
    available: bool
    last_data_unix: Optional[float]
    data_quality: str
    description: str

class SystemStatusResponse(BaseModel):
    activity_mode: str
    mode_description: str
    goes_class: str
    z_score: float
    compute_fraction: float
    alert_armed: bool
    last_update_unix: float
    satellites: list[SatelliteStatus]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/nowcast", response_model=NowcastResponse)
async def get_nowcast():
    """Current solar activity status and flare nowcast (live GOES data)."""
    if state.nowcast_result is None:
        raise HTTPException(503, "Nowcast not yet available — initial NOAA poll pending.")
    return NowcastResponse(
        timestamp=state.last_update_unix or time.time(),
        activity_mode=state.activity_mode,
        goes_class=state.goes_class,
        goes_flux_wm2=state.goes_flux,
        z_score=round(state.z_score, 2),
        flare_probability_pct=state.nowcast_result.get("flare_probability", 0.0),
        flare_prob_uncertainty_pct=state.nowcast_result.get("flare_prob_uncertainty", 0.0),
        class_probabilities=state.nowcast_result.get("class_probs", {}),
        cme_risk_pct=state.nowcast_result.get("cme_risk", 0.0),
        active_modalities=state.nowcast_result.get("active_modalities", []),
        noaa_published=state.nowcast_result.get("noaa_published", {}),
    )


@app.get("/api/forecast", response_model=ForecastResponse)
async def get_forecast():
    """Multi-horizon flare probability forecast (NOAA SWPC + rule-based)."""
    if state.nowcast_result is None:
        raise HTTPException(503, "Forecast not yet available.")
    return ForecastResponse(
        timestamp=state.last_update_unix or time.time(),
        horizons=state.nowcast_result.get("forecast", {}),
        noaa_published=state.nowcast_result.get("noaa_published", {}),
    )


@app.get("/api/lightcurves")
async def get_lightcurves(minutes: int = 30):
    """
    GOES XRS light curve — last N minutes of real NOAA data.
    Also includes any GOES 6-hour archive data loaded in ring buffer.
    """
    minutes = min(minutes, 360)
    # Return the ring buffer (already time-ordered)
    cutoff_ms = (time.time() - minutes * 60) * 1000
    data = [p for p in state.goes_ring if p["t"] >= cutoff_ms]
    return {
        "points": data,
        "minutes": minutes,
        "cadence_min": 1,
        "source": "noaa_swpc_live",
    }


@app.get("/api/solar-wind")
async def get_solar_wind(hours: int = 4):
    """In-situ solar wind time series (MAG + plasma) — real NOAA L1 data."""
    hours = min(hours, 6)
    cutoff_ms = (time.time() - hours * 3600) * 1000
    data = [p for p in state.wind_ring if p["t"] >= cutoff_ms]

    # Also expose current snapshot values
    current = None
    if state.latest and state.latest.wind:
        w = state.latest.wind
        current = {
            "bz": w.bz, "bt": w.bt, "bx": w.bx, "by": w.by,
            "speed": w.speed, "density": w.density,
            "clock_angle": w.clock_angle_deg, "cone_angle": w.cone_angle_deg,
            "dyn_pressure": w.dyn_pressure_npa, "alfven_mach": w.alfven_mach,
        }
    return {"points": data, "current": current, "source": "noaa_swpc_live"}


@app.get("/api/catalog")
async def get_catalog(limit: int = 20, min_class: str = "B"):
    """Real 7-day NOAA GOES flare event catalog."""
    if state.latest is None:
        return {"catalog": [], "total": 0}

    order = {"A": 0, "B": 1, "C": 2, "M": 3, "X": 4}
    min_ord = order.get(min_class[0].upper(), 0)

    events = []
    for e in (state.latest.flares or []):
        cls_ord = order.get(e.goes_class[0].upper(), 0)
        if cls_ord < min_ord:
            continue
        events.append({
            "start_time": e.start_time_utc.isoformat(),
            "peak_time": e.peak_time_utc.isoformat() if e.peak_time_utc else None,
            "end_time": e.end_time_utc.isoformat() if e.end_time_utc else None,
            "goes_class": e.goes_class,
            "location": e.location,
            "region": e.region,
        })

    return {"catalog": events[:limit], "total": len(events), "source": "noaa_swpc_7day"}


@app.get("/api/status", response_model=SystemStatusResponse)
async def get_status():
    """Full system status."""
    mode_descriptions = {
        "QUIET":    "Background Sun — minimal X-ray activity, 60-min NOAA polling",
        "ELEVATED": "C-class activity — enhanced monitoring, 60-s NOAA polling",
        "ACTIVE":   "M-class flare — alert armed, full real-time pipeline",
        "EXTREME":  "X-class — emergency mode, all data sources active",
    }
    now_unix = time.time()
    satellites = [
        SatelliteStatus(name="Aditya-L1", instrument="SoLEXS", agency="ISRO",
            available=state.solexs_available,
            last_data_unix=state.last_update_unix if state.solexs_available else None,
            data_quality="GOOD" if state.solexs_available else "UNAVAILABLE",
            description="Solar Low Energy X-ray Spectrometer, 2.8–12 keV, 1-s cadence"),
        SatelliteStatus(name="Aditya-L1", instrument="HEL1OS", agency="ISRO",
            available=state.hel1os_available,
            last_data_unix=state.last_update_unix if state.hel1os_available else None,
            data_quality="GOOD" if state.hel1os_available else "UNAVAILABLE",
            description="High Energy L1 X-ray Spectrometer, 5–150 keV, 1-s cadence"),
        SatelliteStatus(name="Aditya-L1", instrument="MAG", agency="ISRO",
            available=state.mag_available,
            last_data_unix=state.last_update_unix if state.mag_available else None,
            data_quality="GOOD" if state.mag_available else "UNAVAILABLE",
            description="Dual fluxgate magnetometer, IMF Bx/By/Bz GSE/GSM, 10-s L2"),
        SatelliteStatus(name="Aditya-L1", instrument="ASPEX-SWIS", agency="ISRO/PRL",
            available=state.swis_available,
            last_data_unix=state.last_update_unix if state.swis_available else None,
            data_quality="GOOD" if state.swis_available else "UNAVAILABLE",
            description="Solar Wind Ion Spectrometer, proton density/temp/speed, CDF L2"),
        SatelliteStatus(name="Aditya-L1", instrument="SUIT", agency="ISRO/IUCAA",
            available=state.suit_available, last_data_unix=None,
            data_quality="UNAVAILABLE",
            description="Solar UV Imaging Telescope, NUV 200–400 nm, flare-triggered"),
        SatelliteStatus(name="GOES-16/18", instrument="XRS", agency="NOAA",
            available=state.goes_rt_available,
            last_data_unix=state.last_update_unix if state.goes_rt_available else None,
            data_quality="GOOD" if state.goes_rt_available else "DEGRADED",
            description="X-Ray Sensor, 1–8 Å and 0.5–4 Å, 1-min real-time via NOAA SWPC"),
        SatelliteStatus(name="SDO", instrument="HMI SHARP", agency="NASA",
            available=True, last_data_unix=now_unix - 720,
            data_quality="GOOD",
            description="Helioseismic & Magnetic Imager, 18 AR magnetic params, 12-min cadence"),
    ]
    return SystemStatusResponse(
        activity_mode=state.activity_mode,
        mode_description=mode_descriptions.get(state.activity_mode, "Unknown"),
        goes_class=state.goes_class,
        z_score=round(state.z_score, 2),
        compute_fraction={"QUIET": 0.05, "ELEVATED": 0.25, "ACTIVE": 0.70, "EXTREME": 1.0}.get(state.activity_mode, 0.0),
        alert_armed=state.activity_mode in ("ACTIVE", "EXTREME"),
        last_update_unix=state.last_update_unix,
        satellites=satellites,
    )


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    """Real-time dashboard push via WebSocket. Sends live NOAA data on connect + each poll."""
    await websocket.accept()
    state.active_websockets.append(websocket)
    logger.info("WS client connected. Total: %d", len(state.active_websockets))

    # Send current state immediately on connect
    if state.nowcast_result:
        await _ws_broadcast()

    try:
        while True:
            data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            if data == "ping":
                await websocket.send_text("pong")
    except (WebSocketDisconnect, asyncio.TimeoutError):
        pass
    finally:
        if websocket in state.active_websockets:
            state.active_websockets.remove(websocket)
        logger.info("WS client disconnected. Remaining: %d", len(state.active_websockets))


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "3.0.0",
        "timestamp": time.time(),
        "goes_class": state.goes_class,
        "activity_mode": state.activity_mode,
        "data_age_s": round(time.time() - state.last_update_unix, 0) if state.last_update_unix else None,
        "data_source": "noaa_swpc_live",
    }
