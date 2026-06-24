/**
 * AdityScan v3 — Real-time API client
 *
 * Provides a single `useRealTimeData()` hook that:
 *  1. Connects to the backend WebSocket for live push updates
 *  2. Falls back to REST polling every 60s if WebSocket is unavailable
 *  3. Fetches the light curve and solar wind time series via REST (larger payloads)
 *
 * The hook returns a stable `DashboardData` object that replaces all demo-data.ts imports.
 */

import { useEffect, useRef, useState, useCallback } from "react";

// ── Env configuration ─────────────────────────────────────────────────────────

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000").replace(/\/$/, "");
const WS_URL   = (import.meta.env.VITE_WS_URL   ?? "ws://localhost:8000/ws/live");

// ── Types ─────────────────────────────────────────────────────────────────────

export type ActivityMode = "QUIET" | "ELEVATED" | "ACTIVE" | "EXTREME";
export type SatStatus    = "LIVE" | "OFFLINE" | "DEGRADED";

export interface Satellite {
  id: string;
  name: string;
  agency: string;
  flag: string;
  desc: string;
  status: SatStatus;
  lastDataSec: number;
}

export interface NowcastData {
  flare_probability: number;
  flare_probability_uncertainty: number;
  class_probabilities: Record<string, number>;
  cme_risk: number;
  active_modalities: string[];
  noaa_published: { m_class_pct: number; x_class_pct: number; proton_pct: number };
}

export interface ForecastHorizon {
  mean: number;
  lower: number;
  upper: number;
}

export interface FlareEvent {
  start_time: string;
  peak_time: string | null;
  end_time: string | null;
  goes_class: string;
  location: string;
  region: string;
}

export interface LightCurvePoint {
  t: number;         // Unix ms
  flux_1_8: number;  // W/m²
  flux_0p5_4: number;
  goes_class: string;
  z: number;
}

export interface SolarWindPoint {
  t: number;
  bt: number; bx: number; by: number; bz: number;
  speed: number; density: number;
  dyn_pressure: number; alfven_mach: number;
  clock_angle: number; cone_angle: number;
}

export interface SolarWindCurrent {
  bz: number; bt: number; bx: number; by: number;
  speed: number; density: number;
  clock_angle: number; cone_angle: number;
  dyn_pressure: number; alfven_mach: number;
}

export interface DashboardData {
  // Connection
  isConnected: boolean;
  isLoading: boolean;
  lastUpdated: Date | null;
  dataSource: "live" | "polling" | "stale";
  // Current state
  activityMode: ActivityMode;
  goesClass: string;
  goesFlux: number;
  zScore: number;
  // Satellite status
  satellites: Satellite[];
  // Nowcast
  nowcast: NowcastData;
  // Forecast horizons
  forecast: Record<string, ForecastHorizon>;
  // HOPE precursor flag
  hopeFired: boolean;
  // Time series
  lightCurve: LightCurvePoint[];
  solarWind: SolarWindPoint[];
  solarWindCurrent: SolarWindCurrent | null;
  // Event catalog
  flareEvents: FlareEvent[];
}

// ── Defaults (shown while first load is in progress) ─────────────────────────

const DEFAULT_NOWCAST: NowcastData = {
  flare_probability: 0,
  flare_probability_uncertainty: 0,
  class_probabilities: { B: 0, C: 0, M: 0, X: 0, "X+": 0 },
  cme_risk: 0,
  active_modalities: [],
  noaa_published: { m_class_pct: 0, x_class_pct: 0, proton_pct: 0 },
};

const DEFAULT_DATA: DashboardData = {
  isConnected: false,
  isLoading: true,
  lastUpdated: null,
  dataSource: "polling",
  activityMode: "QUIET",
  goesClass: "—",
  goesFlux: 0,
  zScore: 0,
  satellites: [],
  nowcast: DEFAULT_NOWCAST,
  forecast: {},
  hopeFired: false,
  lightCurve: [],
  solarWind: [],
  solarWindCurrent: null,
  flareEvents: [],
};

// ── Satellite status mapping ──────────────────────────────────────────────────

function buildSatellites(statusData: any): Satellite[] {
  const SAT_META: Record<string, { id: string; flag: string; desc: string }> = {
    SoLEXS:     { id: "solexs", flag: "🇮🇳", desc: "Solar Low Energy X-ray Spectrometer • 2.8–12 keV • 1-s cadence" },
    HEL1OS:     { id: "hel1os", flag: "🇮🇳", desc: "High Energy L1 X-ray Spectrometer • 5–150 keV • 1-s cadence" },
    MAG:        { id: "mag",    flag: "🇮🇳", desc: "Dual fluxgate magnetometer • IMF Bx/By/Bz (GSE/GSM) • 10-s L2" },
    "ASPEX-SWIS": { id: "swis", flag: "🇮🇳", desc: "Solar Wind Ion Spectrometer • Proton density/temp/speed • CDF L2" },
    SUIT:       { id: "suit",   flag: "🇮🇳", desc: "Solar UV Imaging Telescope • NUV 200–400 nm • Flare-triggered" },
    XRS:        { id: "goes",   flag: "🇺🇸", desc: "X-Ray Sensor • 1–8 Å (GOES class) • 1-min real-time, NOAA SWPC" },
    "HMI SHARP": { id: "sharp", flag: "🇺🇸", desc: "Helioseismic Magnetic Imager • 18 AR magnetic params • 12-min" },
  };

  if (!statusData?.satellites) return [];

  const now = Date.now() / 1000;
  return statusData.satellites.map((s: any) => {
    const meta = SAT_META[s.instrument] ?? { id: s.instrument.toLowerCase(), flag: "🛰", desc: s.description };
    const age = s.last_data_unix ? Math.round(now - s.last_data_unix) : 0;

    let status: SatStatus = "OFFLINE";
    if (s.available && s.data_quality === "GOOD") status = "LIVE";
    else if (s.available || s.data_quality === "DEGRADED") status = "DEGRADED";

    return {
      id: meta.id,
      name: `${s.name} / ${s.instrument}`,
      agency: s.agency,
      flag: meta.flag,
      desc: meta.desc,
      status,
      lastDataSec: age,
    };
  });
}

// ── Main hook ─────────────────────────────────────────────────────────────────

export function useRealTimeData(): DashboardData {
  const [data, setData] = useState<DashboardData>(DEFAULT_DATA);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pollTimer = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── REST helpers ──────────────────────────────────────────────────────────

  const fetchStatus = useCallback(async () => {
    const r = await fetch(`${API_BASE}/api/status`);
    if (!r.ok) throw new Error("status " + r.status);
    return r.json();
  }, []);

  const fetchNowcast = useCallback(async () => {
    const r = await fetch(`${API_BASE}/api/nowcast`);
    if (!r.ok) throw new Error("nowcast " + r.status);
    return r.json();
  }, []);

  const fetchForecast = useCallback(async () => {
    const r = await fetch(`${API_BASE}/api/forecast`);
    if (!r.ok) throw new Error("forecast " + r.status);
    return r.json();
  }, []);

  const fetchLightCurve = useCallback(async () => {
    const r = await fetch(`${API_BASE}/api/lightcurves?minutes=30`);
    if (!r.ok) throw new Error("lightcurves " + r.status);
    return r.json();
  }, []);

  const fetchSolarWind = useCallback(async () => {
    const r = await fetch(`${API_BASE}/api/solar-wind?hours=4`);
    if (!r.ok) throw new Error("solar-wind " + r.status);
    return r.json();
  }, []);

  const fetchCatalog = useCallback(async () => {
    const r = await fetch(`${API_BASE}/api/catalog?limit=10&min_class=C`);
    if (!r.ok) throw new Error("catalog " + r.status);
    return r.json();
  }, []);

  // ── Full REST refresh (used on first load + WebSocket fallback) ───────────

  const refreshAll = useCallback(async () => {
    try {
      const [status, nowcast, forecast, lc, sw, catalog] = await Promise.allSettled([
        fetchStatus(), fetchNowcast(), fetchForecast(),
        fetchLightCurve(), fetchSolarWind(), fetchCatalog(),
      ]);

      const statusData  = status.status  === "fulfilled" ? status.value  : null;
      const nowcastData = nowcast.status === "fulfilled" ? nowcast.value : null;
      const forecastData = forecast.status === "fulfilled" ? forecast.value : null;
      const lcData       = lc.status      === "fulfilled" ? lc.value      : null;
      const swData       = sw.status      === "fulfilled" ? sw.value      : null;
      const catData      = catalog.status === "fulfilled" ? catalog.value : null;

      setData(prev => ({
        ...prev,
        isLoading: false,
        lastUpdated: new Date(),
        dataSource: wsRef.current?.readyState === WebSocket.OPEN ? "live" : "polling",

        activityMode: (statusData?.activity_mode ?? prev.activityMode) as ActivityMode,
        goesClass:    statusData?.goes_class   ?? prev.goesClass,
        zScore:       statusData?.z_score      ?? prev.zScore,

        satellites: statusData ? buildSatellites(statusData) : prev.satellites,

        nowcast: nowcastData ? {
          flare_probability: nowcastData.flare_probability_pct ?? 0,
          flare_probability_uncertainty: nowcastData.flare_prob_uncertainty_pct ?? 0,
          class_probabilities: nowcastData.class_probabilities ?? {},
          cme_risk: nowcastData.cme_risk_pct ?? 0,
          active_modalities: nowcastData.active_modalities ?? [],
          noaa_published: nowcastData.noaa_published ?? { m_class_pct: 0, x_class_pct: 0, proton_pct: 0 },
        } : prev.nowcast,

        goesFlux:  nowcastData?.goes_flux_wm2 ?? prev.goesFlux,

        forecast: forecastData?.horizons ?? prev.forecast,

        lightCurve: lcData?.points ?? prev.lightCurve,
        solarWind:  swData?.points ?? prev.solarWind,
        solarWindCurrent: swData?.current ?? prev.solarWindCurrent,

        flareEvents: catData?.catalog ?? prev.flareEvents,
      }));
    } catch (err) {
      console.error("[AdityScan] REST refresh error:", err);
      setData(prev => ({ ...prev, isLoading: false }));
    }
  }, [fetchStatus, fetchNowcast, fetchForecast, fetchLightCurve, fetchSolarWind, fetchCatalog]);

  // ── WebSocket push handler ────────────────────────────────────────────────

  const handleWsMessage = useCallback((raw: string) => {
    try {
      const msg = JSON.parse(raw);
      if (msg.type !== "update") return;

      setData(prev => ({
        ...prev,
        isLoading: false,
        isConnected: true,
        dataSource: "live",
        lastUpdated: new Date(),
        activityMode: (msg.activity_mode ?? prev.activityMode) as ActivityMode,
        goesClass:    msg.goes_class   ?? prev.goesClass,
        goesFlux:     msg.goes_flux    ?? prev.goesFlux,
        zScore:       msg.z_score      ?? prev.zScore,
        hopeFired:    msg.hope_fired   ?? false,

        solarWindCurrent: msg.imf_bz != null ? {
          bz: msg.imf_bz, bt: msg.imf_bt ?? 0,
          bx: 0, by: 0,   // not in WS payload
          speed:       msg.sw_speed   ?? 0,
          density:     msg.sw_density ?? 0,
          clock_angle: msg.clock_angle ?? 0,
          cone_angle:  msg.cone_angle  ?? 0,
          dyn_pressure: msg.dyn_pressure ?? 0,
          alfven_mach:  msg.alfven_mach  ?? 0,
        } : prev.solarWindCurrent,

        nowcast: {
          flare_probability: msg.flare_probability ?? prev.nowcast.flare_probability,
          flare_probability_uncertainty: msg.flare_prob_uncertainty ?? prev.nowcast.flare_probability_uncertainty,
          class_probabilities: msg.class_probabilities ?? prev.nowcast.class_probabilities,
          cme_risk: msg.cme_risk ?? prev.nowcast.cme_risk,
          active_modalities: msg.active_modalities ?? prev.nowcast.active_modalities,
          noaa_published: msg.noaa_published ?? prev.nowcast.noaa_published,
        },

        forecast: msg.forecast ?? prev.forecast,
      }));
    } catch (err) {
      console.error("[AdityScan] WS parse error:", err);
    }
  }, []);

  // ── WebSocket connection ──────────────────────────────────────────────────

  const connectWS = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      console.info("[AdityScan] WebSocket connected to", WS_URL);
      setData(prev => ({ ...prev, isConnected: true, dataSource: "live" }));
      // Ping every 25s to keep connection alive
      const ping = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) ws.send("ping");
      }, 25_000);
      ws.addEventListener("close", () => clearInterval(ping));
    };

    ws.onmessage = (ev) => handleWsMessage(ev.data as string);

    ws.onerror = (err) => {
      console.warn("[AdityScan] WebSocket error — falling back to REST polling", err);
    };

    ws.onclose = () => {
      console.info("[AdityScan] WebSocket closed, reconnecting in 5s...");
      setData(prev => ({ ...prev, isConnected: false, dataSource: "polling" }));
      wsRef.current = null;
      reconnectTimer.current = setTimeout(connectWS, 5_000);
    };
  }, [handleWsMessage]);

  // ── Bootstrap ─────────────────────────────────────────────────────────────

  useEffect(() => {
    // Initial full REST load
    refreshAll();

    // Start WebSocket
    connectWS();

    // REST poll every 60s as fallback / to refresh slow-changing data (catalog, light curve)
    pollTimer.current = setInterval(refreshAll, 60_000);

    return () => {
      wsRef.current?.close();
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      if (pollTimer.current) clearInterval(pollTimer.current);
    };
  }, []);  // eslint-disable-line react-hooks/exhaustive-deps

  return data;
}

// ── Re-export color helper (no demo data dependency) ─────────────────────────

export function goesColor(cls: string): string {
  const c = cls[0];
  if (c === "B") return "var(--goes-b)";
  if (c === "C") return "var(--goes-c)";
  if (c === "M") return "var(--goes-m)";
  if (c === "X") {
    if (cls.includes("+")) return "var(--goes-xp)";
    return "var(--goes-x)";
  }
  return "var(--muted-foreground)";
}

export type { ForecastHorizon as ForecastEntry };
