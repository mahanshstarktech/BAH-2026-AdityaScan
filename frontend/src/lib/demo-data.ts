// Demo data + types for AdityScan v3
export type ActivityMode = "QUIET" | "ELEVATED" | "ACTIVE" | "EXTREME";
export type SatStatus = "LIVE" | "OFFLINE" | "DEGRADED";

export interface Satellite {
  id: string;
  name: string;
  agency: string;
  flag: string;
  desc: string;
  status: SatStatus;
  lastDataSec: number;
}

export const satellites: Satellite[] = [
  { id: "solexs", name: "Aditya-L1 / SoLEXS", agency: "ISRO", flag: "🇮🇳",
    desc: "Solar Low Energy X-ray Spectrometer • 2.8–12 keV • 1-s cadence",
    status: "LIVE", lastDataSec: 2 },
  { id: "hel1os", name: "Aditya-L1 / HEL1OS", agency: "ISRO", flag: "🇮🇳",
    desc: "High Energy L1 X-ray Spectrometer • 5–150 keV • 1-s cadence",
    status: "LIVE", lastDataSec: 3 },
  { id: "mag", name: "Aditya-L1 / MAG", agency: "ISRO", flag: "🇮🇳",
    desc: "Dual fluxgate magnetometer • IMF Bx/By/Bz (GSE/GSM) • 10-s L2",
    status: "LIVE", lastDataSec: 8 },
  { id: "swis", name: "Aditya-L1 / ASPEX-SWIS", agency: "ISRO/PRL", flag: "🇮🇳",
    desc: "Solar Wind Ion Spectrometer • Proton density/temp/speed • CDF L2",
    status: "LIVE", lastDataSec: 12 },
  { id: "suit", name: "Aditya-L1 / SUIT", agency: "ISRO/IUCAA", flag: "🇮🇳",
    desc: "Solar UV Imaging Telescope • NUV 200–400 nm • Flare-triggered",
    status: "OFFLINE", lastDataSec: 0 },
  { id: "goes", name: "GOES-16/18 / XRS", agency: "NOAA", flag: "🇺🇸",
    desc: "X-Ray Sensor • 1–8 Å (GOES class) • 1-min real-time, NOAA SWPC",
    status: "LIVE", lastDataSec: 14 },
  { id: "sharp", name: "SDO / HMI SHARP", agency: "NASA", flag: "🇺🇸",
    desc: "Helioseismic Magnetic Imager • 18 AR magnetic params • 12-min",
    status: "DEGRADED", lastDataSec: 720 },
];

export const nowcast = {
  flare_probability: 61,
  flare_probability_uncertainty: 8,
  class_probabilities: { B: 8, C: 17, M: 45, X: 25, "X+": 5 },
  cme_risk: 42,
  active_modalities: ["solexs", "hel1os", "mag", "swis", "sharp"],
};

export const forecast = [
  { horizon: "5 min",  mean: 58, lower: 48, upper: 68, conf: "BOLD" },
  { horizon: "10 min", mean: 62, lower: 51, upper: 73, conf: "BOLD" },
  { horizon: "15 min", mean: 58, lower: 45, upper: 71, conf: "BOLD" },
  { horizon: "30 min", mean: 51, lower: 36, upper: 66, conf: "NORMAL" },
  { horizon: "60 min", mean: 41, lower: 24, upper: 58, conf: "FADED" },
];

// Generate realistic-ish light curve (last 30 min, 1s cadence ~ subsampled to 6s)
export function genLightCurve() {
  const now = Date.now();
  const points: { t: number; solexs: number; hel1os: number; goes: number }[] = [];
  for (let i = 300; i >= 0; i--) {
    const t = now - i * 6_000;
    const noise = () => (Math.random() - 0.5) * 0.15;
    // Base level around C-class with rising flare near end
    const ramp = Math.max(0, (300 - i) / 300);
    const flare = ramp > 0.7 ? Math.pow(10, (ramp - 0.7) * 3) * 1.4e-5 : 0;
    const base = 6e-6 * (1 + 0.08 * Math.sin(i / 12));
    const solexs = Math.max(1e-7, base + flare + base * noise());
    const hel1os = Math.max(5e-8, base * 0.45 + flare * 0.7 + base * noise());
    const goes = Math.max(1e-7, solexs * 0.9);
    points.push({ t, solexs, hel1os, goes });
  }
  return points;
}

export function genConfidenceTimeline() {
  const now = Date.now();
  const out: { t: number; p5: number; p15: number; p60: number }[] = [];
  for (let i = 72; i >= 0; i--) {
    const t = now - i * 5 * 60_000;
    const wave = Math.sin(i / 7) * 18;
    out.push({
      t,
      p5: Math.max(2, Math.min(95, 45 + wave + (Math.random() - 0.5) * 6)),
      p15: Math.max(2, Math.min(95, 52 + wave * 0.9 + (Math.random() - 0.5) * 5)),
      p60: Math.max(2, Math.min(95, 38 + wave * 0.6 + (Math.random() - 0.5) * 4)),
    });
  }
  return out;
}

export function genSolarWind() {
  const now = Date.now();
  const out: { t: number; bt: number; bz: number; speed: number; density: number }[] = [];
  for (let i = 240; i >= 0; i--) {
    const t = now - i * 60_000;
    const bz = -3.8 + Math.sin(i / 18) * 4 + (Math.random() - 0.5) * 1.2;
    const bt = 7 + Math.abs(Math.sin(i / 22)) * 3 + (Math.random() - 0.5) * 0.6;
    const speed = 428 + Math.sin(i / 30) * 22 + (Math.random() - 0.5) * 8;
    const density = 4.2 + Math.sin(i / 20) * 1.4 + (Math.random() - 0.5) * 0.4;
    out.push({ t, bt, bz, speed, density });
  }
  return out;
}

export const sharpParams = [
  { axis: "USFLUX",   value: 78 },
  { axis: "TOTUSJH",  value: 86 },
  { axis: "MEANPOT",  value: 64 },
  { axis: "R_VALUE",  value: 81 },
  { axis: "SHRGT45",  value: 72 },
];

export const recentEvents = [
  { time: "14:23", cls: "M2.1", dur: "8m",  tmk: 18.4, by: "SoLEXS+HEL1OS", cme: "Possible" },
  { time: "13:47", cls: "C7.4", dur: "5m",  tmk: 12.1, by: "SoLEXS+GOES",   cme: "No" },
  { time: "12:18", cls: "M1.2", dur: "11m", tmk: 16.7, by: "SoLEXS+HEL1OS", cme: "Yes" },
  { time: "11:02", cls: "C4.1", dur: "4m",  tmk: 10.8, by: "GOES",          cme: "No" },
  { time: "09:56", cls: "X1.0", dur: "22m", tmk: 24.3, by: "All",           cme: "Yes" },
  { time: "08:41", cls: "C9.8", dur: "6m",  tmk: 13.9, by: "SoLEXS+GOES",   cme: "No" },
  { time: "07:12", cls: "M3.7", dur: "14m", tmk: 19.5, by: "SoLEXS+HEL1OS", cme: "Possible" },
  { time: "05:48", cls: "C2.3", dur: "3m",  tmk: 9.8,  by: "GOES",          cme: "No" },
  { time: "04:21", cls: "C5.6", dur: "5m",  tmk: 11.2, by: "SoLEXS",        cme: "No" },
  { time: "02:34", cls: "M1.8", dur: "9m",  tmk: 17.0, by: "SoLEXS+HEL1OS", cme: "Possible" },
];

export const current = {
  activity_mode: "ELEVATED" as ActivityMode,
  goes_class: "M1.4",
  goes_flux: 1.4e-5,
  z_score: 4.3,
  sw_speed: 428,
  imf_bz: -3.8,
  active_region: "NOAA AR 3724 — β-γ-δ class",
  compute_load: 25,
  next_inference_sec: 263,
};

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
