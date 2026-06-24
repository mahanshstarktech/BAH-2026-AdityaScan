"""
HOPE (Hard X-ray Onset Precursor Event) Detector.

Physics background:
  The HOPE algorithm detects the horizontal branch in hard X-ray spectrograms —
  a diagnostic of accelerated electrons before the main thermal phase.
  In the Neupert paradigm, HXR emission (non-thermal bremsstrahlung) precedes
  the main SXR peak. HOPE flags when:

  1. HEL1OS high-energy channels (>30 keV) show a statistically significant
     count rate increase (the "precursor burst")
  2. The spectral index γ (power-law photon index) hardens: γ < 4.5 → non-thermal
  3. The rise is impulsive: peak-to-background ratio exceeds threshold in < 60 s

  Physical interpretation:
    - Hard X-rays are from thick-target bremsstrahlung of accelerated electrons
    - Onset of electron acceleration = 30–120 s before SXR peak
    - HOPE flag → immediately escalate to EXTREME mode
    - Used for earliest-possible flare alert (the "precursor window")

  HEL1OS energy bands used here (from HEL1OS Data Analysis User Manual v1.2):
    CdTe: 5–20, 20–30, 30–40, 40–60 keV
    CZT:  20–40, 40–60, 60–80, 80–150 keV

  Algorithm:
    1. Monitor CdTe 30–40 keV and CZT 40–60 keV bands (always-on rolling)
    2. Compute rolling z-score with background window = 10 minutes
    3. If z-score > HOPE_Z_THRESHOLD (default 5σ): candidate onset
    4. Confirm with spectral hardening check (γ_lo < GAMMA_HXR_THRESHOLD)
    5. Confirm impulsive rise time < RISE_TIME_THRESHOLD_S
    6. Fire HOPE flag → ActivityStateMachine.update(hope_fired=True)

  Non-triggered use: When HOPE does NOT fire, compute Neupert ratio
  (SXR derivative vs HXR flux) as a continuous physics feature.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Algorithm parameters ─────────────────────────────────────────────────────
HOPE_Z_THRESHOLD: float = 5.0        # σ threshold for initial trigger
HOPE_GAMMA_THRESHOLD: float = 4.5    # spectral index below this = non-thermal
HOPE_RISE_TIME_THRESHOLD_S: float = 60.0  # max seconds from onset to peak
HOPE_BACKGROUND_WINDOW_S: float = 600.0  # 10-min rolling background window
HOPE_CONFIRM_REQUIRED: int = 3        # N consecutive samples above threshold

# HEL1OS band names used in light curve monitoring (consistent with manual bands)
BAND_CDTE_30_40 = "cdte_30_40"
BAND_CZT_40_60 = "czt_40_60"
BAND_CDTE_20_30 = "cdte_20_30"
BAND_CZT_60_80 = "czt_60_80"


@dataclass
class HOPEEvent:
    """A detected HOPE precursor event."""
    trigger_unix: float           # UNIX time of trigger
    trigger_z_score: float        # z-score at trigger
    gamma_lo: float               # spectral index at trigger (from HEL1OS)
    rise_time_s: float            # time from onset to z-score trigger
    band_triggered: str           # which energy band triggered first
    confirmed: bool               # True if all 3 confirmation criteria met


@dataclass
class HOPEDetector:
    """
    Online, stateless HOPE detector. Call update() with each new HEL1OS sample.

    State is maintained internally. The detector is designed to be called
    from the triage_worker (always-on process) at 1-s cadence.

    Usage
    -----
    detector = HOPEDetector()
    for sample in helios_stream:
        event = detector.update(
            unix_time=sample.time,
            counts_cdte_30_40=sample.cdte_30_40,
            counts_czt_40_60=sample.czt_40_60,
            gamma_lo=sample.gamma_lo,  # from spectral fitter if available
        )
        if event and event.confirmed:
            state_machine.update(..., hope_fired=True)
    """
    background_window_s: float = HOPE_BACKGROUND_WINDOW_S
    z_threshold: float = HOPE_Z_THRESHOLD
    gamma_threshold: float = HOPE_GAMMA_THRESHOLD
    rise_time_threshold_s: float = HOPE_RISE_TIME_THRESHOLD_S
    confirm_required: int = HOPE_CONFIRM_REQUIRED

    # Ring buffers for background estimation
    _buf_cdte: deque = field(default_factory=lambda: deque(maxlen=10000))
    _buf_czt: deque = field(default_factory=lambda: deque(maxlen=10000))
    _buf_times: deque = field(default_factory=lambda: deque(maxlen=10000))

    # Onset tracking
    _onset_time: Optional[float] = field(default=None)
    _onset_count: float = 0.0
    _confirm_streak: int = 0

    # Last confirmed event (to avoid re-triggering)
    _last_event_time: float = field(default=0.0)
    _cooldown_s: float = field(default=600.0)  # 10-min cooldown after event

    def __post_init__(self) -> None:
        pass

    # ── Public API ───────────────────────────────────────────────────────────

    def update(
        self,
        unix_time: float,
        counts_cdte_30_40: float,
        counts_czt_40_60: float,
        gamma_lo: Optional[float] = None,
        counts_czte_20_30: float = 0.0,
    ) -> Optional[HOPEEvent]:
        """
        Feed one 1-second HEL1OS sample into the HOPE detector.

        Parameters
        ----------
        unix_time : float
            UNIX timestamp of this sample.
        counts_cdte_30_40 : float
            CdTe 30–40 keV band count rate (counts/s).
        counts_czt_40_60 : float
            CZT 40–60 keV band count rate (counts/s).
        gamma_lo : float, optional
            Low-energy spectral index from HEL1OS fit (lower = harder = non-thermal).
            May be None when spectral fitting is not running (QUIET mode).
        counts_czte_20_30 : float
            CdTe 20–30 keV for context (not primary trigger, but used in ratio).

        Returns
        -------
        HOPEEvent if triggered and confirmed, else None.
        """
        # Cooldown guard: don't re-trigger within 10 min
        if (unix_time - self._last_event_time) < self._cooldown_s:
            self._update_buffer(unix_time, counts_cdte_30_40, counts_czt_40_60)
            return None

        self._update_buffer(unix_time, counts_cdte_30_40, counts_czt_40_60)

        # Need at least 60 samples for background estimate
        if len(self._buf_times) < 60:
            return None

        bg_mean, bg_std = self._background_stats(unix_time)
        if bg_std < 1e-10:
            return None

        # Primary trigger: CdTe 30–40 keV z-score
        z_cdte = (counts_cdte_30_40 - bg_mean) / bg_std
        z_czt = (counts_czt_40_60 - bg_mean * 0.5) / (bg_std * 0.5 + 1e-10)

        # Use whichever band triggers first
        z_max = max(z_cdte, z_czt)
        triggered_band = BAND_CDTE_30_40 if z_cdte >= z_czt else BAND_CZT_40_60

        if z_max >= self.z_threshold:
            self._confirm_streak += 1
            if self._onset_time is None:
                self._onset_time = unix_time
                self._onset_count = max(counts_cdte_30_40, counts_czt_40_60)
        else:
            self._confirm_streak = 0
            self._onset_time = None

        # Require N consecutive samples above threshold
        if self._confirm_streak < self.confirm_required:
            return None

        rise_time = unix_time - (self._onset_time or unix_time)

        # Confirmation 1: spectral hardening
        spectral_confirmed = (
            gamma_lo is not None and gamma_lo < self.gamma_threshold
        )

        # Confirmation 2: impulsive rise
        rise_confirmed = rise_time <= self.rise_time_threshold_s

        confirmed = spectral_confirmed and rise_confirmed

        event = HOPEEvent(
            trigger_unix=unix_time,
            trigger_z_score=float(z_max),
            gamma_lo=float(gamma_lo) if gamma_lo is not None else float("nan"),
            rise_time_s=float(rise_time),
            band_triggered=triggered_band,
            confirmed=confirmed,
        )

        logger.info(
            "HOPE event: z=%.1f, γ=%.2f, rise=%.0fs, confirmed=%s",
            z_max, event.gamma_lo, rise_time, confirmed
        )

        if confirmed:
            self._last_event_time = unix_time
            self._confirm_streak = 0
            self._onset_time = None

        return event

    def compute_neupert_ratio(
        self,
        sxr_counts_1s: np.ndarray,
        hxr_counts_1s: np.ndarray,
        dt_s: float = 1.0,
    ) -> float:
        """
        Compute the Neupert ratio: ∫HXR dt / SXR_peak.

        In the Neupert effect:
          dSXR/dt ≈ HXR (the SXR derivative matches the HXR profile)
          → ∫HXR dt should equal SXR_peak if Neupert effect holds

        Returns
        -------
        float
            Ratio ∈ [0, ∞). Ratio ≈ 1 → perfect Neupert. Ratio ≫ 1 or ≪ 1
            indicates departure (impulsive HXR without SXR → SEP-rich event).
        """
        if len(sxr_counts_1s) < 2 or len(hxr_counts_1s) < 2:
            return float("nan")

        hxr_integral = float(np.trapz(hxr_counts_1s, dx=dt_s))
        sxr_peak = float(np.max(sxr_counts_1s))

        if sxr_peak < 1e-10 or hxr_integral < 0:
            return float("nan")

        return hxr_integral / sxr_peak

    # ── Private helpers ──────────────────────────────────────────────────────

    def _update_buffer(
        self, unix_time: float, counts_cdte: float, counts_czt: float
    ) -> None:
        """Append to ring buffers for background computation."""
        self._buf_times.append(unix_time)
        self._buf_cdte.append(counts_cdte)
        self._buf_czt.append(counts_czt)

    def _background_stats(self, now: float) -> tuple[float, float]:
        """
        Compute rolling background mean and std, excluding the last 60 s
        (to avoid contaminating background with the event under investigation).
        """
        exclude_recent_s = 60.0
        bg_cutoff = now - exclude_recent_s
        bg_start = now - self.background_window_s

        bg_cdte = [
            c for t, c in zip(self._buf_times, self._buf_cdte)
            if bg_start <= t < bg_cutoff
        ]

        if len(bg_cdte) < 5:
            return 0.0, 0.0

        arr = np.array(bg_cdte, dtype=np.float64)
        return float(np.mean(arr)), float(np.std(arr))


# ── Neupert Engine (standalone) ──────────────────────────────────────────────

class NeupertEngine:
    """
    Continuous Neupert deviation engine.

    Computes dSXR/dt - HXR correlation as a streaming physics feature.
    Non-zero deviation flags: (a) non-Neupert events (direct heating),
    (b) evaporation-driven vs. beam-driven heating distinction.

    Output used as one feature in the X-ray TCN input vector.
    """

    def __init__(self, window_s: float = 300.0, dt_s: float = 1.0) -> None:
        self.window_s = window_s
        self.dt_s = dt_s
        self._sxr_buf: deque = deque(maxlen=int(window_s / dt_s))
        self._hxr_buf: deque = deque(maxlen=int(window_s / dt_s))

    def push(self, sxr_count: float, hxr_count: float) -> None:
        """Append one 1-second sample."""
        self._sxr_buf.append(float(sxr_count))
        self._hxr_buf.append(float(hxr_count))

    def neupert_deviation(self) -> float:
        """
        Compute normalized Neupert deviation.

        Returns: (d/dt SXR - HXR_normalized) normalized by HXR RMS
        Positive = SXR rising faster than HXR predicts (direct heating)
        Negative = HXR burst without SXR rise (SEP-producing, confined)
        0 = perfect Neupert (beam-driven chromospheric evaporation)
        """
        if len(self._sxr_buf) < 3 or len(self._hxr_buf) < 3:
            return 0.0

        sxr = np.array(self._sxr_buf, dtype=np.float64)
        hxr = np.array(self._hxr_buf, dtype=np.float64)

        # SXR time derivative (central differences where possible)
        dsxr_dt = np.gradient(sxr, self.dt_s)

        # Normalize HXR to same scale as dSXR/dt
        hxr_rms = float(np.sqrt(np.mean(hxr**2)))
        dsxr_rms = float(np.sqrt(np.mean(dsxr_dt**2)))

        if hxr_rms < 1e-10 or dsxr_rms < 1e-10:
            return 0.0

        hxr_norm = hxr * (dsxr_rms / hxr_rms)
        deviation = dsxr_dt - hxr_norm

        # Return most recent value (current deviation)
        return float(deviation[-1]) / (dsxr_rms + 1e-10)
