"""
Solar Activity State Machine — AdityScan Adaptive Compute Controller.

Implements a four-mode state machine with hysteresis to prevent rapid
oscillation between modes. Resource allocation scales with activity level:

  QUIET   (~5% peak compute)  → GOES ≤ B5, 60-min ML inference
  ELEVATED (~25%)             → C1 ≤ GOES < M1, 5-min ML inference
  ACTIVE   (~70%)             → M1 ≤ GOES < X1, spectral fitting every 10s
  EXTREME  (100%)             → GOES ≥ X1, every spectrum fitted, 30-s ML

Hysteresis thresholds (prevents flip-flopping):
  QUIET → ELEVATED  : z-score > 3σ sustained for > 60 s
  ELEVATED → QUIET  : z-score < 2σ sustained for > 600 s
  ELEVATED → ACTIVE : GOES proxy > M1 (≥1e-5 W/m²) for > 30 s
  ACTIVE → ELEVATED : GOES proxy < C5 (5e-6 W/m²) for > 300 s
  ACTIVE → EXTREME  : HOPE flag fires OR GOES proxy > X1 (1e-4 W/m²)
  EXTREME → ACTIVE  : T_MK declining AND EM declining for > 600 s

GOES flux thresholds (W/m², 1–8 Å channel):
  B1 = 1e-7, C1 = 1e-6, M1 = 1e-5, X1 = 1e-4
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional

logger = logging.getLogger(__name__)


# ── GOES class flux thresholds (W/m², 1–8 Å) ────────────────────────────────
GOES_B1 = 1e-7
GOES_B5 = 5e-7
GOES_C1 = 1e-6
GOES_C5 = 5e-6
GOES_M1 = 1e-5
GOES_X1 = 1e-4

# ── Activity Mode ────────────────────────────────────────────────────────────

class ActivityMode(Enum):
    QUIET = auto()     # GOES ≤ B5, background Sun
    ELEVATED = auto()  # C-class, watch mode
    ACTIVE = auto()    # M-class, alert mode
    EXTREME = auto()   # X-class, emergency mode


@dataclass
class ModeConfig:
    """Resource allocation configuration per activity mode."""
    mode: ActivityMode

    # Spectral fitting
    solexs_fit_interval_s: Optional[float]   # None = disabled
    hel1os_fit_interval_s: Optional[float]

    # ML inference
    ml_inference_interval_s: float

    # SDO/AIA fetch
    aia_fetch_interval_s: Optional[float]    # None = disabled

    # Image CNN
    image_cnn_interval_s: Optional[float]    # None = disabled

    # WebSocket push
    websocket_push_interval_s: float

    # Database write
    db_write_resolution_s: float             # Aggregate to this resolution

    # Alert arming
    alert_armed: bool

    # CME module
    cme_module_active: bool

    # Relative compute fraction (for monitoring)
    compute_fraction: float


CONFIGS: dict[ActivityMode, ModeConfig] = {
    ActivityMode.QUIET: ModeConfig(
        mode=ActivityMode.QUIET,
        solexs_fit_interval_s=None,          # NO fitting in quiet mode
        hel1os_fit_interval_s=None,
        ml_inference_interval_s=3600.0,      # hourly
        aia_fetch_interval_s=1800.0,         # every 30 min
        image_cnn_interval_s=None,           # disabled
        websocket_push_interval_s=300.0,     # 5-min heartbeat
        db_write_resolution_s=60.0,          # aggregate to 1-min
        alert_armed=False,
        cme_module_active=False,
        compute_fraction=0.05,
    ),
    ActivityMode.ELEVATED: ModeConfig(
        mode=ActivityMode.ELEVATED,
        solexs_fit_interval_s=60.0,          # fit 60-s integrated spectra
        hel1os_fit_interval_s=20.0,          # native HEL1OS cadence
        ml_inference_interval_s=300.0,       # every 5 min
        aia_fetch_interval_s=300.0,          # every 5 min
        image_cnn_interval_s=300.0,          # every 5 min
        websocket_push_interval_s=30.0,
        db_write_resolution_s=1.0,           # full resolution
        alert_armed=False,
        cme_module_active=False,
        compute_fraction=0.25,
    ),
    ActivityMode.ACTIVE: ModeConfig(
        mode=ActivityMode.ACTIVE,
        solexs_fit_interval_s=10.0,          # 10-s integrated spectra
        hel1os_fit_interval_s=20.0,
        ml_inference_interval_s=300.0,
        aia_fetch_interval_s=300.0,
        image_cnn_interval_s=300.0,
        websocket_push_interval_s=10.0,
        db_write_resolution_s=1.0,
        alert_armed=True,
        cme_module_active=True,
        compute_fraction=0.70,
    ),
    ActivityMode.EXTREME: ModeConfig(
        mode=ActivityMode.EXTREME,
        solexs_fit_interval_s=1.0,           # every spectrum (1-s native)
        hel1os_fit_interval_s=20.0,          # native HEL1OS cadence
        ml_inference_interval_s=30.0,        # every 30 seconds
        aia_fetch_interval_s=300.0,
        image_cnn_interval_s=300.0,
        websocket_push_interval_s=5.0,       # 5-s dashboard push
        db_write_resolution_s=1.0,
        alert_armed=True,
        cme_module_active=True,
        compute_fraction=1.0,
    ),
}


# ── Transition timing (hysteresis windows) ────────────────────────────────────

@dataclass
class TransitionHysteresis:
    """
    Hysteresis controller for one mode transition.
    Prevents rapid oscillation by requiring condition to hold for min_hold_s.
    """
    min_hold_s: float             # minimum seconds condition must be satisfied
    triggered_at: float = 0.0    # when condition was first met (UNIX time)
    active: bool = False          # is condition currently being tracked?

    def check(self, condition: bool, now: float) -> bool:
        """
        Returns True if condition has been satisfied for min_hold_s.
        Resets timer if condition drops out.
        Timer fires when elapsed >= min_hold_s (inclusive).
        """
        if condition:
            if not self.active:
                self.active = True
                self.triggered_at = now
                return False  # first tick never fires immediately
            elapsed = now - self.triggered_at
            return elapsed >= self.min_hold_s
        else:
            self.active = False
            return False


@dataclass
class ActivityStateMachine:
    """
    Four-mode solar activity state machine with hysteresis.

    State is driven by three inputs:
      1. z_score   : Rolling z-score from Tier 1 triage (light curve)
      2. goes_flux : GOES XRS proxy flux W/m² (from triage or real NOAA feed)
      3. hope_fired: Boolean from HOPE detector (impulsive phase flag)
      4. t_mk_declining + em_declining: Boolean flags from spectral fitter

    Callers update() every second (always-on Tier 1 loop).
    """
    initial_mode: ActivityMode = ActivityMode.QUIET

    # Internal state
    current_mode: ActivityMode = field(init=False)
    mode_entered_at: float = field(init=False)
    config: ModeConfig = field(init=False)
    transition_history: list[dict] = field(default_factory=list)

    # Hysteresis timers
    _h_quiet_to_elevated: TransitionHysteresis = field(
        default_factory=lambda: TransitionHysteresis(min_hold_s=60.0)
    )
    _h_elevated_to_quiet: TransitionHysteresis = field(
        default_factory=lambda: TransitionHysteresis(min_hold_s=600.0)
    )
    _h_elevated_to_active: TransitionHysteresis = field(
        default_factory=lambda: TransitionHysteresis(min_hold_s=30.0)
    )
    _h_active_to_elevated: TransitionHysteresis = field(
        default_factory=lambda: TransitionHysteresis(min_hold_s=300.0)
    )
    _h_extreme_to_active: TransitionHysteresis = field(
        default_factory=lambda: TransitionHysteresis(min_hold_s=600.0)
    )

    # Callbacks registered by subsystems
    _on_mode_change_callbacks: list[Callable] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.current_mode = self.initial_mode
        self.mode_entered_at = time.time()
        self.config = CONFIGS[self.current_mode]

    # ── Public API ───────────────────────────────────────────────────────────

    def update(
        self,
        z_score: float,
        goes_flux: float,
        hope_fired: bool = False,
        t_mk_declining: bool = False,
        em_declining: bool = False,
        now: Optional[float] = None,
    ) -> ActivityMode:
        """
        Feed one timestep of telemetry into the state machine.
        Called every ~1 second from the triage worker.

        Parameters
        ----------
        z_score : float
            Rolling z-score of X-ray light curve from Tier 1 triage.
        goes_flux : float
            GOES-equivalent flux proxy (W/m², 1–8 Å channel).
        hope_fired : bool
            True if HOPE detector has triggered (impulsive X-ray burst).
        t_mk_declining : bool
            True if plasma temperature is declining (from spectral fitter).
        em_declining : bool
            True if emission measure is declining (flare decay phase).
        now : float, optional
            Current UNIX time. Defaults to time.time().

        Returns
        -------
        ActivityMode
            Current (possibly updated) mode.
        """
        if now is None:
            now = time.time()

        old_mode = self.current_mode
        new_mode = self._evaluate_transitions(
            z_score, goes_flux, hope_fired, t_mk_declining, em_declining, now
        )

        if new_mode != old_mode:
            self._transition_to(new_mode, now, z_score=z_score, goes_flux=goes_flux)

        return self.current_mode

    def register_on_mode_change(self, callback: Callable) -> None:
        """
        Register a callback to be invoked on mode transitions.
        Signature: callback(old_mode: ActivityMode, new_mode: ActivityMode, meta: dict)
        """
        self._on_mode_change_callbacks.append(callback)

    @property
    def mode_duration_s(self) -> float:
        """Seconds since last mode transition."""
        return time.time() - self.mode_entered_at

    def status_dict(self) -> dict:
        """Current machine state as a JSON-serializable dict for API/dashboard."""
        cfg = self.config
        return {
            "mode": self.current_mode.name,
            "mode_duration_s": self.mode_duration_s,
            "compute_fraction": cfg.compute_fraction,
            "alert_armed": cfg.alert_armed,
            "cme_module_active": cfg.cme_module_active,
            "ml_inference_interval_s": cfg.ml_inference_interval_s,
            "solexs_fit_interval_s": cfg.solexs_fit_interval_s,
            "websocket_push_interval_s": cfg.websocket_push_interval_s,
        }

    # ── Private transition logic ──────────────────────────────────────────────

    def _evaluate_transitions(
        self,
        z_score: float,
        goes_flux: float,
        hope_fired: bool,
        t_mk_declining: bool,
        em_declining: bool,
        now: float,
    ) -> ActivityMode:
        mode = self.current_mode

        if mode == ActivityMode.QUIET:
            # → ELEVATED if z-score > 3σ for > 60 s
            if self._h_quiet_to_elevated.check(z_score > 3.0, now):
                return ActivityMode.ELEVATED

        elif mode == ActivityMode.ELEVATED:
            # → QUIET if z-score < 2σ for > 600 s
            if self._h_elevated_to_quiet.check(z_score < 2.0, now):
                return ActivityMode.QUIET
            # → ACTIVE if GOES proxy ≥ M1 for > 30 s
            if self._h_elevated_to_active.check(goes_flux >= GOES_M1, now):
                return ActivityMode.ACTIVE

        elif mode == ActivityMode.ACTIVE:
            # → EXTREME immediately on HOPE flag or GOES ≥ X1
            if hope_fired or goes_flux >= GOES_X1:
                return ActivityMode.EXTREME
            # → ELEVATED if GOES < C5 for > 300 s
            if self._h_active_to_elevated.check(goes_flux < GOES_C5, now):
                return ActivityMode.ELEVATED

        elif mode == ActivityMode.EXTREME:
            # → ACTIVE only when BOTH T_MK and EM declining for > 600 s
            if self._h_extreme_to_active.check(t_mk_declining and em_declining, now):
                return ActivityMode.ACTIVE
            # Do NOT drop out on z-score alone during extreme mode
            # (secondary peak may follow main flare)

        return mode

    def _transition_to(
        self, new_mode: ActivityMode, now: float, **meta
    ) -> None:
        old_mode = self.current_mode
        self.current_mode = new_mode
        self.config = CONFIGS[new_mode]
        self.mode_entered_at = now

        entry = {
            "timestamp": now,
            "from": old_mode.name,
            "to": new_mode.name,
            **meta,
        }
        self.transition_history.append(entry)

        logger.info(
            "Activity mode: %s → %s (goes_flux=%.2e)",
            old_mode.name, new_mode.name, meta.get("goes_flux", 0.0)
        )

        for cb in self._on_mode_change_callbacks:
            try:
                cb(old_mode, new_mode, entry)
            except Exception as exc:
                logger.warning("Mode change callback failed: %s", exc)

        # Reset opposing hysteresis timers on transition
        self._reset_hysteresis_on_transition(old_mode, new_mode)

    def _reset_hysteresis_on_transition(
        self, old_mode: ActivityMode, new_mode: ActivityMode
    ) -> None:
        """Reset timers that are no longer relevant after a transition."""
        if new_mode == ActivityMode.ELEVATED:
            self._h_quiet_to_elevated.active = False
            self._h_elevated_to_quiet.active = False
        elif new_mode == ActivityMode.ACTIVE:
            self._h_elevated_to_active.active = False
            self._h_active_to_elevated.active = False
        elif new_mode == ActivityMode.EXTREME:
            self._h_extreme_to_active.active = False
        elif new_mode == ActivityMode.QUIET:
            self._h_elevated_to_quiet.active = False


# ── Convenience singleton factory ────────────────────────────────────────────

def create_state_machine() -> ActivityStateMachine:
    """Create a production-configured activity state machine."""
    sm = ActivityStateMachine()

    def _log_mode_change(old, new, meta):
        logger.info("[STATE] %s → %s | meta=%s", old.name, new.name, meta)

    sm.register_on_mode_change(_log_mode_change)
    return sm
