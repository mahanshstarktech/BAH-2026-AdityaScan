"""
Tests for activity_mode.py — Solar Activity State Machine.
Tests hysteresis behavior, all four modes, and callback registration.
"""

import time
import pytest
from pipeline.physics.activity_mode import (
    ActivityMode,
    ActivityStateMachine,
    GOES_B5, GOES_C1, GOES_M1, GOES_X1,
    create_state_machine,
)


class TestActivityStateMachine:

    def setup_method(self):
        self.sm = ActivityStateMachine()
        self.now = time.time()

    def _tick(self, n: int, z_score: float, goes_flux: float, **kwargs):
        """Simulate n seconds of updates."""
        mode = None
        for i in range(n):
            mode = self.sm.update(
                z_score=z_score,
                goes_flux=goes_flux,
                now=self.now + i,
                **kwargs,
            )
        self.now += n
        return mode

    def test_initial_mode_is_quiet(self):
        assert self.sm.current_mode == ActivityMode.QUIET

    def test_quiet_to_elevated_requires_sustained_zscore(self):
        # 60 seconds of high z-score: should NOT trigger (timer starts at first, reaches 60 at tick 61)
        mode = self._tick(60, z_score=4.0, goes_flux=GOES_B5)
        assert mode == ActivityMode.QUIET, "Should remain QUIET before 61-s hold"

        # 1 more second (total 61): should now trigger (elapsed=60 >= min_hold_s=60)
        mode = self._tick(1, z_score=4.0, goes_flux=GOES_B5)
        assert mode == ActivityMode.ELEVATED, "Should transition to ELEVATED after 61-s hold"

    def test_elevated_to_quiet_requires_10min_hold(self):
        # First: get to ELEVATED (need 61 ticks)
        self._tick(62, z_score=4.0, goes_flux=GOES_B5)
        assert self.sm.current_mode == ActivityMode.ELEVATED

        # 600 seconds below 2σ: should NOT revert (timer reaches 600 at tick 601)
        mode = self._tick(600, z_score=1.5, goes_flux=GOES_B5 * 0.1)
        assert mode == ActivityMode.ELEVATED, "Should stay ELEVATED before 601-s hold"

        # 1 more second: should revert (elapsed=600 >= min_hold_s=600)
        mode = self._tick(1, z_score=1.5, goes_flux=GOES_B5 * 0.1)
        assert mode == ActivityMode.QUIET

    def test_elevated_to_active_on_m_class(self):
        # Get to ELEVATED first
        self._tick(65, z_score=4.0, goes_flux=GOES_B5)

        # M1 flux for 31 seconds: should trigger ACTIVE
        mode = self._tick(31, z_score=6.0, goes_flux=GOES_M1 * 1.5)
        assert mode == ActivityMode.ACTIVE

    def test_active_to_extreme_on_hope(self):
        # Get to ACTIVE
        self._tick(65, z_score=4.0, goes_flux=GOES_B5)
        self._tick(31, z_score=6.0, goes_flux=GOES_M1 * 1.5)
        assert self.sm.current_mode == ActivityMode.ACTIVE

        # HOPE fires: immediate transition to EXTREME
        mode = self._tick(1, z_score=6.0, goes_flux=GOES_M1, hope_fired=True)
        assert mode == ActivityMode.EXTREME

    def test_active_to_extreme_on_x_class(self):
        # Get to ACTIVE
        self._tick(65, z_score=4.0, goes_flux=GOES_B5)
        self._tick(31, z_score=6.0, goes_flux=GOES_M1 * 1.5)

        # X1 flux: immediate transition
        mode = self._tick(1, z_score=8.0, goes_flux=GOES_X1 * 2)
        assert mode == ActivityMode.EXTREME

    def test_extreme_to_active_requires_decay(self):
        # Get to EXTREME
        self._tick(62, z_score=4.0, goes_flux=GOES_B5)
        self._tick(32, z_score=6.0, goes_flux=GOES_M1 * 1.5)
        self._tick(1, z_score=8.0, goes_flux=GOES_X1 * 2)
        assert self.sm.current_mode == ActivityMode.EXTREME

        # 600 seconds of declining T and EM: NOT enough
        mode = self._tick(600, z_score=2.0, goes_flux=GOES_M1,
                          t_mk_declining=True, em_declining=True)
        assert mode == ActivityMode.EXTREME, "Need 601s for EXTREME → ACTIVE"

        # 1 more second: transition (elapsed=600 >= min_hold_s=600)
        mode = self._tick(1, z_score=2.0, goes_flux=GOES_M1,
                          t_mk_declining=True, em_declining=True)
        assert mode == ActivityMode.ACTIVE

    def test_extreme_does_not_drop_on_zscore_alone(self):
        """EXTREME mode should NOT drop just because z-score falls."""
        self._tick(65, z_score=4.0, goes_flux=GOES_B5)
        self._tick(31, z_score=6.0, goes_flux=GOES_M1 * 1.5)
        self._tick(1, z_score=8.0, goes_flux=GOES_X1 * 2)

        # z-score drops but T_MK and EM not declining
        mode = self._tick(600, z_score=0.5, goes_flux=GOES_M1,
                          t_mk_declining=False, em_declining=False)
        assert mode == ActivityMode.EXTREME, "Should stay EXTREME — secondary peak possible"

    def test_callback_registration(self):
        callbacks = []
        self.sm.register_on_mode_change(
            lambda old, new, meta: callbacks.append((old.name, new.name))
        )
        self._tick(65, z_score=4.0, goes_flux=GOES_B5)
        assert len(callbacks) == 1
        assert callbacks[0] == ("QUIET", "ELEVATED")

    def test_status_dict(self):
        status = self.sm.status_dict()
        assert "mode" in status
        assert "compute_fraction" in status
        assert status["compute_fraction"] == 0.05  # QUIET mode

    def test_mode_configs(self):
        """Verify resource configs are consistent with architecture doc."""
        from pipeline.physics.activity_mode import CONFIGS, ActivityMode
        q = CONFIGS[ActivityMode.QUIET]
        assert q.solexs_fit_interval_s is None, "QUIET mode: no spectral fitting"
        assert q.ml_inference_interval_s == 3600.0

        ex = CONFIGS[ActivityMode.EXTREME]
        assert ex.solexs_fit_interval_s == 1.0, "EXTREME: every 1s spectrum"
        assert ex.ml_inference_interval_s == 30.0
        assert ex.alert_armed is True
