"""
Tests for triage.py — Tier 1 X-ray triage engine.
"""

import pytest
import numpy as np
from pipeline.physics.triage import (
    TriageEngine,
    counts_to_goes_proxy,
    goes_class_from_flux,
    GOES_PROXY_B1, GOES_PROXY_C1, GOES_PROXY_M1, GOES_PROXY_X1,
)


class TestTriageEngine:

    def setup_method(self):
        self.engine = TriageEngine()
        self.t0 = 1_700_000_000.0

    def _feed_background(self, n: int = 70, base_counts: float = 500.0):
        """Seed background with N quiet-Sun samples."""
        np.random.seed(42)
        for i in range(n):
            self.engine.evaluate(
                unix_time=self.t0 + i,
                solexs_sdd2_counts=base_counts + np.random.normal(0, 10.0),
            )
        return self.t0 + n

    def test_quiet_sun_alert_level(self):
        """Quiet Sun should report QUIET alert level."""
        t_now = self._feed_background(70, base_counts=500.0)
        sample = self.engine.evaluate(
            unix_time=t_now,
            solexs_sdd2_counts=505.0,
        )
        assert sample.alert_level == "QUIET"
        assert abs(sample.z_score) < 2.0

    def test_flare_onset_detection(self):
        """Large count rate jump should produce high z-score and alert."""
        t_now = self._feed_background(70, base_counts=500.0)

        # Inject a large burst: ~10σ
        sample = self.engine.evaluate(
            unix_time=t_now,
            solexs_sdd2_counts=500.0 + 10 * 10.0,  # bg=500, std≈10, so z≈10
        )
        assert sample.z_score > 5.0, f"Expected z>5, got {sample.z_score}"
        assert sample.alert_level in ("WARNING", "ALERT")

    def test_insufficient_background_returns_zero_zscore(self):
        """Before 60 background samples, z-score should be 0."""
        sample = self.engine.evaluate(
            unix_time=self.t0,
            solexs_sdd2_counts=1000.0,
        )
        assert sample.z_score == 0.0, "No background = no z-score"

    def test_goes_proxy_conversion(self):
        """GOES proxy should be proportional to counts."""
        low = counts_to_goes_proxy(100.0)
        high = counts_to_goes_proxy(1000.0)
        assert high > low, "Higher counts → higher GOES proxy"

    def test_goes_class_thresholds(self):
        """GOES class string should match known thresholds."""
        assert goes_class_from_flux(5e-7).startswith("B"), f"Got {goes_class_from_flux(5e-7)}"
        assert goes_class_from_flux(3.7e-5).startswith("M"), f"Got {goes_class_from_flux(3.7e-5)}"
        assert goes_class_from_flux(2.1e-4).startswith("X"), f"Got {goes_class_from_flux(2.1e-4)}"
        assert goes_class_from_flux(5e-8).startswith("A"), f"Got {goes_class_from_flux(5e-8)}"

    def test_get_light_curve(self):
        """Light curve output should have correct length."""
        t_now = self._feed_background(300, base_counts=500.0)
        times, counts = self.engine.get_light_curve(last_n_seconds=200)
        assert len(times) > 0
        assert len(times) == len(counts)
        assert len(times) <= 201, "Should not return more than 201 samples (inclusive boundary)"

    def test_recent_statistics(self):
        """Recent statistics should return correct keys."""
        t_now = self._feed_background(100)
        stats = self.engine.recent_statistics(window_s=60)
        assert "mean_counts" in stats
        assert "goes_class_mean" in stats
        assert stats["n_samples"] > 0
