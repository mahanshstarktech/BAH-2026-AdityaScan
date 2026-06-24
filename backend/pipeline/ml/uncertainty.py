"""
Uncertainty Quantification for AdityScan — NASA-grade reliability metrics.

Three-layer uncertainty stack:

Layer 1 — MC Dropout (epistemic uncertainty: model confidence)
  - Keep dropout active at inference
  - T=50 forward passes → mean + std of output probabilities
  - Std interpretation: high std = model uncertain about this input
  - Implementation: AdityScanModel.predict_with_uncertainty()

Layer 2 — Conformal Prediction (frequentist guarantee)
  - Calibration split: holdout set never seen during training
  - Nonconformity score α_i = 1 - P(y_true)_i for each sample
  - At inference: find smallest prediction set S such that
    P(y_true ∈ S) ≥ 1 - α (user-specified coverage, e.g. 90%)
  - Result: guaranteed coverage on any exchangeable test distribution
  - CRITICAL: conformal intervals are for SETS, not continuous bands.
    For regression (probability outputs), we use conformal regression
    intervals: [P_hat - q_alpha * sigma, P_hat + q_alpha * sigma]
    where q_alpha is the (1-alpha) quantile of nonconformity scores.

Layer 3 — Temperature Scaling (calibration)
  - After training, fit single T on validation set
  - Minimizes negative log-likelihood of calibrated probabilities
  - P_calibrated = sigmoid(logit(P_raw) / T)
  - Ensures reliability diagram is on the diagonal (well-calibrated)
  - Implemented as AdityScanModel.temperature parameter

Dashboard output format:
  "P(M+ flare in 15 min) = 73% [62%–84% conformal interval]"
  Confidence display: |interval| < 10% → bold, > 25% → faded
"""

from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.special import expit, logit

logger = logging.getLogger(__name__)


# ── Temperature Scaling ───────────────────────────────────────────────────────

class TemperatureScaler:
    """
    Post-hoc probability calibration via temperature scaling.
    Fits a single scalar T on a validation set after model training.

    Reference: Guo et al. 2017, "On Calibration of Modern Neural Networks"
    """

    def __init__(self) -> None:
        self.temperature: float = 1.0  # initialized to identity

    def fit(
        self,
        logits: np.ndarray,      # (N,) raw model logits (before sigmoid)
        labels: np.ndarray,      # (N,) binary labels {0, 1}
        n_iter: int = 100,
        lr: float = 0.01,
    ) -> float:
        """
        Fit temperature parameter by minimizing binary cross-entropy on validation.
        Uses simple gradient descent (no PyTorch dependency at inference time).

        Returns
        -------
        float
            Optimal temperature T.
        """
        T = 1.0
        for _ in range(n_iter):
            probs = expit(logits / T)
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            # Gradient of NLL w.r.t. T
            # ∂NLL/∂T = (1/N) Σ (p_i - y_i) * logit_i / T²
            grad = np.mean((probs - labels) * logits) / (T**2)
            T = max(0.01, T - lr * grad)

        self.temperature = float(T)
        logger.info("Temperature scaling fit: T = %.4f", self.temperature)
        return self.temperature

    def calibrate(self, probs_raw: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to raw probabilities."""
        log_odds = logit(np.clip(probs_raw, 1e-7, 1 - 1e-7))
        return expit(log_odds / self.temperature)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"temperature": self.temperature}, f)

    @classmethod
    def load(cls, path: str) -> "TemperatureScaler":
        with open(path, "rb") as f:
            data = pickle.load(f)
        scaler = cls()
        scaler.temperature = data["temperature"]
        return scaler


# ── Conformal Prediction ──────────────────────────────────────────────────────

@dataclass
class ConformalInterval:
    """One conformal prediction interval for a probability forecast."""
    horizon_min: int           # forecast horizon (minutes)
    p_mean: float              # calibrated probability estimate
    p_lower: float             # lower bound of conformal interval
    p_upper: float             # upper bound of conformal interval
    coverage: float            # nominal coverage (e.g. 0.9 = 90%)
    n_calibration: int         # number of calibration samples used
    confidence_display: str    # "BOLD", "NORMAL", "FADED" for dashboard

    @property
    def interval_width(self) -> float:
        return self.p_upper - self.p_lower


class ConformalPredictor:
    """
    Regression-style conformal prediction for probability forecasts.

    Calibration: on a holdout set (never seen during training):
      1. Get model predictions P_hat_i for each calibration sample
      2. Get true label y_i ∈ {0, 1}
      3. Compute nonconformity score: s_i = |y_i - P_hat_i|
      4. Store sorted scores for coverage computation

    Inference: for new prediction P_hat:
      1. Find q_alpha = quantile(calibration_scores, 1 - alpha)
      2. Interval = [P_hat - q_alpha, P_hat + q_alpha]  (clipped to [0,1])
      3. Guaranteed: P(y ∈ interval) ≥ 1 - alpha under exchangeability

    Multiple horizons: separate conformal predictor per forecast horizon.
    """

    def __init__(self, coverage: float = 0.9) -> None:
        """
        Parameters
        ----------
        coverage : float
            Nominal coverage of the conformal interval (default 0.9 = 90%).
        """
        self.coverage = coverage
        # Maps horizon_min → sorted array of calibration nonconformity scores
        self._calibration_scores: dict[int, np.ndarray] = {}

    def fit(
        self,
        horizon_min: int,
        p_predictions: np.ndarray,  # (N,) calibration set predictions
        y_true: np.ndarray,         # (N,) binary labels
    ) -> None:
        """
        Compute and store nonconformity scores for one forecast horizon.

        Must be called with a HELD-OUT calibration set (not validation or test).
        The calibration set should reflect the operational distribution
        (e.g. include quiet-Sun samples, not just flare events).
        """
        if len(p_predictions) < 50:
            logger.warning(
                "Conformal calibration for horizon %d min has only %d samples. "
                "Need ≥50 for reliable coverage.",
                horizon_min, len(p_predictions)
            )
        scores = np.abs(y_true.astype(float) - p_predictions.astype(float))
        self._calibration_scores[horizon_min] = np.sort(scores)
        logger.info(
            "Conformal calibration: horizon=%d min, n=%d, median_score=%.3f",
            horizon_min, len(scores), float(np.median(scores))
        )

    def predict(self, horizon_min: int, p_hat: float) -> ConformalInterval:
        """
        Compute conformal interval for a single prediction.

        Parameters
        ----------
        horizon_min : int
            Forecast horizon in minutes.
        p_hat : float
            Model's calibrated probability estimate.

        Returns
        -------
        ConformalInterval
        """
        if horizon_min not in self._calibration_scores:
            # Fallback: uninformative interval
            logger.warning("No calibration for horizon %d min — using uninformative interval", horizon_min)
            return ConformalInterval(
                horizon_min=horizon_min,
                p_mean=p_hat,
                p_lower=max(0.0, p_hat - 0.5),
                p_upper=min(1.0, p_hat + 0.5),
                coverage=self.coverage,
                n_calibration=0,
                confidence_display="FADED",
            )

        scores = self._calibration_scores[horizon_min]
        n = len(scores)

        # Conformal quantile: ceil((n+1)(1-alpha)) / n quantile
        alpha = 1.0 - self.coverage
        q_idx = int(np.ceil((n + 1) * (1 - alpha))) - 1
        q_idx = max(0, min(q_idx, n - 1))
        q_alpha = float(scores[q_idx])

        p_lower = float(np.clip(p_hat - q_alpha, 0.0, 1.0))
        p_upper = float(np.clip(p_hat + q_alpha, 0.0, 1.0))
        interval_width = p_upper - p_lower

        # Dashboard confidence display
        if interval_width < 0.10:
            confidence_display = "BOLD"
        elif interval_width < 0.25:
            confidence_display = "NORMAL"
        else:
            confidence_display = "FADED"

        return ConformalInterval(
            horizon_min=horizon_min,
            p_mean=p_hat,
            p_lower=p_lower,
            p_upper=p_upper,
            coverage=self.coverage,
            n_calibration=n,
            confidence_display=confidence_display,
        )

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({
                "coverage": self.coverage,
                "calibration_scores": self._calibration_scores,
            }, f)

    @classmethod
    def load(cls, path: str) -> "ConformalPredictor":
        with open(path, "rb") as f:
            data = pickle.load(f)
        predictor = cls(coverage=data["coverage"])
        predictor._calibration_scores = data["calibration_scores"]
        return predictor


# ── Unified Uncertainty Output ────────────────────────────────────────────────

@dataclass
class UncertaintyOutput:
    """
    Complete uncertainty-quantified forecast for one inference step.
    Used by the FastAPI /api/nowcast and /api/forecast endpoints.
    """
    # MC Dropout statistics
    flare_prob_mean: float
    flare_prob_std: float

    # Per-horizon conformal intervals
    forecast_intervals: list[ConformalInterval]

    # GOES class probabilities (from nowcast multiclass head)
    class_probs: dict[str, float]  # {"B": 0.1, "C": 0.3, "M": 0.4, "X": 0.15, "X+": 0.05}

    # CME risk
    cme_risk: float

    # Active modalities (for dashboard transparency)
    active_modalities: list[str]  # e.g. ["xray", "sharp", "mag"]

    # Data quality
    mag_quality: Optional[str] = None   # "GOOD", "DEGRADED", "UNAVAILABLE"
    swis_quality: Optional[str] = None

    def to_dashboard_dict(self) -> dict:
        """Serialize to dashboard-ready JSON dict."""
        intervals = {}
        for ci in self.forecast_intervals:
            intervals[f"p_flare_{ci.horizon_min}min"] = {
                "mean": round(ci.p_mean * 100, 1),
                "lower": round(ci.p_lower * 100, 1),
                "upper": round(ci.p_upper * 100, 1),
                "label": f"{round(ci.p_mean*100,0):.0f}% [{round(ci.p_lower*100,0):.0f}%–{round(ci.p_upper*100,0):.0f}%]",
                "confidence": ci.confidence_display,
                "coverage": int(ci.coverage * 100),
            }

        return {
            "nowcast": {
                "flare_probability": round(self.flare_prob_mean * 100, 1),
                "flare_probability_uncertainty": round(self.flare_prob_std * 100, 1),
                "class_probabilities": {k: round(v * 100, 1) for k, v in self.class_probs.items()},
                "cme_risk": round(self.cme_risk * 100, 1),
            },
            "forecast": intervals,
            "data_sources": {
                "active_modalities": self.active_modalities,
                "mag_quality": self.mag_quality,
                "swis_quality": self.swis_quality,
            },
        }


# ── Reliability Diagram (for model validation) ─────────────────────────────────

def compute_reliability_diagram(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Compute calibration curve (reliability diagram) data.
    Returns dict with bin centers, observed frequencies, and ECE.

    Expected Calibration Error (ECE) is the primary calibration metric
    used by NASA/NOAA space weather services.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    bin_acc = []
    bin_conf = []
    bin_counts = []

    for i in range(n_bins):
        mask = (probabilities >= bin_edges[i]) & (probabilities < bin_edges[i + 1])
        if np.sum(mask) == 0:
            bin_acc.append(float("nan"))
            bin_conf.append(float(bin_centers[i]))
            bin_counts.append(0)
        else:
            bin_acc.append(float(np.mean(labels[mask])))
            bin_conf.append(float(np.mean(probabilities[mask])))
            bin_counts.append(int(np.sum(mask)))

    # ECE = weighted mean |confidence - accuracy| per bin
    total = sum(bin_counts)
    ece = 0.0
    for acc, conf, cnt in zip(bin_acc, bin_conf, bin_counts):
        if not np.isnan(acc) and total > 0:
            ece += (cnt / total) * abs(conf - acc)

    return {
        "bin_centers": bin_centers.tolist(),
        "observed_frequency": bin_acc,
        "mean_confidence": bin_conf,
        "bin_counts": bin_counts,
        "ece": ece,
    }
