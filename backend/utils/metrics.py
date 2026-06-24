"""
Skill metrics for solar flare prediction evaluation.

Standard metrics used by the space weather community:
  TSS  — True Skill Statistic (Peirce's Skill Score)
  HSS  — Heidke Skill Score
  FAR  — False Alarm Rate (fraction of warnings that were wrong)
  POD  — Probability of Detection (recall)
  BIAS — Frequency bias (ratio of forecast events to observed)
  ROC AUC — Area Under Receiver Operating Characteristic Curve

These are the metrics used by NOAA SWPC, NASA CCMC, and the
Space Weather Prediction Center to evaluate operational models.

Reference:
  Bloomfield et al. 2012, ApJ Letters (flare forecasting benchmarks)
  Woodcock 1976 (TSS/HSS formulation for 2×2 contingency tables)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class ContingencyTable:
    """2×2 contingency table for binary flare forecasting."""
    TP: int   # True Positive (event forecast + event observed)
    FP: int   # False Positive (event forecast + no event observed)
    FN: int   # False Negative (no forecast + event observed = miss)
    TN: int   # True Negative (no forecast + no event observed)

    @property
    def N(self) -> int:
        return self.TP + self.FP + self.FN + self.TN

    @property
    def POD(self) -> float:
        """Probability of Detection (Hit Rate, Recall)."""
        denom = self.TP + self.FN
        return self.TP / denom if denom > 0 else 0.0

    @property
    def FAR(self) -> float:
        """False Alarm Rate (False Alarm Ratio in some literature)."""
        denom = self.TP + self.FP
        return self.FP / denom if denom > 0 else 0.0

    @property
    def POFD(self) -> float:
        """Probability of False Detection (False Positive Rate)."""
        denom = self.FP + self.TN
        return self.FP / denom if denom > 0 else 0.0

    @property
    def TSS(self) -> float:
        """
        True Skill Statistic (Peirce's Skill Score).
        TSS = POD - POFD  ∈ [-1, 1], perfect = 1, random = 0.
        Preferred metric for rare event forecasting (flares are rare).
        """
        return self.POD - self.POFD

    @property
    def HSS(self) -> float:
        """
        Heidke Skill Score.
        HSS = 2(TP·TN - FP·FN) / [(TP+FN)(FN+TN) + (TP+FP)(FP+TN)]
        ∈ [-∞, 1], perfect = 1, random = 0.
        """
        n1 = 2 * (self.TP * self.TN - self.FP * self.FN)
        n2 = ((self.TP + self.FN) * (self.FN + self.TN) +
               (self.TP + self.FP) * (self.FP + self.TN))
        return n1 / n2 if n2 != 0 else 0.0

    @property
    def BIAS(self) -> float:
        """Frequency bias = (TP + FP) / (TP + FN). >1 over-forecast."""
        denom = self.TP + self.FN
        return (self.TP + self.FP) / denom if denom > 0 else float("nan")

    @property
    def precision(self) -> float:
        denom = self.TP + self.FP
        return self.TP / denom if denom > 0 else 0.0

    def summary_dict(self) -> dict:
        return {
            "TP": self.TP, "FP": self.FP, "FN": self.FN, "TN": self.TN,
            "POD": round(self.POD, 4),
            "FAR": round(self.FAR, 4),
            "TSS": round(self.TSS, 4),
            "HSS": round(self.HSS, 4),
            "BIAS": round(self.BIAS, 4),
            "Precision": round(self.precision, 4),
            "N": self.N,
        }


def build_contingency_table(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    threshold: float = 0.5,
) -> ContingencyTable:
    """
    Build 2×2 contingency table from probability predictions.

    Parameters
    ----------
    y_true : (N,) binary array — 1 = flare, 0 = no flare
    y_pred_prob : (N,) float — model probability estimates
    threshold : float — decision threshold (default 0.5)
    """
    y_pred = (y_pred_prob >= threshold).astype(int)
    TP = int(np.sum((y_true == 1) & (y_pred == 1)))
    FP = int(np.sum((y_true == 0) & (y_pred == 1)))
    FN = int(np.sum((y_true == 1) & (y_pred == 0)))
    TN = int(np.sum((y_true == 0) & (y_pred == 0)))
    return ContingencyTable(TP=TP, FP=FP, FN=FN, TN=TN)


def roc_auc(y_true: np.ndarray, y_pred_prob: np.ndarray) -> float:
    """Compute ROC AUC score."""
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_pred_prob))


def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    metric: str = "TSS",
) -> tuple[float, float]:
    """
    Find the probability threshold that maximizes TSS or HSS.
    Returns (optimal_threshold, optimal_metric_value).
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best_thresh = 0.5
    best_val = -float("inf")

    for t in thresholds:
        ct = build_contingency_table(y_true, y_pred_prob, threshold=t)
        val = ct.TSS if metric == "TSS" else ct.HSS
        if val > best_val:
            best_val = val
            best_thresh = t

    return float(best_thresh), float(best_val)
