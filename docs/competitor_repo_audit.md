# Competitor Repository Audit

Date: 2026-06-29

Scope: local folders `Repo-1` through `Repo-10`, compared against the current
AdityScan repo. The main rule used here is strict: README claims are not treated
as valid metrics unless a model artifact, metrics file, or training/evaluation
script supports them.

## Executive Ranking

| Rank | Repo | Best thing it has | Verified model status |
|---:|---|---|---|
| 1 | Repo-3 `Solar-Flare-Nowcast` | Real GOES historical LightGBM models, chronological evaluation, calibrated probabilities, lead-time metrics | Strongest verified trained artifact |
| 2 | Repo-4 `bah2026-p15` | Best system architecture: O(1) detection, fusion/QC/provenance, leakage-aware CV design, tests | Mostly synthetic/offline, no real trained production artifact found |
| 3 | Repo-8 `aditya-flare-forecast` | Best Aditya-specific compact neural architecture idea: parallel SoLEXS/HEL1OS CNN + BiLSTM + multi-horizon heads | Metrics artifact is invalid/untrustworthy |
| 4 | Repo-10 `aditya-l1-pradan-download` | Useful PRADAN scripts and causal z-score baseline notebooks | No trained model |
| 5 | Repo-1 `ISRO-hackathon` | Good operational dashboard and adaptive-threshold/forecast cascade idea | Claimed metrics are not supported |
| 6 | Repo-6 `aditya-l1-solar-flare-forecasting` | Strong visual/demo frontend | No real model backend; some UI metric claims appear decorative |
| 7 | Repo-9 `adityal1-issdc-access-skill` | Good PRADAN/ISSDC access guidance | Not a forecasting repo |
| 8 | Repo-2 `SolarShield-AI` | Basic skeleton | No meaningful model/metrics found |
| 9 | Repo-5 | Safe-mode / satellite-protection concept | Docs-only skeleton |
| 10 | Repo-7 | Duplicate/near-duplicate of Repo-5 | Docs-only skeleton |

## Current AdityScan Baseline

Current saved report: `models/training_report.json`.

| Metric | Current value |
|---|---:|
| Training window | 2024-05 only |
| TSS | 0.0000 |
| HSS | 0.0000 |
| POD / Recall | 1.0000 |
| FAR | 0.0769 |
| AUC | 0.4167 |
| Confusion matrix | TP=156, FP=13, TN=0, FN=0 |

Interpretation: this was a smoke/update run, not a competitive final model.
TSS is zero because the evaluated split had no true negatives. The model cannot
demonstrate quiet-Sun discrimination until quiet/control months are included.

## Repo-by-Repo Findings

### Repo-1: `ISRO-hackathon`

Claims: SoLEXS+HEL1OS fusion, MAD adaptive thresholding, TCN forecast, POD 0.94,
FAR 0.21, CSI 0.78, and 96.8% accuracy in UI text.

Evidence:
- `scripts/train_models.py` generates JSON weights rather than training from a
  real dataset.
- `scripts/backtest.py` uses synthetic/mock conditions and fallback/default
  metric values.
- API/UI serve validation metrics as defaults/fallbacks.

Better than us:
- Nice two-stage operational idea: adaptive nowcast threshold plus forecast
  confidence.
- Dashboard/replay/API experience is more presentation-ready.

False/unsupported claim:
- README and UI metrics should not be treated as real validation results.

### Repo-2: `SolarShield-AI`

Mostly a project skeleton with placeholder files. No trained artifacts or
credible metric reports found.

Better than us: nothing substantial for model quality.

### Repo-3: `Solar-Flare-Nowcast`

This is the strongest verified competitor.

Model/data:
- Trained LightGBM models are present:
  `flare_model_10min.pkl`, `flare_model_30min.pkl`, `flare_model_60min.pkl`.
- Uses GOES XRS historical data, 570,493 rows.
- Chronological split:
  train `2023-01-01 00:00:00` to `2024-10-13 16:57:00`;
  test `2024-10-13 16:58:00` to `2024-12-31 23:59:00`.
- Includes calibrated probabilities, Brier score, lead-time distribution, and
  persistence baselines.

Verified headline metrics:

| Horizon | TSS | HSS | Precision | Recall | FAR | ROC-AUC | Brier | TP | FP | FN | TN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 min | 0.7429 | 0.8339 | 0.9698 | 0.7443 | 0.0302 | 0.9556 | 0.0140 | 4920 | 153 | 1690 | 107336 |
| 30 min | 0.4946 | 0.6323 | 0.9595 | 0.4967 | 0.0405 | 0.8657 | 0.0409 | 5025 | 212 | 5092 | 103770 |
| 60 min | 0.3453 | 0.4713 | 0.9415 | 0.3486 | 0.0585 | 0.8099 | 0.0741 | 5212 | 324 | 9740 | 98823 |

Lead-time:
- 202 test events at each horizon.
- 10 min median lead time: 8 min.
- 30 min median lead time: 9 min.
- 60 min median lead time: 15 min.

Better than us:
- Proper negative examples and chronological holdout.
- Strong TSS/FAR/AUC with real metric artifacts.
- Calibration/reliability tracking.
- Persistence baseline comparison.
- Multi-horizon models.

Limitations:
- Core model is GOES-only. Aditya-L1 FITS ingestion is fusion-ready but not
  actually the trained predictive signal.

False/incorrect claim:
- README documentation describes TSS with an incorrect formula. The code uses
  the correct definition (`TPR - FPR`), so the implementation is better than the
  README text.

### Repo-4: `bah2026-p15`

Best architecture/science scaffold.

Strengths:
- O(1) streaming detection primitives.
- CUSUM and Poisson FOCuS detection.
- Event finite-state machine and catalog association.
- Multi-sensor fusion, light-travel-time correction, cross-calibration, quality
  scores, provenance, gap filling, and consensus labels.
- Forecast evaluation utilities with TSS, HSS, Brier skill score, POD, FAR,
  ROC-AUC, lead-time, and lead-time-vs-FAR sweep.
- Leakage-aware blocked rolling-origin CV and embargo design.
- Good tests around metrics, labels, CV, fusion, API, and edge parity.

Better than us:
- Much stronger test discipline.
- Much stronger operational/data-quality architecture.
- Better separation between detector, catalog, fusion, forecast, API, and edge.

Limitations:
- Forecasting example is synthetic/offline.
- No real trained model artifact or real-world metric report found.

### Repo-5 and Repo-7

These appear to be duplicates or near-duplicates. Mostly folder structure and
docs around solar flare forecasting and satellite protection/safe-mode ideas.

Better than us:
- Mission-impact framing and safe-mode activation story could help final demo.

Limitations:
- No meaningful implementation, trained model, or metrics found.

### Repo-6: `aditya-l1-solar-flare-forecasting`

Frontend/demo-heavy React + Vite project.

Better than us:
- Stronger immersive mission visualization.
- Useful demo narrative: satellite assembly, solar spectrum, mission control,
  risk dashboard.

Limitations/concerns:
- Main README is still the default Vite README.
- No real model backend or trained artifacts found.
- UI contains hard claims like pre-flare detector accuracy without evidence.

### Repo-8: `aditya-flare-forecast`

Best Aditya-specific neural model concept, but weak evaluation artifact.

Architecture:
- Input `[B, 360, 17]`, 60-min windows at 10-second cadence.
- Parallel SoLEXS and HEL1OS Conv1D branches.
- Overlap Conv for cross-instrument correlation.
- BiLSTM temporal trunk.
- Independent heads for 15/30/60 min probabilities.
- Focal loss and regularization for rare events.

Actual metric artifact:
- `best_val_tss = 0.0`.
- Test set has only one positive example per horizon and zero negatives.
- Reported `best_tss = 1.0` is based on `TP=1, TN=0, FP=0, FN=0`.
- AUC is 0.5, which means no ranking skill.

Better than us:
- The compact parallel-branch architecture is worth testing as a challenger to
  our larger causal TCN.
- Multi-horizon heads are a good product fit.

False/unsupported claim:
- Pipeline docs claim stronger TSS/AUC than the saved artifact supports.
- The saved `TSS=1.0` is not meaningful because there are no negatives.

### Repo-9: `adityal1-issdc-access-skill`

Not a model repo. It is an access/plotting skill for Aditya-L1 and PRADAN.

Better than us:
- Good security guidance: do not store PRADAN passwords; use browser session or
  local files.
- Good payload inventory and first-plot guidance.

### Repo-10: `aditya-l1-pradan-download`

PRADAN scripts and exploratory notebooks.

Better than us:
- Useful causal rolling baseline: rolling median, excess, z-score, gradient,
  hardness features.
- Notebook clearly distinguishes possible false positives from matched flare
  detections.

Limitations:
- No trained model artifact.
- Download scripts are static PRADAN session scripts, not a general reusable
  date-range downloader.

## What Our Idea Covers That Others Mostly Do Not

- Actual multi-modal Aditya-L1 PRADAN ingestion plan for SoLEXS, HEL1OS, MAG,
  with future slots for SWIS/SUIT/SHARP.
- Existing raw HEL1OS L1 ZIP/lightcurve loader support.
- Golden balanced manifest with quiet/control and flare-positive windows.
- Physics-derived features: HXR/SXR ratio, Neupert integral, HEL1OS spectral
  index, QPP/CWT bands, MAG geometry, and Alfvén Mach number slots.
- Cross-modal neural fusion with missing-branch handling.
- Explicit future-sensor architecture rather than a one-off GOES-only model.
- Incremental/update training path with `--resume` and per-month manifests.

The weakness is not the idea. The weakness is that the currently saved model was
trained/evaluated on an imbalanced smoke run. We need to upgrade evaluation
discipline before making performance claims.

## Implementation Plan for AdityScan

### Phase 1: Fix the benchmark layer using Repo-3 ideas

1. Add a GOES-only LightGBM baseline trained on long historical GOES data.
2. Evaluate with strict chronological holdout.
3. Report TSS, HSS, POD/recall, FAR, precision, ROC-AUC, Brier score, and
   confusion matrix for each horizon.
4. Add persistence and climatology baselines.
5. Add reliability/calibration curves and temperature/isotonic calibration.

### Phase 2: Fix AdityScan evaluation before final training

1. Finish the balanced dataset:
   - quiet/control: 2024-03, 2024-06, 2024-08, 2024-09
   - flare-positive: 2024-05
2. Split by day/event, not random windows, to avoid temporal leakage.
3. Require every train/val/test split to contain quiet and flare periods.
4. Reject metric reports when `TN=0`, `FP+TN=0`, or test positives/negatives are
   too small.
5. Store per-month and per-horizon metrics in `training_report.json`.

### Phase 3: Borrow Repo-4 system rigor

1. Add tests for metric formulas, label windows, embargo splits, and leakage.
2. Add a data-quality/provenance table per window:
   source files, missing modalities, gap fraction, cadence, and time alignment.
3. Add simple O(1) nowcast detector as a fallback alongside the neural model.
4. Add lead-time-vs-FAR operating-point sweep for threshold selection.

### Phase 4: Test Repo-8 architecture as a challenger model

1. Implement a compact `ParallelSoLEXSHEL1OSCNNBiLSTM` model behind a config flag.
2. Use 15/30/60 min heads.
3. Use focal loss/class-balanced loss.
4. Compare against our current TCN on the same exact split.
5. Keep the winning architecture; do not replace ours based on claims alone.

### Phase 5: Borrow Repo-10's baseline features

1. Add rolling median/MAD/z-score/gradient/hardness baseline features.
2. Keep them causal only: no future samples in the rolling baseline.
3. Use the baseline detector for explainability and sanity checks.

### Phase 6: Presentation polish only after metrics are real

1. Borrow Repo-6 style only for visual storytelling if needed.
2. Borrow Repo-5/7 safe-mode framing as an impact module.
3. Do not display unsupported accuracy/TSS claims in the UI.

## Recommended Next Build Order

1. Complete quiet/control data ingestion.
2. Add metric validity guards.
3. Add GOES-only LightGBM baseline.
4. Train AdityScan lean model on the balanced manifest.
5. Add Repo-8 compact architecture as an ablation.
6. Choose final model by verified TSS/FAR/AUC, not README claims.

