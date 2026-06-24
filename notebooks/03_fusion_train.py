"""
AdityScan v3 — Phase C/D Training
Multi-modal fusion fine-tuning + calibration + ONNX export.

Phase C: Fine-tune the full AdityScanModel on Aditya-L1 data (when available).
         For now: uses synthetic Aditya-L1-like data or XGBoost head on frozen embeddings.

Phase D: Calibration (temperature scaling + conformal prediction setup).

Run:
  python 03_fusion_train.py --phase c --output models/

Depends on: models/xray_tcn_phase_a_best.pt, models/sharp_lstm_phase_b_best.pt
"""

import os
import sys
import logging
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
from pipeline.ml.fusion import AdityScanModel
from pipeline.ml.uncertainty import (
    TemperatureScaler,
    ConformalPredictor,
    compute_reliability_diagram,
)
from pipeline.utils.metrics import build_contingency_table, find_optimal_threshold, roc_auc

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


FUSION_CONFIG = {
    "batch_size": 64,
    "learning_rate": 5e-5,     # Very small LR for fine-tuning
    "weight_decay": 1e-4,
    "epochs": 30,
    "patience": 8,
    "freeze_backbone_epochs": 10,  # Freeze TCN+LSTM for first 10 epochs
    "mc_dropout_samples": 50,
    "conformal_coverage": 0.90,
    "forecast_horizons_min": [5, 10, 15, 30, 60],
}


# ── Synthetic multi-modal dataset (until real Aditya-L1 data arrives) ─────────

class MultiModalDataset(Dataset):
    """
    Multi-modal fusion dataset.

    In production: loads aligned SoLEXS spectra + SHARP + MAG + SWIS.
    For development: generates realistic synthetic samples.

    Sample:
      xray:   (1800, 11)   — 30 min × 1-s, 11 SoLEXS+HEL1OS features
      sharp:  (120, 21)    — 24h × 12-min, 21 SHARP+MAG features
      insitu: (14,)        — current MAG+SWIS snapshot
      label:  (5,)         — P(M+ flare) at [5,10,15,30,60] min horizons
    """

    def __init__(self, n_samples: int = 5000, seed: int = 42) -> None:
        np.random.seed(seed)
        self.n = n_samples

        # Synthetic features (zero-mean, unit variance after normalization)
        self.xray  = np.random.randn(n_samples, 1800, 11).astype(np.float32)
        self.sharp = np.random.randn(n_samples, 120, 21).astype(np.float32)
        self.insitu = np.random.randn(n_samples, 14).astype(np.float32)

        # Synthetic labels: correlated with xray signal intensity
        xray_signal = self.xray[:, -60:, 0].max(axis=1)   # last-minute X-ray peak
        base_prob = 1.0 / (1.0 + np.exp(-2.0 * (xray_signal - 1.0)))  # sigmoid
        # Multi-horizon: later horizons have lower probability
        self.labels = np.stack([
            np.clip(base_prob * (1 - 0.05 * i) + np.random.normal(0, 0.05, n_samples), 0, 1)
            for i in range(5)
        ], axis=1).astype(np.float32)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.xray[idx]),
            torch.from_numpy(self.sharp[idx]),
            torch.from_numpy(self.insitu[idx]),
            torch.from_numpy(self.labels[idx]),
        )


# ── Phase C: Fusion fine-tuning ────────────────────────────────────────────────

def train_phase_c(
    phase_a_ckpt: str,
    phase_b_ckpt: str,
    output_dir: str = "./models",
    config: dict = FUSION_CONFIG,
) -> str:
    """
    Fine-tune the full AdityScan multi-modal model.

    Strategy:
      1. Load TCN weights from Phase A, LSTM weights from Phase B
      2. Freeze backbone for first K epochs (train only fusion + heads)
      3. Unfreeze all layers for remaining epochs (smaller LR)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Phase C fusion training on device: %s", device)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize model
    model = AdityScanModel(mc_dropout=0.1).to(device)

    # Load Phase A weights into TCN branch
    if phase_a_ckpt and os.path.exists(phase_a_ckpt):
        ckpt_a = torch.load(phase_a_ckpt, map_location="cpu", weights_only=False)
        # Map GOESPretrainModel → XRayTCN (compatible layers)
        tcn_state = {k.replace("tcn_blocks.", "").replace("input_proj.", ""): v
                     for k, v in ckpt_a["model_state_dict"].items()
                     if "classifier" not in k}
        missing, unexpected = model.xray_branch.load_state_dict(tcn_state, strict=False)
        logger.info("Phase A weights loaded: %d missing, %d unexpected keys", len(missing), len(unexpected))

    # Load Phase B weights into SHARP branch
    if phase_b_ckpt and os.path.exists(phase_b_ckpt):
        ckpt_b = torch.load(phase_b_ckpt, map_location="cpu", weights_only=False)
        sharp_state = {k.replace("lstm_branch.", ""): v
                       for k, v in ckpt_b["model_state_dict"].items()
                       if "lstm_branch" in k}
        missing, unexpected = model.sharp_branch.load_state_dict(sharp_state, strict=False)
        logger.info("Phase B weights loaded: %d missing, %d unexpected keys", len(missing), len(unexpected))

    # Dataset
    train_ds = MultiModalDataset(n_samples=4000, seed=42)
    val_ds   = MultiModalDataset(n_samples=1000, seed=99)
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=config["batch_size"], shuffle=False)

    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    criterion = nn.BCELoss()

    best_tss = -float("inf")
    patience_counter = 0
    best_checkpoint = None

    for epoch in range(1, config["epochs"] + 1):
        # Freeze/unfreeze backbone
        if epoch <= config["freeze_backbone_epochs"]:
            for p in model.xray_branch.parameters():
                p.requires_grad = False
            for p in model.sharp_branch.parameters():
                p.requires_grad = False
        else:
            for p in model.parameters():
                p.requires_grad = True

        model.train()
        losses = []
        for xray, sharp, insitu, labels in train_loader:
            xray, sharp, insitu, labels = xray.to(device), sharp.to(device), insitu.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(xray, sharp, insitu)

            # Multi-task loss: BCELoss on each horizon + nowcast
            loss = criterion(out["flare_prob"].squeeze(), labels[:, 2])  # primary: 15-min
            for i, key in enumerate(["p_flare_5min", "p_flare_10min", "p_flare_15min", "p_flare_30min", "p_flare_60min"]):
                loss += 0.5 * criterion(out[key].squeeze(), labels[:, i])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        # Validate on 15-min horizon
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for xray, sharp, insitu, labels in val_loader:
                out = model(xray.to(device), sharp.to(device), insitu.to(device))
                probs = out["p_flare_15min"].squeeze().cpu().numpy()
                all_probs.extend(probs.tolist() if probs.ndim > 0 else [float(probs)])
                all_labels.extend(labels[:, 2].numpy().tolist())

        all_probs = np.array(all_probs)
        all_labels = (np.array(all_labels) >= 0.5).astype(int)  # binarize
        thresh, tss = find_optimal_threshold(all_labels, all_probs, metric="TSS")
        auc = roc_auc(all_labels, all_probs)
        logger.info("Epoch %3d | Loss=%.4f | 15min TSS=%.3f AUC=%.3f | Thresh=%.2f",
                    epoch, np.mean(losses), tss, auc, thresh)

        if tss > best_tss:
            best_tss = tss
            patience_counter = 0
            ckpt_path = os.path.join(output_dir, "adityscan_fusion_phase_c_best.pt")
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "val_tss_15min": tss, "val_auc_15min": auc,
                "optimal_threshold": thresh, "config": config,
            }, ckpt_path)
            best_checkpoint = ckpt_path
            logger.info("✓ New best fusion model (TSS=%.3f)", tss)
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                logger.info("Early stopping at epoch %d", epoch)
                break

    return best_checkpoint


# ── Phase D: Calibration + Conformal Prediction ─────────────────────────────

def run_phase_d(
    fusion_checkpoint: str,
    output_dir: str = "./models",
    config: dict = FUSION_CONFIG,
) -> None:
    """
    Post-training calibration:
      1. Temperature scaling on validation set
      2. Conformal prediction calibration on held-out calibration set
      3. Reliability diagram computation + ECE
      4. ONNX export for production inference
    """
    device = torch.device("cpu")  # Calibration on CPU
    logger.info("Phase D: Calibration and conformal prediction setup")

    # Load trained model
    model = AdityScanModel(mc_dropout=0.1).to(device)
    if os.path.exists(fusion_checkpoint):
        ckpt = torch.load(fusion_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Calibration dataset (holdout — NEVER seen during training or validation)
    cal_ds = MultiModalDataset(n_samples=2000, seed=777)
    cal_loader = DataLoader(cal_ds, batch_size=128, shuffle=False)

    # Collect raw predictions
    all_logits, all_probs_raw, all_labels = [], [], []
    with torch.no_grad():
        for xray, sharp, insitu, labels in cal_loader:
            out = model(xray, sharp, insitu)
            prob = out["p_flare_15min"].squeeze().numpy()
            logit = np.log(prob / (1 - prob + 1e-10) + 1e-10)
            all_logits.extend(logit.tolist() if np.ndim(logit) > 0 else [float(logit)])
            all_probs_raw.extend(prob.tolist() if np.ndim(prob) > 0 else [float(prob)])
            all_labels.extend((labels[:, 2] >= 0.5).numpy().tolist())

    logits = np.array(all_logits)
    probs_raw = np.array(all_probs_raw)
    labels_binary = np.array(all_labels).astype(int)

    # 1. Temperature scaling
    logger.info("Fitting temperature scaling...")
    ts = TemperatureScaler()
    T = ts.fit(logits, labels_binary)
    probs_calibrated = ts.calibrate(probs_raw)
    ts.save(os.path.join(output_dir, "temperature_scaler.pkl"))
    logger.info("Temperature = %.4f", T)

    # 2. Reliability diagram
    reliability = compute_reliability_diagram(probs_calibrated, labels_binary)
    logger.info("ECE after calibration: %.4f", reliability["ece"])

    # 3. Conformal prediction (split calibration)
    # Use first half for conformal calibration
    n_half = len(probs_calibrated) // 2
    conformal = ConformalPredictor(coverage=config["conformal_coverage"])

    for i, horizon_min in enumerate(config["forecast_horizons_min"]):
        # Use a different calibration subset per horizon
        cal_probs = probs_calibrated[:n_half] * (1 - 0.02 * i)  # synthetic variation
        cal_labels = labels_binary[:n_half]
        conformal.fit(horizon_min, cal_probs, cal_labels)

    conformal.save(os.path.join(output_dir, "conformal_predictor.pkl"))
    logger.info("Conformal predictor saved (coverage=%.0f%%)", config["conformal_coverage"] * 100)

    # 4. ONNX export
    from pipeline.ml.fusion import export_to_onnx
    onnx_path = os.path.join(output_dir, "adityscan_v3.onnx")
    export_to_onnx(model, onnx_path)
    logger.info("ONNX model exported: %s", onnx_path)

    # 5. Summary
    thresh, tss = find_optimal_threshold(labels_binary, probs_calibrated, "TSS")
    ct = build_contingency_table(labels_binary, probs_calibrated, thresh)
    auc = roc_auc(labels_binary, probs_calibrated)

    summary = {
        "temperature": T, "ece": reliability["ece"],
        "tss": round(tss, 4), "hss": round(ct.HSS, 4),
        "pod": round(ct.POD, 4), "far": round(ct.FAR, 4),
        "auc_roc": round(auc, 4),
        "optimal_threshold": round(thresh, 3),
        "conformal_coverage": config["conformal_coverage"],
        "model_version": "3.0.0",
    }
    summary_path = os.path.join(output_dir, "model_summary.json")
    import json
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=== FINAL MODEL SUMMARY ===")
    for k, v in summary.items():
        logger.info("  %s: %s", k, v)
    logger.info("All model artifacts saved to: %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AdityScan Phase C/D: Fusion training + calibration")
    parser.add_argument("--phase", choices=["c", "d", "cd"], default="cd")
    parser.add_argument("--phase-a-checkpoint", default="./models/xray_tcn_phase_a_best.pt")
    parser.add_argument("--phase-b-checkpoint", default="./models/sharp_lstm_phase_b_best.pt")
    parser.add_argument("--fusion-checkpoint",  default="./models/adityscan_fusion_phase_c_best.pt")
    parser.add_argument("--output-dir", default="./models")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    fusion_ckpt = args.fusion_checkpoint
    if args.phase in ("c", "cd"):
        fusion_ckpt = train_phase_c(args.phase_a_checkpoint, args.phase_b_checkpoint, args.output_dir)

    if args.phase in ("d", "cd"):
        run_phase_d(fusion_ckpt, args.output_dir)

    print("\n✓ Training pipeline complete.")
    print(f"  Models saved to: {args.output_dir}")
    print("  Production artifacts:")
    print("    adityscan_v3.onnx       — ONNX model for CPU inference")
    print("    temperature_scaler.pkl  — Temperature scaling calibration")
    print("    conformal_predictor.pkl — Conformal prediction intervals")
    print("    model_summary.json      — TSS/HSS/AUC/ECE metrics")
