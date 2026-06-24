"""
AdityScan v3 — Phase A Training Notebook
GOES XRS 50-year pretraining for the X-ray TCN branch.

This script (or notebook) pretrains the XRayTCN on 50 years of GOES
1-minute soft X-ray flux data (1974-2024) from NOAA's NCEI archive.

Run as:
  python 01_goes_pretrain.py --epochs 100 --output models/xray_tcn_phase_a.pt

Dataset:
  GOES XRS 1-minute flux, 1974–2024
  Source: NOAA NCEI — https://www.ngdc.noaa.gov/stp/satellite/goes/
  Total samples: ~26M 1-minute samples (50 years × 365 days × 1440 min)
  Class distribution: ~96% quiet/A-B, ~3% C-class, ~1% M-class, ~0.1% X-class
  → Heavy class weighting required

Architecture:
  Input: (B, 30, 3) — 30-minute window at 1-min cadence
    Features: [flux_1_8, flux_0p5_4, derivative(flux_1_8)]
  Output: P(M+ flare in next 15 min) — binary classification
  Pretraining head: simple linear layer on top of TCN embedding

IMPORTANT: This is pretraining only. The full TCN takes 1-s cadence data.
This phase teaches the temporal patterns, then Phase C fine-tunes on Aditya-L1.
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
from pipeline.ml.fusion import XRayTCN
from pipeline.utils.metrics import (
    build_contingency_table,
    find_optimal_threshold,
    goes_class_from_flux,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────

GOES_PRETRAIN_CONFIG = {
    # Data
    "train_years": list(range(1974, 2020)),    # 46 years for train
    "val_years":   list(range(2020, 2023)),    # 3 years for validation
    "test_years":  list(range(2023, 2025)),    # 2 years for test
    "forecast_horizon_min": 15,               # target: M+ in next 15 min
    "window_min": 30,                          # 30-min input window
    "flare_class_threshold_wm2": 1e-5,        # M1 = 1e-5 W/m²

    # Training
    "batch_size": 512,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "epochs": 100,
    "patience": 10,                            # early stopping patience
    "class_weight_positive": 20.0,             # weight for rare positive class
    "gradient_clip": 1.0,

    # Model (matches XRayTCN but adapted for 1-min GOES input)
    # Note: we reduce N_FEATURES to 3 for GOES-only pretraining
    "dropout": 0.1,
}


# ── GOES Dataset ─────────────────────────────────────────────────────────────

class GOESDataset(Dataset):
    """
    PyTorch Dataset for GOES XRS 1-minute light curves.

    Label: 1 if max(flux[t+1 : t+horizon_min]) >= M1 threshold, else 0.
    Features per timestep: [log10(flux_1_8), log10(flux_0p5_4), d(log10_flux)/dt]
    """

    def __init__(
        self,
        df: pd.DataFrame,
        window_min: int = 30,
        horizon_min: int = 15,
        threshold_wm2: float = 1e-5,
    ):
        self.window = window_min
        self.horizon = horizon_min
        self.threshold = threshold_wm2

        # Precompute log flux
        df = df.copy()
        df["log_flux_1_8"] = np.log10(np.clip(df["flux_1_8"], 1e-9, None))
        df["log_flux_0p5_4"] = np.log10(np.clip(df["flux_0p5_4"], 1e-10, None))
        df["dlog_flux"] = np.gradient(df["log_flux_1_8"].values)

        self.features = df[["log_flux_1_8", "log_flux_0p5_4", "dlog_flux"]].values.astype(np.float32)
        self.flux_raw = df["flux_1_8"].values.astype(np.float64)

        # Precompute labels for all valid windows
        N = len(df)
        total = window_min + horizon_min
        self.valid_starts = np.arange(0, N - total)

    def __len__(self) -> int:
        return len(self.valid_starts)

    def __getitem__(self, idx: int):
        start = self.valid_starts[idx]
        window_end = start + self.window
        horizon_end = window_end + self.horizon

        # Input: (window_min, 3)
        x = self.features[start:window_end]

        # Label: any M+ in forecast horizon
        future_flux = self.flux_raw[window_end:horizon_end]
        y = float(np.any(future_flux >= self.threshold))

        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


def load_goes_csv(data_dir: str | Path, years: list[int]) -> pd.DataFrame:
    """
    Load GOES XRS data from NOAA NCEI CSV files.

    Expected file naming: goes{nn}_{YYYY}.csv or goes_1min_{YYYY}.csv
    Columns expected (NCEI format): time_tag, satellite, flux_0p5_4, flux_1_8
    Falls back to sunpy.timeseries.GOESTimeSeries if CSVs not found.

    Download command:
      for year in $(seq 1974 2024); do
        wget -O goes_${year}.csv "https://www.ngdc.noaa.gov/stp/satellite/goes/..." 
      done
    """
    dfs = []
    data_dir = Path(data_dir)

    for year in years:
        # Try local files first
        patterns = [
            data_dir / f"goes_1min_{year}.csv",
            data_dir / f"goes_{year}.csv",
            data_dir / f"*{year}*.csv",
        ]
        found = None
        for pat in patterns:
            files = list(data_dir.glob(pat.name if "*" not in str(pat) else pat.name))
            if files:
                found = files[0]
                break

        if found:
            df = pd.read_csv(found, parse_dates=["time_tag"])
            df = df.rename(columns={
                "time_tag": "time",
                "A_FLUX": "flux_1_8",
                "B_FLUX": "flux_0p5_4",
            })
            dfs.append(df)
        else:
            logger.warning("GOES data for year %d not found in %s — skipping", year, data_dir)

    if not dfs:
        logger.warning("No GOES data found. Generating synthetic data for testing.")
        return _generate_synthetic_goes(n_days=365)

    combined = pd.concat(dfs, ignore_index=True).sort_values("time")
    combined = combined.dropna(subset=["flux_1_8", "flux_0p5_4"])
    combined["flux_1_8"] = combined["flux_1_8"].clip(1e-9, None)
    combined["flux_0p5_4"] = combined["flux_0p5_4"].clip(1e-10, None)
    logger.info("Loaded GOES data: %d records, %d years", len(combined), len(years))
    return combined


def _generate_synthetic_goes(n_days: int = 365) -> pd.DataFrame:
    """
    Generate synthetic GOES XRS data for development/testing without real data.
    Follows realistic solar cycle statistics (Poisson flare occurrence).
    """
    np.random.seed(42)
    n_minutes = n_days * 1440
    times = pd.date_range("2024-01-01", periods=n_minutes, freq="1min")

    # Background: log-normal solar X-ray background
    bg = np.random.lognormal(mean=np.log(3e-8), sigma=0.3, size=n_minutes)

    # Inject synthetic flares (Poisson process, ~1 M-class per week)
    flux = bg.copy()
    flare_times = np.random.choice(n_minutes - 100, size=n_days // 7, replace=False)
    for ft in flare_times:
        peak_class = np.random.choice(["M", "X"], p=[0.9, 0.1])
        peak_flux = np.random.uniform(1e-5, 5e-5) if peak_class == "M" else np.random.uniform(1e-4, 5e-4)
        duration = np.random.randint(5, 30)
        t = np.arange(duration)
        flare_profile = peak_flux * np.exp(-((t - duration // 3) ** 2) / (2 * (duration // 5) ** 2))
        end_idx = min(ft + duration, n_minutes)
        flux[ft:end_idx] = np.maximum(flux[ft:end_idx], flare_profile[:end_idx - ft])

    return pd.DataFrame({
        "time": times,
        "flux_1_8": flux,
        "flux_0p5_4": flux * 0.3,
    })


# ── Training loop ─────────────────────────────────────────────────────────────

class GOESPretrainModel(nn.Module):
    """Adapter: XRayTCN backbone with 3-feature GOES input for pretraining."""

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        # Override N_FEATURES for GOES (3 features instead of 11)
        from pipeline.ml.fusion import TCNBlock, CausalConv1d
        self.input_proj = nn.Linear(3, 64)
        channels = [64, 64, 128, 128, 256, 256]
        dilations = [1, 2, 4, 8, 16, 32]
        blocks = []
        in_ch = 64
        for ch, dil in zip(channels, dilations):
            blocks.append(TCNBlock(in_ch, ch, kernel_size=8, dilation=dil, dropout=dropout))
            in_ch = ch
        self.tcn_blocks = nn.ModuleList(blocks)
        self.output_proj = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Dropout(dropout))
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(x).transpose(1, 2)
        for block in self.tcn_blocks:
            h = block(h)
        embedding = self.output_proj(h.mean(-1))
        logit = self.classifier(embedding)
        return embedding, logit


def train_phase_a(
    goes_data_dir: str = "./data/goes",
    output_dir: str = "./models",
    config: dict = GOES_PRETRAIN_CONFIG,
) -> str:
    """
    Run Phase A pretraining on GOES 50-year dataset.
    Returns path to saved model checkpoint.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Phase A training on device: %s", device)
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    logger.info("Loading GOES training data...")
    train_df = load_goes_csv(goes_data_dir, config["train_years"])
    val_df   = load_goes_csv(goes_data_dir, config["val_years"])

    train_ds = GOESDataset(train_df, config["window_min"], config["forecast_horizon_min"])
    val_ds   = GOESDataset(val_df,   config["window_min"], config["forecast_horizon_min"])

    logger.info("Train samples: %d | Val samples: %d", len(train_ds), len(val_ds))

    # Class weighting (positive class is rare: ~1%)
    labels = np.array([train_ds[i][1].item() for i in range(min(10000, len(train_ds)))])
    pos_frac = float(np.mean(labels))
    logger.info("Positive class fraction: %.3f%%", pos_frac * 100)

    # Weighted sampler for class balance
    sample_weights = np.where(labels > 0.5, config["class_weight_positive"], 1.0)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(np.tile(sample_weights, len(train_ds) // len(sample_weights) + 1)[:len(train_ds)]),
        num_samples=len(train_ds),
        replacement=True,
    )

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], sampler=sampler, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=config["batch_size"], shuffle=False,  num_workers=4)

    # Model
    model = GOESPretrainModel(dropout=config["dropout"]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(config["class_weight_positive"]).to(device))

    best_val_tss = -float("inf")
    patience_counter = 0
    best_checkpoint = None

    for epoch in range(1, config["epochs"] + 1):
        # Train
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            _, logit = model(x_batch)
            loss = criterion(logit.squeeze(), y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        # Validate
        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                _, logit = model(x_batch.to(device))
                probs = torch.sigmoid(logit.squeeze()).cpu().numpy()
                val_probs.extend(probs.tolist())
                val_labels.extend(y_batch.numpy().tolist())

        val_probs = np.array(val_probs)
        val_labels = np.array(val_labels)
        thresh, tss = find_optimal_threshold(val_labels, val_probs, metric="TSS")
        ct = build_contingency_table(val_labels, val_probs, threshold=thresh)

        logger.info(
            "Epoch %3d | Train Loss=%.4f | Val TSS=%.3f HSS=%.3f | "
            "POD=%.2f FAR=%.2f | Thresh=%.2f",
            epoch, np.mean(train_losses), tss, ct.HSS, ct.POD, ct.FAR, thresh
        )

        # Save checkpoint
        if tss > best_val_tss:
            best_val_tss = tss
            patience_counter = 0
            checkpoint_path = os.path.join(output_dir, "xray_tcn_phase_a_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_tss": tss,
                "config": config,
                "optimal_threshold": thresh,
            }, checkpoint_path)
            best_checkpoint = checkpoint_path
            logger.info("✓ New best model saved (TSS=%.3f)", tss)
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch, config["patience"])
                break

    logger.info("Phase A pretraining complete. Best TSS=%.3f", best_val_tss)
    return best_checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AdityScan Phase A: GOES pretraining")
    parser.add_argument("--goes-data-dir", default="./data/goes", help="Path to GOES CSV files")
    parser.add_argument("--output-dir", default="./models", help="Model output directory")
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs")
    args = parser.parse_args()

    GOES_PRETRAIN_CONFIG["epochs"] = args.epochs
    checkpoint = train_phase_a(args.goes_data_dir, args.output_dir, GOES_PRETRAIN_CONFIG)
    print(f"\nPhase A complete. Checkpoint: {checkpoint}")
    print("Next: Run 02_sharp_lstm_train.py for Phase B")
