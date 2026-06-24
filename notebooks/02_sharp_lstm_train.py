"""
AdityScan v3 — Phase B Training
SHARP LSTM joint training with GOES context (2010–2024).

Trains the SHARPLSTMBranch on SDO/HMI SHARP magnetic parameters
combined with GOES class labels for M+ flare prediction.

Run:
  python 02_sharp_lstm_train.py --output models/sharp_lstm_phase_b.pt

Data:
  SHARP parameters from JSOC (hmi.sharp_cea_720s series)
  GOES event catalog for flare labels (NOAA HEK / SolarMonitor)
  Period: 2010-05-01 to 2024-12-31 (full SDO era)
  Sampling: 12-minute SHARP snapshots for active regions > 100 μHem

Label construction:
  Positive: any M+ flare within the next T hours (forecast horizon)
  Negative: no M+ flare within T hours
  T = [6h, 12h, 24h, 48h] (multi-task learning — one head per horizon)

Class imbalance:
  ~5% of AR-snapshots precede an M+ flare within 24h
  Use stratified sampling + focal loss (γ=2)
"""

import os
import sys
import json
import logging
import argparse
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
from pipeline.ml.fusion import SHARPLSTMBranch
from pipeline.utils.metrics import build_contingency_table, find_optimal_threshold, roc_auc

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────

SHARP_CONFIG = {
    "train_split": 0.70,
    "val_split":   0.15,
    "test_split":  0.15,
    "forecast_horizons_h": [6, 12, 24, 48],
    "window_steps": 120,          # 24h × 12-min cadence
    "n_sharp_features": 16,       # 16 SHARP ML params
    "n_total_features": 21,       # 16 SHARP + 3 MAG-derived + 2 derived
    "batch_size": 128,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 80,
    "patience": 12,
    "focal_gamma": 2.0,           # Focal loss gamma
    "dropout": 0.2,
}

SHARP_ML_FEATURES = [
    "TOTUSJH", "TOTUSJZ", "MEANPOT", "SAVNCPP", "USFLUX",
    "AREA_ACT", "R_VALUE", "SHRGT45", "TOTBSQ", "TOTPOT",
    "TOTFZ", "ABSNJZH", "EPSZ", "TOTFX", "TOTFY", "NACR",
]
EXTRA_FEATURES = ["B_total_mean", "clock_angle_mean", "cone_angle_mean",
                  "goes_flux_log", "goes_flux_derivative"]  # derived from GOES + MAG


# ── Focal Loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Binary focal loss for highly imbalanced datasets.
    FL(p) = -α(1-p)^γ log(p)
    Reduces loss contribution from easy negatives, focuses on hard positives.
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


# ── Dataset ───────────────────────────────────────────────────────────────────

class SHARPDataset(Dataset):
    """
    Dataset for SHARP-based flare prediction.

    Each sample:
      x: (120, 21) — 24h of SHARP + extra features (RobustScaler applied)
      y: (4,) — labels for 6h, 12h, 24h, 48h horizons
    """

    def __init__(
        self,
        df: pd.DataFrame,
        event_df: pd.DataFrame,
        horizons_h: list[int] = [6, 12, 24, 48],
        window_steps: int = 120,
        scaler=None,
    ):
        from sklearn.preprocessing import RobustScaler

        self.window_steps = window_steps
        self.horizons_h = horizons_h
        self.feature_cols = SHARP_ML_FEATURES + EXTRA_FEATURES[:5]

        # Fill NaN with forward fill then zero
        df = df.copy().ffill().fillna(0)

        # Fit/apply RobustScaler
        if scaler is None:
            self.scaler = RobustScaler()
            df[self.feature_cols] = self.scaler.fit_transform(df[self.feature_cols].values)
        else:
            self.scaler = scaler
            df[self.feature_cols] = scaler.transform(df[self.feature_cols].values)

        self.features = df[self.feature_cols].values.astype(np.float32)
        self.times = pd.to_datetime(df["unix_time"], unit="s", utc=True)

        # Build labels for each window
        self.valid_idx = []
        self.labels = []

        for i in range(window_steps, len(df)):
            t_end = self.times.iloc[i]
            row_labels = []
            for h in horizons_h:
                t_horizon = t_end + timedelta(hours=h)
                # Check if any M+ flare in [t_end, t_horizon]
                flares_in_window = event_df[
                    (event_df["start_time"] >= t_end) &
                    (event_df["start_time"] < t_horizon) &
                    (event_df["goes_class"].str.match(r"^[MX]"))
                ]
                row_labels.append(1 if len(flares_in_window) > 0 else 0)
            self.valid_idx.append(i)
            self.labels.append(row_labels)

        self.labels = np.array(self.labels, dtype=np.float32)
        logger.info("SHARP dataset: %d samples, pos_rate_24h=%.2f%%",
                    len(self.valid_idx),
                    100 * self.labels[:, 2].mean())

    def __len__(self) -> int:
        return len(self.valid_idx)

    def __getitem__(self, idx: int):
        i = self.valid_idx[idx]
        x = self.features[i - self.window_steps:i]  # (120, 21)
        y = self.labels[idx]                          # (4,)
        return torch.from_numpy(x), torch.from_numpy(y)


def load_sharp_and_events(data_dir: str | Path) -> tuple:
    """
    Load SHARP parameters and flare event catalog.
    Falls back to synthetic data if files not found.

    Expected files:
      {data_dir}/sharp_cea_720s.parquet  — All active regions, 2010-2024
      {data_dir}/goes_event_catalog.csv  — GOES flare event list
    """
    data_dir = Path(data_dir)
    sharp_path = data_dir / "sharp_cea_720s.parquet"
    events_path = data_dir / "goes_event_catalog.csv"

    if sharp_path.exists():
        df = pd.read_parquet(sharp_path)
        logger.info("Loaded SHARP data: %d rows", len(df))
    else:
        logger.warning("SHARP parquet not found — generating synthetic data for testing")
        df = _generate_synthetic_sharp()

    if events_path.exists():
        event_df = pd.read_csv(events_path, parse_dates=["start_time", "peak_time"])
        logger.info("Loaded event catalog: %d flares", len(event_df))
    else:
        logger.warning("Event catalog not found — generating synthetic events")
        event_df = _generate_synthetic_events(len(df) // 100)

    return df, event_df


def _generate_synthetic_sharp(n_samples: int = 50000) -> pd.DataFrame:
    """Synthetic SHARP data for development testing."""
    np.random.seed(42)
    times = pd.date_range("2010-01-01", periods=n_samples, freq="12min")
    data = {col: np.random.exponential(1.0, n_samples) for col in SHARP_ML_FEATURES}
    data["unix_time"] = times.astype(np.int64) // 10**9
    for feat in EXTRA_FEATURES[:5]:
        data[feat] = np.random.normal(0, 1, n_samples)
    data["HARPNUM"] = np.random.randint(1000, 9000, n_samples)
    return pd.DataFrame(data)


def _generate_synthetic_events(n_events: int = 500) -> pd.DataFrame:
    """Synthetic flare event catalog for testing."""
    np.random.seed(42)
    start_times = pd.date_range("2010-01-01", periods=n_events, freq="5D")
    classes = np.random.choice(["M1.0", "M3.5", "X1.0", "X2.5"], size=n_events, p=[0.5, 0.3, 0.15, 0.05])
    return pd.DataFrame({"start_time": start_times, "goes_class": classes})


# ── Multi-task SHARP model ────────────────────────────────────────────────────

class SHARPMultiHorizonModel(nn.Module):
    """
    SHARP LSTM encoder + 4 forecast heads (one per horizon).
    """
    def __init__(self, n_horizons: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        self.lstm_branch = SHARPLSTMBranch(dropout=dropout)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(128, 32), nn.GELU(), nn.Linear(32, 1))
            for _ in range(n_horizons)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.lstm_branch(x)
        logits = torch.cat([head(embedding) for head in self.heads], dim=-1)  # (B, 4)
        return embedding, logits


# ── Training ──────────────────────────────────────────────────────────────────

def train_phase_b(
    sharp_data_dir: str = "./data/sharp",
    phase_a_checkpoint: str = None,
    output_dir: str = "./models",
    config: dict = SHARP_CONFIG,
) -> str:
    """
    Run Phase B: train SHARP LSTM + optional joint fine-tuning with TCN.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Phase B training on device: %s", device)
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    df, event_df = load_sharp_and_events(sharp_data_dir)

    # Chronological split (NO random split — avoid temporal leakage!)
    n = len(df)
    n_train = int(n * config["train_split"])
    n_val   = int(n * config["val_split"])

    train_df = df.iloc[:n_train]
    val_df   = df.iloc[n_train:n_train + n_val]

    # Split events by time too
    if "start_time" in event_df.columns:
        train_cutoff = pd.Timestamp(df["unix_time"].iloc[n_train], unit="s", tz="UTC")
        val_cutoff   = pd.Timestamp(df["unix_time"].iloc[n_train + n_val], unit="s", tz="UTC")
        train_events = event_df[event_df["start_time"] < train_cutoff]
        val_events   = event_df[(event_df["start_time"] >= train_cutoff) & (event_df["start_time"] < val_cutoff)]
    else:
        train_events = val_events = event_df

    train_ds = SHARPDataset(train_df, train_events, config["forecast_horizons_h"])
    val_ds   = SHARPDataset(val_df, val_events,     config["forecast_horizons_h"], scaler=train_ds.scaler)

    logger.info("Train: %d | Val: %d samples", len(train_ds), len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=config["batch_size"], shuffle=False, num_workers=2)

    # Model
    model = SHARPMultiHorizonModel(n_horizons=len(config["forecast_horizons_h"]), dropout=config["dropout"]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    criterion = FocalLoss(gamma=config["focal_gamma"])

    best_tss_24h = -float("inf")
    patience_counter = 0
    best_checkpoint = None

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        losses = []
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            _, logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        # Validate on 24h horizon (index 2)
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                _, logits = model(x_batch.to(device))
                probs = torch.sigmoid(logits)[:, 2].cpu().numpy()  # 24h horizon
                all_probs.extend(probs.tolist())
                all_labels.extend(y_batch[:, 2].numpy().tolist())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        thresh, tss = find_optimal_threshold(all_labels, all_probs, metric="TSS")
        auc = roc_auc(all_labels, all_probs)
        ct = build_contingency_table(all_labels, all_probs, threshold=thresh)

        logger.info(
            "Epoch %3d | Loss=%.4f | 24h TSS=%.3f HSS=%.3f AUC=%.3f | POD=%.2f FAR=%.2f",
            epoch, np.mean(losses), tss, ct.HSS, auc, ct.POD, ct.FAR
        )

        if tss > best_tss_24h:
            best_tss_24h = tss
            patience_counter = 0
            checkpoint_path = os.path.join(output_dir, "sharp_lstm_phase_b_best.pt")
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "val_tss_24h": tss, "val_auc_24h": auc,
                "config": config, "optimal_threshold": thresh,
                "scaler_state": train_ds.scaler.__dict__,
            }, checkpoint_path)
            best_checkpoint = checkpoint_path
            logger.info("✓ New best model (24h TSS=%.3f)", tss)
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                logger.info("Early stopping at epoch %d", epoch)
                break

    logger.info("Phase B complete. Best 24h TSS=%.3f", best_tss_24h)
    return best_checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sharp-data-dir", default="./data/sharp")
    parser.add_argument("--phase-a-checkpoint", default=None)
    parser.add_argument("--output-dir", default="./models")
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()
    SHARP_CONFIG["epochs"] = args.epochs
    checkpoint = train_phase_b(args.sharp_data_dir, args.phase_a_checkpoint, args.output_dir, SHARP_CONFIG)
    print(f"\nPhase B complete. Checkpoint: {checkpoint}")
    print("Next: Run 03_fusion_train.py for multi-modal fusion training")
