"""
AdityScan — Real Data Training Pipeline (SoLEXS + HEL1OS primary)
===================================================================
Uses actual Aditya-L1 PRADAN Level-1 data as the PRIMARY training source.

Data on disk:
  SoLEXS L1 (.lc.gz):  TIME(s) + COUNTS(cts/s) — 86400 rows/day, 1-s cadence
  HEL1OS L1 (.fits):   MJD + ISOT + CTR(cts/s) — multi-band, ~42000 rows/obs
                        Bands: CdTe 5-20, 20-30, 30-40, 40-60 keV
                               CZT  20-40, 40-60, 60-80, 80-150 keV

Supplementary: GOES XRS (downloaded via sunpy) for additional temporal context
               and extended flare label generation.

Training strategy (adapting to 4-day Aditya-L1 dataset):
  - We have ~4 days of simultaneous SoLEXS + HEL1OS data
  - GOES flare catalog cross-matched to identify any flare events in those windows
  - For the primary ML model: sliding window (30 s) → predict flare in next N minutes
  - Phase A: Train XRayTCN on SoLEXS counts + HEL1OS multi-band (real data ONLY)
  - Phase B: Train LSTM on combined SoLEXS+HEL1OS time-series
  - Phase C: Fusion + calibration
  - Outputs: trained .pt checkpoints + ONNX model + model_summary.json

Run:
  python notebooks/04_real_data_train.py
  (from adityscan/ directory, with adityscan_env activated)
"""

import os
import sys
import json
import gzip
import io
import logging
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

# ── Data Paths ───────────────────────────────────────────────────────────────
DATA_ROOT = Path(__file__).parent.parent / "data" / "pradan_cache"

SOLEXS_FILES = {
    "20260621": [
        DATA_ROOT / "slx_20260621/AL1_SLX_L1_20260621_v1.0/SDD2/AL1_SOLEXS_20260621_SDD2_L1.lc.gz",
    ],
    "20260622": [
        DATA_ROOT / "slx_20260622/AL1_SLX_L1_20260622_v1.0/SDD2/AL1_SOLEXS_20260622_SDD2_L1.lc.gz",
    ],
}

HEL1OS_CDTE_FILES = {
    "20260623": DATA_ROOT / "hel1os_20260623/2026/06/23/HLS_20260623_121027_42566sec_lev1_V111/cdte/lightcurve_cdte2.fits",
    "20240629": DATA_ROOT / "hel1os_20240629/2024/06/29/HLS_20240629_161229_28046sec_lev1_V111/cdte/lightcurve_cdte2.fits",
}

HEL1OS_CZT_FILES = {
    "20260623": DATA_ROOT / "hel1os_20260623/2026/06/23/HLS_20260623_121027_42566sec_lev1_V111/czt/lightcurve_czt2.fits",
    "20240629": DATA_ROOT / "hel1os_20240629/2024/06/29/HLS_20240629_161229_28046sec_lev1_V111/czt/lightcurve_czt2.fits",
}


# ── Configuration ─────────────────────────────────────────────────────────────
CONFIG = {
    # Windows
    "window_seconds": 60,         # 60-second input window (1-s cadence → 60 steps)
    "forecast_horizons_s": [60, 120, 300, 600],  # predict flare in next 1,2,5,10 min
    # Flare detection thresholds (in SoLEXS counts/s, calibrated against GOES)
    "flare_count_threshold": 50.0,  # counts/s above background = candidate flare
    "flare_sigma_threshold": 5.0,   # sigma above rolling median = flare
    # Training
    "batch_size": 64,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "epochs_tcn": 80,
    "epochs_lstm": 60,
    "epochs_fusion": 30,
    "patience": 10,
    "class_weight_pos": 15.0,
    "dropout": 0.15,
    "gradient_clip": 1.0,
    # Val split (chronological, no leakage)
    "val_fraction": 0.20,
    "output_dir": "./models",
}


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_solexs_lc(filepath: Path) -> pd.DataFrame:
    """
    Load SoLEXS L1 light curve from .lc.gz FITS file.
    Columns: TIME (UNIX), COUNTS (cts/s)
    """
    from astropy.io import fits

    logger.info("Loading SoLEXS: %s", filepath.name)
    with gzip.open(str(filepath), "rb") as f:
        with fits.open(io.BytesIO(f.read())) as hdul:
            data = hdul[1].data
            times = data["TIME"].astype(np.float64)
            counts = data["COUNTS"].astype(np.float64)

    df = pd.DataFrame({"time_unix": times, "solexs_counts": counts})
    # Clean: drop NaN, clip negatives
    df = df.dropna().copy()
    df["solexs_counts"] = df["solexs_counts"].clip(0)
    df = df.sort_values("time_unix").reset_index(drop=True)
    logger.info("  → %d valid SoLEXS rows (%.1f hours)", len(df), len(df) / 3600)
    return df


def load_helios_lc(cdte_path: Path, czt_path: Path) -> pd.DataFrame:
    """
    Load HEL1OS L1 light curve from CdTe + CZT FITS files.
    Merges all energy bands into a single DataFrame aligned on MJD.
    Columns: time_unix, hel_cdte_5_20, hel_cdte_20_30, hel_cdte_30_40,
             hel_cdte_40_60, hel_czte_20_40, hel_czt_40_60, hel_czt_60_80,
             hel_czt_80_150
    """
    from astropy.io import fits
    from astropy.time import Time

    def _load_file(fpath: Path, prefix: str) -> pd.DataFrame:
        dfs = []
        with fits.open(str(fpath)) as hdul:
            for i in range(1, len(hdul)):
                hdu = hdul[i]
                ext_name = hdu.name  # e.g. CDTE2_LC_BAND_5.00KEV_TO_20.00KEV
                # Parse band name from extension
                band_tag = ext_name.replace("CDTE2_LC_BAND_", "").replace("CZT2_LC_BAND_", "")
                band_tag = band_tag.replace(".00KEV_TO_", "_").replace(".00KEV", "").lower()
                col_name = f"{prefix}_{band_tag}"

                data = hdu.data
                mjd = data["MJD"].astype(np.float64)
                ctr = data["CTR"].astype(np.float64)

                df = pd.DataFrame({"mjd": mjd, col_name: ctr})
                df = df.dropna().copy()
                df[col_name] = df[col_name].clip(0)
                dfs.append(df)

        # Merge all bands on MJD
        merged = dfs[0]
        for d in dfs[1:]:
            merged = merged.merge(d, on="mjd", how="outer")
        merged = merged.sort_values("mjd").reset_index(drop=True)

        # Convert MJD → UNIX
        t = Time(merged["mjd"].values, format="mjd", scale="utc")
        merged["time_unix"] = t.unix
        merged = merged.drop(columns=["mjd"])
        return merged

    logger.info("Loading HEL1OS CdTe: %s", cdte_path.name)
    cdte_df = _load_file(cdte_path, "hel_cdte")
    logger.info("  → %d CdTe rows (%.1f hours)", len(cdte_df), len(cdte_df) / 3600)

    logger.info("Loading HEL1OS CZT:  %s", czt_path.name)
    czt_df = _load_file(czt_path, "hel_czt")
    logger.info("  → %d CZT rows (%.1f hours)", len(czt_df), len(czt_df) / 3600)

    # Merge CdTe + CZT on time (nearest second)
    merged = pd.merge_asof(
        cdte_df.sort_values("time_unix"),
        czt_df.sort_values("time_unix"),
        on="time_unix",
        tolerance=2.0,  # 2-second tolerance
        direction="nearest",
    )
    merged = merged.fillna(0.0).sort_values("time_unix").reset_index(drop=True)
    logger.info("Combined HEL1OS: %d rows", len(merged))
    return merged


def load_all_data() -> pd.DataFrame:
    """
    Load and combine all available SoLEXS + HEL1OS data into one DataFrame.
    """
    # --- SoLEXS ---
    slx_frames = []
    for date, paths in SOLEXS_FILES.items():
        for p in paths:
            if p.exists():
                slx_frames.append(load_solexs_lc(p))
            else:
                logger.warning("SoLEXS file not found: %s", p)
    
    if not slx_frames:
        raise FileNotFoundError("No SoLEXS data files found!")
    
    slx_df = pd.concat(slx_frames).sort_values("time_unix").reset_index(drop=True)
    logger.info("Total SoLEXS: %d rows", len(slx_df))

    # --- HEL1OS ---
    hel_frames = []
    for date in sorted(HEL1OS_CDTE_FILES.keys()):
        c = HEL1OS_CDTE_FILES[date]
        z = HEL1OS_CZT_FILES[date]
        if c.exists() and z.exists():
            hel_frames.append(load_helios_lc(c, z))
        else:
            logger.warning("HEL1OS files not found for date %s", date)
    
    if not hel_frames:
        logger.warning("No HEL1OS data found — using SoLEXS only")
        # Add zero HEL1OS columns so pipeline still runs
        slx_df["hel_cdte_5_20"] = 0.0
        slx_df["hel_cdte_20_30"] = 0.0
        slx_df["hel_cdte_30_40"] = 0.0
        slx_df["hel_cdte_40_60"] = 0.0
        slx_df["hel_czt_20_40"] = 0.0
        slx_df["hel_czt_40_60"] = 0.0
        slx_df["hel_czt_60_80"] = 0.0
        slx_df["hel_czt_80_150"] = 0.0
        return slx_df

    hel_df = pd.concat(hel_frames).sort_values("time_unix").reset_index(drop=True)
    logger.info("Total HEL1OS: %d rows", len(hel_df))

    # --- Merge SoLEXS + HEL1OS on time ---
    merged = pd.merge_asof(
        slx_df.sort_values("time_unix"),
        hel_df.sort_values("time_unix"),
        on="time_unix",
        tolerance=2.0,
        direction="nearest",
    )
    merged = merged.fillna(0.0).sort_values("time_unix").reset_index(drop=True)
    logger.info("Merged SoLEXS+HEL1OS: %d rows", len(merged))
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived features from raw SoLEXS + HEL1OS counts.
    
    Features:
      1.  solexs_counts              — raw SoLEXS SDD2 total count rate
      2.  solexs_log                 — log10(counts + 1)
      3.  solexs_derivative          — d(counts)/dt
      4.  solexs_zscore_60s          — z-score vs 60-s rolling median
      5.  solexs_zscore_300s         — z-score vs 5-min rolling median
      6.  hel_cdte_5_20              — CdTe 5-20 keV (soft X-ray)
      7.  hel_cdte_20_30             — CdTe 20-30 keV
      8.  hel_cdte_30_40             — CdTe 30-40 keV (HOPE trigger)
      9.  hel_cdte_40_60             — CdTe 40-60 keV
      10. hel_czt_40_60              — CZT 40-60 keV (HOPE trigger)
      11. hel_czt_80_150             — CZT 80-150 keV (high-energy)
      12. hxr_ratio                  — HXR/SXR ratio (non-thermal indicator)
      13. neupert_integral           — running integral of HXR (Neupert proxy)
    """
    df = df.copy()

    # SoLEXS features
    df["solexs_log"] = np.log10(df["solexs_counts"].clip(0.01))
    df["solexs_derivative"] = np.gradient(df["solexs_counts"].values)

    # Rolling statistics (z-score)
    counts = df["solexs_counts"].values
    for window in [60, 300]:
        roll_med = pd.Series(counts).rolling(window, min_periods=5, center=False).median().bfill().fillna(0)
        roll_std = pd.Series(counts).rolling(window, min_periods=5, center=False).std().bfill().fillna(1)
        roll_std = roll_std.clip(lower=0.5)
        df[f"solexs_zscore_{window}s"] = ((counts - roll_med.values) / roll_std.values).clip(-10, 10)

    # HEL1OS band normalization (log scale)
    hel_cols = [c for c in df.columns if c.startswith("hel_")]
    for col in hel_cols:
        df[f"{col}_log"] = np.log10(df[col].clip(0.01))

    # HXR/SXR ratio
    hxr = df.get("hel_cdte_30_40", pd.Series(np.zeros(len(df))))
    sxr = df["solexs_counts"].clip(0.01)
    df["hxr_sxr_ratio"] = (hxr / sxr).clip(0, 100)

    # Neupert proxy (running integral of CdTe 30-40 keV)
    hxr_arr = df.get("hel_cdte_30_40", pd.Series(np.zeros(len(df)))).values
    running_integral = np.cumsum(hxr_arr)
    df["neupert_integral"] = (running_integral - running_integral.mean()) / (running_integral.std() + 1e-6)

    df = df.fillna(0.0)
    return df


def compute_flare_labels(df: pd.DataFrame, horizons_s: list) -> np.ndarray:
    """
    Compute binary flare labels for each timestep using threshold detection.
    
    Flare = SoLEXS counts exceed background by >= flare_sigma_threshold sigma
            in the next N seconds.
    
    Returns labels array: (N, len(horizons_s)) — binary 0/1
    """
    counts = df["solexs_counts"].values
    N = len(counts)

    # Compute local background using rolling median (5-min window)
    roll_med = pd.Series(counts).rolling(300, min_periods=30, center=False).median().bfill().values
    roll_std = pd.Series(counts).rolling(300, min_periods=30, center=False).std().bfill().clip(lower=0.5).values

    # Flare candidate = sigma > threshold or absolute count threshold
    sigma = CONFIG["flare_sigma_threshold"]
    abs_thresh = CONFIG["flare_count_threshold"]
    is_flare_second = ((counts - roll_med) > sigma * roll_std) | (counts > abs_thresh)

    labels = np.zeros((N, len(horizons_s)), dtype=np.float32)
    for j, h in enumerate(horizons_s):
        for i in range(N - h):
            labels[i, j] = float(np.any(is_flare_second[i + 1 : i + h + 1]))

    n_pos = int(labels[:, -1].sum())
    logger.info("Flare labels: %d positives / %d total (horizon=%ds) = %.2f%%",
                n_pos, N, horizons_s[-1], 100 * n_pos / max(N, 1))
    return labels


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_COLS_TCN = [
    "solexs_counts", "solexs_log", "solexs_derivative",
    "solexs_zscore_60s", "solexs_zscore_300s",
    "hxr_sxr_ratio", "neupert_integral",
]
# Dynamic: also include all hel_* columns found after feature engineering
FEATURE_COLS_HEL = []  # filled at runtime


class SoLEXSHEL1OSDataset(Dataset):
    """
    Sliding-window dataset over SoLEXS + HEL1OS real data.
    
    x:     (window_seconds, n_features)  — multi-channel time-series window
    y:     (n_horizons,)                 — binary labels for each forecast horizon
    """

    def __init__(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        feature_cols: list,
        window_s: int = 60,
        horizons_s: list = None,
        scaler=None,
    ):
        from sklearn.preprocessing import RobustScaler
        horizons_s = horizons_s or [60, 120, 300, 600]

        self.window = window_s
        self.n_horizons = len(horizons_s)
        self.feature_cols = feature_cols

        # Apply scaler
        X = df[feature_cols].values.astype(np.float32)
        if scaler is None:
            self.scaler = RobustScaler()
            X = self.scaler.fit_transform(X)
        else:
            self.scaler = scaler
            X = scaler.transform(X)

        self.X = X
        self.Y = labels  # (N, n_horizons)

        # Valid starts: must have a full window before AND at least max horizon after
        max_h = max(horizons_s)
        self.valid_starts = np.arange(window_s, len(df) - max_h)
        logger.info("Dataset: %d samples | %d features | window=%ds",
                    len(self.valid_starts), len(feature_cols), window_s)

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        i = self.valid_starts[idx]
        x = self.X[i - self.window : i]       # (window, n_features)
        y = self.Y[i]                           # (n_horizons,)
        return torch.from_numpy(x), torch.from_numpy(y)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURES (lightweight, CPU-friendly for hackathon)
# ══════════════════════════════════════════════════════════════════════════════

class SoLEXSTCN(nn.Module):
    """
    Lightweight TCN for SoLEXS + HEL1OS 1-second light curves.
    n_features: number of input channels (from FEATURE_COLS)
    """
    def __init__(self, n_features: int, n_horizons: int = 4, dropout: float = 0.15):
        super().__init__()
        # Input projection
        self.input_proj = nn.Linear(n_features, 32)
        # TCN blocks (causal convolutions)
        channels = [32, 64, 64, 128]
        dilations = [1,  2,  4,  8 ]
        blocks = []
        in_ch = 32
        for ch, d in zip(channels, dilations):
            blocks.append(nn.Sequential(
                nn.Conv1d(in_ch, ch, kernel_size=4, dilation=d, padding=3 * d),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(ch, ch, kernel_size=1),
                nn.GELU(),
            ))
            in_ch = ch
        self.tcn = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.embedding_size = 128

        # Multi-horizon heads
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(128, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1))
            for _ in range(n_horizons)
        ])

    def forward(self, x: torch.Tensor):
        # x: (B, T, F)
        h = self.input_proj(x).transpose(1, 2)  # (B, 32, T)
        for block in self.tcn:
            h_new = block(h)
            h_new = h_new[:, :, :h.shape[2]]  # causal crop
            if h_new.shape[1] == h.shape[1]:
                h = h + h_new
            else:
                h = h_new
        emb = self.pool(h).squeeze(-1)  # (B, 128)
        logits = torch.cat([head(emb) for head in self.heads], dim=-1)  # (B, n_horizons)
        return emb, logits


class SoLEXSLSTM(nn.Module):
    """
    Bidirectional LSTM over SoLEXS+HEL1OS time-series.
    """
    def __init__(self, n_features: int, n_horizons: int = 4, dropout: float = 0.15):
        super().__init__()
        self.input_proj = nn.Linear(n_features, 32)
        self.lstm = nn.LSTM(32, 64, num_layers=2, batch_first=True,
                             dropout=dropout, bidirectional=True)
        self.attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, dropout=dropout, batch_first=True)
        self.embedding_size = 128
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(128, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1))
            for _ in range(n_horizons)
        ])

    def forward(self, x: torch.Tensor):
        h = self.input_proj(x)               # (B, T, 32)
        out, _ = self.lstm(h)                # (B, T, 128)
        context, _ = self.attn(out, out, out)
        emb = context.mean(dim=1)            # (B, 128)
        logits = torch.cat([head(emb) for head in self.heads], dim=-1)
        return emb, logits


class FusionModel(nn.Module):
    """
    Late fusion: TCN embedding + LSTM embedding → shared MLP → heads.
    """
    def __init__(self, n_features: int, n_horizons: int = 4, dropout: float = 0.15):
        super().__init__()
        self.tcn = SoLEXSTCN(n_features, n_horizons, dropout)
        self.lstm = SoLEXSLSTM(n_features, n_horizons, dropout)
        fused_dim = 128 + 128  # TCN emb + LSTM emb
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
        )
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
            for _ in range(n_horizons)
        ])

    def forward(self, x: torch.Tensor):
        emb_tcn, _ = self.tcn(x)
        emb_lstm, _ = self.lstm(x)
        fused = torch.cat([emb_tcn, emb_lstm], dim=-1)
        shared = self.fusion(fused)
        probs = torch.cat([head(shared) for head in self.heads], dim=-1)
        return shared, probs


# ══════════════════════════════════════════════════════════════════════════════
# FOCAL LOSS (for imbalanced flare classes)
# ══════════════════════════════════════════════════════════════════════════════

class FocalBCELoss(nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight: float = 10.0):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, n_horizons), targets: (B, n_horizons)
        probs = torch.sigmoid(logits)
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=torch.tensor(self.pos_weight, device=logits.device),
            reduction="none",
        )
        pt = torch.where(targets == 1, probs, 1 - probs)
        weight = (1 - pt) ** self.gamma
        return (weight * bce).mean()


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING UTILS
# ══════════════════════════════════════════════════════════════════════════════

from utils.metrics import build_contingency_table, find_optimal_threshold, roc_auc


def evaluate(model, loader, device, primary_horizon_idx=2):
    """Evaluate model and return TSS, HSS, AUC, POD, FAR for primary horizon."""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            _, logits = model(x)
            probs = torch.sigmoid(logits[:, primary_horizon_idx]).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(y[:, primary_horizon_idx].numpy().tolist())

    probs_arr = np.array(all_probs)
    labels_arr = np.array(all_labels)

    if labels_arr.sum() < 2:
        return 0.0, 0.0, 0.5, 0.0, 1.0, 0.5

    thresh, tss = find_optimal_threshold(labels_arr, probs_arr, metric="TSS")
    ct = build_contingency_table(labels_arr, probs_arr, threshold=thresh)
    auc = roc_auc(labels_arr, probs_arr)
    return tss, ct.HSS, auc, ct.POD, ct.FAR, thresh


def train_model(
    model, train_loader, val_loader, optimizer, criterion, scheduler,
    epochs, patience, device, phase_name, output_dir, primary_horizon_idx=2
):
    """Generic training loop with early stopping."""
    os.makedirs(output_dir, exist_ok=True)
    best_tss = -float("inf")
    patience_counter = 0
    best_ckpt = None
    best_metrics = {}

    logger.info("=" * 60)
    logger.info("Starting %s training: %d epochs, patience=%d", phase_name, epochs, patience)
    logger.info("Device: %s | Train batches: %d | Val batches: %d",
                device, len(train_loader), len(val_loader))
    logger.info("=" * 60)

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            _, logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CONFIG["gradient_clip"])
            optimizer.step()
            losses.append(loss.item())
        
        if scheduler:
            scheduler.step()

        tss, hss, auc, pod, far, thresh = evaluate(model, val_loader, device, primary_horizon_idx)
        train_loss = np.mean(losses)

        logger.info(
            "[%s] Epoch %3d/%d | Loss=%.4f | TSS=%.3f HSS=%.3f AUC=%.3f | POD=%.2f FAR=%.2f | Thresh=%.2f",
            phase_name, epoch, epochs, train_loss, tss, hss, auc, pod, far, thresh
        )

        if tss > best_tss:
            best_tss = tss
            patience_counter = 0
            best_metrics = {
                "epoch": epoch, "train_loss": round(float(train_loss), 4),
                "tss": round(float(tss), 4), "hss": round(float(hss), 4),
                "auc": round(float(auc), 4), "pod": round(float(pod), 4),
                "far": round(float(far), 4), "threshold": round(float(thresh), 4),
            }
            ckpt_path = os.path.join(output_dir, f"{phase_name}_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": best_metrics,
                "config": CONFIG,
            }, ckpt_path)
            best_ckpt = ckpt_path
            logger.info("  ✓ NEW BEST — saved to %s", ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("  Early stopping at epoch %d", epoch)
                break

    logger.info("[%s] Training complete. Best TSS=%.3f (epoch %d)",
                phase_name, best_tss, best_metrics.get("epoch", -1))
    return best_ckpt, best_metrics


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline():
    device = torch.device("cpu")  # CPU training (no GPU on Mac)
    logger.info("AdityScan Real-Data Training Pipeline")
    logger.info("Device: %s", device)

    # ── Step 1: Load & engineer features ─────────────────────────────────────
    logger.info("\n─── STEP 1: Loading Real Data ───────────────────────────────")
    df = load_all_data()
    df = engineer_features(df)

    # Determine feature columns (all engineered)
    base_cols = FEATURE_COLS_TCN.copy()
    hel_log_cols = [c for c in df.columns if c.startswith("hel_") and c.endswith("_log")]
    feat_cols = [c for c in base_cols + hel_log_cols if c in df.columns]
    logger.info("Feature columns (%d): %s", len(feat_cols), feat_cols)

    # ── Step 2: Flare labels ──────────────────────────────────────────────────
    logger.info("\n─── STEP 2: Computing Flare Labels ──────────────────────────")
    horizons_s = CONFIG["forecast_horizons_s"]
    labels = compute_flare_labels(df, horizons_s)

    # Print label statistics per horizon
    for j, h in enumerate(horizons_s):
        pos = labels[:, j].sum()
        logger.info("  Horizon %ds: %d positives / %d total (%.2f%%)",
                    h, int(pos), len(labels), 100 * pos / len(labels))

    # ── Step 3: Train/Val split (chronological) ───────────────────────────────
    logger.info("\n─── STEP 3: Chronological Train/Val Split ───────────────────")
    n = len(df)
    n_train = int(n * (1 - CONFIG["val_fraction"]))
    train_df = df.iloc[:n_train].copy()
    val_df   = df.iloc[n_train:].copy()
    train_labels = labels[:n_train]
    val_labels   = labels[n_train:]
    logger.info("Train: %d rows | Val: %d rows", n_train, n - n_train)

    # ── Step 4: Build datasets ────────────────────────────────────────────────
    logger.info("\n─── STEP 4: Building Datasets ───────────────────────────────")
    window = CONFIG["window_seconds"]
    train_ds = SoLEXSHEL1OSDataset(train_df, train_labels, feat_cols, window, horizons_s)
    val_ds   = SoLEXSHEL1OSDataset(val_df,   val_labels,   feat_cols, window, horizons_s,
                                    scaler=train_ds.scaler)

    n_features = len(feat_cols)
    n_horizons = len(horizons_s)
    primary_horizon_idx = 2  # 300 seconds (5 min)

    # Weighted sampler (for class balance)
    train_y_primary = train_labels[window:, primary_horizon_idx]
    pos_frac = train_y_primary.mean()
    logger.info("Positive class fraction (primary horizon): %.3f%%", 100 * pos_frac)
    sample_weights = np.where(train_y_primary > 0.5, CONFIG["class_weight_pos"], 1.0)
    # Trim/pad to match dataset length
    ds_len = len(train_ds)
    if len(sample_weights) > ds_len:
        sample_weights = sample_weights[:ds_len]
    elif len(sample_weights) < ds_len:
        sample_weights = np.concatenate([sample_weights, np.ones(ds_len - len(sample_weights))])
    
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights.astype(np.float32)),
        num_samples=ds_len,
        replacement=True,
    )

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"], shuffle=False,   num_workers=0)

    criterion = FocalBCELoss(gamma=2.0, pos_weight=CONFIG["class_weight_pos"])
    output_dir = CONFIG["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    all_metrics = {}

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE A: TCN on real SoLEXS + HEL1OS
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("\n══════════════════════════════════════════════════")
    logger.info("PHASE A: TCN on Real SoLEXS + HEL1OS Data")
    logger.info("══════════════════════════════════════════════════")

    tcn_model = SoLEXSTCN(n_features, n_horizons, CONFIG["dropout"]).to(device)
    total_params = sum(p.numel() for p in tcn_model.parameters())
    logger.info("TCN parameters: %d", total_params)

    opt_a = optim.AdamW(tcn_model.parameters(), lr=CONFIG["learning_rate"],
                        weight_decay=CONFIG["weight_decay"])
    sched_a = optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=CONFIG["epochs_tcn"])

    ckpt_a, metrics_a = train_model(
        tcn_model, train_loader, val_loader, opt_a, criterion, sched_a,
        CONFIG["epochs_tcn"], CONFIG["patience"], device, "phase_a_tcn",
        output_dir, primary_horizon_idx,
    )
    all_metrics["phase_a_tcn"] = metrics_a
    logger.info("Phase A complete: %s", json.dumps(metrics_a, indent=2))

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE B: LSTM on real SoLEXS + HEL1OS
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("\n══════════════════════════════════════════════════")
    logger.info("PHASE B: LSTM on Real SoLEXS + HEL1OS Data")
    logger.info("══════════════════════════════════════════════════")

    lstm_model = SoLEXSLSTM(n_features, n_horizons, CONFIG["dropout"]).to(device)
    total_params = sum(p.numel() for p in lstm_model.parameters())
    logger.info("LSTM parameters: %d", total_params)

    opt_b = optim.AdamW(lstm_model.parameters(), lr=CONFIG["learning_rate"],
                        weight_decay=CONFIG["weight_decay"])
    sched_b = optim.lr_scheduler.CosineAnnealingLR(opt_b, T_max=CONFIG["epochs_lstm"])

    ckpt_b, metrics_b = train_model(
        lstm_model, train_loader, val_loader, opt_b, criterion, sched_b,
        CONFIG["epochs_lstm"], CONFIG["patience"], device, "phase_b_lstm",
        output_dir, primary_horizon_idx,
    )
    all_metrics["phase_b_lstm"] = metrics_b
    logger.info("Phase B complete: %s", json.dumps(metrics_b, indent=2))

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE C: Fusion (TCN + LSTM ensemble)
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("\n══════════════════════════════════════════════════")
    logger.info("PHASE C: Fusion (TCN + LSTM Late Fusion)")
    logger.info("══════════════════════════════════════════════════")

    fusion_model = FusionModel(n_features, n_horizons, CONFIG["dropout"]).to(device)

    # Load pretrained TCN backbone
    if ckpt_a and os.path.exists(ckpt_a):
        state_a = torch.load(ckpt_a, map_location="cpu")["model_state_dict"]
        fusion_model.tcn.load_state_dict(state_a, strict=False)
        logger.info("Loaded Phase A weights into fusion TCN branch")
    
    if ckpt_b and os.path.exists(ckpt_b):
        state_b = torch.load(ckpt_b, map_location="cpu")["model_state_dict"]
        fusion_model.lstm.load_state_dict(state_b, strict=False)
        logger.info("Loaded Phase B weights into fusion LSTM branch")

    total_params = sum(p.numel() for p in fusion_model.parameters())
    logger.info("Fusion model parameters: %d", total_params)

    # Freeze backbone initially
    for p in list(fusion_model.tcn.parameters()) + list(fusion_model.lstm.parameters()):
        p.requires_grad = False

    opt_c = optim.AdamW(filter(lambda p: p.requires_grad, fusion_model.parameters()),
                        lr=CONFIG["learning_rate"] * 0.5, weight_decay=CONFIG["weight_decay"])
    sched_c = optim.lr_scheduler.CosineAnnealingLR(opt_c, T_max=CONFIG["epochs_fusion"])

    # For fusion, we need a slight adaptation — probs instead of logits
    class FusionCriterion(nn.Module):
        def forward(self, logits, targets):
            return nn.functional.binary_cross_entropy(logits.clamp(1e-6, 1 - 1e-6), targets)

    fusion_criterion = FusionCriterion()

    # Custom train loop for fusion (uses sigmoid outputs not logits)
    def fusion_train_model(model, train_loader, val_loader, optimizer, scheduler,
                            epochs, patience, output_dir):
        best_tss = -float("inf")
        patience_counter = 0
        best_ckpt = None
        best_metrics = {}
        unfreeze_done = False

        for epoch in range(1, epochs + 1):
            # Unfreeze backbone after halfway
            if epoch > epochs // 2 and not unfreeze_done:
                for p in model.parameters():
                    p.requires_grad = True
                optimizer = optim.AdamW(model.parameters(),
                                        lr=CONFIG["learning_rate"] * 0.1,
                                        weight_decay=CONFIG["weight_decay"])
                unfreeze_done = True
                logger.info("  Backbone unfrozen at epoch %d", epoch)

            model.train()
            losses = []
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                _, probs = model(x)
                loss = fusion_criterion(probs, y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), CONFIG["gradient_clip"])
                optimizer.step()
                losses.append(loss.item())

            # Evaluate
            model.eval()
            all_p, all_l = [], []
            with torch.no_grad():
                for x, y in val_loader:
                    _, probs = model(x.to(device))
                    all_p.extend(probs[:, primary_horizon_idx].cpu().numpy().tolist())
                    all_l.extend(y[:, primary_horizon_idx].numpy().tolist())

            p_arr, l_arr = np.array(all_p), np.array(all_l)
            if l_arr.sum() < 2:
                tss, hss, auc_, pod, far_, thresh_ = 0, 0, 0.5, 0, 1, 0.5
            else:
                thresh_, tss = find_optimal_threshold(l_arr, p_arr, "TSS")
                ct = build_contingency_table(l_arr, p_arr, threshold=thresh_)
                auc_ = roc_auc(l_arr, p_arr)
                hss, pod, far_ = ct.HSS, ct.POD, ct.FAR

            logger.info(
                "[phase_c_fusion] Epoch %3d/%d | Loss=%.4f | TSS=%.3f HSS=%.3f AUC=%.3f | "
                "POD=%.2f FAR=%.2f | Thresh=%.2f",
                epoch, epochs, np.mean(losses), tss, hss, auc_, pod, far_, thresh_
            )

            if tss > best_tss:
                best_tss = tss
                patience_counter = 0
                best_metrics = {
                    "epoch": epoch,
                    "train_loss": round(float(np.mean(losses)), 4),
                    "tss": round(float(tss), 4),
                    "hss": round(float(hss), 4),
                    "auc": round(float(auc_), 4),
                    "pod": round(float(pod), 4),
                    "far": round(float(far_), 4),
                    "threshold": round(float(thresh_), 4),
                }
                ckpt_path = os.path.join(output_dir, "phase_c_fusion_best.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "metrics": best_metrics,
                    "feature_cols": feat_cols,
                    "scaler": train_ds.scaler,
                    "config": CONFIG,
                }, ckpt_path)
                best_ckpt = ckpt_path
                logger.info("  ✓ NEW BEST FUSION — saved")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("  Early stopping at epoch %d", epoch)
                    break

        return best_ckpt, best_metrics

    ckpt_c, metrics_c = fusion_train_model(
        fusion_model, train_loader, val_loader, opt_c, sched_c,
        CONFIG["epochs_fusion"], CONFIG["patience"], output_dir
    )
    all_metrics["phase_c_fusion"] = metrics_c
    logger.info("Phase C complete: %s", json.dumps(metrics_c, indent=2))

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE D: Calibration + Export
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("\n══════════════════════════════════════════════════")
    logger.info("PHASE D: Calibration + ONNX Export")
    logger.info("══════════════════════════════════════════════════")

    # Load best fusion model
    if ckpt_c and os.path.exists(ckpt_c):
        state = torch.load(ckpt_c, map_location="cpu")
        fusion_model.load_state_dict(state["model_state_dict"])
    fusion_model.eval()

    # Compute calibration on validation set
    cal_probs, cal_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            _, probs = fusion_model(x)
            cal_probs.extend(probs[:, primary_horizon_idx].numpy().tolist())
            cal_labels.extend(y[:, primary_horizon_idx].numpy().tolist())

    cal_probs_arr = np.array(cal_probs)
    cal_labels_arr = np.array(cal_labels)

    # Temperature scaling
    from pipeline.ml.uncertainty import TemperatureScaler, ConformalPredictor
    ts = TemperatureScaler()
    logits_cal = np.log(cal_probs_arr.clip(1e-6) / (1 - cal_probs_arr.clip(max=1 - 1e-6) + 1e-10))
    T = ts.fit(logits_cal, cal_labels_arr.astype(int))
    cal_probs_calib = ts.calibrate(cal_probs_arr)
    logger.info("Temperature scaling T=%.4f", T)
    ts.save(os.path.join(output_dir, "temperature_scaler.pkl"))

    # Conformal prediction
    conformal = ConformalPredictor(coverage=0.90)
    for i, h in enumerate(horizons_s):
        conformal.fit(h, cal_probs_calib[:len(cal_probs_calib)//2], cal_labels_arr[:len(cal_labels_arr)//2].astype(int))
    conformal.save(os.path.join(output_dir, "conformal_predictor.pkl"))
    logger.info("Conformal predictor saved (coverage=90%%)")

    # ONNX export
    dummy_input = torch.zeros(1, window, n_features)
    onnx_path = os.path.join(output_dir, "adityscan_solexs_helios.onnx")
    torch.onnx.export(
        fusion_model,
        dummy_input,
        onnx_path,
        input_names=["x"],
        output_names=["embedding", "flare_probs"],
        dynamic_axes={"x": {0: "batch_size"}, "flare_probs": {0: "batch_size"}},
        opset_version=17,
    )
    logger.info("ONNX model exported: %s", onnx_path)

    # Save scaler for inference
    import pickle
    with open(os.path.join(output_dir, "feature_scaler.pkl"), "wb") as f:
        pickle.dump(train_ds.scaler, f)

    # ─── FINAL SUMMARY ────────────────────────────────────────────────────────
    final_summary = {
        "model_name": "AdityScan SoLEXS+HEL1OS Flare Predictor",
        "model_version": "4.0.0",
        "data_primary": ["Aditya-L1 SoLEXS L1", "Aditya-L1 HEL1OS L1"],
        "data_dates": list(SOLEXS_FILES.keys()) + list(HEL1OS_CDTE_FILES.keys()),
        "features": feat_cols,
        "n_features": n_features,
        "forecast_horizons_s": horizons_s,
        "window_seconds": window,
        "temperature_T": round(float(T), 4),
        "phase_a_tcn": metrics_a,
        "phase_b_lstm": metrics_b,
        "phase_c_fusion": metrics_c,
        "artifacts": {
            "phase_a_checkpoint": "phase_a_tcn_best.pt",
            "phase_b_checkpoint": "phase_b_lstm_best.pt",
            "phase_c_checkpoint": "phase_c_fusion_best.pt",
            "onnx_model": "adityscan_solexs_helios.onnx",
            "scaler": "feature_scaler.pkl",
            "temperature_scaler": "temperature_scaler.pkl",
            "conformal_predictor": "conformal_predictor.pkl",
        },
    }

    summary_path = os.path.join(output_dir, "model_summary.json")
    with open(summary_path, "w") as f:
        json.dump(final_summary, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info("Phase A (TCN)    → TSS=%.3f  HSS=%.3f  AUC=%.3f  POD=%.2f  FAR=%.2f",
                metrics_a.get("tss", 0), metrics_a.get("hss", 0), metrics_a.get("auc", 0),
                metrics_a.get("pod", 0), metrics_a.get("far", 0))
    logger.info("Phase B (LSTM)   → TSS=%.3f  HSS=%.3f  AUC=%.3f  POD=%.2f  FAR=%.2f",
                metrics_b.get("tss", 0), metrics_b.get("hss", 0), metrics_b.get("auc", 0),
                metrics_b.get("pod", 0), metrics_b.get("far", 0))
    logger.info("Phase C (Fusion) → TSS=%.3f  HSS=%.3f  AUC=%.3f  POD=%.2f  FAR=%.2f",
                metrics_c.get("tss", 0), metrics_c.get("hss", 0), metrics_c.get("auc", 0),
                metrics_c.get("pod", 0), metrics_c.get("far", 0))
    logger.info("Temperature T=%.4f | Conformal coverage=90%%", T)
    logger.info("All artifacts saved to: %s/", output_dir)
    logger.info("Summary: %s", summary_path)
    logger.info("=" * 60)

    return final_summary


if __name__ == "__main__":
    summary = run_pipeline()
    print("\n✓ All phases complete. Results:")
    print(json.dumps({
        "phase_a_tss": summary["phase_a_tcn"].get("tss"),
        "phase_b_tss": summary["phase_b_lstm"].get("tss"),
        "phase_c_tss": summary["phase_c_fusion"].get("tss"),
    }, indent=2))
