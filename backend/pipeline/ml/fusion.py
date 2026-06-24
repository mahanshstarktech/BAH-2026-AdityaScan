"""
Multi-Modal Fusion Model — AdityScan v3 Core Neural Architecture.

Architecture:
  - X-ray TCN branch (SoLEXS + HEL1OS): 256-dim embedding
  - SHARP LSTM branch (SDO/HMI magnetic): 128-dim embedding
  - In-situ MLP branch (MAG + ASPEX-SWIS): 64-dim embedding
  - Image EfficientNet-B0 branch (SUIT/AIA): 256-dim embedding (optional)
  - Cross-modal attention fusion (4 heads, 256-dim)
  - Output: nowcast + forecast + CME risk + confidence

Key design decisions:
  1. X-ray embedding is the primary "query" in attention — it's always available
  2. Missing modalities → zero embedding + mask token (model trained with masking)
  3. MC Dropout (p=0.1) at inference for epistemic uncertainty
  4. Temperature scaling calibration layer (single learnable param)
  5. ONNX-exportable — no dynamic control flow in forward pass

Hardware target: Apple M4 Air (CPU inference via ONNX Runtime)
  - Forward pass: ~50-100ms per sample (benchmark before deployment)
  - MC Dropout T=50 → ~2.5-5s for full uncertainty estimate (run async)

Training phases (from architecture doc):
  Phase A: GOES pretraining (50 years, X-ray TCN only)
  Phase B: SHARP+GOES joint (SDO era 2010-2024)
  Phase C: Aditya-L1 fine-tuning (XGBoost head on frozen TCN embedding)
  Phase D: CME association model (LASCO catalog cross-match)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Temporal Convolutional Network (X-ray Branch) ────────────────────────────

class CausalConv1d(nn.Module):
    """Causal (non-leaky) 1D convolution with dilation support."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation  # causal padding
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=self.padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Remove future samples from output (causal masking)
        return self.conv(x)[:, :, :-self.padding] if self.padding > 0 else self.conv(x)


class TCNBlock(nn.Module):
    """One dilated causal TCN block with residual connection."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 8,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        residual = self.residual(x)
        out = self.conv1(x)
        out = self.norm1(out.transpose(1, 2)).transpose(1, 2)
        out = F.gelu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.norm2(out.transpose(1, 2)).transpose(1, 2)
        out = F.gelu(out)
        out = self.dropout(out)
        return out + residual


class XRayTCN(nn.Module):
    """
    Temporal Convolutional Network for X-ray time series.

    Input: (B, T, F) where
      B = batch size
      T = 1800 (30 min × 1 s/sample)
      F = 11 features (5 SoLEXS + 5 HEL1OS + 1 Neupert ratio)
          [T_MK, EM_norm, logT_err, norm_err, chi2_red,
           T_MK_hxr, gamma_lo, gamma_hi, break_E_keV, chi2_red_hxr,
           neupert_ratio]
    Output: (B, 256) embedding

    Architecture:
      Dilated TCN: channels [64, 128, 256], dilation [1, 2, 4, 8, 16, 32]
      Kernel size: 8
      Causal padding: no future leakage
    """
    N_FEATURES = 11
    OUTPUT_DIM = 256

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        # Input projection
        self.input_proj = nn.Linear(self.N_FEATURES, 64)

        # Dilated TCN layers
        channels = [64, 64, 128, 128, 256, 256]
        dilations = [1, 2, 4, 8, 16, 32]
        blocks = []
        in_ch = 64
        for ch, dil in zip(channels, dilations):
            blocks.append(TCNBlock(in_ch, ch, kernel_size=8, dilation=dil, dropout=dropout))
            in_ch = ch
        self.tcn_blocks = nn.ModuleList(blocks)

        # Global context pooling + projection
        self.output_proj = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, 11) float32 — time series features
        mask : (B, T) bool, optional — True where features are valid

        Returns
        -------
        embedding : (B, 256)
        """
        B, T, F = x.shape
        # Project to channel space: (B, T, 64) → (B, 64, T) for conv
        h = self.input_proj(x)  # (B, T, 64)
        h = h.transpose(1, 2)   # (B, 64, T)

        for block in self.tcn_blocks:
            h = block(h)

        # Mean pool over time (masked if provided)
        if mask is not None:
            m = mask.float().unsqueeze(1)  # (B, 1, T)
            h = (h * m).sum(-1) / (m.sum(-1) + 1e-8)
        else:
            h = h.mean(-1)  # (B, 256)

        return self.output_proj(h)


# ── SHARP LSTM Branch ────────────────────────────────────────────────────────

class SHARPLSTMBranch(nn.Module):
    """
    LSTM encoder for SDO/HMI SHARP magnetic parameters.

    Input: (B, T, 21) where T = 120 (24 hr × 12-min cadence)
      18 SHARP parameters:
        TOTUSJH, TOTUSJZ, MEANPOT, SAVNCPP, USFLUX, AREA_ACT,
        R_VALUE, SHRGT45, TOTBSQ, TOTPOT, TOTFZ, ABSNJZH,
        EPSZ, TOTBSQ (alias), MEANPOT (alias), SHRGT45 (alias),
        USFLUX (alias), TOTPOT (alias)  [using 18 unique names from SHARP CEA]
      + 3 MAG-derived: B_total_mean, clock_angle_mean, cone_angle_mean

    Output: (B, 128)
    """
    N_FEATURES = 21
    OUTPUT_DIM = 128

    def __init__(self, dropout: float = 0.2) -> None:
        super().__init__()
        # RobustScaler applied OFFLINE (not in model). Raw features enter here.
        self.input_proj = nn.Linear(self.N_FEATURES, 64)
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.output_proj = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, 21) — SHARP + MAG features (pre-scaled by RobustScaler)
        lengths : (B,) int — valid sequence length per batch item (for packing)

        Returns
        -------
        embedding : (B, 128)
        """
        h = self.input_proj(x)  # (B, T, 64)

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                h, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (h_n, _) = self.lstm(packed)
        else:
            _, (h_n, _) = self.lstm(h)

        # h_n: (2, B, 128) — take last layer
        last_hidden = h_n[-1]  # (B, 128)
        return self.output_proj(last_hidden)


# ── In-situ MLP Branch (MAG + ASPEX-SWIS) ───────────────────────────────────

class InSituMLPBranch(nn.Module):
    """
    MLP for in-situ solar wind / IMF features.

    Input: (B, 14) concatenated features:
      MAG (8):  B_total_mean, B_total_std, Bx_mean, By_mean, Bz_mean,
                clock_angle_mean, cone_angle_mean, B_variance
      SWIS (6): proton_density_mean, proton_temperature_mean, proton_speed_mean,
                density_std, speed_std, dynamic_pressure_mean

    Output: (B, 64)
    """
    N_FEATURES = 14
    OUTPUT_DIM = 64

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.N_FEATURES, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 14) → (B, 64)"""
        return self.net(x)


# ── Cross-Modal Attention Fusion ─────────────────────────────────────────────

class CrossModalAttentionFusion(nn.Module):
    """
    Multi-head attention fusion of branch embeddings.

    All embeddings projected to 256-dim common space.
    X-ray embedding is the primary query (always available).
    Other branches are key-value pairs — masked when not available.

    4 attention heads. Output: 256-dim fused representation.
    """
    COMMON_DIM = 256
    N_HEADS = 4

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        d = self.COMMON_DIM

        # Project each branch to common space
        self.proj_xray = nn.Linear(256, d)       # already 256
        self.proj_sharp = nn.Linear(128, d)       # 128 → 256
        self.proj_insitu = nn.Linear(64, d)       # 64 → 256
        self.proj_image = nn.Linear(256, d)       # 256 → 256 (EfficientNet output)

        # Cross-attention: query from X-ray, keys/values from all modalities
        self.attention = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=self.N_HEADS,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d * 2, d),
        )
        self.norm2 = nn.LayerNorm(d)

    def forward(
        self,
        xray_emb: torch.Tensor,                      # (B, 256) — REQUIRED
        sharp_emb: Optional[torch.Tensor] = None,    # (B, 128)
        insitu_emb: Optional[torch.Tensor] = None,   # (B, 64)
        image_emb: Optional[torch.Tensor] = None,    # (B, 256)
    ) -> torch.Tensor:
        """
        Fuse all available branch embeddings.
        Missing branches are replaced with learned zero embeddings.
        Returns (B, 256) fused representation.
        """
        B = xray_emb.shape[0]
        d = self.COMMON_DIM
        device = xray_emb.device

        # Project all to common space; use zeros for missing modalities
        q = self.proj_xray(xray_emb).unsqueeze(1)  # (B, 1, 256) — query

        kvs = [self.proj_xray(xray_emb)]  # X-ray is always in key-value set

        if sharp_emb is not None:
            kvs.append(self.proj_sharp(sharp_emb))
        else:
            kvs.append(torch.zeros(B, d, device=device))

        if insitu_emb is not None:
            kvs.append(self.proj_insitu(insitu_emb))
        else:
            kvs.append(torch.zeros(B, d, device=device))

        if image_emb is not None:
            kvs.append(self.proj_image(image_emb))
        else:
            kvs.append(torch.zeros(B, d, device=device))

        # Stack: (B, n_modalities, 256)
        kv = torch.stack(kvs, dim=1)

        # Multi-head cross-attention
        attn_out, _ = self.attention(q, kv, kv)  # (B, 1, 256)
        attn_out = attn_out.squeeze(1)           # (B, 256)

        # Residual + FFN
        attn_out = self.norm(attn_out + xray_emb)
        out = self.ffn(attn_out)
        return self.norm2(out + attn_out)


# ── Output Heads ─────────────────────────────────────────────────────────────

class NowcastHead(nn.Module):
    """
    Nowcast output: "Is there a flare happening RIGHT NOW?"
    Outputs:
      - binary: P(flare now)
      - class: P(B, C, M, X, X+) — 5-class GOES classification
      - cme_risk: P(CME-associated) conditioned on spectral hardness
    """
    def __init__(self, input_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.binary = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.multiclass = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 5),    # B, C, M, X, X+
        )
        self.cme_risk = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, fused: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "flare_prob": torch.sigmoid(self.binary(fused)),       # (B, 1)
            "class_logits": self.multiclass(fused),                # (B, 5) — use softmax at inference
            "cme_risk": torch.sigmoid(self.cme_risk(fused)),       # (B, 1)
        }


class ForecastHead(nn.Module):
    """
    Forecast output: P(M+ flare | next X minutes) for 5 horizons.
    Horizons: 5, 10, 15, 30, 60 minutes
    """
    HORIZONS = [5, 10, 15, 30, 60]  # minutes

    def __init__(self, input_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(input_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )
            for _ in self.HORIZONS
        ])

    def forward(self, fused: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            f"p_flare_{h}min": torch.sigmoid(head(fused))
            for h, head in zip(self.HORIZONS, self.heads)
        }


# ── Full AdityScan Multi-Modal Model ─────────────────────────────────────────

class AdityScanModel(nn.Module):
    """
    Complete AdityScan multi-modal flare forecasting model.

    All branches can be independently masked at inference time
    when their data source is unavailable.

    MC Dropout is achieved by keeping dropout active at inference via
    calling model.train() before inference loops — or use enable_mc_dropout().
    """

    def __init__(self, mc_dropout: float = 0.1) -> None:
        super().__init__()
        # Branch encoders
        self.xray_branch = XRayTCN(dropout=mc_dropout)
        self.sharp_branch = SHARPLSTMBranch(dropout=mc_dropout)
        self.insitu_branch = InSituMLPBranch(dropout=mc_dropout)

        # Image branch (EfficientNet-B0) — loaded separately if available
        self.image_branch: Optional[nn.Module] = None

        # Fusion
        self.fusion = CrossModalAttentionFusion(dropout=mc_dropout)

        # Output heads
        self.nowcast_head = NowcastHead(dropout=mc_dropout)
        self.forecast_head = ForecastHead(dropout=mc_dropout)

        # Temperature scaling (calibration, single scalar)
        self.temperature = nn.Parameter(torch.ones(1) * 1.0, requires_grad=True)

    def enable_image_branch(self) -> None:
        """Load EfficientNet-B0 image branch (requires torchvision)."""
        try:
            from torchvision.models import efficientnet_b0
            backbone = efficientnet_b0(pretrained=False)
            # Replace first conv to accept single-channel input
            backbone.features[0][0] = nn.Conv2d(
                1, 32, kernel_size=3, stride=2, padding=1, bias=False
            )
            # Replace classifier head with linear projection to 256
            backbone.classifier = nn.Linear(1280, 256)
            self.image_branch = backbone
            print("Image branch (EfficientNet-B0) loaded.")
        except ImportError:
            print("torchvision not available — image branch disabled.")

    def enable_mc_dropout(self) -> None:
        """Enable MC Dropout at inference (call model.eval() keeps BN in eval, but dropout active)."""
        # Set dropout layers to training mode explicitly
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def forward(
        self,
        xray: torch.Tensor,
        sharp: Optional[torch.Tensor] = None,
        insitu: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        sharp_lengths: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through the complete model.

        Parameters
        ----------
        xray : (B, 1800, 11) — X-ray time series (REQUIRED)
        sharp : (B, 120, 21) — SHARP+MAG features (optional)
        insitu : (B, 14) — in-situ MAG+SWIS features (optional)
        image : (B, 1, 512, 512) — SUIT/AIA image (optional)
        sharp_lengths : (B,) — valid LSTM sequence lengths (optional)

        Returns
        -------
        dict with keys from NowcastHead + ForecastHead outputs.
        """
        # X-ray branch (always runs)
        xray_emb = self.xray_branch(xray)  # (B, 256)

        # Optional branches
        sharp_emb = self.sharp_branch(sharp, sharp_lengths) if sharp is not None else None
        insitu_emb = self.insitu_branch(insitu) if insitu is not None else None
        image_emb = None
        if image is not None and self.image_branch is not None:
            image_emb = self.image_branch(image)

        # Fusion
        fused = self.fusion(xray_emb, sharp_emb, insitu_emb, image_emb)

        # Temperature scaling of fused logits (calibration)
        fused_calibrated = fused / self.temperature.clamp(min=0.1)

        # Output heads
        nowcast = self.nowcast_head(fused_calibrated)
        forecast = self.forecast_head(fused_calibrated)

        return {**nowcast, **forecast}

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        xray: torch.Tensor,
        sharp: Optional[torch.Tensor] = None,
        insitu: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        n_samples: int = 50,
    ) -> dict[str, dict[str, float]]:
        """
        MC Dropout inference: run T=50 stochastic forward passes.
        Returns mean and std of each output probability.

        Returns dict like:
        {
          "p_flare_15min": {"mean": 0.73, "std": 0.06},
          "flare_prob": {"mean": 0.85, "std": 0.04},
          ...
        }
        """
        self.enable_mc_dropout()
        all_outputs: list[dict] = []

        for _ in range(n_samples):
            out = self(xray, sharp, insitu, image)
            all_outputs.append({k: v.cpu().numpy() for k, v in out.items()})

        # Aggregate
        result = {}
        for key in all_outputs[0]:
            vals = [o[key][0][0] for o in all_outputs]  # (50,)
            import numpy as np
            result[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "p05": float(np.percentile(vals, 5)),
                "p95": float(np.percentile(vals, 95)),
            }

        return result


# ── ONNX Export utility ───────────────────────────────────────────────────────

def export_to_onnx(
    model: AdityScanModel,
    output_path: str,
    opset: int = 17,
) -> None:
    """
    Export the AdityScan model to ONNX for production CPU inference.

    Does NOT export MC Dropout (single deterministic pass for ONNX).
    Run MC Dropout ensemble externally if uncertainty is needed at inference.
    """
    import torch

    model.eval()
    dummy_xray = torch.zeros(1, 1800, 11)
    dummy_sharp = torch.zeros(1, 120, 21)
    dummy_insitu = torch.zeros(1, 14)

    with torch.no_grad():
        dummy_out = model(dummy_xray, dummy_sharp, dummy_insitu)

    # Build output names from model output keys
    output_names = list(dummy_out.keys())

    torch.onnx.export(
        model,
        (dummy_xray, dummy_sharp, dummy_insitu),
        output_path,
        opset_version=opset,
        input_names=["xray", "sharp", "insitu"],
        output_names=output_names,
        dynamic_axes={
            "xray": {0: "batch"},
            "sharp": {0: "batch"},
            "insitu": {0: "batch"},
        },
    )
    print(f"Model exported to ONNX: {output_path}")
