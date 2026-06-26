#!/bin/bash
# =============================================================================
# AdityScan v4 — M4 Mac Setup & Training Runner
# Hand this to your friend. Tell them to run ONLY this file.
# =============================================================================
# Usage:
#   chmod +x run_training.sh
#   ./run_training.sh
#
# What this does:
#   1. Checks Python 3.11+ is installed
#   2. Creates a virtual environment
#   3. Installs ALL required packages (M4 Metal GPU versions)
#   4. Runs the complete training pipeline
#   5. Tells you where the model is saved
# =============================================================================

set -e  # Exit on any error

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║         AdityScan v4 — SOTA Training Pipeline Setup          ║${NC}"
echo -e "${CYAN}║         Optimized for Apple M4 MacBook Air                   ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ── Step 1: Check Python version ─────────────────────────────────────────────
echo -e "${YELLOW}[1/5] Checking Python version...${NC}"
PYTHON_BIN=""
for py in python3.12 python3.11 python3; do
    if command -v $py &>/dev/null; then
        version=$($py -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        major=$(echo $version | cut -d. -f1)
        minor=$(echo $version | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 11 ]; then
            PYTHON_BIN=$py
            echo -e "${GREEN}  ✓ Found Python $version at $(which $py)${NC}"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo -e "${RED}  ✗ Python 3.11+ not found!${NC}"
    echo "  Install it: brew install python@3.12"
    exit 1
fi

# ── Step 2: Create virtual environment ───────────────────────────────────────
echo ""
echo -e "${YELLOW}[2/5] Creating virtual environment...${NC}"
VENV_DIR="$(pwd)/.venv_training"

if [ -d "$VENV_DIR" ]; then
    echo -e "${GREEN}  ✓ Virtual environment already exists${NC}"
else
    $PYTHON_BIN -m venv "$VENV_DIR"
    echo -e "${GREEN}  ✓ Created: $VENV_DIR${NC}"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip --quiet

# ── Step 3: Install packages (M4 Metal GPU versions) ─────────────────────────
echo ""
echo -e "${YELLOW}[3/5] Installing packages (this may take 5-10 min first time)...${NC}"
echo "    PyTorch with Metal (MPS) support..."

# PyTorch for M4 Mac (Metal Performance Shaders)
pip install --quiet \
    torch>=2.2.0 \
    torchvision>=0.17.0

echo "    ML + Scientific packages..."
pip install --quiet \
    numpy>=1.26.0 \
    pandas>=2.0.0 \
    scikit-learn>=1.4.0 \
    scipy>=1.11.0

echo "    ONNX Runtime..."
pip install --quiet \
    onnx>=1.17.0 \
    onnxruntime>=1.17.0

echo "    Astropy + SunPy (for real PRADAN FITS files)..."
pip install --quiet \
    astropy>=6.0.0 \
    sunpy>=5.0.0 \
    cdflib>=1.2.0 \
    scikit-image>=0.22.0

echo -e "${GREEN}  ✓ All packages installed${NC}"

# ── Step 4: Verify Metal GPU is available ────────────────────────────────────
echo ""
echo -e "${YELLOW}[4/5] Verifying M4 Metal GPU...${NC}"
python -c "
import torch
if torch.backends.mps.is_available():
    print('  \033[0;32m✓ Apple Metal GPU (MPS) is available — training will be fast!\033[0m')
    # Quick benchmark
    x = torch.randn(1000, 1000, device='mps')
    y = x @ x.T
    print(f'  ✓ MPS matrix multiply test passed: shape={y.shape}')
else:
    print('  \033[1;33m⚠ Metal GPU not available — will use CPU (slower)\033[0m')
"

# ── Step 5: Run training ──────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}[5/5] Starting training pipeline...${NC}"
echo ""
echo -e "${CYAN}  Expected time: 30-60 minutes on M4 Air${NC}"
echo -e "${CYAN}  Output model: ./models/adityscan_v4.onnx${NC}"
echo ""
echo -e "${YELLOW}  TIP: Keep your MacBook plugged in during training.${NC}"
echo -e "${YELLOW}  You can see live logs in training.log${NC}"
echo ""

# Navigate to project root
cd "$(dirname "$0")"

# Run the training
python notebooks/05_multimodal_sota_train.py \
    --phase-a-samples 40000 \
    --phase-b-samples 25000 \
    --phase-c-samples 30000 \
    --phase-a-epochs 20 \
    --phase-b-epochs 15 \
    --phase-c-epochs 25 \
    --batch-size-a 64 \
    --batch-size-b 128 \
    --batch-size-c 32

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    TRAINING COMPLETE! 🎉                    ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}  Model saved: $(pwd)/models/adityscan_v4.onnx${NC}"
echo -e "${GREEN}  Summary:     $(pwd)/models/model_summary.json${NC}"
echo ""
echo -e "${CYAN}  Now send these 2 files back to Mahan:${NC}"
echo -e "${CYAN}    1. models/adityscan_v4.onnx${NC}"
echo -e "${CYAN}    2. models/model_summary.json${NC}"
echo ""
echo -e "${CYAN}  Or: git push (if you have access to the repo)${NC}"
echo ""
