#!/bin/bash
# =============================================================================
# AdityScan v4 — M4 Mac Setup & REAL Incremental Training Runner
# =============================================================================
# Hand this to your friend. They run ONLY this file.
#
# BEFORE RUNNING: Set your PRADAN login credentials:
#   export PRADAN_USER=your_username
#   export PRADAN_PASS=your_password
#
# Then run:
#   chmod +x run_training.sh
#   ./run_training.sh
#
# What this does:
#   1. Checks Python 3.11+ is installed
#   2. Creates a virtual environment, installs all packages
#   3. Downloads REAL Aditya-L1 data from PRADAN month by month
#   4. Trains on each month → saves checkpoint → deletes month data
#   5. All 15 months trained → exports adityscan_v4.onnx
#
# DATA SOURCES (ALL REAL):
#   SoLEXS L1    → PRADAN (requires login)
#   HEL1OS L1    → PRADAN (requires login)
#   MAG L2       → PRADAN (requires login)
#   ASPEX-SWIS   → PRADAN (requires login)
#   SHARP        → NASA JSOC (FREE, no login)
#   GOES XRS     → NOAA NCEI (FREE, no login)
#
# MEMORY: Never exceeds 5 GB RAM. Safe for 16 GB M4 Air.
# DISK:   Never exceeds 2 GB at once (deletes each month after training).
# TIME:   ~45-90 minutes total for all 15 months.
# =============================================================================

set -e

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

echo ""
echo -e "${CYAN}${BOLD}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}${BOLD}║      AdityScan v4 — REAL Data Incremental Training           ║${NC}"
echo -e "${CYAN}${BOLD}║      Apple M4 MacBook Air Optimized                          ║${NC}"
echo -e "${CYAN}${BOLD}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ── Step 1: Check Python ──────────────────────────────────────────────────────
echo -e "${YELLOW}[1/6] Checking Python version...${NC}"
PYTHON_BIN=""
for py in python3.12 python3.11 python3; do
    if command -v $py &>/dev/null; then
        version=$($py -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        major=$(echo $version | cut -d. -f1)
        minor=$(echo $version | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 11 ]; then
            PYTHON_BIN=$py
            echo -e "${GREEN}  ✓ Python $version${NC}"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo -e "${RED}  ✗ Python 3.11+ not found. Install: brew install python@3.12${NC}"
    exit 1
fi

# ── Step 2: Virtual environment ───────────────────────────────────────────────
echo ""
echo -e "${YELLOW}[2/6] Setting up virtual environment...${NC}"
VENV_DIR="$(pwd)/.venv_training"

if [ ! -d "$VENV_DIR" ]; then
    $PYTHON_BIN -m venv "$VENV_DIR"
    echo -e "${GREEN}  ✓ Created virtual environment${NC}"
else
    echo -e "${GREEN}  ✓ Virtual environment already exists${NC}"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip --quiet

# ── Step 3: Install ALL required packages ────────────────────────────────────
echo ""
echo -e "${YELLOW}[3/6] Installing packages (first time: ~10 min)...${NC}"

echo "    Core ML (PyTorch with M4 Metal GPU)..."
pip install --quiet torch>=2.2.0 torchvision>=0.17.0

echo "    Scientific stack..."
pip install --quiet numpy>=1.26.0 pandas>=2.0.0 scipy>=1.11.0 scikit-learn>=1.4.0

echo "    Astronomy & solar physics..."
pip install --quiet "astropy>=6.0.0" "sunpy[all]>=5.0.0" cdflib>=1.2.0 scikit-image>=0.22.0

echo "    NASA JSOC access (for SHARP data)..."
pip install --quiet drms>=0.7.0

echo "    ONNX export..."
pip install --quiet onnx>=1.17.0 onnxruntime>=1.17.0

echo -e "${GREEN}  ✓ All packages installed${NC}"

# ── Step 4: Check PRADAN credentials ─────────────────────────────────────────
echo ""
echo -e "${YELLOW}[4/6] Checking PRADAN credentials...${NC}"

if [ -z "$PRADAN_USER" ] || [ -z "$PRADAN_PASS" ]; then
    echo -e "${YELLOW}  ⚠  PRADAN credentials not found in environment.${NC}"
    echo ""
    echo -e "${YELLOW}  Without credentials, the script will:${NC}"
    echo -e "${YELLOW}    - Skip SoLEXS, HEL1OS, MAG, ASPEX data download${NC}"
    echo -e "${YELLOW}    - Still download GOES (free) and SHARP (free)${NC}"
    echo -e "${YELLOW}    - Still train on whatever data is already in data/pradan_cache/${NC}"
    echo ""
    echo -e "${CYAN}  To add credentials, run before this script:${NC}"
    echo -e "${CYAN}    export PRADAN_USER=your_username${NC}"
    echo -e "${CYAN}    export PRADAN_PASS=your_password${NC}"
    echo ""
    echo -e "${YELLOW}  Continuing without PRADAN credentials...${NC}"
else
    echo -e "${GREEN}  ✓ PRADAN credentials found (user: $PRADAN_USER)${NC}"
fi

# ── Step 5: Verify Metal GPU ──────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}[5/6] Verifying M4 Metal GPU...${NC}"
python -c "
import torch
if torch.backends.mps.is_available():
    print('  \033[0;32m✓ Apple Metal GPU (MPS) is ACTIVE — training is GPU-accelerated!\033[0m')
    x = torch.randn(512, 512, device='mps')
    y = x @ x.T
    print(f'  ✓ MPS verified: {y.shape}')
else:
    print('  \033[1;33m⚠  Metal GPU not available — using CPU (slower)\033[0m')
"

# ── Step 6: Run training ──────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}[6/6] Starting incremental training...${NC}"
echo ""
echo -e "${CYAN}  TRAINING PLAN:${NC}"
echo -e "${CYAN}  • 15 months of real Aditya-L1 data (Jan 2024 – Apr 2025)${NC}"
echo -e "${CYAN}  • Best months first: May 2024 (X8.7), Oct 2024 (X9.0)${NC}"
echo -e "${CYAN}  • Each month: download → train → save → delete (max 2 GB disk)${NC}"
echo -e "${CYAN}  • ALL branches: SoLEXS+HEL1OS TCN + SHARP LSTM + MAG+SWIS MLP${NC}"
echo -e "${CYAN}  • Wavelet transform (QPP detection) applied to SoLEXS${NC}"
echo ""
echo -e "${YELLOW}  Keep your MacBook plugged in. Estimated time: 45–90 minutes.${NC}"
echo -e "${YELLOW}  Progress is saved after each month — safe to interrupt and resume.${NC}"
echo ""

cd "$(dirname "$0")"

# Run with --delete-after-month to save disk space
python notebooks/06_incremental_real_train.py \
    --epochs-per-month 15 \
    --batch-size 32 \
    --delete-after-month

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}${BOLD}║              TRAINING COMPLETE! 🎉                           ║${NC}"
echo -e "${GREEN}${BOLD}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}  Files to send back to Mahan:${NC}"
echo -e "${GREEN}    models/adityscan_v4.onnx          ← Production model${NC}"
echo -e "${GREEN}    models/adityscan_v4_incremental.pt← PyTorch checkpoint${NC}"
echo -e "${GREEN}    models/training_report.json       ← Metrics & data sources${NC}"
echo ""
echo -e "${CYAN}  Or: git add models/ && git commit -m 'trained v4' && git push${NC}"
echo ""
