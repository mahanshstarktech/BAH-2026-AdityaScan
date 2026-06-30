#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

PLAN="${PLAN:-config/training_windows_available_balanced.json}"
PYTHON_BIN="${PYTHON_BIN:-.venv_training/bin/python}"
ENV_FILE="${ENV_FILE:-.env.training.local}"

if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

if [ -z "${PRADAN_USER:-}" ] || [ -z "${PRADAN_PASS:-}" ] || \
   [ "${PRADAN_USER:-}" = "your_pradan_username" ] || \
   [ "${PRADAN_PASS:-}" = "your_pradan_password" ]; then
  echo "PRADAN_USER and PRADAN_PASS must be set in your terminal or $ENV_FILE."
  echo "The current values look empty or still set to the example placeholders."
  echo "Example:"
  echo "  PRADAN_USER=\"actual_username\""
  echo "  PRADAN_PASS=\"actual_password\""
  echo
  echo "Or create $ENV_FILE from .env.training.example and fill it in."
  exit 1
fi

echo "Step 1/2: Downloading PRADAN SoLEXS + HEL1OS into data/pradan_cache..."
"$PYTHON_BIN" scripts/download_pradan_range.py \
  --plan "$PLAN" \
  --instruments solexs,helios \
  --env-file "$ENV_FILE" \
  --sleep-seconds 2

echo "Step 2/2: Updating model from latest checkpoint with balanced windows..."
"$PYTHON_BIN" notebooks/06_incremental_real_train.py \
  --training-manifest "$PLAN" \
  --skip-download \
  --require-all-modalities \
  --required-modalities solexs,helios,goes \
  --no-sharp --no-suit --no-swis \
  --resume \
  "$@"
