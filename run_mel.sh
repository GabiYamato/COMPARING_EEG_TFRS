#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

echo "===================================================="
echo "  Starting MEL run"
echo "===================================================="

# Run all models and ablations for MEL
$PYTHON "$SCRIPT_DIR/main.py" --dataset mel --mode all

# Update CSV summaries & notebooks
$PYTHON "$SCRIPT_DIR/csv_generator.py"
$PYTHON "$SCRIPT_DIR/notebook_generator.py"

echo "✓ MEL run complete."
