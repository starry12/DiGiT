#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ $# -ne 1 ]]; then
    echo "Usage: sudo bash $0 /path/to/fresh-result-directory" >&2
    exit 2
fi
if [[ $EUID -ne 0 ]]; then
    echo "BaM device mapping requires root; run this command with sudo." >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="$ROOT/bam/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PYTHON_BIN="${DIGIT_PYTHON:-/home/embed/miniconda3/envs/gids/bin/python}"
unset PYTHONOPTIMIZE
cd "$ROOT"
"$PYTHON_BIN" -u ae/freeze.py verify frozen/pa_sage_formal_20260921_v4 --live
exec "$PYTHON_BIN" -u ae/run_pa_sage.py native-smoke --output "$1"
