#!/usr/bin/env bash
set -euo pipefail
umask 022
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ $EUID -ne 0 || -z ${TMUX:-} || $# -lt 1 ]]; then
  echo 'Usage inside tmux: sudo --preserve-env=TMUX bash ae/run_pa_sage_formal.sh FRESH_OUTPUT [--seeds 0 1 2] [--repeats 0 1]' >&2
  exit 1
fi
OUTPUT=$1
shift
export CUDA_VISIBLE_DEVICES=2 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="$ROOT/bam/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset PYTHONOPTIMIZE
PYTHON_BIN="${DIGIT_PYTHON:-/home/embed/miniconda3/envs/gids/bin/python}"
cd "$ROOT"
"$PYTHON_BIN" ae/freeze.py verify frozen/pa_sage_formal_20260921_v4 --live
exec "$PYTHON_BIN" -u ae/run_pa_sage.py experiment --output "$OUTPUT" "$@"
