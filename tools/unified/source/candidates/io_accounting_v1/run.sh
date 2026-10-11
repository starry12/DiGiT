#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
[[ -n ${TMUX:-} ]] || { echo 'Run inside tmux' >&2; exit 1; }
export CUDA_VISIBLE_DEVICES=2 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="$ROOT/bam/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset PYTHONOPTIMIZE
cd "$ROOT"
exec "${DIGIT_PYTHON:-/home/embed/miniconda3/envs/gids/bin/python}" -u -m candidates.io_accounting_v1.run \
 --output "${1:-$ROOT/results/io_accounting_native_20260921_v1}"
