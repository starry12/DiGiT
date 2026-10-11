#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
[[ -n ${TMUX:-} ]] || { echo 'Run inside tmux' >&2; exit 1; }
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=2 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="$ROOT/bam/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset PYTHONOPTIMIZE
exec "${DIGIT_PYTHON:-/home/embed/miniconda3/envs/gids/bin/python}" -u -m candidates.pa_sage_bidir_native_v2.run "$@"
