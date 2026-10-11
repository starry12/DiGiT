#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1
export DIGIT_PYTHON="${DIGIT_PYTHON:-/home/embed/miniconda3/envs/gids/bin/python}"
: "${TMUX:?Run in tmux with local sudo}"
if [[ $EUID -ne 0 ]]; then echo 'Native IG needs local sudo inside tmux' >&2; exit 1; fi
for model in sage gcn gat; do
  bash submission/v9/run.sh representative --dataset IG --model "$model" --gpu 2 --output "results/ig_${model}_window_20260922_v1"
done
