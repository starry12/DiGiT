#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
export PYTHONDONTWRITEBYTECODE=1
exec "${DIGIT_PYTHON:-python3}" -B "$ROOT/artifact.py" "$@"
