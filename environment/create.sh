#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
[[ $# == 1 ]] || { echo 'Usage: bash environment/create.sh NEW_ABSOLUTE_ENV_PREFIX' >&2; exit 2; }
[[ $1 == /* && ! -e $1 ]] || { echo 'A fresh absolute environment prefix is required' >&2; exit 2; }
CONDA_BIN=${DIGIT_CONDA:-conda}
"$CONDA_BIN" create -y -p "$1" --file "$HERE/conda-linux-64.lock"
"$1/bin/python" -m pip --isolated install --index-url https://pypi.org/simple --require-hashes --no-deps -r "$HERE/requirements.lock"
"$1/bin/python" -m pip check
"$1/bin/python" "$HERE/check.py"
