#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
ROOT=$(cd "$HERE/../.." && pwd -P)
BASE="$ROOT/candidates/io_accounting_v1/native/bam"
PY=/home/embed/miniconda3/envs/gids/bin/python
mkdir -p "$HERE/runtime/BAM_Feature_Store"
read -r -a INCLUDES <<< "$("$PY" -m pybind11 --includes)"
/usr/local/cuda-12.4/bin/nvcc -std=c++11 -O3 -arch=sm_89 --default-stream per-thread -shared -Xcompiler -fPIC \
  "${INCLUDES[@]}" -I"$HERE/native/gids_module/include" -I"$BASE/include" -I"$BASE/include/freestanding/include" \
  "$HERE/native/gids_module/gids_nvme.cu" "$ROOT/bam/build/lib/libnvm.so" \
  -Xlinker -rpath -Xlinker '$ORIGIN/../../../../bam/build/lib' -Xcompiler -pthread \
  -o "$HERE/runtime/BAM_Feature_Store/BAM_Feature_Store.so"
printf 'from .BAM_Feature_Store import *\n' > "$HERE/runtime/BAM_Feature_Store/__init__.py"
