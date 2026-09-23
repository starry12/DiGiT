#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
ROOT=$(cd "$HERE/../.." && pwd -P)
PY=${DIGIT_PYTHON:-python3}
NVCC=${DIGIT_NVCC:-nvcc}
mkdir -p "$HERE/runtime/BAM_Feature_Store"
read -r -a INCLUDES <<< "$("$PY" -m pybind11 --includes)"
"$NVCC" -std=c++11 -O3 -arch=sm_89 --default-stream per-thread -shared -Xcompiler -fPIC \
  "${INCLUDES[@]}" -I"$HERE/native/gids_module/include" \
  -I"$HERE/native/include" -I"$ROOT/third_party/bam/include" -I"$ROOT/third_party/bam/include/freestanding/include" \
  "$HERE/native/gids_module/gids_nvme.cu" "$ROOT/third_party/bam/build/lib/libnvm.so" \
  -Xlinker -rpath -Xlinker '$ORIGIN/../../../../third_party/bam/build/lib' -Xcompiler -pthread \
  -o "$HERE/runtime/BAM_Feature_Store/BAM_Feature_Store.so"
printf 'from .BAM_Feature_Store import *\n' > "$HERE/runtime/BAM_Feature_Store/__init__.py"
