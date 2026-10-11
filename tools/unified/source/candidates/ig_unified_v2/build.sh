#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
ROOT=$(cd "$HERE/../.." && pwd -P)
PY=${DIGIT_PYTHON:-/home/embed/miniconda3/envs/gids/bin/python}
NVCC=${DIGIT_NVCC:-/usr/local/cuda-12.4/bin/nvcc}
read -r -a INCLUDES <<< "$("$PY" -m pybind11 --includes)"
"$NVCC" -std=c++11 -O3 -arch=sm_89 --default-stream per-thread -shared -Xcompiler -fPIC   "${INCLUDES[@]}" -I"$HERE/native/include" -I"$ROOT/candidates/io_accounting_v1/native/gids_module/include"   -I"$ROOT/candidates/io_accounting_v1/native/bam/include"   -I"$ROOT/candidates/io_accounting_v1/native/bam/include/freestanding/include"   "$HERE/native/gids_nvme.cu" "$ROOT/bam/build/lib/libnvm.so"   -Xlinker -rpath -Xlinker '$ORIGIN/../../../bam/build/lib' -Xcompiler -pthread   -o "$HERE/runtime/IGPerfNative.so"
