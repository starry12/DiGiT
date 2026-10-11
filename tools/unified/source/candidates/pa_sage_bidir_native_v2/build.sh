#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PY=${DIGIT_PYTHON:-/home/embed/miniconda3/envs/gids/bin/python}
NVCC=${DIGIT_NVCC:-/usr/local/cuda-12.4/bin/nvcc}
read -r -a INC <<< "$("$PY" -m pybind11 --includes)"
"$NVCC" -std=c++17 -O3 -arch=sm_89 --default-stream per-thread -shared -Xcompiler -fPIC \
 "${INC[@]}" "$HERE/native/digit_sampler_cuda.cu" -o "$HERE/runtime/digit/DiGiTSamplerCUDA.so"
