#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
PY=${DIGIT_PYTHON:-python3}
CMAKE=${DIGIT_CMAKE:-cmake}
NVCC=${DIGIT_NVCC:-nvcc}
BUILD=${DIGIT_BUILD_DIR:-"$ROOT/build/ae"}
PYBIND=$("$PY" -m pybind11 --cmakedir)
compile_extension() {
  local source=$1 output=$2 name=$3
  "$CMAKE" -S "$source" -B "$BUILD/$name" -Dpybind11_DIR="$PYBIND" \
    -DPYTHON_EXECUTABLE="$(command -v "$PY")" -DCMAKE_CUDA_COMPILER="$(command -v "$NVCC")" \
    -DDIGIT_EXTENSION_OUTPUT="$output"
  "$CMAKE" --build "$BUILD/$name" -j "${DIGIT_BUILD_JOBS:-2}"
}
compile_extension "$ROOT/ae/native/papers/sampler" "$ROOT/ae/papers/runtime/digit" papers_sampler
compile_extension "$ROOT/ae/native/papers/output" "$ROOT/ae/papers/runtime/digit" papers_output
compile_extension "$ROOT/ae/native/igb/sampler" "$ROOT/ae/igb" igb_sampler
"$CMAKE" -S "$ROOT/ae/native/papers/gids_module" -B "$BUILD/feature_store" \
  -Dpybind11_DIR="$PYBIND" -DPYTHON_EXECUTABLE="$(command -v "$PY")" \
  -DCMAKE_CUDA_COMPILER="$(command -v "$NVCC")" -DDIGIT_BAM_LIBRARY="$ROOT/bam/build/lib/libnvm.so"
"$CMAKE" --build "$BUILD/feature_store" -j "${DIGIT_BUILD_JOBS:-2}"
cp "$BUILD/feature_store/BAM_Feature_Store/BAM_Feature_Store.so" "$ROOT/ae/papers/runtime/BAM_Feature_Store/"
read -r -a INCLUDES <<< "$("$PY" -m pybind11 --includes)"
"$NVCC" -std=c++11 -O3 -arch=sm_89 --default-stream per-thread -shared -Xcompiler -fPIC \
  "${INCLUDES[@]}" -I"$ROOT/ae/native/papers/gids_module/include" \
  -I"$ROOT/ae/native/papers/bam/include" -I"$ROOT/ae/native/papers/bam/include/freestanding/include" \
  "$ROOT/ae/native/igb/feature_store/gids_nvme.cu" "$ROOT/bam/build/lib/libnvm.so" \
  -Xlinker -rpath -Xlinker '$ORIGIN/.digit_lib' -Xcompiler -pthread -o "$ROOT/ae/igb/IGSageNative.so"
"$PY" "$ROOT/ae/refresh_manifest.py"
