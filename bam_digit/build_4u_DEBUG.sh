set -e

source /home/embed/anaconda3/etc/profile.d/conda.sh
conda activate bam
cd /home/embed/bam_gids
rm -rf build
mkdir -p build
cd build
cmake -D CMAKE_CUDA_COMPILER=/usr/local/cuda-12.3/bin/nvcc \
 -D NVIDIA=/usr/src/nvidia-550.54.14/ -D driver_include:PATH=/usr/src/nvidia-550.54.14/nvidia \
 -D CMAKE_BUILD_TYPE=DEBUG ..
make libnvm -j$(nproc) VERBOSE=1
make benchmarks -j$(nproc) 2>/home/embed/bam_gids/build_err.txt VERBOSE=1
cd module
make -j$(nproc)