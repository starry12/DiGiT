set -e

source /home/embed/anaconda3/etc/profile.d/conda.sh
conda activate gids
cd /home/embed/GIDS

cd bam
mkdir -p build
cd build
cmake -D CMAKE_CUDA_COMPILER=/usr/local/cuda-12.3/bin/nvcc -D NVIDIA=/usr/src/nvidia-550.54.14/ -D driver_include:PATH=/usr/src/nvidia-550.54.14/nvidia ..
make libnvm -j$(nproc)
make benchmarks -j$(nproc)
cd module
make -j$(nproc)

cd /home/embed/GIDS
cd gids_module
mkdir -p build
cd build
cmake .. \
-DPYTHON_INCLUDE_DIR=$(python -c "import sysconfig; print(sysconfig.get_path('include'))")  \
-DPYTHON_LIBRARY=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
make -j$(nproc)
cd BAM_Feature_Store
pip install .
cd ../../../
cd GIDS_Setup
pip install .
cd ../evaluation