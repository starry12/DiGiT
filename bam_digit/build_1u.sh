set -e

cd /home/embed/Documents/gids_bam
mkdir -p build
cd build
cmake -D CMAKE_BUILD_TYPE=Debug ..
# make libnvm -j$(nproc) VERBOSE=1
make libnvm -j$(nproc)
# make benchmarks -j$(nproc) 2>/home/embed/Documents/gids_bam/build_err.txt VERBOSE=1
make benchmarks -j$(nproc) 2>/home/embed/Documents/gids_bam/build_err.txt
cd module
make -j$(nproc)