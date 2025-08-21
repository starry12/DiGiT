cmake_minimum_required (VERSION 3.1)
project (libnvm-benchmarks)

set (THREADS_PREFER_PTHREAD_FLAG ON)
find_package (Threads REQUIRED)

find_package (CUDA 8.0 REQUIRED)



make_benchmark (block-benchmark-module-cache block-bench-cache "main-float-cache.cu;")
make_benchmark (block-benchmark-module-cache-jc block-bench-cache-jc "main-float-cache_jc.cu;")
# make_benchmark (block-benchmark-module-libnvm0 block-bench-libnvm0 "main-libnvm0.cu;")
# make_benchmark (block-benchmark-module-libnvm1 block-bench-libnvm1 "main-libnvm1.cu;")

# make_benchmark_choice (block-benchmark block-benchmark-smartio block-benchmark-module)
