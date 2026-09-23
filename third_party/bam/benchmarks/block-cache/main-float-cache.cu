#include <buffer.h>
#include <byteswap.h>
#include <ctrl.h>
#include <cuda.h>
#include <event.h>
#include <fcntl.h>
#include <nvm_admin.h>
#include <nvm_cmd.h>
#include <nvm_ctrl.h>
#include <nvm_error.h>
#include <nvm_io.h>
#include <nvm_parallel_queue.h>
#include <nvm_queue.h>
#include <nvm_types.h>
#include <nvm_util.h>
#include <page_cache.h>
#include <queue.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <util.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "settings.h"
#ifdef __DIS_CLUSTER__
#include <sisci_api.h>
#endif

using error = std::runtime_error;
using std::string;

// uint32_t n_ctrls = 1;
const char *const sam_ctrls_paths[] = {"/dev/libnvm0", "/dev/libnvm1", "/dev/libnvm9", "/dev/libnvm2", "/dev/libnvm3",
                                       "/dev/libnvm5", "/dev/libnvm6", "/dev/libnvm7", "/dev/libnvm8"};
const char *const intel_ctrls_paths[] = {"/dev/libinvm0", "/dev/libinvm1", "/dev/libinvm9", "/dev/libinvm2", "/dev/libinvm3",
                                         "/dev/libinvm5", "/dev/libinvm6", "/dev/libinvm7", "/dev/libinvm8"};
// const char* const ctrls_paths[] = {"/dev/libnvm0", "/dev/libnvm1",
// "/dev/libnvm2", "/dev/libnvm3", "/dev/libnvm4", "/dev/libnvm5",
// "/dev/libnvm6", "/dev/libnvm7", "/dev/libnvm8", "/dev/libnvm9",
// "/dev/libnvm10", "/dev/libnvm11", "/dev/libnvm12", "/dev/libnvm13",
// "/dev/libnvm14", "/dev/libnvm15", "/dev/libnvm16", "/dev/libnvm17",
// "/dev/libnvm18", "/dev/libnvm19", "/dev/libnvm20", "/dev/libnvm21",
// "/dev/libnvm22", "/dev/libnvm23", "/dev/libnvm24","/dev/libnvm25",
// "/dev/libnvm26", "/dev/libnvm27", "/dev/libnvm28", "/dev/libnvm29",
// "/dev/libnvm30", "/dev/libnvm31"};

#define SIZE (8 * 4096)

// feature type and dimension
typedef float FEATURE_TYPE;
#define FEATURE_DIM feature_dim

__global__ void print_cache_kernel(page_cache_d_t *pc) {
  uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid == 0) {
    hexdump(pc->base_addr, SIZE);
  }
}

__global__ void new_kernel(ulonglong4 *dst, ulonglong4 *src, size_t num) { warp_memcpy<ulonglong4>(dst, src, num); }

int check_numpy_header(const char *file_path) {
  int fd = open(file_path, O_RDONLY);
  if (fd == -1) {
    fprintf(stderr, "File cannot be opened\n");
    return -1;
  }
  char header[10];
  if (read(fd, header, 10) != 10) {
    fprintf(stderr, "Failed to read file header\n");
    close(fd);
    return -1;
  }
  close(fd);
  if (memcmp(header, "\x93NUMPY", 6) == 0) {
    int16_t header_len;
    memcpy(&header_len, header + 8, sizeof(int16_t));
    return 10 + header_len;
  }
  return 0;
}

// 写一个返回智能指针shared_ptr的函数，函数接收一个文件路径，一个文件读写偏移量作为参数，将文件mmap到内存中并返回一个shared_ptr；
// 函数使用类型模板作为文件中读取内容的类型。

template <typename T>
std::shared_ptr<T> mmap_file_with_offset(const char *file_path, std::size_t offset) {
  int fd = open(file_path, O_RDWR);
  if (fd == -1) {
    throw std::runtime_error("Failed to open file");
  }

  struct stat sb;
  if (fstat(fd, &sb) == -1) {
    close(fd);
    throw std::runtime_error("Failed to get file size");
  }

  size_t length = sb.st_size;

  void *mapped = mmap(nullptr, length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  void *map_in = (uint8_t *)mapped + offset;
  close(fd);

  if (mapped == MAP_FAILED) {
    throw std::runtime_error("Failed to mmap file");
  }

  auto deleter = [mapped, length](T *ptr) { munmap(mapped, length); };

  return std::shared_ptr<T>(static_cast<T *>(map_in), deleter);
}

template <typename T>
int file_element_count(const char *file_path, std::size_t offset) {
  struct stat sb;
  if (stat(file_path, &sb) == -1) {
    throw std::runtime_error("Failed to get file size");
  }
  return (sb.st_size - offset) / sizeof(T);
}

template <typename T>
std::shared_ptr<T> get_batch_start_offset(const char *batch_size, std::shared_ptr<T> &map_batch_size, int batch_len) {
  uint64_t *batch_start_offset_ptr = new uint64_t[batch_len];
  std::shared_ptr<T> batch_start_offset = std::shared_ptr<T>(batch_start_offset_ptr, [](T *ptr) { delete[] ptr; });

  for (size_t i = 0; i < batch_len; i++) {
    if (i == 0) {
      batch_start_offset.get()[0] = 0;
      continue;
    }
    int prev_node_count = map_batch_size.get()[i - 1];
    batch_start_offset.get()[i] = batch_start_offset.get()[i - 1] + prev_node_count;
  }
  return batch_start_offset;
}

template <typename T = float>
__global__ __launch_bounds__(128, 4) 
void read_feature_kernel(array_d_t<T> *dr, T *out_tensor_ptr, int64_t *index_ptr, int dim, int64_t num_idx, int cache_dim, uint64_t key_off) {
  uint64_t bid = blockIdx.x;
  int num_warps = blockDim.x / 32;
  int warp_id = threadIdx.x / 32;
  int idx_idx = bid * num_warps + warp_id;
  if (idx_idx < num_idx) {  // 确保当前的全局索引不超出总索引范围
    bam_ptr<T> ptr(dr);

    uint64_t row_index = index_ptr[idx_idx] + key_off;
    uint64_t tid = threadIdx.x % 32;

    for (; tid < dim; tid += 32) {
      // 每一个warp负责读取一个feature，一个warp里面总共32个thread，tid是thread在warp中的索引
      if (tid == 0) {
        // atomicAdd(d_gpu_access, 1);
      }
      T temp = ptr[(row_index)*cache_dim + tid];
      // printf("data: %llu\n",  (row_index) * cache_dim + tid);
      out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = temp;
      // 将读取的数据存储到输出张量的对应位置
    }
  }
}

int main(int argc, char **argv) {
  Settings settings;
  try {
    settings.parseArguments(argc, argv);
  } catch (const string &e) {
    fprintf(stderr, "%s\n", e.c_str());
    fprintf(stderr, "%s\n", Settings::usageString(argv[0]).c_str());
    return 1;
  }

  cudaDeviceProp properties;
  if (cudaGetDeviceProperties(&properties, settings.cudaDevice) != cudaSuccess) {
    fprintf(stderr, "Failed to get CUDA device properties\n");
    return 1;
  }

  try {
    const char *node_sequence;
    const char *batch_size;
    node_sequence = settings.node_sequence;
    batch_size = settings.batch_size;
    const int ssd_num = settings.ssd_num;

    int node_seq_offset = check_numpy_header(node_sequence);

    if (node_seq_offset == -1) {
      fprintf(stderr, "Failed to get npy header length\n");
      return 1;
    }

    std::cout << "Node sequence offset: " << node_seq_offset << std::endl;

    std::shared_ptr<uint64_t> map_node_sequence = mmap_file_with_offset<uint64_t>(node_sequence, node_seq_offset);

    int batch_size_offset = check_numpy_header(batch_size);
    if (batch_size_offset == -1) {
      fprintf(stderr, "Failed to get {Batch-Size} npy header length\n");
      exit(0);
    }
    std::cout << "Batch size offset: " << batch_size_offset << std::endl;
    int batch_len = file_element_count<uint64_t>(batch_size, batch_size_offset);

    std::shared_ptr<uint64_t> map_batch_size = mmap_file_with_offset<uint64_t>(batch_size, batch_size_offset);
    std::shared_ptr<uint64_t> batch_start_offset = get_batch_start_offset<uint64_t>(batch_size, map_batch_size, batch_len);

    std::cout << "Batch length(how many batch): " << batch_len << std::endl;
    // std::cout << "Batch start offset: " << batch_start_offset.get()[0] <<
    // std::endl; std::cout << "Batch start offset: " <<
    // batch_start_offset.get()[1] << std::endl;

    cuda_err_chk(cudaSetDevice(settings.cudaDevice));

    std::vector<Controller *> ctrls(1);
    ctrls[0] = new Controller(settings.ssdtype == 0 ? sam_ctrls_paths[ssd_num] : intel_ctrls_paths[ssd_num], settings.nvmNamespace, settings.cudaDevice,
                              settings.queueDepth, settings.numQueues);

    std::cout << "Ctrl page size: " << ctrls[0]->ctrl->page_size << std::endl;
    uint64_t page_size = settings.pageSize;
    uint64_t num_pages = settings.num_page;
    uint64_t numElems = num_pages * page_size / sizeof(FEATURE_TYPE);
    // 默认一个feature占据一个page
    unsigned int feature_dim = page_size / sizeof(FEATURE_TYPE);
    // cache里能够承载的条目数量 (1024*1024 = MB)
    uint64_t cache_n_pages = settings.cache_size * 1024LL * 1024 / page_size;
    std::cout << "Cache npages: " << cache_n_pages << std::endl;

    uint32_t cudaDevice = 0;
    page_cache_t *h_pc = new page_cache_t(page_size, cache_n_pages, cudaDevice, ctrls[0][0], (uint64_t)64, ctrls);
    page_cache_t *d_pc = (page_cache_t *)(h_pc->d_pc_ptr);
    // bam的寻址空间
    range_t<FEATURE_TYPE> *h_range =
        new range_t<FEATURE_TYPE>((uint64_t)0, numElems, (uint64_t)0, (uint64_t)num_pages, (uint64_t)0, (uint64_t)page_size, h_pc, cudaDevice,
                                  // REPLICATE
                                  STRIPE);

    range_d_t<FEATURE_TYPE> *d_range = (range_d_t<FEATURE_TYPE> *)h_range->d_range_ptr;
    std::vector<range_t<FEATURE_TYPE> *> vr;
    vr.push_back(h_range);
    array_t<FEATURE_TYPE> *a = new array_t<FEATURE_TYPE>(numElems, 0, vr, cudaDevice);

    double total_time = 0.0;
    uint64_t total_data = 0;
    int scale_factor = 16;
    for (size_t i = 0; i < (int)1; i++) {
      uint64_t node_count = 0;
      for (size_t j = 0; j < scale_factor; j++) {
        node_count += map_batch_size.get()[j * scale_factor + i];
      }

      uint64_t start_offset = batch_start_offset.get()[i * scale_factor];

      uint64_t b_size = settings.blkSize;  // 128;
      std::cout << std::dec << "settings.blkSize " << settings.blkSize << std::endl;
      std::cout << std::dec << "b_size " << b_size << std::endl;
      // 每个块包含的warp数（一个warp通常包含32个线程）
      uint64_t n_warp = b_size / 32;
      // 计算所需的网格大小，确保每个索引都被处理
      uint64_t g_size = (node_count + n_warp - 1) / n_warp;

      // 为要取出来的feature在GPU上分配内存
      // FEATURE_TYPE *h_data = new FEATURE_TYPE[node_count * FEATURE_DIM];
      FEATURE_TYPE *d_data;
      // memset(h_data, 0, node_count * FEATURE_DIM * sizeof(FEATURE_TYPE));
      cuda_err_chk(cudaMalloc(&d_data, node_count * FEATURE_DIM * sizeof(FEATURE_TYPE)));
      // cuda_err_chk(cudaMemcpy(d_data, h_data, node_count * FEATURE_DIM *
      // sizeof(FEATURE_TYPE), cudaMe`mcpyHostToDevice));

      // 为索引在GPU上分配内存
      int64_t *d_index_ptr;
      cuda_err_chk(cudaMalloc(&d_index_ptr, node_count * sizeof(uint64_t)));
      cuda_err_chk(cudaMemcpy(d_index_ptr, map_node_sequence.get() + start_offset, node_count * sizeof(uint64_t), cudaMemcpyHostToDevice));

      cuda_err_chk(cudaDeviceSynchronize());

      Event before;
      read_feature_kernel<FEATURE_TYPE><<<g_size, b_size>>>(a->d_array_ptr, d_data, d_index_ptr, FEATURE_DIM, node_count, FEATURE_DIM, 0);
      Event after;
      cuda_err_chk(cudaDeviceSynchronize());

      double elapsed = after - before;
      total_time += elapsed;
      uint64_t data = node_count * FEATURE_DIM * sizeof(FEATURE_TYPE);
      total_data += data;

      cudaFree(d_data);
      cudaFree(d_index_ptr);
      std::cout << "Batch " << i << " done" << std::endl;
    }
    a->print_reset_stats();
    double bandwidth = (((double)total_data) / (total_time / 1000000)) / (1024ULL * 1024ULL * 1024ULL);
    // std::cout << std::dec << "Elapsed Time: " << elapsed << "\tNumber of Ops:
    // "<< ios << "\tData Size (bytes): " << data << std::endl; std::cout <<
    // std::dec << "Ops/sec: " << iops << "\tEffective Bandwidth(GB/S): " <<
    // bandwidth << std::endl;
    std::cout << std::dec << "Elapsed Time(s): " << total_time / (1000 * 1000) << "\tTotal Data Size (GBytes): " << total_data / 1024.0 / 1024.0 / 1024.0 << std::endl;
    std::cout << std::dec << "Effective Bandwidth(GB/S): " << bandwidth << std::endl;
  } catch (const error &e) {
    fprintf(stderr, "Unexpected error: %s\n", e.what());
    return 1;
  }
}
