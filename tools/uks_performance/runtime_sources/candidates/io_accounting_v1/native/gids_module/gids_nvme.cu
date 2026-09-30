#include <pybind11/pybind11.h>

#include <cstdint>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <iomanip>


#include <stdio.h>
#include <vector>

#include <bam_nvme.h>
#include <pybind11/stl.h>
#include "gids_kernel.cu"
//#include <bafs_ptr.h>


typedef std::chrono::high_resolution_clock Clock;

std::vector<uint64_t> probe_mixed_io_geometry_cuda(
    const std::vector<uint64_t>& storage_rows,
    const std::vector<uint8_t>& group_flags,
    uint32_t feature_row_bytes, uint32_t minimum_transfer_bytes,
    uint32_t group_size, uint32_t cache_slot_bytes,
    uint64_t full_valid_mask, uint64_t payload_offset_bytes) {
  int device_count = 0;
  cudaError_t device_status = cudaGetDeviceCount(&device_count);
  if (device_status != cudaSuccess || device_count == 0) {
    throw std::runtime_error(
        "Phase 9B CUDA geometry probe requires a visible CUDA device");
  }
  if (storage_rows.size() != group_flags.size()) {
    throw std::invalid_argument("geometry probe rows and flags differ in length");
  }
  if (feature_row_bytes == 0 || minimum_transfer_bytes == 0 ||
      group_size == 0 || cache_slot_bytes == 0 ||
      feature_row_bytes > minimum_transfer_bytes ||
      minimum_transfer_bytes % feature_row_bytes != 0 ||
      cache_slot_bytes % minimum_transfer_bytes != 0) {
    throw std::invalid_argument("geometry probe received unsupported dimensions");
  }
  uint32_t subrows = cache_slot_bytes / minimum_transfer_bytes;
  if (subrows == 0 || subrows > 64) {
    throw std::invalid_argument("geometry probe requires 1..64 validity bits");
  }
  uint64_t expected_mask = subrows == 64
      ? ~0ULL : ((1ULL << subrows) - 1ULL);
  if (full_valid_mask != expected_mask ||
      ((static_cast<uint64_t>(feature_row_bytes) * group_size
        + minimum_transfer_bytes - 1) / minimum_transfer_bytes)
          * minimum_transfer_bytes != cache_slot_bytes ||
      payload_offset_bytes % minimum_transfer_bytes != 0) {
    throw std::invalid_argument("geometry probe contract is inconsistent");
  }
  for (size_t index = 0; index < storage_rows.size(); ++index) {
    if (group_flags[index] > 1) {
      throw std::invalid_argument("geometry probe group flag is not boolean");
    }
    uint64_t row_in_slot =
        (storage_rows[index] * feature_row_bytes % cache_slot_bytes)
        / feature_row_bytes;
    if (group_flags[index] && row_in_slot >= group_size) {
      throw std::invalid_argument("geometry probe group row points into padding");
    }
  }
  if (storage_rows.empty()) return {};

  uint64_t* device_rows = nullptr;
  uint8_t* device_flags = nullptr;
  uint64_t* device_output = nullptr;
  size_t rows_bytes = storage_rows.size() * sizeof(uint64_t);
  size_t flags_bytes = group_flags.size() * sizeof(uint8_t);
  size_t output_bytes = storage_rows.size() * 6 * sizeof(uint64_t);
  cuda_err_chk(cudaMalloc(&device_rows, rows_bytes));
  cuda_err_chk(cudaMalloc(&device_flags, flags_bytes));
  cuda_err_chk(cudaMalloc(&device_output, output_bytes));
  cuda_err_chk(cudaMemcpy(
      device_rows, storage_rows.data(), rows_bytes, cudaMemcpyHostToDevice));
  cuda_err_chk(cudaMemcpy(
      device_flags, group_flags.data(), flags_bytes, cudaMemcpyHostToDevice));
  uint32_t threads = 128;
  uint32_t blocks = static_cast<uint32_t>(
      (storage_rows.size() + threads - 1) / threads);
  probe_mixed_io_geometry_kernel<<<blocks, threads>>>(
      device_rows, device_flags, device_output, storage_rows.size(),
      feature_row_bytes, minimum_transfer_bytes, group_size,
      cache_slot_bytes, full_valid_mask, payload_offset_bytes);
  cuda_err_chk(cudaGetLastError());
  cuda_err_chk(cudaDeviceSynchronize());
  std::vector<uint64_t> output(storage_rows.size() * 6);
  cuda_err_chk(cudaMemcpy(
      output.data(), device_output, output_bytes, cudaMemcpyDeviceToHost));
  cuda_err_chk(cudaFree(device_output));
  cuda_err_chk(cudaFree(device_flags));
  cuda_err_chk(cudaFree(device_rows));
  return output;
}

void GIDS_Controllers::init_GIDS_controllers(uint32_t num_ctrls, uint64_t q_depth, uint64_t num_q, 
                          const std::vector<int>& ssd_list){

  n_ctrls = num_ctrls;
  queueDepth = q_depth; // 1024
  numQueues = num_q; // 128

  for (size_t i = 0; i < n_ctrls; i++) {
 	printf("SSD index: %i\n", ssd_list[i]);
       	  ctrls.push_back(new Controller(ctrls_paths[ssd_list[i]], nvmNamespace, cudaDevice, queueDepth, numQueues));
  }
}


// 在主机上分配一块页锁定内存（大小为 dim × len 的 TYPE 数组）
// 并在设备端为这块内存创建映射，获取对应的 GPU 访问指针
// 这样，GPU 内核便可以通过 device_cpu_buffer 直接访问或修改这段主机内存，实现“零拷贝”（zero-copy）操作
template <typename TYPE>
void BAM_Feature_Store<TYPE>::cpu_backing_buffer(uint64_t dim, uint64_t len){
  TYPE* cpu_buffer_ptr; // cpu_buffer_ptr：将指向分配在CPU内存中的缓冲区
  TYPE* d_cpu_buffer_ptr; // d_cpu_buffer_ptr：将指向同一块内存在GPU侧的映射地址

  cuda_err_chk(cudaHostAlloc((TYPE **)&cpu_buffer_ptr, sizeof(TYPE) * dim * len, cudaHostAllocMapped));
  // 返回一个可以在 GPU 上使用的 设备指针 d_cpu_buffer_ptr
  cuda_err_chk(cudaHostGetDevicePointer((TYPE **)&d_cpu_buffer_ptr, (TYPE *)cpu_buffer_ptr, 0));

  CPU_buffer.cpu_buffer_dim = dim;
  CPU_buffer.cpu_buffer_len = len;
  CPU_buffer.cpu_buffer = cpu_buffer_ptr;
  CPU_buffer.device_cpu_buffer = d_cpu_buffer_ptr;
  cpu_buffer_flag = true;
}

template <typename TYPE>
void BAM_Feature_Store<TYPE>::set_cpu_feature_path(
    uint32_t path, uint64_t reserve_rows, uint32_t gather_threads) {
  if (path > 1) {
    throw std::invalid_argument(
        "CPU feature path must be 0 (mapped) or 1 (staged)");
  }
  if (gather_threads == 0) {
    throw std::invalid_argument("CPU staging gather_threads must be positive");
  }
  if (path == 1 && !cpu_buffer_flag) {
    throw std::runtime_error("CPU staging requires an initialized CPU buffer");
  }
  cpu_feature_path = path;
  cpu_staging_threads = gather_threads;
  cpu_staging_reserve_rows = std::max<uint64_t>(1, reserve_rows);
  if (cpu_feature_path == 1) {
    if (seq_flag || cpu_page_offsets.empty()) {
      throw std::runtime_error(
          "CPU staging requires a populated page-aligned CPU cache");
    }
    ensure_cpu_staging_runtime(cpu_staging_reserve_rows,
                               cpu_staging_reserve_rows);
  }
}

template <typename TYPE>
void BAM_Feature_Store<TYPE>::set_mixed_io(bool enabled) {
  if (!enabled) {
    mixed_io_enabled = false;
    return;
  }
  if (pageSize != 8192 || h_pc->pdt.ctrl_page_size != 4096) {
    throw std::invalid_argument("legacy mixed I/O requires 4 KiB rows in an 8 KiB slot");
  }
  set_mixed_io_geometry(true, 4096, 4096, 2, 2, 0x3ULL,
                        "legacy-phase8d");
}

template <typename TYPE>
void BAM_Feature_Store<TYPE>::set_mixed_io_geometry(
    bool enabled, uint32_t feature_row_bytes,
    uint32_t minimum_transfer_bytes, uint32_t group_size,
    uint32_t requested_subrow_count, uint64_t requested_full_valid_mask,
    const std::string& geometry_sha256) {
  if (!enabled) {
    mixed_io_enabled = false;
    return;
  }
  if (feature_row_bytes == 0 || minimum_transfer_bytes == 0 || group_size == 0) {
    throw std::invalid_argument("mixed-I/O geometry fields must be positive");
  }
  if (feature_row_bytes > minimum_transfer_bytes ||
      minimum_transfer_bytes % feature_row_bytes != 0) {
    throw std::invalid_argument(
        "native Phase 9B currently requires feature rows to divide the minimum transfer");
  }
  if (minimum_transfer_bytes != h_pc->pdt.ctrl_page_size ||
      pageSize % minimum_transfer_bytes != 0) {
    throw std::invalid_argument(
        "mixed-I/O minimum transfer must match the controller PRP page");
  }
  uint64_t logical_group_bytes =
      static_cast<uint64_t>(feature_row_bytes) * group_size;
  uint64_t rounded_group_bytes =
      ((logical_group_bytes + minimum_transfer_bytes - 1) /
       minimum_transfer_bytes) * minimum_transfer_bytes;
  if (rounded_group_bytes != pageSize) {
    throw std::invalid_argument(
        "mixed-I/O rounded group bytes must equal the cache slot size");
  }
  uint32_t subrows = pageSize / minimum_transfer_bytes;
  if (subrows == 0 || subrows > 64) {
    throw std::invalid_argument(
        "mixed-I/O cache slot requires between 1 and 64 validity bits");
  }
  uint64_t full_mask = subrows == 64 ? ~0ULL : ((1ULL << subrows) - 1ULL);
  uint32_t expected_cache_subrows = mixed_split_cache_entries ? 1 : subrows;
  if (requested_subrow_count != subrows ||
      requested_full_valid_mask != full_mask ||
      expected_cache_subrows != h_pc->pdt.mixed_subrow_count ||
      h_pc->pdt.mixed_subrow_bytes != minimum_transfer_bytes ||
      h_pc->pdt.mixed_subrow_prps == nullptr ||
      h_pc->pdt.mixed_split_entries != mixed_split_cache_entries) {
    throw std::invalid_argument("BaM page cache does not match mixed-I/O geometry");
  }
  if (geometry_sha256.empty()) {
    throw std::invalid_argument("mixed-I/O geometry hash cannot be empty");
  }
  mixed_feature_row_bytes = feature_row_bytes;
  mixed_min_transfer_bytes = minimum_transfer_bytes;
  mixed_group_size = group_size;
  mixed_subrow_count = subrows;
  mixed_full_mask = full_mask;
  mixed_geometry_sha256 = geometry_sha256;
  mixed_io_enabled = true;
}

template <typename TYPE>
void BAM_Feature_Store<TYPE>::set_device_io_stats(bool enabled) {
  if (h_pc == nullptr || h_range == nullptr || a == nullptr) {
    throw std::runtime_error(
        "device I/O statistics require initialized controllers");
  }
  h_pc->set_device_io_stats(enabled);

  // array_d_t does not dereference h_pc->d_pc_ptr during feature reads.  Each
  // range contains a by-value snapshot of page_cache_d_t, and array_t keeps a
  // second device copy of that range.  Propagate the updated flag and counter
  // pointers to both copies; updating only d_pc_ptr leaves the actual NVMe
  // path permanently disabled even though the host API reports it enabled.
  h_range->rdt.cache = h_pc->pdt;
  cuda_err_chk(cudaMemcpy(
      &(h_range->d_range_ptr->cache), &(h_range->rdt.cache),
      sizeof(page_cache_d_t), cudaMemcpyHostToDevice));
  cuda_err_chk(cudaMemcpy(
      &(a->adt.d_ranges[0].cache), &(h_range->rdt.cache),
      sizeof(page_cache_d_t), cudaMemcpyHostToDevice));
}

template <typename TYPE>
void BAM_Feature_Store<TYPE>::ensure_cpu_staging_runtime(
    uint64_t index_rows, uint64_t payload_rows) {
  if (!staging_runtime_initialized) {
    cuda_err_chk(cudaStreamCreateWithFlags(&staging_ssd_stream,
                                           cudaStreamNonBlocking));
    cuda_err_chk(cudaStreamCreateWithFlags(&staging_copy_stream,
                                           cudaStreamNonBlocking));
    cuda_err_chk(cudaEventCreate(&staging_h2d_start));
    cuda_err_chk(cudaEventCreate(&staging_h2d_end));
    cuda_err_chk(cudaEventCreate(&staging_scatter_end));
    staging_runtime_initialized = true;
  }

  if (index_rows > staging_index_capacity) {
    uint64_t next_capacity = std::max(
        index_rows,
        std::max(cpu_staging_reserve_rows, staging_index_capacity * 2));
    if (staging_host_indices != nullptr) {
      cuda_err_chk(cudaFreeHost(staging_host_indices));
    }
    cuda_err_chk(cudaHostAlloc(
        reinterpret_cast<void **>(&staging_host_indices),
        next_capacity * sizeof(int64_t), cudaHostAllocDefault));
    staging_index_capacity = next_capacity;
    ++staging_reallocations;
  }

  if (payload_rows > staging_payload_capacity) {
    uint64_t next_capacity = std::max(
        payload_rows,
        std::max(cpu_staging_reserve_rows, staging_payload_capacity * 2));
    if (staging_host_payload != nullptr) {
      cuda_err_chk(cudaFreeHost(staging_host_payload));
    }
    if (staging_device_payload != nullptr) {
      cuda_err_chk(cudaFree(staging_device_payload));
    }
    size_t row_bytes = sizeof(uint64_t) +
                       CPU_buffer.cpu_buffer_dim * sizeof(TYPE);
    size_t allocation_bytes = next_capacity * row_bytes;
    cuda_err_chk(cudaHostAlloc(
        reinterpret_cast<void **>(&staging_host_payload), allocation_bytes,
        cudaHostAllocDefault));
    cuda_err_chk(cudaMalloc(
        reinterpret_cast<void **>(&staging_device_payload), allocation_bytes));
    staging_payload_capacity = next_capacity;
    staging_source_rows.reserve(next_capacity);
    staging_output_positions.reserve(next_capacity);
    ++staging_reallocations;
  }
}

template <typename TYPE>
void BAM_Feature_Store<TYPE>::init_controllers(GIDS_Controllers GIDS_ctrl, uint32_t ps, uint64_t read_off, 
                                                uint64_t cache_size, uint64_t num_ele, uint64_t num_ssd,
                                                uint32_t replacement_policy,
                                                uint32_t cache_entry_bytes) {

  if (replacement_policy > 1) {
    throw std::invalid_argument("replacement_policy must be 0 (legacy) or 1 (fifo)");
  }
  if (cache_entry_bytes != 0) {
    if (num_ssd != 1) {
      throw std::invalid_argument(
          "split mixed-I/O cache entries currently require exactly one SSD");
    }
    if (cache_entry_bytes != 4096 || ps != 2 * cache_entry_bytes) {
      throw std::invalid_argument(
          "split mixed-I/O cache currently requires two adjacent 4-KiB entries per storage slot");
    }
    if ((read_off * static_cast<uint64_t>(ps)) % cache_entry_bytes != 0) {
      throw std::invalid_argument(
          "split mixed-I/O cache offset is not entry aligned");
    }
    if ((num_ele * sizeof(TYPE)) % ps != 0) {
      throw std::invalid_argument(
          "split mixed-I/O payload must contain complete two-entry slots");
    }
  }

  numElems = num_ele;
  read_offset = read_off;
  n_ctrls = num_ssd;
  this -> pageSize = ps;
  this -> cache_entry_size = cache_entry_bytes == 0 ? ps : cache_entry_bytes;
  this -> mixed_split_cache_entries = cache_entry_bytes != 0;
  this -> dim = ps / sizeof(TYPE);
  this -> total_access = 0; 

  ctrls = GIDS_ctrl.ctrls;

  std::cout << "Ctrl sizes: " << ctrls.size() << std::endl;
  uint64_t page_size = this->cache_entry_size;
  uint64_t n_pages = cache_size * 1024LL*1024/page_size; // cache里能够承载的条目数量 (1024*1024)
  if (mixed_split_cache_entries && n_pages < 2) {
    throw std::invalid_argument(
        "split mixed-I/O cache requires at least two 4-KiB entries");
  }
  this -> numPages = n_pages;

  std::cout << "n pages: " << (int)(this->numPages) <<std::endl;
  std::cout << "storage slot size: " << (int)(this->pageSize) << std::endl;
  std::cout << "cache entry size: " << (int)(this->cache_entry_size) << std::endl;
  std::cout << "num elements: " << this->numElems << std::endl; // 4096000000

  // page_cache是以range为单位的
  // page cache是 Bam software cache
  this -> h_pc = new page_cache_t(page_size, n_pages, cudaDevice, ctrls[0][0],
                                  (uint64_t)64, ctrls, replacement_policy,
                                  256, false, 0,
                                  mixed_split_cache_entries);
  std::cout << "GPU cache policy: " << (replacement_policy == 1 ? "fifo" : "legacy") << std::endl;
  page_cache_t *d_pc = (page_cache_t *)(h_pc->d_pc_ptr);


  uint64_t t_size = numElems * sizeof(TYPE);

  uint64_t cache_read_off = read_off * static_cast<uint64_t>(pageSize) /
                            cache_entry_size;
  this -> h_range = new range_t<TYPE>((uint64_t)0, (uint64_t)numElems, cache_read_off,
                              (uint64_t)(t_size / page_size), (uint64_t)0,
                              (uint64_t)page_size, h_pc, cudaDevice, 
			      //REPLICATE
			      STRIPE
			      );


  this -> d_range = (range_d_t<TYPE> *)h_range->d_range_ptr;

  this -> vr.push_back(nullptr);
  this -> vr[0] = h_range;
  this -> a = new array_t<TYPE>(numElems, 0, vr, cudaDevice);

  cudaMalloc(&d_cpu_access, sizeof(uint64_t));
  cudaMemset(d_cpu_access, 0 , sizeof(uint64_t));

  cudaMalloc(&d_gpu_access, sizeof(uint64_t));
  cudaMemset(d_gpu_access, 0 , sizeof(uint64_t));

  cudaMalloc(&d_mixed_io_stats,
             sizeof(unsigned long long int) * MIXED_IO_COUNTER_COUNT);
  cudaMemset(d_mixed_io_stats, 0,
             sizeof(unsigned long long int) * MIXED_IO_COUNTER_COUNT);

  return;
}





template <typename TYPE>
void  BAM_Feature_Store<TYPE>::set_window_buffering(uint64_t id_idx,  int64_t num_pages, int hash_off = 0){
	 uint64_t* idx_ptr = (uint64_t*) id_idx;
	 uint64_t page_size = pageSize;
	 set_window_buffering_kernel<TYPE><<<num_pages, 32>>>(a->d_array_ptr,idx_ptr, page_size, hash_off);
	 cuda_err_chk(cudaDeviceSynchronize())
}


template <typename TYPE>
void BAM_Feature_Store<TYPE>::print_stats_no_ctrl(){

  std::cout << "print stats: ";
  this->h_pc->print_reset_stats();
  std::cout << std::endl;

  std::cout << "print array reset: ";
  this->a->print_reset_stats();
  std::cout << std::endl;
}


template <typename TYPE>
void BAM_Feature_Store<TYPE>::print_stats(){
  std::vector<uint64_t> feature_access = get_feature_access_stats();
  std::cout << std::endl;
  std::cout << "Total feature access: " << std::dec << total_access << std::endl;
  std::cout << "cpu access total: " << std::dec << feature_access[0] << std::endl;
  std::cout << "gpu+ssd access total: " << std::dec << feature_access[1] << std::endl;
  std::cout << "Data access Time(ms): " << std::fixed << std::setprecision(5) << data_access_time << std::endl;

  std::cout << "print stats: ";
  this->h_pc->print_reset_stats();
  std::cout << std::endl;

  std::cout << "print array reset: ";
  this->a->print_reset_stats();
  std::cout << std::endl;

  for(int i = 0; i < n_ctrls; i++){
 	std::cout << "print ctrl reset " << i << ": ";
  	(this->ctrls[i])->print_reset_stats();
  	std::cout << std::endl;

  }
 
  std::cout << "Kernel Time: \t " << this->kernel_time << std::endl;

  //this->kernel_time = 0;
  std::cout << "Total Access: \t " << this->total_access << std::endl;
  this->total_access = 0;
}


template <typename TYPE>
void BAM_Feature_Store<TYPE>::read_feature_cpu_staged(
    uint64_t i_ptr, uint64_t i_index_ptr, uint64_t i_group_flags_ptr,
    int64_t num_index, int dim, int cache_dim, uint64_t key_off) {
  if (num_index <= 0) {
    return;
  }
  if (cpu_feature_path != 1 || !cpu_buffer_flag || seq_flag) {
    throw std::runtime_error("CPU staged read requested without staged cache setup");
  }
  if (dim <= 0 || cache_dim <= 0 ||
      static_cast<uint64_t>(cache_dim) != CPU_buffer.cpu_buffer_dim ||
      dim > cache_dim) {
    throw std::invalid_argument("CPU staged read received incompatible dimensions");
  }

  uint64_t requested_rows = static_cast<uint64_t>(num_index);
  ensure_cpu_staging_runtime(requested_rows, requested_rows);
  TYPE *tensor_ptr = reinterpret_cast<TYPE *>(i_ptr);
  int64_t *index_ptr = reinterpret_cast<int64_t *>(i_index_ptr);
  bool *group_ptr = reinterpret_cast<bool *>(i_group_flags_ptr);
  uint64_t b_size = blkSize;
  uint64_t n_warp = b_size / 32;
  uint64_t g_size = (requested_rows + n_warp - 1) / n_warp;

  if (mixed_io_enabled) {
    if (group_ptr == nullptr) {
      throw std::invalid_argument("mixed staged read requires group flags");
    }
    read_feature_kernel_mixed_ssd_only_with_cpu_map<TYPE>
        <<<g_size, b_size, 0, staging_ssd_stream>>>(
            a->d_array_ptr, d_range, tensor_ptr, index_ptr, group_ptr, dim,
            num_index, cache_dim, key_off, d_gpu_access, pageSize,
            mixed_min_transfer_bytes, cache_entry_size, mixed_full_mask,
            d_mixed_io_stats);
  } else {
    read_feature_kernel_ssd_only_with_cpu_map<TYPE>
        <<<g_size, b_size, 0, staging_ssd_stream>>>(
            a->d_array_ptr, d_range, tensor_ptr, index_ptr, dim, num_index,
            cache_dim, key_off, d_gpu_access, pageSize);
  }

  auto index_copy_start = Clock::now();
  cuda_err_chk(cudaMemcpyAsync(
      staging_host_indices, index_ptr, requested_rows * sizeof(int64_t),
      cudaMemcpyDeviceToHost, staging_copy_stream));
  cuda_err_chk(cudaStreamSynchronize(staging_copy_stream));
  auto index_copy_end = Clock::now();
  staging_index_copy_ns += static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          index_copy_end - index_copy_start).count());
  staging_index_bytes += requested_rows * sizeof(int64_t);

  auto gather_start = Clock::now();
  staging_source_rows.clear();
  staging_output_positions.clear();
  uint64_t feature_bytes = static_cast<uint64_t>(cache_dim) * sizeof(TYPE);
  uint64_t rows_per_page = pageSize / feature_bytes;
  for (uint64_t output_row = 0; output_row < requested_rows; ++output_row) {
    int64_t raw_index = staging_host_indices[output_row];
    if (raw_index < 0) {
      throw std::runtime_error("CPU staging received a negative storage row");
    }
    uint64_t storage_row = static_cast<uint64_t>(raw_index) + key_off;
    uint64_t storage_page = storage_row / rows_per_page;
    if (storage_page >= cpu_page_offsets.size()) {
      throw std::runtime_error("CPU staging storage row exceeds configured range");
    }
    uint32_t encoded_offset = cpu_page_offsets[storage_page];
    if ((encoded_offset & 0x1) == 1) {
      staging_output_positions.push_back(output_row);
      staging_source_rows.push_back(
          static_cast<uint64_t>(encoded_offset >> 1) +
          storage_row % rows_per_page);
    }
  }

  uint64_t cpu_rows = staging_source_rows.size();
  uint64_t *payload_positions =
      reinterpret_cast<uint64_t *>(staging_host_payload);
  TYPE *payload_features = reinterpret_cast<TYPE *>(
      staging_host_payload + cpu_rows * sizeof(uint64_t));
  if (cpu_rows != 0) {
    std::memcpy(payload_positions, staging_output_positions.data(),
                cpu_rows * sizeof(uint64_t));
    uint32_t worker_count = static_cast<uint32_t>(
        std::min<uint64_t>(cpu_staging_threads, cpu_rows));
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (uint32_t worker = 0; worker < worker_count; ++worker) {
      uint64_t begin = cpu_rows * worker / worker_count;
      uint64_t end = cpu_rows * (worker + 1) / worker_count;
      workers.emplace_back([=]() {
        for (uint64_t staged_row = begin; staged_row < end; ++staged_row) {
          const TYPE *source = CPU_buffer.cpu_buffer +
              staging_source_rows[staged_row] * cache_dim;
          TYPE *destination = payload_features + staged_row * dim;
          std::memcpy(destination, source,
                      static_cast<size_t>(dim) * sizeof(TYPE));
        }
      });
    }
    for (auto &worker : workers) {
      worker.join();
    }
  }
  auto gather_end = Clock::now();
  staging_host_gather_ns += static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          gather_end - gather_start).count());

  if (cpu_rows != 0) {
    size_t payload_bytes = cpu_rows *
        (sizeof(uint64_t) + static_cast<size_t>(dim) * sizeof(TYPE));
    cuda_err_chk(cudaEventRecord(staging_h2d_start, staging_copy_stream));
    cuda_err_chk(cudaMemcpyAsync(
        staging_device_payload, staging_host_payload, payload_bytes,
        cudaMemcpyHostToDevice, staging_copy_stream));
    cuda_err_chk(cudaEventRecord(staging_h2d_end, staging_copy_stream));
    scatter_staged_cpu_features<TYPE><<<cpu_rows, 256, 0,
                                        staging_copy_stream>>>(
        staging_device_payload, tensor_ptr, cpu_rows, dim);
    cuda_err_chk(cudaEventRecord(staging_scatter_end, staging_copy_stream));
    staging_payload_bytes += payload_bytes;
  }

  cuda_err_chk(cudaStreamSynchronize(staging_copy_stream));
  cuda_err_chk(cudaStreamSynchronize(staging_ssd_stream));
  if (cpu_rows != 0) {
    float h2d_ms = 0.0f;
    float scatter_ms = 0.0f;
    cuda_err_chk(cudaEventElapsedTime(&h2d_ms, staging_h2d_start,
                                      staging_h2d_end));
    cuda_err_chk(cudaEventElapsedTime(&scatter_ms, staging_h2d_end,
                                      staging_scatter_end));
    staging_h2d_ns += static_cast<uint64_t>(h2d_ms * 1000000.0f);
    staging_scatter_ns += static_cast<uint64_t>(scatter_ms * 1000000.0f);
  }
  cpu_access_count_total += cpu_rows;
  ++staging_batches;
  staging_cpu_rows += cpu_rows;
}





template <typename TYPE>
void BAM_Feature_Store<TYPE>::read_feature(uint64_t i_ptr, uint64_t i_index_ptr,
                                     int64_t num_index, int dim, int cache_dim,
                                     uint64_t key_off,
                                     uint64_t i_group_flags_ptr) {

  if (useful_io_enabled && (dim != cache_dim ||
      (dim * sizeof(TYPE) != 512 && dim * sizeof(TYPE) != 4096) ||
      cache_entry_size % (dim * sizeof(TYPE)) != 0))
      throw std::invalid_argument("Useful I/O requires aligned, complete 512/4096 B rows");

  // num_index表示这次读取的feature个数
  // dim是数据的维度
  TYPE *tensor_ptr = (TYPE *)i_ptr; // 指向数据的指针
  int64_t *index_ptr = (int64_t *)i_index_ptr; // 指向索引的指针
  bool *group_ptr = reinterpret_cast<bool *>(i_group_flags_ptr);

  uint64_t b_size = blkSize; // 每个CUDA块的线程数
  uint64_t n_warp = b_size / 32; // 每个块包含的warp数（一个warp通常包含32个线程）
  uint64_t g_size = (num_index+n_warp - 1) / n_warp; // 计算所需的网格大小，确保每个索引都被处理

  cuda_err_chk(cudaDeviceSynchronize());

  auto t1 = Clock::now();

  if(cpu_buffer_flag == false){
    if (mixed_io_enabled) {
      if (group_ptr == nullptr) {
        throw std::invalid_argument("mixed read requires group flags");
      }
      read_feature_kernel_mixed<TYPE><<<g_size, b_size>>>(
          a->d_array_ptr, tensor_ptr, index_ptr, group_ptr, dim, num_index,
          cache_dim, key_off, pageSize, mixed_min_transfer_bytes,
          mixed_full_mask, d_gpu_access, d_mixed_io_stats);
    } else {
      read_feature_kernel<TYPE><<<g_size, b_size>>>(
          a->d_array_ptr, tensor_ptr, index_ptr, dim, num_index, cache_dim,
          key_off, d_gpu_access);
    }
  }
  else if(cpu_feature_path == 1){
    read_feature_cpu_staged(i_ptr, i_index_ptr, i_group_flags_ptr, num_index,
                            dim, cache_dim, key_off);
  }
  else{
    if (mixed_io_enabled) {
      if (group_ptr == nullptr) {
        throw std::invalid_argument("mixed CPU-aware read requires group flags");
      }
      if (seq_flag) {
        throw std::runtime_error(
            "mixed I/O requires a populated page-mapped CPU cache");
      }
      read_feature_kernel_mixed_with_cpu_backing_memory<TYPE>
          <<<g_size, b_size>>>(
              a->d_array_ptr, d_range, tensor_ptr, index_ptr, group_ptr, dim,
              num_index, cache_dim, CPU_buffer, d_cpu_access, key_off,
              d_gpu_access, pageSize, mixed_min_transfer_bytes,
              cache_entry_size, mixed_full_mask, d_mixed_io_stats);
    } else {
      read_feature_kernel_with_cpu_backing_memory<TYPE><<<g_size, b_size>>>(
          a->d_array_ptr, d_range, tensor_ptr, index_ptr, dim, num_index,
          cache_dim, CPU_buffer, seq_flag, d_cpu_access, key_off,
          d_gpu_access, pageSize);
    }
  }

  cuda_err_chk(cudaDeviceSynchronize());
  auto t3 = Clock::now();

  auto us = std::chrono::duration_cast<std::chrono::microseconds>( t3 - t1); // Microsecond (as int)
  // auto ms = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1); // Microsecond (as int)

  const float ms_fractional = static_cast<float>(us.count()) / 1000; // Milliseconds (as float)
  data_access_time += ms_fractional;

  //std::cout << "Kernel Time once: \t " << ms_fractional << std::endl;

  kernel_time += ms_fractional;
  total_access += num_index;

  return;
}

template <typename TYPE>
void BAM_Feature_Store<TYPE>::get_io_stat(uint64_t i_ptr){
  uint64_t *tensor_ptr = (uint64_t *)i_ptr; // 指向数据的指针
  
  range_d_t<TYPE> range_d;
  this->a->get_io_stat(range_d);

  tensor_ptr[0] = range_d.access_cnt;
  tensor_ptr[1] = range_d.miss_cnt;
  tensor_ptr[2] = range_d.hit_cnt;
}

template <typename TYPE>
void BAM_Feature_Store<TYPE>::begin_useful_io_region() {
    if (!h_pc) throw std::runtime_error("Controllers not initialized");
    cuda_err_chk(cudaDeviceSynchronize());
    auto device = h_pc->get_device_io_stats();
    if (!device[0] || device[1]) throw std::runtime_error("Need enabled, quiescent device counters");
    auto& cache = h_pc->pdt;
    cuda_err_chk(cudaMemset(cache.useful_masks, 255, cache.n_pages * sizeof(uint64_t)));
    uint64_t values[6] = {1, ++useful_region_id, 0, 0, 0, 0};
    cuda_err_chk(cudaMemcpy(cache.useful_counters, values, sizeof(values), cudaMemcpyHostToDevice));
    useful_io_enabled = true;
}
template <typename TYPE>
std::vector<uint64_t> BAM_Feature_Store<TYPE>::get_useful_io_stats() {
    if (!h_pc) throw std::runtime_error("Controllers not initialized");
    cuda_err_chk(cudaDeviceSynchronize());
    std::vector<uint64_t> result(6, 0);
    cuda_err_chk(cudaMemcpy(result.data(), h_pc->pdt.useful_counters,
                           6 * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    return result;
}

template <typename TYPE>
std::vector<uint64_t> BAM_Feature_Store<TYPE>::get_gpu_cache_stats(){
  return this->h_pc->get_cache_stats();
}

template <typename TYPE>
std::vector<uint64_t> BAM_Feature_Store<TYPE>::get_feature_access_stats(){
  cuda_err_chk(cudaDeviceSynchronize());
  cudaMemcpy(&cpu_access_count, d_cpu_access, sizeof(unsigned long long int),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(&gpu_access_count, d_gpu_access, sizeof(unsigned long long int),
             cudaMemcpyDeviceToHost);
  return {cpu_access_count_total + cpu_access_count, gpu_access_count};
}

template <typename TYPE>
std::vector<uint64_t> BAM_Feature_Store<TYPE>::get_mixed_io_stats(){
  cuda_err_chk(cudaDeviceSynchronize());
  std::vector<uint64_t> values(MIXED_IO_COUNTER_COUNT, 0);
  cuda_err_chk(cudaMemcpy(values.data(), d_mixed_io_stats,
                          sizeof(unsigned long long int) *
                              MIXED_IO_COUNTER_COUNT,
                          cudaMemcpyDeviceToHost));
  values.insert(values.begin(), mixed_io_enabled ? 1ULL : 0ULL);
  values.push_back(mixed_feature_row_bytes);
  values.push_back(mixed_min_transfer_bytes);
  values.push_back(mixed_group_size);
  values.push_back(mixed_subrow_count);
  values.push_back(pageSize);
  values.push_back(mixed_full_mask);
  values.push_back(cache_entry_size);
  return values;
}

template <typename TYPE>
std::vector<uint64_t> BAM_Feature_Store<TYPE>::get_mixed_io_geometry(){
  return {mixed_feature_row_bytes, mixed_min_transfer_bytes, mixed_group_size,
          mixed_subrow_count, pageSize, mixed_full_mask, cache_entry_size};
}

template <typename TYPE>
std::string BAM_Feature_Store<TYPE>::get_mixed_io_geometry_hash(){
  return mixed_geometry_sha256;
}

template <typename TYPE>
std::vector<uint64_t> BAM_Feature_Store<TYPE>::get_device_io_stats(){
  if (h_pc == nullptr) {
    throw std::runtime_error(
        "device I/O statistics require initialized controllers");
  }
  return h_pc->get_device_io_stats();
}

template <typename TYPE>
void BAM_Feature_Store<TYPE>::reset_device_io_stats(){
  if (h_pc == nullptr) {
    throw std::runtime_error(
        "device I/O statistics require initialized controllers");
  }
  h_pc->reset_device_io_stats();
}

template <typename TYPE>
std::vector<uint64_t> BAM_Feature_Store<TYPE>::get_cpu_staging_stats(){
  cuda_err_chk(cudaDeviceSynchronize());
  return {
      cpu_feature_path,
      cpu_staging_reserve_rows,
      cpu_staging_threads,
      staging_batches,
      staging_cpu_rows,
      staging_payload_bytes,
      staging_index_bytes,
      staging_reallocations,
      staging_index_copy_ns,
      staging_host_gather_ns,
      staging_h2d_ns,
      staging_scatter_ns,
  };
}

template <typename TYPE>
void BAM_Feature_Store<TYPE>::read_feature_hetero(int num_iter, const std::vector<uint64_t>&  i_ptr_list, const std::vector<uint64_t>&  i_index_ptr_list,
                                     const std::vector<uint64_t>&   num_index, int dim, int cache_dim, const std::vector<uint64_t>& key_off) {

  if (cpu_feature_path == 1) {
    throw std::runtime_error("CPU staging does not support heterogeneous reads");
  }

  cudaStream_t streams[num_iter];
  for (int i = 0; i < num_iter; i++) {
      cudaStreamCreate(&streams[i]);
  }

  cuda_err_chk(cudaDeviceSynchronize());
  auto t1 = Clock::now();

  for(uint64_t i = 0;  i < num_iter; i++){
    uint64_t i_ptr = i_ptr_list[i];
    uint64_t    i_index_ptr =  i_index_ptr_list[i];  
    TYPE *tensor_ptr = (TYPE *) i_ptr;
    int64_t *index_ptr = (int64_t *)i_index_ptr;

    uint64_t b_size = blkSize;
    uint64_t n_warp = b_size / 32;
    uint64_t g_size = (num_index[i]+n_warp - 1) / n_warp;

    if(cpu_buffer_flag == false){
      read_feature_kernel<TYPE><<<g_size, b_size, 0, streams[i] >>>(a->d_array_ptr, tensor_ptr,
                                                    index_ptr, dim, num_index[i], cache_dim, key_off[i], d_gpu_access);
    }
    else{
      read_feature_kernel_with_cpu_backing_memory<<<g_size, b_size, 0, streams[i] >>>(a->d_array_ptr, d_range ,tensor_ptr,
                                                    index_ptr, dim, num_index[i], cache_dim, CPU_buffer, seq_flag, 
                                                    d_cpu_access,  key_off[i], d_gpu_access,
                                                    pageSize);
    }
    total_access += num_index[i];
  }

  for (int i = 0; i < num_iter; i++) {
    cudaStreamSynchronize(streams[i]);
  }

  cuda_err_chk(cudaDeviceSynchronize());
  cuda_err_chk(cudaDeviceSynchronize());

  auto t2 = Clock::now();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(
      t2 - t1); // Microsecond (as int)
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      t2 - t1); // Microsecond (as int)
  const float ms_fractional =
      static_cast<float>(us.count()) / 1000; // Milliseconds (as float)

  //std::cout << "Duration = " << us.count() << "µs (" << ms_fractional << "ms)"
    //        << std::endl;
 
  kernel_time += ms_fractional;

  for (int i = 0; i < num_iter; i++) {
      cudaStreamDestroy(streams[i]);
  }
  

  return;
}


// 同时读多个batch的feature
template <typename TYPE>
void BAM_Feature_Store<TYPE>::read_feature_merged(int num_iter, const std::vector<uint64_t>&  i_ptr_list, const std::vector<uint64_t>&  i_index_ptr_list,
                                     const std::vector<uint64_t>&   num_index, int dim, int cache_dim=1024) {

  if (cpu_feature_path == 1) {
    throw std::runtime_error("CPU staging does not support merged accumulator reads");
  }

  cudaStream_t streams[num_iter];
  for (int i = 0; i < num_iter; i++) {
      cudaStreamCreate(&streams[i]);
  }
  cuda_err_chk(cudaDeviceSynchronize());
  auto t1 = Clock::now();

  for(uint64_t i = 0;  i < num_iter; i++){
    uint64_t i_ptr = i_ptr_list[i];
    uint64_t    i_index_ptr =  i_index_ptr_list[i];         
    TYPE *tensor_ptr = (TYPE *) i_ptr;
    int64_t *index_ptr = (int64_t *)i_index_ptr;

    uint64_t b_size = blkSize;
    uint64_t n_warp = b_size / 32;
    uint64_t g_size = (num_index[i]+n_warp - 1) / n_warp;
    

    if(cpu_buffer_flag == false){
      read_feature_kernel<TYPE><<<g_size, b_size, 0, streams[i] >>>(a->d_array_ptr, tensor_ptr,
                                                    index_ptr, dim, num_index[i], cache_dim, 0, d_gpu_access);
    }
    else{
      read_feature_kernel_with_cpu_backing_memory<<<g_size, b_size, 0, streams[i] >>>(a->d_array_ptr, d_range ,tensor_ptr,
                                                    index_ptr, dim, num_index[i], cache_dim, CPU_buffer, seq_flag, 
                                                    d_cpu_access, 0, d_gpu_access,
                                                    pageSize);
    }
    total_access += num_index[i];
  }

  for (int i = 0; i < num_iter; i++) {
    cudaStreamSynchronize(streams[i]);
  }

  cuda_err_chk(cudaDeviceSynchronize());
  cuda_err_chk(cudaDeviceSynchronize());

  auto t2 = Clock::now();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(
      t2 - t1); // Microsecond (as int)
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      t2 - t1); // Microsecond (as int)
  const float ms_fractional =
      static_cast<float>(us.count()) / 1000; // Milliseconds (as float)

  data_access_time += ms_fractional;

  //std::cout << "Duration = " << us.count() << "µs (" << ms_fractional << "ms)"
    //        << std::endl;
 
  kernel_time += ms_fractional;

  for (int i = 0; i < num_iter; i++) {
      cudaStreamDestroy(streams[i]);
  }
  return;
}





template <typename TYPE>
void BAM_Feature_Store<TYPE>::read_feature_merged_hetero(int num_iter, const std::vector<uint64_t>&  i_ptr_list, const std::vector<uint64_t>&  i_index_ptr_list,
                                     const std::vector<uint64_t>&   num_index, int dim, int cache_dim, const std::vector<uint64_t>& key_off) {

  if (cpu_feature_path == 1) {
    throw std::runtime_error("CPU staging does not support merged heterogeneous reads");
  }

  cudaStream_t streams[num_iter];
  for (int i = 0; i < num_iter; i++) {
      cudaStreamCreate(&streams[i]);
  }

  cuda_err_chk(cudaDeviceSynchronize());
  auto t1 = Clock::now();

  for(uint64_t i = 0;  i < num_iter; i++){
    uint64_t i_ptr = i_ptr_list[i];
    uint64_t    i_index_ptr =  i_index_ptr_list[i];         
    TYPE *tensor_ptr = (TYPE *) i_ptr;
    int64_t *index_ptr = (int64_t *)i_index_ptr;

    uint64_t b_size = blkSize;
    uint64_t n_warp = b_size / 32;
    uint64_t g_size = (num_index[i]+n_warp - 1) / n_warp;
    

    if(cpu_buffer_flag == false){
      read_feature_kernel<TYPE><<<g_size, b_size, 0, streams[i] >>>(a->d_array_ptr, tensor_ptr,
                                                    index_ptr, dim, num_index[i], cache_dim, key_off[i], d_gpu_access);
    }
    else{
      read_feature_kernel_with_cpu_backing_memory<<<g_size, b_size, 0, streams[i] >>>(a->d_array_ptr, d_range ,tensor_ptr,
                                                    index_ptr, dim, num_index[i], cache_dim, CPU_buffer, seq_flag, 
                                                    d_cpu_access, key_off[i], d_gpu_access,
                                                    pageSize);
    }
    total_access += num_index[i];
  }

  for (int i = 0; i < num_iter; i++) {
    cudaStreamSynchronize(streams[i]);
  }

  cuda_err_chk(cudaDeviceSynchronize());
  cuda_err_chk(cudaDeviceSynchronize());

  auto t2 = Clock::now();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(
      t2 - t1); // Microsecond (as int)
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      t2 - t1); // Microsecond (as int)
  const float ms_fractional =
      static_cast<float>(us.count()) / 1000; // Milliseconds (as float)

  //std::cout << "Duration = " << us.count() << "µs (" << ms_fractional << "ms)"
    //        << std::endl;
 
  kernel_time += ms_fractional;

  for (int i = 0; i < num_iter; i++) {
      cudaStreamDestroy(streams[i]);
  }
  return;
}







template <typename TYPE>
void  BAM_Feature_Store<TYPE>::store_tensor(uint64_t tensor_ptr, uint64_t num, uint64_t offset){


//__global__ void write_feature_kernel2(Controller** ctrls, page_cache_d_t* pc, array_d_t<T> *dr, T* in_tensor_ptr, uint64_t dim, uint32_t num_ctrls) {
	TYPE* t_ptr = (TYPE*) tensor_ptr;
	page_cache_d_t* d_pc = (page_cache_d_t*) (h_pc -> d_pc_ptr);
	size_t b_size = 128;
	printf("num of writing node data: %llu dim: %llu\n", num, dim);
	write_feature_kernel2<TYPE><<<num, b_size>>>(h_pc->pdt.d_ctrls, d_pc, a->d_array_ptr, t_ptr, dim,  n_ctrls, offset/sizeof(TYPE));
	cuda_err_chk(cudaDeviceSynchronize());
  	h_pc->flush_cache();
   	cuda_err_chk(cudaDeviceSynchronize());
/*
  uint64_t s_offset = 0; 
  
  uint64_t total_cache_size = (pageSize * numPages);
  uint64_t total_tensor_size = (sizeof(TYPE) * num);
  uint64_t num_pages = total_tensor_size / pageSize;

  uint32_t n_tsteps = ceil((float)(total_tensor_size)/(float)total_cache_size);  
  printf("total iter: %llu\n", (unsigned long long) n_tsteps);
  TYPE* t_ptr = (TYPE*) tensor_ptr;
  
  page_cache_d_t* d_pc = (page_cache_d_t*) (h_pc -> d_pc_ptr);
  size_t b_size = 128;
  size_t g_size = (((total_tensor_size + pageSize -1) / pageSize)  + b_size - 1)/b_size;

  for (uint32_t cstep =0; cstep < n_tsteps; cstep++) {
    uint64_t cpysize = std::min(total_cache_size, (total_tensor_size-s_offset));


   // printf("first ele:%f\n", t_ptr[0]);
    cuda_err_chk(cudaMemcpy(h_pc->pdt.base_addr, t_ptr+s_offset+offset, cpysize, cudaMemcpyHostToDevice));
    printf("g size: %i num: %llu\n", g_size, num);
    write_feature_kernel<TYPE><<<100, b_size>>>(h_pc->pdt.d_ctrls, d_pc, a->d_array_ptr, t_ptr, num_pages, pageSize, offset, s_offset, n_ctrls);
    cuda_err_chk(cudaDeviceSynchronize());
    
  // printf("CALLLING FLUSH\n");
  // h_pc->flush_cache();
    //cuda_err_chk(cudaDeviceSynchronize());
    s_offset = s_offset + cpysize; 

  }
*/
}


template <typename TYPE>
void  BAM_Feature_Store<TYPE>::flush_cache(){
  h_pc->flush_cache();
  cuda_err_chk(cudaDeviceSynchronize());
}



template <typename TYPE>
void  BAM_Feature_Store<TYPE>::set_cpu_buffer(uint64_t idx_buffer, int num){
  // GIDS_Loader.set_cpu_buffer(pr_ten, num_pinned_nodes)
  int bsize = 1024;
  int grid = (num + bsize - 1) / bsize;

  // idx_buffer 本质上是一个指向 uint64_t 数组的地址
  uint64_t* idx_ptr = (uint64_t* ) idx_buffer;
  uint64_t feature_dim = CPU_buffer.cpu_buffer_dim;
  uint64_t feature_bytes = feature_dim * sizeof(TYPE);
  if(feature_dim == 0 || pageSize % feature_bytes != 0){
    throw std::runtime_error("CPU feature rows must divide the storage page size exactly");
  }
  set_cpu_buffer_kernel<TYPE><<<grid,bsize>>>(
      d_range, idx_ptr, num, pageSize, feature_dim, cache_entry_size);
  cuda_err_chk(cudaDeviceSynchronize());
  
  set_cpu_buffer_data_kernel<TYPE><<<num,32>>>(a->d_array_ptr, CPU_buffer.device_cpu_buffer, idx_ptr, feature_dim, num);
  cuda_err_chk(cudaDeviceSynchronize());

  std::vector<uint64_t> cached_rows(num);
  if (num != 0) {
    cuda_err_chk(cudaMemcpy(cached_rows.data(), idx_ptr,
                            static_cast<size_t>(num) * sizeof(uint64_t),
                            cudaMemcpyDeviceToHost));
  }
  uint64_t rows_per_page = pageSize / feature_bytes;
  uint64_t page_count =
      (numElems * sizeof(TYPE) + pageSize - 1) / pageSize;
  cpu_page_offsets.assign(page_count, 0);
  for (int cached_index = 0; cached_index < num; ++cached_index) {
    uint64_t storage_row = cached_rows[cached_index];
    if (storage_row % rows_per_page == 0) {
      uint64_t storage_page = storage_row / rows_per_page;
      if (storage_page >= cpu_page_offsets.size()) {
        throw std::runtime_error("CPU cache row exceeds configured storage range");
      }
      if (static_cast<uint64_t>(cached_index) >
          (static_cast<uint64_t>(UINT32_MAX) >> 1)) {
        throw std::runtime_error("CPU cache offset exceeds Phase 8C encoding");
      }
      cpu_page_offsets[storage_page] =
          (static_cast<uint32_t>(cached_index) << 1) | 0x1;
    }
  }

  // 1) 每个节点 copy dim 个 TYPE；TYPE 一般是 float (4 字节) 或 double (8 字节)
  // 2) num 表示一共拷了多少个节点
  size_t bytes_copied = (size_t)num * (size_t)feature_dim * sizeof(TYPE);

    // 3) 计算 GB 数 （1 GB = 1024^3 字节）
  double gb_copied = bytes_copied / (1024.0 * 1024.0 * 1024.0);

    // 4) 打印到标准输出
  std::cout << "===== Debug: 已向 CPU 页锁内存 写入 "
            << std::dec << bytes_copied << " 字节"
            << " （约 " << std::fixed << std::setprecision(3)
            << gb_copied << " GB） ====="
            << std::endl;

  seq_flag = false;


}



template <typename TYPE>
void  BAM_Feature_Store<TYPE>::set_offsets(uint64_t in_off, uint64_t index_off, uint64_t data_off){

 offset_array = new uint64_t[3];
    printf("set offset: in_off: %llu index_off: %llu data_off: %llu offset_ptr:%llu\n", in_off, index_off, data_off, (uint64_t) offset_array);

  offset_array[0] = (in_off);
  offset_array[1] = (index_off);
  offset_array[2] = (data_off);

}


template <typename TYPE>
uint64_t BAM_Feature_Store<TYPE>::get_offset_array(){
  return ((uint64_t) offset_array);
}

template <typename TYPE>
uint64_t BAM_Feature_Store<TYPE>::get_array_ptr(){
	return ((uint64_t) (a->d_array_ptr));
}


template <typename TYPE>
void  BAM_Feature_Store<TYPE>::read_tensor(uint64_t num, uint64_t offset){
  printf("offset:%llu\n", (unsigned long long) offset);
  seq_read_kernel<TYPE><<<1, 1>>>(a->d_array_ptr, num, offset);
  cuda_err_chk(cudaDeviceSynchronize());

}


template <typename TYPE>
unsigned long long int BAM_Feature_Store<TYPE>::get_cpu_access_count(){
	cuda_err_chk(cudaDeviceSynchronize());
	cudaMemcpy(&cpu_access_count, d_cpu_access, sizeof(unsigned long long int),
	           cudaMemcpyDeviceToHost);
	return cpu_access_count;
}

template <typename TYPE>
unsigned long long int BAM_Feature_Store<TYPE>::get_cpu_access_count_total(){
	return get_feature_access_stats()[0];
}

template <typename TYPE>
unsigned long long int BAM_Feature_Store<TYPE>::get_gpu_access_count_total(){
	return get_feature_access_stats()[1];
}

template <typename TYPE>
void BAM_Feature_Store<TYPE>::flush_cpu_access_count(){
  cpu_access_count_total += get_cpu_access_count();
	cpu_access_count = 0;
  cudaMemset(d_cpu_access, 0 , sizeof(unsigned long long int));
}

template <typename T>
BAM_Feature_Store<T> create_BAM_Feature_Store() {
    return BAM_Feature_Store<T>();
}



// GPU-only protocol gate fixture: no controller, page allocation, or SSD I/O.
__global__ void full_resident_probe_fixture(data_page_t* pages, uint32_t* out) {
  if (threadIdx.x || blockIdx.x) return;
  range_d_t<float> range;
  range.pages = pages;
  range.cache.mixed_full_mask = 1;
  const uint32_t states[6] = {VALID, VALID | BUSY, INVALID, VALID, VALID | 4, VALID | CNT_MASK};
  const uint64_t masks[6] = {1, 1, 1, 0, 1, 1};
  for (int i = 0; i < 6; ++i) {
    pages[i].state.store(states[i], simt::memory_order_release);
    pages[i].valid_subrows = masks[i];
    bool hit = full_try_pin_resident(&range, i);
    out[i*3] = hit;
    out[i*3+1] = pages[i].state.load(simt::memory_order_acquire);
    if (hit) pages[i].state.fetch_sub(1, simt::memory_order_release);
    out[i*3+2] = pages[i].state.load(simt::memory_order_acquire);
  }
}
std::vector<uint32_t> full_resident_probe_test() {
  data_page_t* pages = nullptr; uint32_t* output = nullptr;
  auto check = [](cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); };
  check(cudaMalloc(&pages, sizeof(data_page_t)*6));
  try {
    check(cudaMalloc(&output, sizeof(uint32_t)*18));
    full_resident_probe_fixture<<<1,32>>>(pages, output);
    check(cudaGetLastError()); check(cudaDeviceSynchronize());
    std::vector<uint32_t> result(18);
    check(cudaMemcpy(result.data(), output, sizeof(uint32_t)*18, cudaMemcpyDeviceToHost));
    cudaFree(output); cudaFree(pages); return result;
  } catch (...) { cudaFree(output); cudaFree(pages); throw; }
}

// Uses the exact production bitmap/fill/consume helpers; no NVMe controller.
__global__ void useful_counter_probe_kernel(page_cache_d_t cache, uint64_t op,
                                           uint64_t slot, uint64_t offset, uint64_t bytes) {
    if (op == 1 || op == 2) cache.useful_fill(slot, offset, bytes, op == 1);
    else cache.useful_consume(slot, offset, bytes);
}
std::vector<uint64_t> useful_counter_probe(
    const std::vector<std::vector<uint64_t>>& events, uint64_t slots, uint64_t page_bytes) {
    if (!slots || slots > 1024 || page_bytes > 16384 || !page_bytes || page_bytes % 512)
        throw std::invalid_argument("Bad probe geometry");
    // Validate before allocating so invalid probes do not leak device memory.
    for (const auto& e : events) {
        if (e.size() != 4 || e[0] > 5 || e[1] >= slots || e[2] % 512 ||
            e[3] % 512 || !e[3] || e[2] >= page_bytes || e[3] > page_bytes-e[2])
            throw std::invalid_argument("Bad probe event");
    }
    page_cache_d_t cache = {};
    cache.n_pages = slots; cache.page_size = page_bytes;
    auto masks = createBuffer(slots * sizeof(uint64_t), 0);
    auto counts = createBuffer(6 * sizeof(uint64_t), 0);
    cache.useful_masks = (unsigned long long*)masks.get();
    cache.useful_counters = (unsigned long long*)counts.get();
    cuda_err_chk(cudaMemset(cache.useful_masks, 255, slots * sizeof(uint64_t)));
    uint64_t region = 1;
    uint64_t values[6] = {1, region, 0, 0, 0, 0};
    cuda_err_chk(cudaMemcpy(cache.useful_counters, values, sizeof(values), cudaMemcpyHostToDevice));
    std::vector<uint64_t> result;
    for (const auto& e : events) {
        if (e[0] == 0) {
            cuda_err_chk(cudaMemset(cache.useful_masks, 255, slots * sizeof(uint64_t)));
            uint64_t reset[6] = {1, ++region, 0, 0, 0, 0};
            cuda_err_chk(cudaMemcpy(cache.useful_counters, reset, sizeof(reset), cudaMemcpyHostToDevice));
        } else if (e[0] == 5) {
            cuda_err_chk(cudaMemset(cache.useful_counters, 0, sizeof(uint64_t)));
        } else {
            useful_counter_probe_kernel<<<e[0] == 4 ? 4 : 1, e[0] == 4 ? 256 : 1>>>(
                cache, e[0], e[1], e[2], e[3]);
            cuda_err_chk(cudaGetLastError());
            cuda_err_chk(cudaDeviceSynchronize());
        }
        cuda_err_chk(cudaMemcpy(values, cache.useful_counters, sizeof(values), cudaMemcpyDeviceToHost));
        result.insert(result.end(), values, values+6);
    }
    return result;
}

PYBIND11_MODULE(BAM_Feature_Store, m) {
  m.def("full_resident_probe_test", &full_resident_probe_test);
  m.def("useful_counter_probe", &useful_counter_probe);
    namespace py = pybind11;
    m.def("probe_mixed_io_geometry_cuda", &probe_mixed_io_geometry_cuda,
          py::arg("storage_rows"), py::arg("group_flags"),
          py::arg("feature_row_bytes"), py::arg("minimum_transfer_bytes"),
          py::arg("group_size"), py::arg("cache_slot_bytes"),
          py::arg("full_valid_mask"), py::arg("payload_offset_bytes"));
  m.doc() = "Python bindings for an example library";

  //py::class_<BAM_Feature_Store<>, std::unique_ptr<BAM_Feature_Store<float>, py::nodelete>>(m, "BAM_Feature_Store")
    py::class_<BAM_Feature_Store<float>>(m, "BAM_Feature_Store_float")
      .def(py::init<>())
      .def("init_controllers", &BAM_Feature_Store<float>::init_controllers,
           py::arg("controllers"), py::arg("page_size"), py::arg("read_offset"),
           py::arg("cache_size"), py::arg("num_elements"), py::arg("num_ssd"),
           py::arg("replacement_policy") = 0,
           py::arg("cache_entry_bytes") = 0)
      .def("get_io_stat", &BAM_Feature_Store<float>::get_io_stat)
      .def("get_gpu_cache_stats", &BAM_Feature_Store<float>::get_gpu_cache_stats)
      .def("get_feature_access_stats", &BAM_Feature_Store<float>::get_feature_access_stats)
      .def("set_mixed_io", &BAM_Feature_Store<float>::set_mixed_io)
      .def("set_mixed_io_geometry", &BAM_Feature_Store<float>::set_mixed_io_geometry)
      .def("get_mixed_io_stats", &BAM_Feature_Store<float>::get_mixed_io_stats)
      .def("get_mixed_io_geometry", &BAM_Feature_Store<float>::get_mixed_io_geometry)
      .def("get_mixed_io_geometry_hash", &BAM_Feature_Store<float>::get_mixed_io_geometry_hash)
      .def("set_device_io_stats", &BAM_Feature_Store<float>::set_device_io_stats)
      .def("get_device_io_stats", &BAM_Feature_Store<float>::get_device_io_stats)
      .def("reset_device_io_stats", &BAM_Feature_Store<float>::reset_device_io_stats)
      .def("begin_useful_io_region", &BAM_Feature_Store<float>::begin_useful_io_region)
      .def("get_useful_io_stats", &BAM_Feature_Store<float>::get_useful_io_stats)
      .def("set_cpu_feature_path", &BAM_Feature_Store<float>::set_cpu_feature_path,
           py::arg("path"), py::arg("reserve_rows") = 131072,
           py::arg("gather_threads") = 8)
      .def("get_cpu_staging_stats", &BAM_Feature_Store<float>::get_cpu_staging_stats)
      .def("read_feature", &BAM_Feature_Store<float>::read_feature,
           py::arg("tensor_ptr"), py::arg("index_ptr"),
           py::arg("num_index"), py::arg("dim"), py::arg("cache_dim"),
           py::arg("key_off"), py::arg("group_flags_ptr") = 0)
      .def("read_feature_hetero", &BAM_Feature_Store<float>::read_feature_hetero)

      .def("read_feature_merged_hetero", &BAM_Feature_Store<float>::read_feature_merged_hetero)
      .def("read_feature_merged", &BAM_Feature_Store<float>::read_feature_merged)
      .def("set_window_buffering", &BAM_Feature_Store<float>::set_window_buffering)
      .def("cpu_backing_buffer", &BAM_Feature_Store<float>::cpu_backing_buffer)
      .def("set_cpu_buffer", &BAM_Feature_Store<float>::set_cpu_buffer)

      .def("flush_cache", &BAM_Feature_Store<float>::flush_cache)
      .def("store_tensor",  &BAM_Feature_Store<float>::store_tensor)
      .def("read_tensor",  &BAM_Feature_Store<float>::read_tensor)

      .def("get_array_ptr", &BAM_Feature_Store<float>::get_array_ptr)
      .def("get_offset_array", &BAM_Feature_Store<float>::get_offset_array)
      .def("set_offsets", &BAM_Feature_Store<float>::set_offsets)
      .def("get_cpu_access_count", &BAM_Feature_Store<float>::get_cpu_access_count)
      .def("get_cpu_access_count_total", &BAM_Feature_Store<float>::get_cpu_access_count_total)
      .def("get_gpu_access_count_total", &BAM_Feature_Store<float>::get_gpu_access_count_total)
      .def("flush_cpu_access_count", &BAM_Feature_Store<float>::flush_cpu_access_count)

      .def("print_stats", &BAM_Feature_Store<float>::print_stats);



    py::class_<BAM_Feature_Store<int64_t>>(m, "BAM_Feature_Store_long")
      .def(py::init<>())
      .def("init_controllers", &BAM_Feature_Store<int64_t>::init_controllers,
           py::arg("controllers"), py::arg("page_size"), py::arg("read_offset"),
           py::arg("cache_size"), py::arg("num_elements"), py::arg("num_ssd"),
           py::arg("replacement_policy") = 0,
           py::arg("cache_entry_bytes") = 0)
      .def("get_io_stat", &BAM_Feature_Store<int64_t>::get_io_stat)
      .def("get_gpu_cache_stats", &BAM_Feature_Store<int64_t>::get_gpu_cache_stats)
      .def("get_feature_access_stats", &BAM_Feature_Store<int64_t>::get_feature_access_stats)
      .def("set_mixed_io", &BAM_Feature_Store<int64_t>::set_mixed_io)
      .def("set_mixed_io_geometry", &BAM_Feature_Store<int64_t>::set_mixed_io_geometry)
      .def("get_mixed_io_stats", &BAM_Feature_Store<int64_t>::get_mixed_io_stats)
      .def("get_mixed_io_geometry", &BAM_Feature_Store<int64_t>::get_mixed_io_geometry)
      .def("get_mixed_io_geometry_hash", &BAM_Feature_Store<int64_t>::get_mixed_io_geometry_hash)
      .def("set_device_io_stats", &BAM_Feature_Store<int64_t>::set_device_io_stats)
      .def("get_device_io_stats", &BAM_Feature_Store<int64_t>::get_device_io_stats)
      .def("reset_device_io_stats", &BAM_Feature_Store<int64_t>::reset_device_io_stats)
      .def("begin_useful_io_region", &BAM_Feature_Store<int64_t>::begin_useful_io_region)
      .def("get_useful_io_stats", &BAM_Feature_Store<int64_t>::get_useful_io_stats)
      .def("set_cpu_feature_path", &BAM_Feature_Store<int64_t>::set_cpu_feature_path,
           py::arg("path"), py::arg("reserve_rows") = 131072,
           py::arg("gather_threads") = 8)
      .def("get_cpu_staging_stats", &BAM_Feature_Store<int64_t>::get_cpu_staging_stats)
      .def("read_feature", &BAM_Feature_Store<int64_t>::read_feature,
           py::arg("tensor_ptr"), py::arg("index_ptr"),
           py::arg("num_index"), py::arg("dim"), py::arg("cache_dim"),
           py::arg("key_off"), py::arg("group_flags_ptr") = 0)
      .def("read_feature_hetero", &BAM_Feature_Store<int64_t>::read_feature_hetero)

      .def("read_feature_merged", &BAM_Feature_Store<int64_t>::read_feature_merged)
      .def("read_feature_merged_hetero", &BAM_Feature_Store<int64_t>::read_feature_merged_hetero)


      .def("set_window_buffering", &BAM_Feature_Store<int64_t>::set_window_buffering)
      .def("cpu_backing_buffer", &BAM_Feature_Store<int64_t>::cpu_backing_buffer)
      .def("set_cpu_buffer", &BAM_Feature_Store<int64_t>::set_cpu_buffer)

      .def("flush_cache", &BAM_Feature_Store<int64_t>::flush_cache)
      .def("store_tensor",  &BAM_Feature_Store<int64_t>::store_tensor)
      .def("read_tensor",  &BAM_Feature_Store<int64_t>::read_tensor)

      .def("get_array_ptr", &BAM_Feature_Store<int64_t>::get_array_ptr)
      .def("get_offset_array", &BAM_Feature_Store<int64_t>::get_offset_array)
      .def("set_offsets", &BAM_Feature_Store<int64_t>::set_offsets)
      .def("get_cpu_access_count", &BAM_Feature_Store<int64_t>::get_cpu_access_count)
      .def("get_cpu_access_count_total", &BAM_Feature_Store<int64_t>::get_cpu_access_count_total)
      .def("get_gpu_access_count_total", &BAM_Feature_Store<int64_t>::get_gpu_access_count_total)
      .def("flush_cpu_access_count", &BAM_Feature_Store<int64_t>::flush_cpu_access_count)


      .def("print_stats", &BAM_Feature_Store<int64_t>::print_stats);




      py::class_<GIDS_Controllers>(m, "GIDS_Controllers")
      .def(py::init<>())
      .def("init_GIDS_controllers", &GIDS_Controllers::init_GIDS_controllers);

}

//gids
