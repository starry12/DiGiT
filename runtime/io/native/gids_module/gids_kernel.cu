template <typename T>
__device__ __forceinline__ void count_useful_row(array_d_t<T>* dr, int64_t range_id,
                                                T* address, uint64_t row, int dim, int cache_dim) {
    if ((threadIdx.x & 31) != 0 || address == nullptr) return;
    const auto& cache = dr->d_ranges[range_id].cache;
    uint64_t slot = (reinterpret_cast<uint8_t*>(address) - cache.base_addr) / cache.page_size;
    uint64_t offset = (row * static_cast<uint64_t>(cache_dim) * sizeof(T)) % cache.page_size;
    cache.useful_consume(slot, offset, static_cast<uint64_t>(dim) * sizeof(T));
}

// Full 4-KiB routing: reserve an existing complete page without I/O.
// The normal reader still records the hit. The extra reference prevents
// eviction until that read finishes; a busy or partial page uses CPU fallback.
template <typename T>
__device__ __forceinline__ bool full_try_pin_resident(range_d_t<T>* range, uint64_t page) {
  uint32_t state = range->pages[page].state.load(simt::memory_order_acquire);
  while ((state & (VALID | BUSY)) == VALID && (state & CNT_MASK) != CNT_MASK) {
    if (range->pages[page].state.compare_exchange_weak(state, state + 1,
          simt::memory_order_acquire, simt::memory_order_relaxed)) {
      if (range->pages[page].valid_subrows == range->cache.mixed_full_mask) return true;
      range->pages[page].state.fetch_sub(1, simt::memory_order_release);
      return false;
    }
  }
  return false;
}
template <typename T>
__device__ __forceinline__ void full_unpin_resident(range_d_t<T>* range, uint64_t page) {
  __syncwarp();
  if ((threadIdx.x & 31) == 0)
    range->pages[page].state.fetch_sub(1, simt::memory_order_release);
}


__global__ void probe_mixed_io_geometry_kernel(
    const uint64_t* storage_rows, const uint8_t* group_flags,
    uint64_t* output, uint64_t count, uint32_t feature_row_bytes,
    uint32_t minimum_transfer_bytes, uint32_t group_size,
    uint32_t cache_slot_bytes, uint64_t full_valid_mask,
    uint64_t payload_offset_bytes) {
  uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  uint64_t byte_offset = storage_rows[index] * feature_row_bytes;
  uint64_t slot = byte_offset / cache_slot_bytes;
  uint64_t byte_in_slot = byte_offset % cache_slot_bytes;
  uint64_t row_in_slot = byte_in_slot / feature_row_bytes;
  uint64_t first_subrow = byte_in_slot / minimum_transfer_bytes;
  uint64_t last_subrow =
      (byte_in_slot + feature_row_bytes - 1) / minimum_transfer_bytes;
  uint64_t raw_subrows = last_subrow - first_subrow + 1;
  bool is_group = group_flags[index] != 0;
  uint64_t requested_mask = is_group
      ? full_valid_mask
      : (((1ULL << raw_subrows) - 1ULL) << first_subrow);
  uint64_t* row = output + index * 6;
  row[0] = slot;
  row[1] = row_in_slot;
  row[2] = byte_in_slot;
  row[3] = requested_mask;
  row[4] = payload_offset_bytes + slot * cache_slot_bytes
      + (is_group ? 0 : first_subrow * minimum_transfer_bytes);
  row[5] = is_group ? cache_slot_bytes
                    : raw_subrows * minimum_transfer_bytes;
}


template <typename T = float>
__global__ void read_feature_kernel(array_d_t<T> *dr, T *out_tensor_ptr,
                                    int64_t *index_ptr, int dim,
                                    int64_t num_idx, int cache_dim, uint64_t key_off,
                                    unsigned long long int* d_gpu_access) {
  uint64_t bid = blockIdx.x;
  int num_warps = blockDim.x / 32;
  int warp_id = threadIdx.x / 32;
  int idx_idx = bid * num_warps + warp_id;
  if (idx_idx < num_idx) {  //确保当前的全局索引不超出总索引范围
 	    bam_ptr<T> ptr(dr);

        uint64_t row_index = index_ptr[idx_idx] + key_off;
      	uint64_t tid = threadIdx.x % 32;


    for (; tid < dim; tid += 32) {
      // 每一个warp负责读取一个feature，一个warp里面总共32个thread，tid是thread在warp中的索引
      if (tid == 0)
        atomicAdd(d_gpu_access, 1);
	    T temp = ptr.load((row_index) * cache_dim + tid);
      // printf("data: %llu\n",  (row_index) * cache_dim + tid);
	    out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = temp;
      // 将读取的数据存储到输出张量的对应位置
    }
    __syncwarp();
    count_useful_row(dr, ptr.range_id, ptr.addr, row_index, dim, cache_dim);

  }
}

template <typename T = float>
__device__ __forceinline__ void read_mixed_feature_row(
    array_d_t<T>* dr, T* out_tensor_ptr, int64_t output_row,
    uint64_t storage_row, int dim, int cache_dim, bool is_group,
    uint32_t storage_page_size, uint32_t subrow_bytes,
    uint64_t full_valid_mask, unsigned long long* mixed_io_stats) {
  uint64_t lane = threadIdx.x & 0x1F;
  uint64_t element_index = storage_row * static_cast<uint64_t>(cache_dim);
  uint64_t feature_bytes = static_cast<uint64_t>(cache_dim) * sizeof(T);
  uint64_t byte_in_page =
      (storage_row * feature_bytes) % storage_page_size;
  uint32_t first_subrow = byte_in_page / subrow_bytes;
  uint32_t last_subrow =
      (byte_in_page + feature_bytes - 1) / subrow_bytes;
  uint32_t raw_subrows = last_subrow - first_subrow + 1;
  uint64_t requested_parts = is_group
      ? full_valid_mask
      : (((1ULL << raw_subrows) - 1ULL) << first_subrow);
  data_page_t* page = nullptr;
  size_t start = 0;
  size_t end = 0;
  int64_t range_id = -1;
  T* page_addr = static_cast<T*>(dr->acquire_page_mask(
      element_index, requested_parts, is_group, mixed_io_stats, page, start,
      end, range_id));
  // start is derived from the actual cache-entry geometry.  This remains
  // correct when an 8-KiB storage slot is represented by two independent
  // 4-KiB cache entries; byte_in_page is intentionally only used to derive
  // the storage validity mask above.
  T* row_addr = page_addr + (element_index - start);
  if (lane == 0) {
    atomicAdd(mixed_io_stats +
                  (is_group ? MIXED_GROUP_ROWS : MIXED_RAW_ROWS),
              1ULL);
  }
  for (uint64_t feature = lane; feature < static_cast<uint64_t>(dim);
       feature += 32) {
    out_tensor_ptr[output_row * dim + feature] = row_addr[feature];
  }
  __syncwarp();
  count_useful_row(dr, range_id, page_addr, storage_row, dim, cache_dim);
  dr->release_page(page, range_id, element_index);
}

template <typename T = float>
__global__ void read_feature_kernel_mixed(
    array_d_t<T>* dr, T* out_tensor_ptr, int64_t* index_ptr,
    const bool* group_ptr, int dim, int64_t num_idx, int cache_dim,
    uint64_t key_off, uint32_t storage_page_size, uint32_t subrow_bytes,
    uint64_t full_valid_mask, unsigned long long* d_gpu_access,
    unsigned long long* mixed_io_stats) {
  uint64_t bid = blockIdx.x;
  int num_warps = blockDim.x / 32;
  int warp_id = threadIdx.x / 32;
  int64_t idx_idx = static_cast<int64_t>(bid * num_warps + warp_id);
  if (idx_idx >= num_idx) {
    return;
  }
  if ((threadIdx.x & 0x1F) == 0) {
    atomicAdd(d_gpu_access, 1ULL);
  }
  uint64_t storage_row = static_cast<uint64_t>(index_ptr[idx_idx]) + key_off;
  read_mixed_feature_row(dr, out_tensor_ptr, idx_idx, storage_row, dim,
                         cache_dim, group_ptr[idx_idx], storage_page_size,
                         subrow_bytes, full_valid_mask, mixed_io_stats);
}


template <typename T = float>
__global__ void read_feature_kernel_with_cpu_backing_memory(array_d_t<T> *dr, range_d_t<T> *range, T *out_tensor_ptr,
                                    int64_t *index_ptr, int dim,
                                    int64_t num_idx, int cache_dim, GIDS_CPU_buffer<T> CPU_buffer, bool cpu_seq, 
                                    unsigned long long int* d_cpu_access, uint64_t key_off, unsigned long long int* d_gpu_access,
                                    uint32_t storage_page_size) {

  uint64_t bid = blockIdx.x;

  int num_warps = blockDim.x / 32;
  int warp_id = threadIdx.x / 32;
  int idx_idx = bid * num_warps + warp_id;
  if (idx_idx < num_idx) {
 	    bam_ptr<T> ptr(dr);

      uint64_t row_index = index_ptr[idx_idx] + key_off;
      uint64_t tid = threadIdx.x % 32;
      unsigned lane = threadIdx.x & 0x1F;  // 等价于 threadIdx.x % 32

      uint64_t feature_bytes = (uint64_t)cache_dim * sizeof(T);
      uint64_t rows_per_page = storage_page_size / feature_bytes;
      uint64_t storage_page = row_index / rows_per_page;
      uint64_t row_in_page = row_index % rows_per_page;
      uint32_t cpu_off = range -> get_cpu_offset(storage_page);
      //printf("CPU_buffer.cpu_buffer_len: %llu\n", CPU_buffer.cpu_buffer_len);
      // 200000
      //printf("cpu_seq: %s\n", cpu_seq ? "true" : "false");
      // false
      if(cpu_seq){
        if(row_index < CPU_buffer.cpu_buffer_len){
          printf("row_index: %llu\n", row_index);
          if(tid == 0)
            atomicAdd(d_cpu_access, 1);
          for (; tid < dim; tid += 32) {
            T temp = CPU_buffer.device_cpu_buffer[(row_index) * cache_dim + tid];
            out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = temp;
            }
        }

        else{
        for (; tid < dim; tid += 32) {
          if (tid == 0){
            atomicAdd(d_gpu_access, 1);
            // std::cout << "gpu access" << *d_gpu_access << std::endl; 
          }
          T temp = ptr.load((row_index) * cache_dim + tid);
          out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = temp;
        }
      }
      }
      else{
        int resident = 0;
        if ((cpu_off & 1) && storage_page_size == 4096 && cache_dim == 128) {
          if (lane == 0) resident = full_try_pin_resident(range, storage_page);
          resident = __shfl_sync(0xffffffff, resident, 0);
        }
        if ((cpu_off & 1) && !resident){ // CPU fallback after GPU probe
          if(lane == 0)
            atomicAdd(d_cpu_access, 1);
          for (; tid < dim; tid += 32) {
            uint64_t cpu_row = (cpu_off >> 1) + row_in_page;
            T temp = CPU_buffer.device_cpu_buffer[cpu_row * cache_dim + tid];
            out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = temp; // 结果存入输出 out_tensor_ptr[...]
          }
        }else{
          // 每一个warp负责读取一个feature，一个warp里面总共32个thread，tid是thread在warp中的索引
          if (lane == 0)
            atomicAdd(d_gpu_access, 1);
          for (; tid < dim; tid += 32) {
            T temp = ptr.load((row_index) * cache_dim + tid);
            out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = temp;
          }
          if (resident) full_unpin_resident(range, storage_page);
        }
      }
    __syncwarp();
    count_useful_row(dr, ptr.range_id, ptr.addr, row_index, dim, cache_dim);

  }
}

template <typename T = float>
__global__ void read_feature_kernel_mixed_with_cpu_backing_memory(
    array_d_t<T>* dr, range_d_t<T>* range, T* out_tensor_ptr,
    int64_t* index_ptr, const bool* group_ptr, int dim, int64_t num_idx,
    int cache_dim, GIDS_CPU_buffer<T> CPU_buffer,
    unsigned long long* d_cpu_access, uint64_t key_off,
    unsigned long long* d_gpu_access, uint32_t storage_page_size,
    uint32_t subrow_bytes, uint32_t cache_entry_bytes,
    uint64_t full_valid_mask,
    unsigned long long* mixed_io_stats) {
  uint64_t bid = blockIdx.x;
  int num_warps = blockDim.x / 32;
  int warp_id = threadIdx.x / 32;
  int64_t idx_idx = static_cast<int64_t>(bid * num_warps + warp_id);
  if (idx_idx >= num_idx) {
    return;
  }

  uint64_t lane = threadIdx.x & 0x1F;
  uint64_t storage_row = static_cast<uint64_t>(index_ptr[idx_idx]) + key_off;
  uint64_t feature_bytes = static_cast<uint64_t>(cache_dim) * sizeof(T);
  uint64_t rows_per_page = storage_page_size / feature_bytes;
  uint64_t storage_page = storage_row / rows_per_page;
  uint64_t row_in_page = storage_row % rows_per_page;
  uint64_t cache_page = storage_page *
      (static_cast<uint64_t>(storage_page_size) / cache_entry_bytes);
  uint32_t cpu_off = range->get_cpu_offset(cache_page);
  int resident = 0;
  if ((cpu_off & 1) && storage_page_size == 4096 && cache_entry_bytes == 4096 && cache_dim == 128) {
    if (lane == 0) resident = full_try_pin_resident(range, cache_page);
    resident = __shfl_sync(0xffffffff, resident, 0);
  }
  if ((cpu_off & 1) && !resident) {
    if (lane == 0) {
      atomicAdd(d_cpu_access, 1ULL);
    }
    uint64_t cpu_row = (cpu_off >> 1) + row_in_page;
    for (uint64_t feature = lane; feature < static_cast<uint64_t>(dim);
         feature += 32) {
      out_tensor_ptr[idx_idx * dim + feature] =
          CPU_buffer.device_cpu_buffer[cpu_row * cache_dim + feature];
    }
    return;
  }

  if (lane == 0) {
    atomicAdd(d_gpu_access, 1ULL);
  }
  read_mixed_feature_row(dr, out_tensor_ptr, idx_idx, storage_row, dim,
                         cache_dim, group_ptr[idx_idx], storage_page_size,
                         subrow_bytes, full_valid_mask, mixed_io_stats);
  if (resident) full_unpin_resident(range, cache_page);
}


template <typename T = float>
__global__ void read_feature_kernel_ssd_only_with_cpu_map(
    array_d_t<T> *dr, range_d_t<T> *range, T *out_tensor_ptr,
    int64_t *index_ptr, int dim, int64_t num_idx, int cache_dim,
    uint64_t key_off, unsigned long long int *d_gpu_access,
    uint32_t storage_page_size) {
  uint64_t bid = blockIdx.x;
  int num_warps = blockDim.x / 32;
  int warp_id = threadIdx.x / 32;
  int64_t idx_idx = static_cast<int64_t>(bid * num_warps + warp_id);
  if (idx_idx >= num_idx) {
    return;
  }

  bam_ptr<T> ptr(dr);
  uint64_t row_index = static_cast<uint64_t>(index_ptr[idx_idx]) + key_off;
  uint64_t feature_bytes = static_cast<uint64_t>(cache_dim) * sizeof(T);
  uint64_t rows_per_page = storage_page_size / feature_bytes;
  uint64_t storage_page = row_index / rows_per_page;
  uint32_t cpu_off = range->get_cpu_offset(storage_page);
  if ((cpu_off & 0x1) == 1) {
    return;
  }

  uint64_t lane = threadIdx.x & 0x1F;
  if (lane == 0) {
    atomicAdd(d_gpu_access, 1);
  }
  for (uint64_t feature = lane; feature < static_cast<uint64_t>(dim);
       feature += 32) {
    out_tensor_ptr[idx_idx * dim + feature] =
        ptr.load(row_index * cache_dim + feature);
  }
    __syncwarp();
    count_useful_row(dr, ptr.range_id, ptr.addr, row_index, dim, cache_dim);

}

template <typename T = float>
__global__ void read_feature_kernel_mixed_ssd_only_with_cpu_map(
    array_d_t<T>* dr, range_d_t<T>* range, T* out_tensor_ptr,
    int64_t* index_ptr, const bool* group_ptr, int dim, int64_t num_idx,
    int cache_dim, uint64_t key_off, unsigned long long* d_gpu_access,
    uint32_t storage_page_size, uint32_t subrow_bytes,
    uint32_t cache_entry_bytes, uint64_t full_valid_mask,
    unsigned long long* mixed_io_stats) {
  uint64_t bid = blockIdx.x;
  int num_warps = blockDim.x / 32;
  int warp_id = threadIdx.x / 32;
  int64_t idx_idx = static_cast<int64_t>(bid * num_warps + warp_id);
  if (idx_idx >= num_idx) {
    return;
  }

  uint64_t storage_row = static_cast<uint64_t>(index_ptr[idx_idx]) + key_off;
  uint64_t feature_bytes = static_cast<uint64_t>(cache_dim) * sizeof(T);
  uint64_t rows_per_page = storage_page_size / feature_bytes;
  uint64_t storage_page = storage_row / rows_per_page;
  uint64_t cache_page = storage_page *
      (static_cast<uint64_t>(storage_page_size) / cache_entry_bytes);
  uint32_t cpu_off = range->get_cpu_offset(cache_page);
  if ((cpu_off & 0x1) == 1) {
    return;
  }
  if ((threadIdx.x & 0x1F) == 0) {
    atomicAdd(d_gpu_access, 1ULL);
  }
  read_mixed_feature_row(dr, out_tensor_ptr, idx_idx, storage_row, dim,
                         cache_dim, group_ptr[idx_idx], storage_page_size,
                         subrow_bytes, full_valid_mask, mixed_io_stats);
}


template <typename T = float>
__global__ void scatter_staged_cpu_features(
    const uint8_t *payload, T *out_tensor_ptr, uint64_t cpu_rows, int dim) {
  uint64_t staged_row = blockIdx.x;
  if (staged_row >= cpu_rows) {
    return;
  }
  const uint64_t *positions = reinterpret_cast<const uint64_t *>(payload);
  const T *features = reinterpret_cast<const T *>(
      payload + cpu_rows * sizeof(uint64_t));
  uint64_t output_row = positions[staged_row];
  for (uint64_t feature = threadIdx.x; feature < static_cast<uint64_t>(dim);
       feature += blockDim.x) {
    out_tensor_ptr[output_row * dim + feature] =
        features[staged_row * dim + feature];
  }
}



template <typename T = float>
__global__ void set_cpu_buffer_kernel(range_d_t<T> *d_range, uint64_t* idx_ptr,
                                      int num, uint32_t pageSize,
                                      uint64_t feature_dim,
                                      uint32_t cache_entry_size) {

  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
	uint64_t bid = blockIdx.x;
  // RJC：未看完全懂！为了保持不超出索引范围，这里的idx < num是必要的
  if(idx <  num){
    uint64_t rows_per_page = pageSize / (feature_dim * sizeof(T));
    uint64_t storage_row = idx_ptr[idx];
    uint64_t row_in_page = storage_row % rows_per_page;
    uint64_t storage_page = storage_row / rows_per_page;
    if(row_in_page == 0){
      uint64_t cache_page = storage_page *
          (static_cast<uint64_t>(pageSize) / cache_entry_size);
      d_range -> set_cpu_buffer(cache_page, idx);
    }
  }

}


template <typename T = float>
__global__ void set_cpu_buffer_data_kernel(array_d_t<T> *dr, T* CPU_buffer, uint64_t* idx_ptr, uint64_t dim, int num) {
	uint64_t bid = blockIdx.x;
	bam_ptr<T> ptr(dr);
	if(bid <  num){
		uint64_t idx = idx_ptr[bid];
		for(uint64_t i  = threadIdx.x; i < dim; i += blockDim.x){
      // 将 dr[idx][i] 拷贝到 CPU_buffer[bid][i]
			CPU_buffer[bid * dim + i] = ptr.load(idx * dim + i);
		}
	}

}


template <typename T = float>
__global__
void set_window_buffering_kernel(array_d_t<T>* dr, uint64_t *index_ptr, uint64_t page_size, int hash_off){
	bam_ptr<T> ptr(dr);
	if(threadIdx.x == 0){
		uint64_t page_idx = index_ptr[blockIdx.x] + hash_off;
		ptr.set_window_buffer_counter(page_idx * page_size/sizeof(T), 1);
	}
}

template <typename T = float>
__global__ void read_kernel(array_d_t<T> *dr,
                                    uint64_t num, uint64_t offset) {
      bam_ptr<T> ptr(dr);
     if(threadIdx.x == 0 && blockIdx.x == 0){
        for(uint64_t i = 0; i < num; i++){
              if(i == 0) printf("idx: %llu type size:%i \n", offset,  (int) sizeof(T));
             // T temp = ptr[i + offset];
	              printf("read data: %llu\n",  (unsigned long long) ptr.load(i + offset));
             // printf("float read data: %f\n", temp);

        }
     }                           
}


template <typename T = float>
__global__ void seq_read_kernel(array_d_t<T> *dr,
                                    uint64_t num, uint64_t offset) {
    bam_ptr<T> ptr(dr);
     if(threadIdx.x == 0 && blockIdx.x == 0){
        for(uint64_t i = 0; i < num; i++){
             // if(i == 0) printf("idx: %llu type size:%i \n", offset,  (int) sizeof(T));
	              T temp = ptr.load(i + offset);
              //printf("read data: %llu\n",  (unsigned long long) ptr[i + offset]);
	              printf("read data: %f\n",  (float) ptr.load(i + offset));
             // printf("float read data: %f\n", temp);

        }
     }                           
}


template <typename T = float>
__global__ void write_feature_kernel(Controller** ctrls, page_cache_d_t* pc, array_d_t<T> *dr, T* in_tensor_ptr,
                                    uint64_t num, uint64_t page_size,  uint64_t o_offset,  uint64_t s_offset, uint32_t num_ctrls) {

    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t ctrl = (tid) % (num_ctrls);
    uint64_t pc_idx = tid / num_ctrls;

    uint32_t queue = (tid) % (ctrls[ctrl]->n_qps);

    if(tid < num){
    	uint64_t start_block = ((o_offset+s_offset + pc_idx*page_size)) >> ctrls[ctrl]->d_qps[queue].block_size_log ;

    	uint64_t n_blocks = page_size >> ctrls[ctrl]->d_qps[queue].block_size_log; /// ctrls[ctrl].ns.lba_data_size;;
    	write_data(pc, (ctrls[ctrl]->d_qps)+(queue),start_block, n_blocks, tid);
    }
}

template <typename T = float>
__global__ void write_feature_kernel2(Controller** ctrls, page_cache_d_t* pc, array_d_t<T> *dr, T* in_tensor_ptr, uint64_t dim, uint32_t num_ctrls, uint64_t offset) {


	bam_ptr<T> ptr(dr);
	uint64_t row_index = blockIdx.x;

	for(int i = threadIdx.x; i < dim; i += blockDim.x){
		ptr[(row_index) * dim + i] = in_tensor_ptr[(row_index) * dim + i + offset];
	}
}
