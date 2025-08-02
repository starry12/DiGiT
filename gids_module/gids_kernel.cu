

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
	    T temp = ptr[(row_index) * cache_dim + tid];
      // printf("data: %llu\n",  (row_index) * cache_dim + tid);
	    out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = temp;
      // 将读取的数据存储到输出张量的对应位置
    }
  }
}


template <typename T = float>
__global__ void read_feature_kernel_with_cpu_backing_memory(array_d_t<T> *dr, range_d_t<T> *range, T *out_tensor_ptr,
                                    int64_t *index_ptr, int dim,
                                    int64_t num_idx, int cache_dim, GIDS_CPU_buffer<T> CPU_buffer, bool cpu_seq, 
                                    unsigned long long int* d_cpu_access, uint64_t key_off, unsigned long long int* d_gpu_access, 
                                    unsigned long long int* d_cpu_clock, unsigned long long int* d_gpu_clock) {

  uint64_t bid = blockIdx.x;

  int num_warps = blockDim.x / 32;
  int warp_id = threadIdx.x / 32;
  int idx_idx = bid * num_warps + warp_id;
  if (idx_idx < num_idx) {
 	    bam_ptr<T> ptr(dr);

      uint64_t row_index = index_ptr[idx_idx] + key_off;
      uint64_t tid = threadIdx.x % 32;
      unsigned lane = threadIdx.x & 0x1F;  // 等价于 threadIdx.x % 32

      uint32_t cpu_off = range -> get_cpu_offset(row_index);
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
          T temp = ptr[(row_index) * cache_dim + tid];
          out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = temp;
        }
      }
      }
      else{
        if((cpu_off & 0x1) == 1){ // 命中cpu的cache
          clock_t start_clock = 0, end_clock = 0;
          if(lane == 0){
            atomicAdd(d_cpu_access, 1);
            start_clock = clock();
          }
          for (; tid < dim; tid += 32) {
            T temp = CPU_buffer.device_cpu_buffer[(cpu_off >> 1) * cache_dim + tid];
            out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = temp; // 结果存入输出 out_tensor_ptr[...]
          }
          if (lane == 0) {
            // 循环后再记录时钟，累加到 d_cpu_clock
            end_clock = clock();
            atomicAdd(d_cpu_clock, end_clock - start_clock);
          }
        }else{
          clock_t start_clock = 0, end_clock = 0;
          // 每一个warp负责读取一个feature，一个warp里面总共32个thread，tid是thread在warp中的索引
          if (lane == 0){
            atomicAdd(d_gpu_access, 1);
            start_clock = clock();
            // std::cout << "gpu access" << *d_gpu_access << std::endl; 
          }
          for (; tid < dim; tid += 32) {
            T temp = ptr[(row_index) * cache_dim + tid];
            out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = temp;
          }
          if (lane == 0) {
            // 循环后再记录时钟，累加到 d_cpu_clock
            end_clock = clock();
            atomicAdd(d_gpu_clock, end_clock - start_clock);
            //printf("After this warp, d_gpu_clock = %llu\n", d_gpu_clock);
          }
        }
      }
  }
}



template <typename T = float>
__global__ void set_cpu_buffer_kernel(range_d_t<T> *d_range, uint64_t* idx_ptr, int num, uint32_t pageSize) {

  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
	uint64_t bid = blockIdx.x;
  // RJC：未看完全懂！为了保持不超出索引范围，这里的idx < num是必要的
  if(idx <  num){
    d_range -> set_cpu_buffer(idx_ptr[idx], idx );
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
			CPU_buffer[bid * dim + i] = ptr[idx * dim + i];
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
              printf("read data: %llu\n",  (unsigned long long) ptr[i + offset]);
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
              T temp = ptr[i + offset];
              //printf("read data: %llu\n",  (unsigned long long) ptr[i + offset]);
              printf("read data: %f\n",  (float) ptr[i + offset]);
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




