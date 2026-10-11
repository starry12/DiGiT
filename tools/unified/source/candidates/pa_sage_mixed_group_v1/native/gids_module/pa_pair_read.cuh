__device__ __forceinline__ uint64_t pair_hash_key(uint64_t row) {
    row ^= row >> 33; row *= 0xff51afd7ed558ccdULL;
    row ^= row >> 33; return row;
}
__global__ void policy_pair_hash(const int64_t* ids,const bool* flags,uint64_t n,
    const uint32_t* cpu_map,uint64_t extent,int* table,uint64_t capacity,unsigned long long* errors) {
    uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=n)return;
    if(ids[i]<0 || uint64_t(ids[i])>=extent){atomicAdd(errors,1ULL);return;}
    uint64_t row=ids[i];
    if(!flags[i])return;
    if((row&7ULL)>1 || (row&~7ULL)+1>=extent){atomicAdd(errors,1ULL);return;}
    if(cpu_map[row])return;
    uint64_t h=pair_hash_key(row)&(capacity-1);
    for(uint64_t step=0;step<capacity;++step,h=(h+1)&(capacity-1)){
        int previous=atomicCAS(table+h,-1,int(i));
        if(previous==-1)return;
        if(ids[previous]==ids[i]){atomicAdd(errors,1ULL);return;}
    }
    atomicAdd(errors,1ULL);
}
__device__ __forceinline__ int policy_partner(const int64_t* ids,const int* table,
    uint64_t capacity,uint64_t wanted){
    uint64_t h=pair_hash_key(wanted)&(capacity-1);
    for(uint64_t step=0;step<capacity;++step,h=(h+1)&(capacity-1)){
        int idx=table[h]; if(idx<0) return -1;
        if(uint64_t(ids[idx])==wanted) return idx;
    } return -1;
}

template <typename T>
__device__ __forceinline__ void policy_scatter_pair(const volatile T* tmp,T* a,T* b){
    for(unsigned f=threadIdx.x&31;f<128;f+=32){a[f]=tmp[f];b[f]=tmp[128+f];}
}

// One warp owns both output rows and both logical-page BUSY locks. Reserve
// slots in ascending logical address order. FIFO must skip pinned/BUSY victims.
template <typename T>
__device__ void policy_read_pair(range_d_t<T>* range,T* output,uint64_t first,
    uint64_t out0,uint64_t out1,uint64_t scratch_slot,
    unsigned long long* route_count,unsigned long long* mixed_counts){
    const uint32_t lane=threadIdx.x&31;auto& c=range->cache;
    uint32_t state0=0,state1=0,slot0=0,slot1=0;
    if(!lane){
        unsigned ns=8;
        while(true){
            state0=range->pages[first].state.fetch_or(BUSY,simt::memory_order_acquire);
            if(!(state0&BUSY)){
                state1=range->pages[first+1].state.fetch_or(BUSY,simt::memory_order_acquire);
                if(!(state1&BUSY))break;
                range->pages[first].state.fetch_and(DISABLE_BUSY_MASK,simt::memory_order_release);
            }
            __nanosleep(ns);if(ns<256)ns*=2;
        }
        for(uint64_t pg=first;pg<=first+1;++pg){
            uint32_t protect=0;
            uint8_t pf=range->pages[pg].prefetch_count.load(simt::memory_order_acquire);
            if(pf && !(pf&0x80)){protect=1;range->pages[pg].prefetch_count.store(pf|0x80,simt::memory_order_release);}
            uint8_t wb=range->pages[pg].prefetch_counter.load(simt::memory_order_acquire);
            if(wb>1 && !(wb&0x80)){protect=1;range->pages[pg].prefetch_counter.store(wb|0x80,simt::memory_order_release);}
            range->pages[pg].state.fetch_add(32+protect,simt::memory_order_acquire);
        }
        const uint32_t queue=get_smid()%c.d_ctrls[0]->n_qps;
        if(state0&VALID)slot0=range->pages[first].offset;
        else {slot0=c.find_slot(first,range->range_id,queue,range->read_io_cnt,range->evicted_p_array,true);range->pages[first].offset=slot0;range->pages[first].valid_subrows=1;}
        if(state1&VALID)slot1=range->pages[first+1].offset;
        else {slot1=c.find_slot(first+1,range->range_id,queue,range->read_io_cnt,range->evicted_p_array,true);range->pages[first+1].offset=slot1;range->pages[first+1].valid_subrows=1;}
        unsigned hits=unsigned(bool(state0&VALID))+unsigned(bool(state1&VALID));
        c.cache_hits->fetch_add(hits,simt::memory_order_relaxed);
        range->access_cnt.fetch_add(64,simt::memory_order_relaxed);
        range->hit_cnt.fetch_add(hits*32,simt::memory_order_relaxed);
        range->miss_cnt.fetch_add((2-hits)*32,simt::memory_order_relaxed);
        atomicAdd(route_count,2ULL);atomicAdd(c.pair_counts+6,1ULL);
        atomicAdd(c.pair_counts+7,(unsigned long long)hits);
        if(hits<2){
            Controller* ctrl=c.d_ctrls[0];ctrl->access_counter.fetch_add(1,simt::memory_order_relaxed);
            if(!hits){
                // PRP points at a 4-KiB-aligned contiguous staging buffer.
                // Two arbitrary 1-KiB resident slots cannot be NVMe PRP1/PRP2.
                read_data_prp(&c,ctrl->d_qps+queue,
                    range->get_backing_page(first)*c.n_blocks_per_page,
                    2*c.n_blocks_per_page,c.pair_scratch_prps[scratch_slot],0);
                atomicAdd(mixed_counts+MIXED_GROUP_FULL_COMMANDS,1ULL);
                atomicAdd(mixed_counts+MIXED_GROUP_BYTES,1024ULL);
                atomicAdd(mixed_counts+MIXED_PHYSICAL_BYTES,1024ULL);
            }else{
                uint64_t pg=(state0&VALID)?first+1:first;
                read_data(&c,ctrl->d_qps+queue,range->get_backing_page(pg)*c.n_blocks_per_page,
                    c.n_blocks_per_page,(state0&VALID)?slot1:slot0);
                atomicAdd(mixed_counts+MIXED_GROUP_PARTIAL_COMMANDS,1ULL);
                atomicAdd(mixed_counts+MIXED_GROUP_BYTES,512ULL);
                atomicAdd(mixed_counts+MIXED_PHYSICAL_BYTES,512ULL);
            }
        }
        if(!(state0&VALID))c.useful_fill(slot0,0,512,true);
        if(!(state1&VALID))c.useful_fill(slot1,0,512,true);
    }
    state0=__shfl_sync(0xffffffff,state0,0);state1=__shfl_sync(0xffffffff,state1,0);
    slot0=__shfl_sync(0xffffffff,slot0,0);slot1=__shfl_sync(0xffffffff,slot1,0);
    T* a=reinterpret_cast<T*>(c.base_addr+uint64_t(slot0)*512);
    T* b=reinterpret_cast<T*>(c.base_addr+uint64_t(slot1)*512);
    if(!(state0&VALID)&&!(state1&VALID)){
        const volatile T* tmp=reinterpret_cast<const volatile T*>(c.base_addr+c.n_pages*512+scratch_slot*4096);
        policy_scatter_pair(tmp,a,b);
    }
    __syncwarp();
    for(unsigned f=lane;f<128;f+=32){output[out0*128+f]=a[f];output[out1*128+f]=b[f];}
    __threadfence();__syncwarp();
    if(!lane){
        c.useful_consume(slot0,0,512);c.useful_consume(slot1,0,512);
        atomicAdd(mixed_counts+MIXED_GROUP_ROWS,2ULL);
        if(state0&VALID)range->pages[first].state.fetch_and(DISABLE_BUSY_MASK,simt::memory_order_release);
        else range->pages[first].state.fetch_xor(DISABLE_BUSY_ENABLE_VALID,simt::memory_order_release);
        if(state1&VALID)range->pages[first+1].state.fetch_and(DISABLE_BUSY_MASK,simt::memory_order_release);
        else range->pages[first+1].state.fetch_xor(DISABLE_BUSY_ENABLE_VALID,simt::memory_order_release);
        range->release_page(first,32);
        range->release_page(first+1,32);
    }
    __syncwarp();
}

template <typename T = float>
__device__ void pair_single_fallback(array_d_t<T> *dr, range_d_t<T> *range, T *out_tensor_ptr,
                                    int64_t *index_ptr, int dim,
                                    int64_t num_idx, int cache_dim, GIDS_CPU_buffer<T> CPU_buffer, bool cpu_seq, 
                                    unsigned long long int* d_cpu_access, uint64_t key_off, unsigned long long int* d_gpu_access,
                                    uint32_t storage_page_size, int requested) {

  uint64_t bid = blockIdx.x;

  int num_warps = blockDim.x / 32;
  int warp_id = threadIdx.x / 32;
  int idx_idx = requested;
  if (idx_idx < num_idx) {
 	    bam_ptr<T> ptr(dr);

      uint64_t row_index = index_ptr[idx_idx] + key_off;
      uint64_t tid = threadIdx.x % 32;
      unsigned lane = threadIdx.x & 0x1F;  // 等价于 threadIdx.x % 32

      uint64_t feature_bytes = (uint64_t)cache_dim * sizeof(T);
      uint64_t rows_per_page = storage_page_size / feature_bytes;
      uint64_t storage_page = row_index / rows_per_page;
      uint64_t row_in_page = row_index % rows_per_page;
      uint32_t cpu_off = CPU_buffer.row_map ? exact_cpu_offset(CPU_buffer, row_index) : range -> get_cpu_offset(storage_page);
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
            out_tensor_ptr[idx_idx * dim + tid] = temp;
            }
        }

        else{
        for (; tid < dim; tid += 32) {
          if (tid == 0){
            atomicAdd(d_gpu_access, 1);
            // std::cout << "gpu access" << *d_gpu_access << std::endl; 
          }
          T temp = ptr.load((row_index) * cache_dim + tid);
          out_tensor_ptr[idx_idx * dim + tid] = temp;
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
            uint64_t cpu_row = (cpu_off >> 1) + (CPU_buffer.row_map ? 0 : row_in_page);
            T temp = CPU_buffer.device_cpu_buffer[cpu_row * cache_dim + tid];
            out_tensor_ptr[idx_idx * dim + tid] = temp; // 结果存入输出 out_tensor_ptr[...]
          }
        }else{
          // 每一个warp负责读取一个feature，一个warp里面总共32个thread，tid是thread在warp中的索引
          if (lane == 0)
            atomicAdd(d_gpu_access, 1);
          for (; tid < dim; tid += 32) {
            T temp = ptr.load((row_index) * cache_dim + tid);
            out_tensor_ptr[idx_idx * dim + tid] = temp;
          }
          if (resident) full_unpin_resident(range, storage_page);
        }
      }
    __syncwarp();
    count_useful_row(dr, ptr.range_id, ptr.addr, row_index, dim, cache_dim);

  }
}


template <typename T>
__global__ void pa_pair_read(array_d_t<T>* array,range_d_t<T>* range,T* output,
    int64_t* ids,const bool* flags,uint64_t n,GIDS_CPU_buffer<T> cpu,
    const int* table,uint64_t capacity,unsigned child,unsigned long long* cpu_count,
    unsigned long long* route_count,unsigned long long* mixed_counts){
    unsigned lane=threadIdx.x&31;
    uint64_t warp=(uint64_t(blockIdx.x)*blockDim.x+threadIdx.x)/32;
    uint64_t stride=gridDim.x*blockDim.x/32;
    for(uint64_t i=warp;i<n;i+=stride){
        int partner=-1;
        if(!lane && ids[i]>=0 && uint64_t(ids[i])<cpu.row_map_len && flags[i] && (ids[i]&7)<=1 && !cpu.row_map[ids[i]])
            partner=policy_partner(ids,table,capacity,uint64_t(ids[i])^1ULL);
        partner=__shfl_sync(0xffffffff,partner,0);
        if(partner>=0){
            if(!(ids[i]&1))policy_read_pair(range,output,ids[i],i,partner,child*4096+warp,route_count,mixed_counts);
        }else if(ids[i]>=0 && uint64_t(ids[i])<cpu.row_map_len){
            pair_single_fallback(array,range,output,ids,128,n,128,cpu,false,cpu_count,0,route_count,512,i);
        }
        __syncwarp();
    }
}
