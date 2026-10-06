// Included after the frozen parent's kernels. Never changes BaM page-cache ABI.
#include "include/policy_math.h"

// mode: 0 = CPU preload, 1 = static bypass, 2 = FIFO with exact CPU fallback.
// One scratch page per launched warp; strided iterations reuse only that page.
// read_data submits actual NVMe READs and retains BaM's visibility replay.
// Bypass NEVER calls the cache lookup, insertion, eviction or resident probe.
template <typename T>
__global__ void policy_read_rows(
    array_d_t<T>* array, range_d_t<T>* range, T* output,
    const int64_t* indices, uint64_t count, uint64_t storage_rows,
    GIDS_CPU_buffer<T> cpu, uint32_t mode, uint64_t key,
    unsigned long long* cpu_count, unsigned long long* route_count,
    unsigned long long* policy_counts, unsigned long long* mixed_counts) {
    const uint64_t lane = threadIdx.x & 31;
    const uint64_t warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const uint64_t stride = gridDim.x * blockDim.x / 32;
    for (uint64_t i = warp; i < count; i += stride) {
        if (!digit_policy::row_valid(indices[i], key, storage_rows)) {
            if (!lane) atomicAdd(policy_counts + 1, 1ULL);
            continue; // No out-of-range DMA; host fails the batch after sync.
        }
        const uint64_t row = static_cast<uint64_t>(indices[i]) + key;
        const uint64_t page = digit_policy::page_of(row);
        const uint32_t slot = mode == 0 ? 0 : cpu.row_map[row];
        if (!digit_policy::slot_valid(slot, cpu.cpu_buffer_len)) {
            if (!lane) atomicAdd(policy_counts + 1, 1ULL);
            continue;
        }
        int resident = 0;
        if ((mode == 2 || mode == 3) && slot) {
            if (!lane) resident = full_try_pin_resident(range, page);
            resident = __shfl_sync(0xffffffff, resident, 0);
        }
        if (slot && !resident) {
            if (!lane) atomicAdd(cpu_count, 1ULL);
            for (uint64_t f = lane; f < 128; f += 32)
                output[i * 128 + f] = cpu.device_cpu_buffer[(uint64_t(slot)-1) * 128 + f];
        } else if ((mode == 2 || mode == 3)) {
            if (!lane) atomicAdd(route_count, 1ULL);
            // Independent one-node cache lines; graph groups retain their original addresses.
            read_mixed_feature_row(array, output, i, row, 128, 128, true,
                                  512, 512, 1, mixed_counts);
            if (resident) full_unpin_resident(range, page);
        } else {
            auto& cache = range->cache;
            if (!lane) {
                Controller* controller = cache.d_ctrls[0];
                const uint32_t queue = get_smid() % controller->n_qps;
                read_data(&cache, controller->d_qps + queue,
                          range->get_backing_page(page) * cache.n_blocks_per_page,
                          cache.n_blocks_per_page, warp);
                if (mode == 1) {
                    atomicAdd(route_count, 1ULL);
                    atomicAdd(policy_counts, 1ULL);
                    cache.useful_fill(warp, 0, 512, true);
                    cache.useful_consume(warp, digit_policy::byte_in_page(row), 512);
                }
            }
            __syncwarp();
            // Volatile reads avoid retaining stale values across DMA slot reuse.
            const volatile T* payload = reinterpret_cast<volatile T*>(
                cache.base_addr + warp * 512 + digit_policy::byte_in_page(row));
            for (uint64_t f = lane; f < 128; f += 32)
                output[i * 128 + f] = payload[f];
        }
        __syncwarp(); // all lanes finish before this warp reuses its DMA page
    }
}

// Pair only two requested group members, never padding, CPU-hot aliases, or a
// missing sibling. Explicit group bases support odd UKL replica starts.
__device__ __forceinline__ uint64_t pair_hash_key(uint64_t row) {
    row ^= row >> 33; row *= 0xff51afd7ed558ccdULL;
    row ^= row >> 33; return row;
}
__global__ void policy_pair_hash(const int64_t* ids,const int64_t* bases,uint64_t n,
    const uint32_t* cpu_map,uint64_t extent,int* table,uint64_t capacity,unsigned long long* errors) {
    const uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=n || ids[i]<0 || uint64_t(ids[i])>=extent) return;
    const uint64_t row=ids[i];
    if(bases[i]<0 || uint64_t(bases[i])+1>=extent || (row!=uint64_t(bases[i]) && row!=uint64_t(bases[i])+1) || cpu_map[row]) return;
    uint64_t h=pair_hash_key(row)&(capacity-1);
    for(uint64_t step=0;step<capacity;++step,h=(h+1)&(capacity-1)){
        int prev=atomicCAS(table+h,-1,int(i));
        if(prev==-1)return;
        if(ids[prev]==ids[i]){atomicAdd(errors,1ULL);return;}
    }
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
        atomicAdd(route_count,2ULL);atomicAdd(c.request_size_counts+3,1ULL);
        atomicAdd(c.request_size_counts+4,(unsigned long long)hits);
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
    }
    __syncwarp();
}

template <typename T>
__device__ void policy_single_row(
    array_d_t<T>* array, range_d_t<T>* range, T* output,
    const int64_t* indices, uint64_t count, uint64_t storage_rows,
    GIDS_CPU_buffer<T> cpu, uint32_t mode, uint64_t key,
    unsigned long long* cpu_count, unsigned long long* route_count,
    unsigned long long* policy_counts, unsigned long long* mixed_counts, uint64_t request) {
    const uint64_t lane = threadIdx.x & 31;
    const uint64_t warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const uint64_t stride = gridDim.x * blockDim.x / 32;
    for (uint64_t i = request; i < request+1; ++i) {
        if (!digit_policy::row_valid(indices[i], key, storage_rows)) {
            if (!lane) atomicAdd(policy_counts + 1, 1ULL);
            continue; // No out-of-range DMA; host fails the batch after sync.
        }
        const uint64_t row = static_cast<uint64_t>(indices[i]) + key;
        const uint64_t page = digit_policy::page_of(row);
        const uint32_t slot = mode == 0 ? 0 : cpu.row_map[row];
        if (!digit_policy::slot_valid(slot, cpu.cpu_buffer_len)) {
            if (!lane) atomicAdd(policy_counts + 1, 1ULL);
            continue;
        }
        int resident = 0;
        if ((mode == 2 || mode == 3) && slot) {
            if (!lane) resident = full_try_pin_resident(range, page);
            resident = __shfl_sync(0xffffffff, resident, 0);
        }
        if (slot && !resident) {
            if (!lane) atomicAdd(cpu_count, 1ULL);
            for (uint64_t f = lane; f < 128; f += 32)
                output[i * 128 + f] = cpu.device_cpu_buffer[(uint64_t(slot)-1) * 128 + f];
        } else if ((mode == 2 || mode == 3)) {
            if (!lane) atomicAdd(route_count, 1ULL);
            // Independent one-node cache lines; graph groups retain their original addresses.
            read_mixed_feature_row(array, output, i, row, 128, 128, true,
                                  512, 512, 1, mixed_counts);
            if (resident) full_unpin_resident(range, page);
        } else {
            auto& cache = range->cache;
            if (!lane) {
                Controller* controller = cache.d_ctrls[0];
                const uint32_t queue = get_smid() % controller->n_qps;
                read_data(&cache, controller->d_qps + queue,
                          range->get_backing_page(page) * cache.n_blocks_per_page,
                          cache.n_blocks_per_page, warp);
                if (mode == 1) {
                    atomicAdd(route_count, 1ULL);
                    atomicAdd(policy_counts, 1ULL);
                    cache.useful_fill(warp, 0, 512, true);
                    cache.useful_consume(warp, digit_policy::byte_in_page(row), 512);
                }
            }
            __syncwarp();
            // Volatile reads avoid retaining stale values across DMA slot reuse.
            const volatile T* payload = reinterpret_cast<volatile T*>(
                cache.base_addr + warp * 512 + digit_policy::byte_in_page(row));
            for (uint64_t f = lane; f < 128; f += 32)
                output[i * 128 + f] = payload[f];
        }
        __syncwarp(); // all lanes finish before this warp reuses its DMA page
    }
}


template <typename T>
__global__ void policy_mixed_rows(array_d_t<T>* array,range_d_t<T>* range,T* output,
    const int64_t* ids,const int64_t* bases,uint64_t n,GIDS_CPU_buffer<T> cpu,
    const int* table,uint64_t capacity,unsigned long long* cpu_count,
    unsigned long long* route_count,unsigned long long* policy_counts,unsigned long long* mixed_counts){
    const uint64_t warp=(uint64_t(blockIdx.x)*blockDim.x+threadIdx.x)/32;
    const uint64_t stride=gridDim.x*blockDim.x/32;
    const uint32_t lane=threadIdx.x&31;
    for(uint64_t i=warp;i<n;i+=stride){
        int partner=-1;
        if(!lane && ids[i]>=0 && uint64_t(ids[i])<cpu.row_map_len && bases[i]>=0 && uint64_t(bases[i])+1<cpu.row_map_len && (ids[i]==bases[i] || ids[i]==bases[i]+1) && !cpu.row_map[ids[i]])
            partner=policy_partner(ids,table,capacity,uint64_t(bases[i])+(ids[i]==bases[i]?1:0));
        partner=__shfl_sync(0xffffffff,partner,0);
        if(partner>=0 && bases[partner]==bases[i]){
            if(ids[i]==bases[i])policy_read_pair(range,output,ids[i],i,partner,warp,route_count,mixed_counts);
        }else policy_single_row(array,range,output,ids,n,cpu.row_map_len,cpu,2,0,cpu_count,route_count,policy_counts,mixed_counts,i);
    }
}

__global__ void probe_partner_kernel(const int64_t* ids,const int64_t* bases,const uint32_t* cpu,
 uint64_t n,uint64_t extent,const int* table,uint64_t cap,int* partners){
 uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
 partners[i]=(ids[i]>=0 && uint64_t(ids[i])<extent && bases[i]>=0 && uint64_t(bases[i])+1<extent && (ids[i]==bases[i] || ids[i]==bases[i]+1) && !cpu[ids[i]])?policy_partner(ids,table,cap,uint64_t(bases[i])+(ids[i]==bases[i]?1:0)):-1;
 if(partners[i]>=0 && bases[partners[i]]!=bases[i])partners[i]=-1;
}
__global__ void probe_scatter_kernel(const float* src,float* a,float* b){policy_scatter_pair(src,a,b);}
