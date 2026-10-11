// Per-batch physical demand only. No host lookup and no graph-sized table.
// CPU mappings are immutable after preload. GPU validity is deliberately
// checked later, under the existing pair locks, not snapshotted here.
__device__ __forceinline__ uint64_t demand_bucket(uint64_t row,uint64_t mask) {
    row ^= row >> 30; row *= 0xbf58476d1ce4e5b9ULL;
    row ^= row >> 27; row *= 0x94d049bb133111ebULL;
    return (row ^ (row >> 31)) & mask;
}
__global__ void demand_insert(const int64_t* rows,uint64_t n,const data_page_t* pages,
                              uint64_t hot_start,unsigned long long* table,uint64_t mask) {
    uint64_t j=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(j>=n)return;
    uint64_t row=rows[j];
    if(row>=hot_start || (pages[row].cpu_feature_offset & 1U))return;
    uint64_t slot=demand_bucket(row,mask);
    for(uint64_t probe=0;probe<=mask;probe++,slot=(slot+1)&mask){
        auto old=atomicCAS(table+slot,~0ULL,row);
        if(old==~0ULL || old==row)return;
    }
}
__device__ __forceinline__ bool demand_contains(uint64_t row,const unsigned long long* table,uint64_t mask) {
    uint64_t slot=demand_bucket(row,mask);
    for(uint64_t probe=0;probe<=mask;probe++,slot=(slot+1)&mask){
        auto found=table[slot];if(found==row)return true;if(found==~0ULL)return false;
    }
    return false;
}
__global__ void demand_filter(const int64_t* rows,const bool* flags,uint64_t n,
                              const data_page_t* pages,uint64_t hot_start,
                              const unsigned long long* table,uint64_t mask,bool* filtered) {
    uint64_t j=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(j>=n)return;
    uint64_t row=rows[j],sibling=row^1ULL;
    filtered[j]=flags[j] && row<hot_start && sibling<hot_start &&
        !(pages[row].cpu_feature_offset & 1U) && demand_contains(sibling,table,mask);
}
static void filter_demand(cudaStream_t stream,const int64_t* rows,const bool* flags,uint64_t n,
                           const data_page_t* pages,uint64_t hot_start,
                           unsigned long long* table,uint64_t capacity,bool* filtered) {
    need(capacity>=2*n && !(capacity&(capacity-1)) && capacity<=4194304,
         "Demand hash capacity must be power of two, at least twice the request count");
    ck(cudaMemsetAsync(table,255,capacity*8,stream));
    demand_insert<<<(n+255)/256,256,0,stream>>>(rows,n,pages,hot_start,table,capacity-1);
    demand_filter<<<(n+255)/256,256,0,stream>>>(rows,flags,n,pages,hot_start,table,capacity-1,filtered);
    ck(cudaGetLastError());
}
