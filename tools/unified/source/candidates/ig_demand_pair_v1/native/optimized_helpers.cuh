// Candidate-local changes. Retain existing feature kernels, cache policies,
// primary/replay accounting and SSD geometry; never change frozen backends.
static std::map<Store*,uint64_t> alias_cursor;
static std::map<Store*,unsigned long long*> request_errors;
__global__ void alias_write(data_page_t* pages,const uint32_t* slots,uint64_t first,uint64_t n) {
    uint64_t j=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(j<n) pages[first+j].cpu_feature_offset=slots[j] ? ((slots[j]-1)<<1)|1U : 0;
}
static void alias_chunk(Store& s,uint64_t first,py::array_t<uint32_t,py::array::c_style> slots) {
    auto b=slots.request();auto p=static_cast<uint32_t*>(b.ptr);uint64_t n=b.size;
    need(short_loaded[&s]==s.CPU_buffer.cpu_buffer_len,"Alias install before preload");
    need(b.ndim==1 && n>0 && n<=(1ULL<<20) && first==alias_cursor[&s] &&
         first<=s.cpu_page_offsets.size() && n<=s.cpu_page_offsets.size()-first,"Alias chunk gap/extent");
    for(uint64_t j=0;j<n;j++)need(p[j]<=s.CPU_buffer.cpu_buffer_len,"Alias slot outside cache");
    uint32_t* gpu;ck(cudaMalloc(&gpu,n*4));ck(cudaMemcpy(gpu,p,n*4,cudaMemcpyHostToDevice));
    alias_write<<<(n+255)/256,256>>>(s.h_range->rdt.pages,gpu,first,n);
    ck(cudaGetLastError());ck(cudaDeviceSynchronize());ck(cudaFree(gpu));
    for(uint64_t j=0;j<n;j++)s.cpu_page_offsets[first+j]=p[j]?((p[j]-1)<<1)|1U:0;
    alias_cursor[&s]+=n;
}
static void alias_finish(Store& s) {
    need(alias_cursor[&s]==s.cpu_page_offsets.size(),"Incomplete alias map");
    need(short_verify_map(s)==0,"Alias map verification failed");
}
__global__ void request_guard(const int64_t* rows,const bool* flags,uint64_t n,
                              uint64_t extent,uint64_t hot_start,bool mixed,
                              unsigned long long* errors) {
    uint64_t j=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(j<n && (rows[j]<0 || uint64_t(rows[j])>=extent ||
                (flags[j] && (!mixed || uint64_t(rows[j])>=hot_start)))) atomicAdd(errors,1ULL);
}
#include "demand_helpers.cuh"
static void device_read(Store& s,uint64_t output,uint64_t ids,uint64_t flags,uint64_t n,
                        uint64_t hot_start,uint64_t stream_ptr,uint64_t demand_table,
                        uint64_t demand_capacity,uint64_t demand_flags) {
    need(s.cpu_buffer_flag && !s.seq_flag && s.cpu_feature_path==0 && s.cache_entry_size==4096 &&
         short_loaded[&s]==s.CPU_buffer.cpu_buffer_len && n>0 && n<=2097152 && output && ids && flags,
         "Invalid direct device read state/extent");
    if(s.mixed_io_enabled)need(alias_cursor[&s]==s.cpu_page_offsets.size(),"Alias map not installed");
    auto stream=reinterpret_cast<cudaStream_t>(stream_ptr);
    auto& errors=request_errors[&s];if(!errors)ck(cudaMalloc(&errors,8));
    ck(cudaMemsetAsync(errors,0,8,stream));
    request_guard<<<(n+255)/256,256,0,stream>>>(reinterpret_cast<int64_t*>(ids),
        reinterpret_cast<bool*>(flags),n,s.numElems/1024,hot_start,s.mixed_io_enabled,errors);
    ck(cudaGetLastError());unsigned long long bad=0;
    ck(cudaMemcpyAsync(&bad,errors,8,cudaMemcpyDeviceToHost,stream));
    ck(cudaStreamSynchronize(stream));need(bad==0,"Device request outside verified layout");
    if(s.mixed_io_enabled){
        need(demand_table && demand_flags,"Missing demand scratch");
        filter_demand(stream,reinterpret_cast<int64_t*>(ids),reinterpret_cast<bool*>(flags),n,
                      s.h_range->rdt.pages,hot_start,reinterpret_cast<unsigned long long*>(demand_table),
                      demand_capacity,reinterpret_cast<bool*>(demand_flags));
        flags=demand_flags;
    }
    // Preserve the accepted 16K launch bound and its FIFO order. Each kernel
    // writes its slice of the final tensor; no per-chunk allocation or copy.
    for(uint64_t at=0;at<n;at+=16384){
        uint64_t count=std::min(uint64_t(16384),n-at),threads=s.blkSize;
        uint64_t blocks=(count+threads/32-1)/(threads/32);
        auto out=reinterpret_cast<float*>(output)+at*1024;
        auto rows=reinterpret_cast<int64_t*>(ids)+at;auto group=reinterpret_cast<bool*>(flags)+at;
        if(s.mixed_io_enabled)
            read_feature_kernel_mixed_with_cpu_backing_memory<float><<<blocks,threads,0,stream>>>(
                s.a->d_array_ptr,s.d_range,out,rows,group,1024,count,1024,s.CPU_buffer,s.d_cpu_access,0,
                s.d_gpu_access,s.pageSize,s.mixed_min_transfer_bytes,s.cache_entry_size,s.mixed_full_mask,s.d_mixed_io_stats);
        else
            read_feature_kernel_with_cpu_backing_memory<float><<<blocks,threads,0,stream>>>(
                s.a->d_array_ptr,s.d_range,out,rows,1024,count,1024,s.CPU_buffer,s.seq_flag,s.d_cpu_access,0,s.d_gpu_access,s.pageSize);
        ck(cudaGetLastError());
    }
    // Keep native timing honest and owners alive through all I/O completion.
    ck(cudaStreamSynchronize(stream));
}
__global__ void annotate_groups(const int64_t* sources,const int64_t* rows,const bool* flags,
                               uint64_t edges,uint64_t count,uint64_t extent,int64_t* preferred,
                               unsigned long long* errors) {
    uint64_t j=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(j<edges && flags[j]){
        if(sources[j]<0 || uint64_t(sources[j])>=count || rows[j]<0 || uint64_t(rows[j])>=extent){atomicAdd(errors,1ULL);return;}
        atomicMin(reinterpret_cast<unsigned long long*>(preferred+sources[j]),static_cast<unsigned long long>(rows[j]));
    }
}
__global__ void annotate_finish(const int64_t* logical,const int32_t* primary,uint64_t count,
                               uint64_t nodes,uint64_t extent,int64_t* rows,bool* flags,unsigned long long* errors) {
    uint64_t j=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(j>=count)return;
    bool grouped=rows[j]>=0;flags[j]=grouped;
    if(logical[j]<0 || uint64_t(logical[j])>=nodes){atomicAdd(errors,1ULL);rows[j]=0;flags[j]=false;return;}
    if(!grouped)rows[j]=primary[logical[j]];
    if(rows[j]<0 || uint64_t(rows[j])>=extent){atomicAdd(errors,1ULL);rows[j]=0;flags[j]=false;}
}
static void annotate(uint64_t logical,uint64_t primary,uint64_t sources,uint64_t edge_rows,uint64_t edge_flags,
                     uint64_t count,uint64_t edges,uint64_t nodes,uint64_t extent,uint64_t rows,uint64_t flags,
                     uint64_t errors,uint64_t stream_ptr) {
    need(count>0 && count<=2097152 && edges<=20971520 && primary && errors,"Annotation extent/pointers");
    auto stream=reinterpret_cast<cudaStream_t>(stream_ptr);
    ck(cudaMemsetAsync(reinterpret_cast<void*>(rows),0xff,count*8,stream));
    if(edges)annotate_groups<<<(edges+255)/256,256,0,stream>>>(reinterpret_cast<int64_t*>(sources),
        reinterpret_cast<int64_t*>(edge_rows),reinterpret_cast<bool*>(edge_flags),edges,count,extent,
        reinterpret_cast<int64_t*>(rows),reinterpret_cast<unsigned long long*>(errors));
    annotate_finish<<<(count+255)/256,256,0,stream>>>(reinterpret_cast<int64_t*>(logical),
        reinterpret_cast<int32_t*>(primary),count,nodes,extent,reinterpret_cast<int64_t*>(rows),
        reinterpret_cast<bool*>(flags),reinterpret_cast<unsigned long long*>(errors));
    ck(cudaGetLastError());
}
static std::vector<float> alias_fixture(bool mixed) {
    data_page_t* pages;range_d_t<float>* range;float *cache,*mapped,*out;int64_t* ids;bool* flags;
    uint32_t* slots;unsigned long long* stats;
    ck(cudaMalloc(&pages,8*sizeof(data_page_t)));ck(cudaMemset(pages,0,8*sizeof(data_page_t)));
    range_d_t<float> host{};host.pages=pages;host.cache.mixed_full_mask=1;
    ck(cudaMalloc(&range,sizeof(host)));ck(cudaMemcpy(range,&host,sizeof(host),cudaMemcpyHostToDevice));
    ck(cudaHostAlloc(&cache,3*4096,cudaHostAllocMapped));ck(cudaHostGetDevicePointer(&mapped,cache,0));
    for(int j=0;j<3072;j++)cache[j]=float(j)/32;
    uint32_t hs[]={1,1,0,2,3,3,0,1};ck(cudaMalloc(&slots,32));ck(cudaMemcpy(slots,hs,32,cudaMemcpyHostToDevice));
    alias_write<<<1,32>>>(pages,slots,0,8);
    int64_t hi[]={7,0,1,5,4,3};ck(cudaMalloc(&ids,48));ck(cudaMemcpy(ids,hi,48,cudaMemcpyHostToDevice));
    ck(cudaMalloc(&flags,6));ck(cudaMemset(flags,1,6));ck(cudaMalloc(&out,6*4096));
    ck(cudaMalloc(&stats,16));ck(cudaMemset(stats,0,16));GIDS_CPU_buffer<float> buf{cache,mapped,1024,3};
    if(mixed)read_feature_kernel_mixed_with_cpu_backing_memory<float><<<2,128>>>(nullptr,range,out,ids,flags,1024,6,1024,buf,stats,0,stats+1,8192,4096,4096,3,nullptr);
    else read_feature_kernel_with_cpu_backing_memory<float><<<2,128>>>(nullptr,range,out,ids,1024,6,1024,buf,false,stats,0,stats+1,4096);
    ck(cudaGetLastError());ck(cudaDeviceSynchronize());uint64_t counts[2];ck(cudaMemcpy(counts,stats,16,cudaMemcpyDeviceToHost));
    need(counts[0]==6 && counts[1]==0,"Alias fixture CPU/SSD counts");std::vector<float> result(6144);
    ck(cudaMemcpy(result.data(),out,6*4096,cudaMemcpyDeviceToHost));
    ck(cudaFree(stats));ck(cudaFree(out));ck(cudaFree(flags));ck(cudaFree(ids));ck(cudaFree(slots));
    ck(cudaFreeHost(cache));ck(cudaFree(range));ck(cudaFree(pages));return result;
}
#include "demand_fixtures.cuh"
static void bind_optimized(py::module_& m){
    m.attr("DEMAND_PAIR_ABI")=1;
    m.def("demand_fixture",&demand_fixture);m.def("pair_hit_fixture",&pair_hit_fixture);
    m.attr("OPTIMIZED_IG_ABI")=1;m.def("alias_chunk",&alias_chunk);m.def("alias_finish",&alias_finish);
    m.def("device_read",&device_read);m.def("annotate",&annotate);m.def("alias_fixture",&alias_fixture);
}
