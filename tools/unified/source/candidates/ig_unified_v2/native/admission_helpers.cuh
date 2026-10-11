#include <pybind11/numpy.h>
#include <map>
namespace py = pybind11;
using Store = BAM_Feature_Store<float>;
struct AdmissionState { uint64_t loaded=0, last=0, start=0; bool begun=false; };
static std::map<Store*,AdmissionState> admission_states;
static void ck(cudaError_t x) {if(x!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(x));}
static void need(bool x,const char* msg) {if(!x)throw std::invalid_argument(msg);}
__global__ void admission_map(range_d_t<float>* r,uint64_t first,uint64_t count,uint64_t base) {
    uint64_t j=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(j<count)r->set_cpu_buffer(first+j,base+j);
}
__global__ void admission_scan(data_page_t* pages,uint64_t count,uint64_t start,uint64_t hot,unsigned long long* errors) {
    for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=uint64_t(gridDim.x)*blockDim.x) {
        uint32_t expected=(i>=start && i-start<hot)?uint32_t(((i-start)<<1)|1):0;
        if(pages[i].cpu_feature_offset!=expected)atomicAdd(errors,1ULL);
        if(pages[i].state.load(simt::memory_order_relaxed)!=INVALID || pages[i].valid_subrows!=0)atomicAdd(errors+1,1ULL);
    }
}
static std::map<std::string,uint64_t> admission_info(Store& s) {
    return {{"storage_rows",s.numElems/1024},{"range_pages",s.h_range->rdt.page_count},
      {"data_page_bytes",sizeof(data_page_t)},{"range_bytes",s.h_range->rdt.page_count*sizeof(data_page_t)},
      {"range_pointer",uint64_t(s.h_range->rdt.pages)},{"gpu_cache_pointer",uint64_t(s.h_pc->pdt.base_addr)},
      {"gpu_cache_bytes",s.numPages*s.cache_entry_size},{"cache_entries",s.numPages},
      {"cache_entry_bytes",s.cache_entry_size},{"storage_slot_bytes",s.pageSize},
      {"cpu_pointer",s.cpu_buffer_flag?uint64_t(s.CPU_buffer.cpu_buffer):0},
      {"cpu_device_pointer",s.cpu_buffer_flag?uint64_t(s.CPU_buffer.device_cpu_buffer):0},
      {"cpu_cache_bytes",s.cpu_buffer_flag?s.CPU_buffer.cpu_buffer_len*s.CPU_buffer.cpu_buffer_dim*4:0},
      {"cpu_loaded_rows",admission_states[&s].loaded},{"native_fifo",s.h_pc->pdt.replacement_policy},
      {"native_split_entries",s.mixed_split_cache_entries}};
}
static void admission_begin(Store& s,uint64_t count) {
    need(s.cache_entry_size==4096 && s.pageSize==8192 && s.mixed_io_enabled,"requires IG split mixed geometry");
    need(count>0 && count<(1ULL<<31) && !s.cpu_buffer_flag,"invalid or repeated CPU allocation");
    s.cpu_backing_buffer(1024,count);s.seq_flag=false;
    // Retain the real host table used by the existing staged metadata accounting.
    s.cpu_page_offsets.assign(s.numElems/1024,0);
    ck(cudaMemset(s.h_pc->pdt.base_addr,0,s.numPages*s.cache_entry_size));
    ck(cudaDeviceSynchronize());
}
static void admission_fill(Store& s,uint64_t first,py::array_t<float,py::array::c_style> data) {
    auto& st=admission_states[&s];auto b=data.request();
    need(s.cpu_buffer_flag && b.ndim==2 && b.shape[1]==1024,"invalid feature block shape");uint64_t count=b.shape[0];
    need(count>0 && count<=16384,"invalid bounded feature block");
    need(first<s.numElems/1024 && count<=s.numElems/1024-first && st.loaded+count<=s.CPU_buffer.cpu_buffer_len,"CPU fill bounds");
    need(!st.begun || first==st.last+1,"CPU fill must be contiguous, increasing, unique");
    if(!st.begun)st.start=first;
    std::memcpy(s.CPU_buffer.cpu_buffer+st.loaded*1024,b.ptr,count*4096);
    need(std::memcmp(s.CPU_buffer.cpu_buffer+st.loaded*1024,b.ptr,count*4096)==0,"host cache copy mismatch");
    admission_map<<<(count+255)/256,256>>>(s.d_range,first,count,st.loaded);ck(cudaGetLastError());ck(cudaDeviceSynchronize());
    for(uint64_t j=0;j<count;j++)s.cpu_page_offsets[first+j]=uint32_t(((st.loaded+j)<<1)|1);
    st.loaded+=count;st.last=first+count-1;st.begun=true;
}
static std::vector<uint64_t> admission_verify_pages(Store& s) {
    auto& st=admission_states[&s];need(s.cpu_buffer_flag && st.loaded==s.CPU_buffer.cpu_buffer_len,"cache incomplete");
    unsigned long long* e;ck(cudaMalloc(&e,16));ck(cudaMemset(e,0,16));
    admission_scan<<<4096,256>>>(s.h_range->rdt.pages,s.h_range->rdt.page_count,st.start,st.loaded,e);
    ck(cudaGetLastError());ck(cudaDeviceSynchronize());std::vector<uint64_t> r(2);ck(cudaMemcpy(r.data(),e,16,cudaMemcpyDeviceToHost));ck(cudaFree(e));
    return r;
}
static void admission_read_cpu(Store& s,uint64_t output,uint64_t indices,uint64_t count,uint64_t flags,
                               py::array_t<int64_t,py::array::c_style> host_ids) {
    auto& st=admission_states[&s];auto b=host_ids.request();
    need(st.loaded==s.CPU_buffer.cpu_buffer_len && b.ndim==1 && uint64_t(b.size)==count && count<=16384,"invalid cached-only request");
    if(!count)return;
    auto ids=static_cast<int64_t*>(b.ptr);std::vector<int64_t> actual(count);
    ck(cudaMemcpy(actual.data(),reinterpret_cast<void*>(indices),count*8,cudaMemcpyDeviceToHost));
    need(std::memcmp(ids,actual.data(),count*8)==0,"GPU and host IDs differ");
    for(uint64_t j=0;j<count;j++)need(ids[j]>=0 && uint64_t(ids[j])>=st.start && uint64_t(ids[j])-st.start<st.loaded,"cached-only reader rejects a cold row before launch");
    s.read_feature(output,indices,count,1024,1024,0,flags);
}
static std::vector<float> admission_row_fixture() {
    // Exercise the changed production reader without a controller. Every requested
    // entry is CPU-mapped/INVALID, so full_try_pin_resident must not dereference dr.
    data_page_t* pages=nullptr;range_d_t<float>* range=nullptr;float *cache=nullptr,*output=nullptr;
    int64_t* ids=nullptr;bool* flags=nullptr;unsigned long long* counters=nullptr;
    ck(cudaMalloc(&pages,8*sizeof(data_page_t)));ck(cudaMemset(pages,0,8*sizeof(data_page_t)));
    range_d_t<float> host_range{};host_range.pages=pages;host_range.cache.mixed_full_mask=1;
    ck(cudaMalloc(&range,sizeof(host_range)));ck(cudaMemcpy(range,&host_range,sizeof(host_range),cudaMemcpyHostToDevice));
    ck(cudaHostAlloc(&cache,3*4096,cudaHostAllocMapped));float* mapped;ck(cudaHostGetDevicePointer(&mapped,cache,0));
    for(int i=0;i<3072;i++)cache[i]=float(i)/16;
    GIDS_CPU_buffer<float> buf{cache,mapped,1024,3};
    admission_map<<<1,32>>>(range,2,3,0);
    int64_t host_ids[]={4,2,3,4,2};ck(cudaMalloc(&ids,40));ck(cudaMemcpy(ids,host_ids,40,cudaMemcpyHostToDevice));
    ck(cudaMalloc(&flags,5));ck(cudaMemset(flags,0,5));ck(cudaMalloc(&output,5*4096));
    ck(cudaMalloc(&counters,16));ck(cudaMemset(counters,0,16));
    read_feature_kernel_mixed_with_cpu_backing_memory<float><<<2,128>>>(nullptr,range,output,ids,flags,1024,5,1024,buf,counters,0,counters+1,8192,4096,4096,3,nullptr);
    ck(cudaGetLastError());ck(cudaDeviceSynchronize());
    std::vector<float> result(5120);ck(cudaMemcpy(result.data(),output,5*4096,cudaMemcpyDeviceToHost));
    uint64_t counts[2];ck(cudaMemcpy(counts,counters,16,cudaMemcpyDeviceToHost));need(counts[0]==5 && counts[1]==0,"fixture routing counter");
    ck(cudaFree(counters));ck(cudaFree(output));ck(cudaFree(flags));ck(cudaFree(ids));ck(cudaFreeHost(cache));ck(cudaFree(range));ck(cudaFree(pages));
    return result;
}
static void bind_admission(py::module_& m) {
    m.def("admission_info",&admission_info);m.def("admission_begin",&admission_begin);
    m.def("admission_fill",&admission_fill);m.def("admission_verify_pages",&admission_verify_pages);
    m.def("admission_read_cpu",&admission_read_cpu);
    m.def("admission_row_fixture",&admission_row_fixture);
    m.attr("ADMISSION_API")=1;
}
