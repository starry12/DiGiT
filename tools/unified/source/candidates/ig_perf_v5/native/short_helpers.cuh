#include "transfer_plan.h"
static std::map<Store*,uint64_t> short_loaded;
static std::map<Store*,uint64_t> short_last;
__global__ void short_map(range_d_t<float>* r,const int64_t* ids,uint64_t count,uint64_t base) {
    uint64_t j=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(j<count)r->set_cpu_buffer(ids[j],base+j);
}
__global__ void short_scan(data_page_t* pages,const uint32_t* expected,uint64_t first,uint64_t count,unsigned long long* errors) {
    uint64_t j=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(j<count && pages[first+j].cpu_feature_offset!=expected[j])atomicAdd(errors,1ULL);
}
static void short_begin(Store& s,uint64_t count) {
    need(s.cache_entry_size==4096 && (s.pageSize==4096 || s.pageSize==8192) && !s.cpu_buffer_flag,"short cache geometry/repeated allocation");
    need(count>0 && count<(1ULL<<31),"short cache row count");s.cpu_backing_buffer(1024,count);s.seq_flag=false;s.cpu_page_offsets.assign(s.numElems/1024,0);
    ck(cudaMemset(s.h_pc->pdt.base_addr,0,s.numPages*s.cache_entry_size));ck(cudaDeviceSynchronize());short_loaded[&s]=0;
}
static void short_fill(Store& s,py::array_t<int64_t,py::array::c_style> ids,py::array_t<float,py::array::c_style> values) {
    auto ib=ids.request();auto vb=values.request();uint64_t count=ib.size,loaded=short_loaded[&s];auto rows=static_cast<int64_t*>(ib.ptr);
    need(s.cpu_buffer_flag && ib.ndim==1 && vb.ndim==2 && vb.shape[1]==1024 && uint64_t(vb.shape[0])==count && count>0 && count<=16384,"short fill shapes");
    need(loaded+count<=s.CPU_buffer.cpu_buffer_len,"short fill exceeds allocation");
    for(uint64_t j=0;j<count;j++)need(rows[j]>=0 && uint64_t(rows[j])<s.numElems/1024 && (j?rows[j]>rows[j-1]:!loaded || uint64_t(rows[j])>short_last[&s]),"short rows must increase uniquely and be in range");
    std::memcpy(s.CPU_buffer.cpu_buffer+loaded*1024,vb.ptr,count*4096);need(std::memcmp(s.CPU_buffer.cpu_buffer+loaded*1024,vb.ptr,count*4096)==0,"short cache copy mismatch");
    int64_t* gpu;ck(cudaMalloc(&gpu,count*8));ck(cudaMemcpy(gpu,rows,count*8,cudaMemcpyHostToDevice));
    short_map<<<(count+255)/256,256>>>(s.d_range,gpu,count,loaded);ck(cudaGetLastError());ck(cudaDeviceSynchronize());ck(cudaFree(gpu));
    for(uint64_t j=0;j<count;j++)s.cpu_page_offsets[rows[j]]=uint32_t(((loaded+j)<<1)|1);
    short_loaded[&s]+=count;short_last[&s]=rows[count-1];admission_states[&s].loaded=short_loaded[&s];
}
static uint64_t short_verify_map(Store& s) {
    need(s.cpu_buffer_flag && short_loaded[&s]==s.CPU_buffer.cpu_buffer_len,"short cache incomplete");
    unsigned long long *errors,result;uint32_t* expected;uint64_t chunk=1<<20;ck(cudaMalloc(&errors,8));ck(cudaMemset(errors,0,8));ck(cudaMalloc(&expected,chunk*4));
    for(uint64_t at=0;at<s.cpu_page_offsets.size();at+=chunk){uint64_t count=std::min(chunk,s.cpu_page_offsets.size()-at);ck(cudaMemcpy(expected,s.cpu_page_offsets.data()+at,count*4,cudaMemcpyHostToDevice));
        short_scan<<<(count+255)/256,256>>>(s.h_range->rdt.pages,expected,at,count,errors);ck(cudaGetLastError());ck(cudaDeviceSynchronize());}
    ck(cudaMemcpy(&result,errors,8,cudaMemcpyDeviceToHost));ck(cudaFree(expected));ck(cudaFree(errors));return result;
}
static void short_prime(Store& s,uint64_t output,uint64_t ids,uint64_t count,uint64_t flags) {
    need(s.cpu_buffer_flag && count>0 && count<=16384,"invalid routing prime");
    s.cpu_buffer_flag=false;
    try{s.read_feature(output,ids,count,1024,1024,0,flags);}catch(...){s.cpu_buffer_flag=true;throw;}
    s.cpu_buffer_flag=true;
}
static bool transfer_range_ok(uint64_t offset,uint64_t bytes,bool writing) {
    if(!bytes || bytes>64ULL*1024*1024 || bytes%4096 || offset%4096)return false;
    bool inside=(offset>=IG_BASE_START && offset<=IG_BASE_END && bytes<=IG_BASE_END-offset) || (offset>=IG_FULL_START && offset<=IG_FULL_END && bytes<=IG_FULL_END-offset);
    if(inside)return true;
    return !writing && bytes==4096 && (offset==0 || offset==IG_BASE_START-4096 || offset==IG_BASE_END || offset==IG_FULL_START-4096 || offset==IG_FULL_END);
}
__global__ void short_transfer(page_cache_d_t* pc,uint64_t offset,uint64_t count,bool writing) {
    uint64_t j=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(j<count){Controller* c=pc->d_ctrls[0];uint64_t queue=j%c->n_qps;QueuePair* qp=c->d_qps+queue;
        uint64_t start=(offset+j*4096)>>qp->block_size_log;uint64_t blocks=4096>>qp->block_size_log;
        if(writing)write_data(pc,qp,start,blocks,j);else read_data(pc,qp,start,blocks,j);}
}
static void short_payload_io(Store& s,uint64_t offset,py::array_t<uint8_t,py::array::c_style> data,bool writing) {
    auto b=data.request();uint64_t bytes=b.size;
    need(b.ndim==1 && transfer_range_ok(offset,bytes,writing),"payload transfer outside frozen ranges/alignment/chunk");
    need(s.n_ctrls==1 && s.cache_entry_size==4096 && bytes<=s.numPages*4096 && !s.cpu_buffer_flag,"payload transfer geometry/state");
    need(writing || data.writeable(),"read output must be writable");
    if(writing)ck(cudaMemcpy(s.h_pc->pdt.base_addr,b.ptr,bytes,cudaMemcpyHostToDevice));
    else ck(cudaMemset(s.h_pc->pdt.base_addr,0xa5,bytes)); // A failed read cannot reuse the preceding write's bytes.
    short_transfer<<<(bytes/4096+127)/128,128>>>((page_cache_d_t*)s.h_pc->d_pc_ptr,offset,bytes/4096,writing);ck(cudaGetLastError());ck(cudaDeviceSynchronize());
    if(!writing)ck(cudaMemcpy(b.ptr,s.h_pc->pdt.base_addr,bytes,cudaMemcpyDeviceToHost));
}
static std::vector<float> short_raw_fixture() {
    data_page_t* pages;range_d_t<float>* range;float *cache,*mapped,*output;int64_t* ids;unsigned long long* stats;
    ck(cudaMalloc(&pages,8*sizeof(data_page_t)));ck(cudaMemset(pages,0,8*sizeof(data_page_t)));
    range_d_t<float> host{};host.pages=pages;host.cache.mixed_full_mask=1;
    ck(cudaMalloc(&range,sizeof(host)));ck(cudaMemcpy(range,&host,sizeof(host),cudaMemcpyHostToDevice));
    ck(cudaHostAlloc(&cache,3*4096,cudaHostAllocMapped));ck(cudaHostGetDevicePointer(&mapped,cache,0));
    for(int i=0;i<3072;i++)cache[i]=float(i)/32;
    int64_t hot[]={0,3,7};ck(cudaMalloc(&ids,40));ck(cudaMemcpy(ids,hot,24,cudaMemcpyHostToDevice));short_map<<<1,32>>>(range,ids,3,0);
    int64_t requests[]={7,0,3,7,0};ck(cudaMemcpy(ids,requests,40,cudaMemcpyHostToDevice));ck(cudaMalloc(&output,5*4096));ck(cudaMalloc(&stats,16));ck(cudaMemset(stats,0,16));
    GIDS_CPU_buffer<float> buffer{cache,mapped,1024,3};
    read_feature_kernel_with_cpu_backing_memory<float><<<2,128>>>(nullptr,range,output,ids,1024,5,1024,buffer,false,stats,0,stats+1,4096);
    ck(cudaGetLastError());ck(cudaDeviceSynchronize());std::vector<float> result(5120);ck(cudaMemcpy(result.data(),output,5*4096,cudaMemcpyDeviceToHost));
    uint64_t counts[2];ck(cudaMemcpy(counts,stats,16,cudaMemcpyDeviceToHost));need(counts[0]==5 && counts[1]==0,"raw fixture counters");
    ck(cudaFree(stats));ck(cudaFree(output));ck(cudaFree(ids));ck(cudaFreeHost(cache));ck(cudaFree(range));ck(cudaFree(pages));return result;
}
static void bind_short(py::module_& m) {
    m.def("short_begin",&short_begin);m.def("short_fill",&short_fill);m.def("short_verify_map",&short_verify_map);m.def("short_prime",&short_prime);
    m.def("transfer_range_ok",&transfer_range_ok);m.def("short_payload_io",&short_payload_io);m.def("short_raw_fixture",&short_raw_fixture);
}
