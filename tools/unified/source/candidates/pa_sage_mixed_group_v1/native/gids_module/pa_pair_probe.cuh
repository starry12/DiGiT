// GPU-only fixtures: production hash/scatter and resident paired cache lifecycle.
// No NVMe controller is initialized and no SSD command can be issued here.
__global__ void pair_probe_partners(const int64_t* ids,const bool* flags,const uint32_t* cpu,
    uint64_t n,uint64_t extent,const int* table,uint64_t cap,int64_t* out){
    uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
    out[i]=(ids[i]>=0 && uint64_t(ids[i])<extent && flags[i] && (ids[i]&7)<=1 && !cpu[ids[i]])?
        policy_partner(ids,table,cap,uint64_t(ids[i])^1ULL):-1;
}
std::vector<int64_t> pair_plan_probe(std::vector<int64_t> ids,std::vector<uint8_t> flags,std::vector<uint32_t> cpu){
    if(ids.empty() || ids.size()!=flags.size() || cpu.empty())throw std::invalid_argument("Invalid fixture inputs");
    uint64_t n=ids.size(),cap=1;while(cap<2*n)cap*=2;
    auto di=createBuffer(n*8,0),df=createBuffer(n,0),dc=createBuffer(cpu.size()*4,0),dt=createBuffer(cap*4,0),de=createBuffer(8,0),out=createBuffer(n*8,0);
    cuda_err_chk(cudaMemcpy(di.get(),ids.data(),n*8,cudaMemcpyHostToDevice));
    cuda_err_chk(cudaMemcpy(df.get(),flags.data(),n,cudaMemcpyHostToDevice));
    cuda_err_chk(cudaMemcpy(dc.get(),cpu.data(),cpu.size()*4,cudaMemcpyHostToDevice));
    cuda_err_chk(cudaMemset(dt.get(),0xff,cap*4));cuda_err_chk(cudaMemset(de.get(),0,8));
    policy_pair_hash<<<(n+255)/256,256>>>((int64_t*)di.get(),(bool*)df.get(),n,(uint32_t*)dc.get(),cpu.size(),(int*)dt.get(),cap,(unsigned long long*)de.get());
    pair_probe_partners<<<(n+255)/256,256>>>((int64_t*)di.get(),(bool*)df.get(),(uint32_t*)dc.get(),n,cpu.size(),(int*)dt.get(),cap,(int64_t*)out.get());
    cuda_err_chk(cudaGetLastError());cuda_err_chk(cudaDeviceSynchronize());
    std::vector<int64_t> result(n+1);
    cuda_err_chk(cudaMemcpy(result.data(),out.get(),n*8,cudaMemcpyDeviceToHost));
    cuda_err_chk(cudaMemcpy(result.data()+n,de.get(),8,cudaMemcpyDeviceToHost));return result;
}
__global__ void pair_resident_probe_kernel(page_cache_d_t cache,data_page_t* pages,Controller* dummy,
    Controller** ptrs,float* payload,float* output,unsigned long long* counts,uint64_t* result,int hints){
    unsigned lane=threadIdx.x;__shared__ range_d_t<float> range;
    if(!lane){
        dummy->n_qps=1;ptrs[0]=dummy;cache.d_ctrls=ptrs;
        range.pages=pages;range.cache=cache;range.range_id=0;
        range.access_cnt.store(0);range.hit_cnt.store(0);range.miss_cnt.store(0);
        for(int i=0;i<2;++i){pages[i].state.store(VALID);pages[i].offset=i;pages[i].valid_subrows=1;
            pages[i].prefetch_count.store(0);pages[i].prefetch_counter.store(hints);}
    }
    for(int i=lane;i<256;i+=32)payload[i]=float(i);
    __syncwarp();
    for(int iteration=0;iteration<hints;++iteration){
        policy_read_pair(&range,output,0,0,1,0,counts+9,counts+10);
        if(!lane){result[iteration*2]=pages[0].state.load()&CNT_MASK;result[iteration*2+1]=pages[0].prefetch_counter.load();}
    }
    __syncwarp();
    for(int i=lane;i<256;i+=32)if(output[i]!=float(i))atomicAdd(counts+30,1ULL);
    if(!lane){result[4]=pages[1].state.load()&CNT_MASK;result[5]=pages[1].prefetch_counter.load();result[6]=cache.cache_hits->load();}
}
std::vector<uint64_t> pair_resident_probe(int hints){
    if(hints<1 || hints>2)throw std::invalid_argument("Expected one or two window hints");
    auto pages=createBuffer(2*sizeof(data_page_t),0),payload=createBuffer(1024,0),out=createBuffer(1024,0);
    auto dummy=createBuffer(sizeof(Controller),0),ptrs=createBuffer(sizeof(Controller*),0),counts=createBuffer(40*8,0),result=createBuffer(8*8,0);
    cuda_err_chk(cudaMemset(counts.get(),0,40*8));cuda_err_chk(cudaMemset(result.get(),0,8*8));
    page_cache_d_t cache={};cache.page_size=512;cache.n_pages=2;cache.n_ctrls=1;cache.n_blocks_per_page=1;
    cache.base_addr=(uint8_t*)payload.get();cache.cache_hits=(simt::atomic<uint64_t,simt::thread_scope_device>*)counts.get();cache.pair_counts=(unsigned long long*)counts.get();
    cache.useful_counters=(unsigned long long*)counts.get()+32;
    pair_resident_probe_kernel<<<1,32>>>(cache,(data_page_t*)pages.get(),(Controller*)dummy.get(),(Controller**)ptrs.get(),(float*)payload.get(),(float*)out.get(),(unsigned long long*)counts.get(),(uint64_t*)result.get(),hints);
    cuda_err_chk(cudaGetLastError());cuda_err_chk(cudaDeviceSynchronize());std::vector<uint64_t> host(8);
    cuda_err_chk(cudaMemcpy(host.data(),result.get(),8*8,cudaMemcpyDeviceToHost));
    cuda_err_chk(cudaMemcpy(host.data()+7,(uint64_t*)counts.get()+30,8,cudaMemcpyDeviceToHost));return host;
}
