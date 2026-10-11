// No controller, SSD or large graph. Exercise the actual production filter
// and the locked cache-hit path using bounded synthetic metadata.
static std::vector<uint8_t> demand_fixture(std::vector<int64_t> rows,std::vector<uint8_t> flags,
                                          std::vector<uint32_t> cpu,uint64_t hot_start) {
    uint64_t n=rows.size(),extent=cpu.size(),capacity=2;
    need(n && n<=131072 && flags.size()==n && extent && extent<=262144 && hot_start<=extent,
         "Bad demand fixture extent");
    for(uint64_t j=0;j<n;j++)need(rows[j]>=0 && uint64_t(rows[j])<extent && flags[j]<=1,"Bad fixture request");
    while(capacity<2*n)capacity*=2;
    std::vector<data_page_t> host(extent);
    for(uint64_t j=0;j<extent;j++)host[j].cpu_feature_offset=cpu[j]?1:0;
    data_page_t* pages;int64_t* ids;bool *input,*output;unsigned long long* table;
    ck(cudaMalloc(&pages,extent*sizeof(data_page_t)));ck(cudaMemcpy(pages,host.data(),extent*sizeof(data_page_t),cudaMemcpyHostToDevice));
    ck(cudaMalloc(&ids,n*8));ck(cudaMemcpy(ids,rows.data(),n*8,cudaMemcpyHostToDevice));
    ck(cudaMalloc(&input,n));ck(cudaMemcpy(input,flags.data(),n,cudaMemcpyHostToDevice));
    ck(cudaMalloc(&output,n));ck(cudaMalloc(&table,capacity*8));
    filter_demand(0,ids,input,n,pages,hot_start,table,capacity,output);
    ck(cudaDeviceSynchronize());std::vector<uint8_t> result(n);
    ck(cudaMemcpy(result.data(),output,n,cudaMemcpyDeviceToHost));
    ck(cudaFree(table));ck(cudaFree(output));ck(cudaFree(input));ck(cudaFree(ids));ck(cudaFree(pages));return result;
}
__global__ void pair_hit_probe(range_d_t<float>* range,uint64_t requested,bool sibling_valid,
                              unsigned long long* stats) {
    for(int j=0;j<2;j++){
        range->pages[j].state.store((j==requested || sibling_valid)?VALID:0,simt::memory_order_relaxed);
        range->pages[j].offset=7+j;range->pages[j].valid_subrows=1;
    }
    uint64_t slot=range->acquire_page_pair(requested,1,0,stats);
    stats[12]=slot;stats[13]=range->hit_cnt.load(simt::memory_order_relaxed);
    range->pages[requested].state.fetch_sub(1,simt::memory_order_release);
    stats[14]=range->pages[0].state.load(simt::memory_order_relaxed);
    stats[15]=range->pages[1].state.load(simt::memory_order_relaxed);
}
static std::vector<uint64_t> pair_hit_fixture(uint64_t requested,bool sibling_valid){
    need(requested<2,"Fixture row");data_page_t* pages;range_d_t<float>* range;unsigned long long* stats;
    ck(cudaMalloc(&pages,2*sizeof(data_page_t)));ck(cudaMemset(pages,0,2*sizeof(data_page_t)));
    ck(cudaMalloc(&stats,17*8));ck(cudaMemset(stats,0,17*8));
    range_d_t<float> host{};host.pages=pages;
    host.cache.cache_hits=reinterpret_cast<simt::atomic<uint64_t,simt::thread_scope_device>*>(stats+16);
    ck(cudaMalloc(&range,sizeof(host)));ck(cudaMemcpy(range,&host,sizeof(host),cudaMemcpyHostToDevice));
    pair_hit_probe<<<1,1>>>(range,requested,sibling_valid,stats);ck(cudaGetLastError());ck(cudaDeviceSynchronize());
    std::vector<uint64_t> result(17);ck(cudaMemcpy(result.data(),stats,17*8,cudaMemcpyDeviceToHost));
    ck(cudaFree(range));ck(cudaFree(stats));ck(cudaFree(pages));return result;
}
