// Tiny GPU-only calls to the production WB acquire/release and FIFO victim code.
// Every resident victim is clean. No controller is created or dereferenced.
__global__ void fifo_window_probe_kernel(page_cache_d_t cache, data_page_t* pages,
                                       uint64_t* out, int mode) {
  if (threadIdx.x || blockIdx.x) return;
  range_d_t<float> range;
  range.pages=pages;range.cache=cache;range.range_id=0;
  range.access_cnt.store(0);range.hit_cnt.store(0);range.read_io_cnt.store(0);
  for (int i=0;i<8;++i) {
    pages[i].state.store(i<3?VALID:INVALID);
    pages[i].prefetch_count.store(0);pages[i].prefetch_counter.store(0);
    pages[i].valid_subrows=i<3?1:0;pages[i].offset=i;
  }
  for (int i=0;i<3;++i) {
    cache.cache_pages[i].page_take_lock.store(UNLOCKED);
    cache.cache_pages[i].page_translation=(uint64_t)i<<cache.n_ranges_bits;
  }
  cache.cache_resident_pages->store(3);
  if (mode==0) {
    range.set_window_buffer_counter(0,1);range.set_window_buffer_counter(0,1);
    range.acquire_page(0,1,false,0,0);range.release_page(0,1);
  } else if (mode==1) pages[0].state.store(VALID|BUSY);
  else if (mode==2) cache.cache_pages[0].page_take_lock.store(LOCKED);
  out[1]=pages[0].state.load()&CNT_MASK;
  out[2]=pages[0].prefetch_counter.load();
  uint32_t slot=cache.find_slot(3,0,0,range.read_io_cnt,nullptr);
  out[0]=slot;
  pages[3].offset=slot;pages[3].state.store(VALID);pages[3].valid_subrows=1;
  // The pending read releases the window reference, making slot 0 eligible.
  if (mode==0) {range.acquire_page(0,1,false,0,0);range.release_page(0,1);}
  else if (mode==1) pages[0].state.store(VALID);
  else if (mode==2) cache.cache_pages[0].page_take_lock.store(UNLOCKED);
  out[3]=pages[0].state.load()&CNT_MASK;
  out[4]=pages[0].prefetch_counter.load();
  for (int address=4;address<=6;++address) {
    slot=cache.find_slot(address,0,0,range.read_io_cnt,nullptr);
    pages[address].offset=slot;pages[address].state.store(VALID);
    pages[address].valid_subrows=1;
  }
  out[5]=slot;
  out[6]=cache.fifo_ticket->load();out[7]=cache.cache_physical_inserts->load();
  out[8]=cache.cache_evictions->load();out[9]=cache.cache_resident_pages->load();
}

std::vector<uint64_t> fifo_window_probe(int mode) {
  if (mode<0 || mode>3) throw std::invalid_argument("Unknown FIFO fixture mode");
  auto pages=createBuffer(sizeof(data_page_t)*8,0);
  auto slots=createBuffer(sizeof(cache_page_t)*3,0);
  auto ranges=createBuffer(sizeof(pages_t),0);
  auto counts=createBuffer(sizeof(uint64_t)*8,0);
  auto output=createBuffer(sizeof(uint64_t)*10,0);
  cuda_err_chk(cudaMemset(counts.get(),0,sizeof(uint64_t)*8));
  pages_t ptr=(pages_t)pages.get();
  cuda_err_chk(cudaMemcpy(ranges.get(),&ptr,sizeof(ptr),cudaMemcpyHostToDevice));
  page_cache_d_t cache={};cache.page_size=512;cache.n_pages=3;
  cache.replacement_policy=CACHE_FIFO;cache.mixed_split_entries=false;
  cache.n_ranges_bits=1;cache.n_ranges_mask=1;cache.mixed_full_mask=1;
  cache.ranges=(pages_t*)ranges.get();cache.cache_pages=(cache_page_t*)slots.get();
  auto values=(simt::atomic<uint64_t,simt::thread_scope_device>*)counts.get();
  cache.cache_hits=values;cache.cache_inserts=values+1;
  cache.cache_physical_inserts=values+2;cache.cache_evictions=values+3;
  cache.cache_resident_pages=values+4;cache.fifo_ticket=values+5;
  fifo_window_probe_kernel<<<1,1>>>(cache,ptr,(uint64_t*)output.get(),mode);
  cuda_err_chk(cudaGetLastError());cuda_err_chk(cudaDeviceSynchronize());
  std::vector<uint64_t> result(10);
  cuda_err_chk(cudaMemcpy(result.data(),output.get(),sizeof(uint64_t)*10,cudaMemcpyDeviceToHost));
  return result;
}
