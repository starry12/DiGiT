// Single live ticket. All pointer owners stay in Python until completion.
template<typename T> void BAM_Feature_Store<T>::prepare_policy_async(uint64_t stream_ptr,uint64_t max_rows) {
 if(async_ready || policy_state!=3 || policy_mode!=2 || !stream_ptr || !max_rows || max_rows>1048576)
  throw std::runtime_error("Invalid async preparation");
 async_stream=reinterpret_cast<cudaStream_t>(stream_ptr);async_max_rows=max_rows;
 uint64_t cap=1;while(cap<max_rows*2)cap<<=1;
 pair_hash_buffer=createBuffer(cap*sizeof(int),cudaDevice);pair_hash_capacity=cap;
 cuda_err_chk(cudaEventCreate(&async_begin));cuda_err_chk(cudaEventCreate(&async_end));
 cuda_err_chk(cudaMallocHost((void**)&async_errors,sizeof(unsigned long long)));async_ready=true;
}
template<typename T> uint64_t BAM_Feature_Store<T>::submit_policy_async(uint64_t out,uint64_t ids,uint64_t flags,int64_t n,uint64_t stream_ptr) {
 if(!async_ready || async_pending || n<=0 || uint64_t(n)>async_max_rows || !out || !ids || !flags || reinterpret_cast<cudaStream_t>(stream_ptr)!=async_stream)
  throw std::runtime_error("Invalid async read/overlapping ticket");
 async_pending=true;async_rows=n;++async_ticket;
 cuda_err_chk(cudaEventRecord(async_begin,async_stream));
 int* table=(int*)pair_hash_buffer.get();
 cuda_err_chk(cudaMemsetAsync(table,0xff,pair_hash_capacity*sizeof(int),async_stream));
 policy_pair_hash<<<(n+255)/256,256,0,async_stream>>>((int64_t*)ids,(int64_t*)flags,n,CPU_buffer.row_map,CPU_buffer.row_map_len,table,pair_hash_capacity,d_policy_counts+1);
 cuda_err_chk(cudaGetLastError());
 policy_mixed_rows<T><<<std::min<uint64_t>(1024,digit_policy::blocks_for(n,false)),128,0,async_stream>>>(a->d_array_ptr,d_range,(T*)out,(int64_t*)ids,(int64_t*)flags,n,CPU_buffer,table,pair_hash_capacity,d_cpu_access,d_gpu_access,d_policy_counts,d_mixed_io_stats);
 cuda_err_chk(cudaGetLastError());
 cuda_err_chk(cudaMemcpyAsync(async_errors,d_policy_counts+1,sizeof(unsigned long long),cudaMemcpyDeviceToHost,async_stream));
 cuda_err_chk(cudaEventRecord(async_end,async_stream));return async_ticket;
}
template<typename T> void BAM_Feature_Store<T>::collect_policy_async(uint64_t ticket) {
 if(!async_ready || !async_pending || ticket!=async_ticket)throw std::runtime_error("Wrong async ticket");
 cuda_err_chk(cudaEventSynchronize(async_end));
 float elapsed=0;cuda_err_chk(cudaEventElapsedTime(&elapsed,async_begin,async_end));
 data_access_time+=elapsed;kernel_time+=elapsed;total_access+=async_rows;
 async_pending=false;
 if(*async_errors)throw std::runtime_error("Async policy rejected feature row/CPU slot");
}
template<typename T> void BAM_Feature_Store<T>::close_policy_async() {
 if(!async_ready)return;
 cuda_err_chk(cudaStreamSynchronize(async_stream));
 // Exception cleanup must finish DMA before Python releases tensor owners.
 async_pending=false;
 cuda_err_chk(cudaEventDestroy(async_begin));cuda_err_chk(cudaEventDestroy(async_end));
 cuda_err_chk(cudaFreeHost(async_errors));async_errors=nullptr;async_ready=false;
}
