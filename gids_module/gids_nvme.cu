#include <pybind11/pybind11.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <iomanip>


#include <stdio.h>
#include <vector>

#include <bam_nvme.h>
#include <pybind11/stl.h>
#include "gids_kernel.cu"
//#include <bafs_ptr.h>


typedef std::chrono::high_resolution_clock Clock;

void GIDS_Controllers::init_GIDS_controllers(uint32_t num_ctrls, uint64_t q_depth, uint64_t num_q, 
                          const std::vector<int>& ssd_list){

  n_ctrls = num_ctrls;
  queueDepth = q_depth; // 1024
  numQueues = num_q; // 128

  for (size_t i = 0; i < n_ctrls; i++) {
 	printf("SSD index: %i\n", ssd_list[i]);
       	  ctrls.push_back(new Controller(ctrls_paths[ssd_list[i]], nvmNamespace, cudaDevice, queueDepth, numQueues));
  }
}


// 在主机上分配一块页锁定内存（大小为 dim × len 的 TYPE 数组）
// 并在设备端为这块内存创建映射，获取对应的 GPU 访问指针
// 这样，GPU 内核便可以通过 device_cpu_buffer 直接访问或修改这段主机内存，实现“零拷贝”（zero-copy）操作
template <typename TYPE>
void BAM_Feature_Store<TYPE>::cpu_backing_buffer(uint64_t dim, uint64_t len){
  TYPE* cpu_buffer_ptr; // cpu_buffer_ptr：将指向分配在CPU内存中的缓冲区
  TYPE* d_cpu_buffer_ptr; // d_cpu_buffer_ptr：将指向同一块内存在GPU侧的映射地址

  cuda_err_chk(cudaHostAlloc((TYPE **)&cpu_buffer_ptr, sizeof(TYPE) * dim * len, cudaHostAllocMapped));
  // 返回一个可以在 GPU 上使用的 设备指针 d_cpu_buffer_ptr
  cuda_err_chk(cudaHostGetDevicePointer((TYPE **)&d_cpu_buffer_ptr, (TYPE *)cpu_buffer_ptr, 0));

  CPU_buffer.cpu_buffer_dim = dim;
  CPU_buffer.cpu_buffer_len = len;
  CPU_buffer.cpu_buffer = cpu_buffer_ptr;
  CPU_buffer.device_cpu_buffer = d_cpu_buffer_ptr;
  cpu_buffer_flag = true;
}

template <typename TYPE>
void BAM_Feature_Store<TYPE>::init_controllers(GIDS_Controllers GIDS_ctrl, uint32_t ps, uint64_t read_off, 
                                                uint64_t cache_size, uint64_t num_ele, uint64_t num_ssd = 1) {

  numElems = num_ele;
  read_offset = read_off;
  n_ctrls = num_ssd;
  this -> pageSize = ps;
  this -> dim = ps / sizeof(TYPE);
  this -> total_access = 0; 

  ctrls = GIDS_ctrl.ctrls;

  std::cout << "Ctrl sizes: " << ctrls.size() << std::endl;
  uint64_t page_size = pageSize;  // 4KB
  uint64_t n_pages = cache_size * 1024LL*1024/page_size; // cache里能够承载的条目数量 (1024*1024)
  this -> numPages = n_pages;

  std::cout << "n pages: " << (int)(this->numPages) <<std::endl;
  std::cout << "page size: " << (int)(this->pageSize) << std::endl;
  std::cout << "num elements: " << this->numElems << std::endl; // 4096000000

  // page_cache是以range为单位的
  // page cache是 Bam software cache
  this -> h_pc = new page_cache_t(page_size, n_pages, cudaDevice, ctrls[0][0],(uint64_t)64, ctrls);
  page_cache_t *d_pc = (page_cache_t *)(h_pc->d_pc_ptr);


  uint64_t t_size = numElems * sizeof(TYPE);

  this -> h_range = new range_t<TYPE>((uint64_t)0, (uint64_t)numElems, (uint64_t)read_off,
                              (uint64_t)(t_size / page_size), (uint64_t)0,
                              (uint64_t)page_size, h_pc, cudaDevice, 
			      //REPLICATE
			      STRIPE
			      );

  
  this -> d_range = (range_d_t<TYPE> *)h_range->d_range_ptr;

  this -> vr.push_back(nullptr);
  this -> vr[0] = h_range;
  this -> a = new array_t<TYPE>(numElems, 0, vr, cudaDevice);

  cudaMalloc(&d_cpu_access, sizeof(uint64_t));
  cudaMemset(d_cpu_access, 0 , sizeof(uint64_t));

  cudaMalloc(&d_gpu_access, sizeof(uint64_t));
  cudaMemset(d_gpu_access, 0 , sizeof(uint64_t));

  cudaMalloc(&d_cpu_clock, sizeof(uint64_t));
  cudaMemset(d_cpu_clock, 0 , sizeof(uint64_t));

  cudaMalloc(&d_gpu_clock, sizeof(uint64_t));
  cudaMemset(d_gpu_clock, 0 , sizeof(uint64_t));
 
  return;
}





template <typename TYPE>
void  BAM_Feature_Store<TYPE>::set_window_buffering(uint64_t id_idx,  int64_t num_pages, int hash_off = 0){
	 uint64_t* idx_ptr = (uint64_t*) id_idx;
	 uint64_t page_size = pageSize;
	 set_window_buffering_kernel<TYPE><<<num_pages, 32>>>(a->d_array_ptr,idx_ptr, page_size, hash_off);
	 cuda_err_chk(cudaDeviceSynchronize())
}


template <typename TYPE>
void BAM_Feature_Store<TYPE>::print_stats_no_ctrl(){

  std::cout << "print stats: ";
  this->h_pc->print_reset_stats();
  std::cout << std::endl;

  std::cout << "print array reset: ";
  this->a->print_reset_stats();
  std::cout << std::endl;
}


template <typename TYPE>
void BAM_Feature_Store<TYPE>::print_stats(){
  std::cout << std::endl;
  std::cout << "Total feature access: " << std::dec << total_access << std::endl;
  std::cout << "cpu access total: " << std::dec << cpu_access_count_total << std::endl;
  std::cout << "gpu+ssd access total: " << std::dec << gpu_access_count_total << std::endl;
  std::cout << "cpu clock total: " << std::dec << cpu_access_time_total << std::endl;
  std::cout << "gpu clock total: " << std::dec << gpu_access_time_total << std::endl;
  std::cout << "Data access Time(ms): " << std::fixed << std::setprecision(5) << data_access_time << std::endl;

  //std::cout << "cpu_access_clock_time: " << std::dec << cpu_access_clock_time << std::endl;

  std::cout << "print stats: ";
  this->h_pc->print_reset_stats();
  std::cout << std::endl;

  std::cout << "print array reset: ";
  this->a->print_reset_stats();
  std::cout << std::endl;

  for(int i = 0; i < n_ctrls; i++){
 	std::cout << "print ctrl reset " << i << ": ";
  	(this->ctrls[i])->print_reset_stats();
  	std::cout << std::endl;

  }
 
  std::cout << "Kernel Time: \t " << this->kernel_time << std::endl;
  
  //this->kernel_time = 0;
  std::cout << "Total Access: \t " << this->total_access << std::endl;
  this->total_access = 0;
}





template <typename TYPE>
void BAM_Feature_Store<TYPE>::read_feature(uint64_t i_ptr, uint64_t i_index_ptr,
                                     int64_t num_index, int dim, int cache_dim, uint64_t key_off = 0) {

  // num_index表示这次读取的feature个数
  // dim是数据的维度
  TYPE *tensor_ptr = (TYPE *)i_ptr; // 指向数据的指针
  int64_t *index_ptr = (int64_t *)i_index_ptr; // 指向索引的指针

  uint64_t b_size = blkSize; // 每个CUDA块的线程数
  uint64_t n_warp = b_size / 32; // 每个块包含的warp数（一个warp通常包含32个线程）
  uint64_t g_size = (num_index+n_warp - 1) / n_warp; // 计算所需的网格大小，确保每个索引都被处理

  unsigned long long int prev_cpu_access_count,prev_gpu_access_count;
  cudaMemcpy(&prev_cpu_access_count, d_cpu_access, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&prev_gpu_access_count, d_gpu_access, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);

  cuda_err_chk(cudaDeviceSynchronize());

  auto t1 = Clock::now();

  if(cpu_buffer_flag == false){
    read_feature_kernel<TYPE><<<g_size, b_size>>>(a->d_array_ptr, tensor_ptr,
                                                  index_ptr, dim, num_index, cache_dim, key_off, d_gpu_access);
  }
  else{
    read_feature_kernel_with_cpu_backing_memory<<<g_size, b_size>>>(a->d_array_ptr, d_range, tensor_ptr,
                                                  index_ptr, dim, num_index, cache_dim, CPU_buffer, seq_flag,
                                                  d_cpu_access, key_off, d_gpu_access, d_cpu_clock, d_gpu_clock);
  }

  auto t3 = Clock::now();

  cuda_err_chk(cudaDeviceSynchronize());

  cudaMemcpy(&cpu_access_count, d_cpu_access, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&gpu_access_count, d_gpu_access, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
  // 将GPU上的d_cpu_access内存复制回主机变量cpu_access_count，以便获取CPU访问计数
  cpu_access_count_total += cpu_access_count - prev_cpu_access_count;
  gpu_access_count_total += gpu_access_count - prev_gpu_access_count;

  cudaMemcpy(&cpu_access_time, d_cpu_clock, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&gpu_access_time, d_gpu_clock, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);

  double gpu_frequency_hz= 735e6;

  auto cpu_access_clock_time = cpu_access_time / gpu_frequency_hz;

  //std::cout << "cpu_access_clock_time: \t " << cpu_access_clock_time << std::endl;


  auto read_feature_kernel_time = std::chrono::duration_cast<std::chrono::microseconds>( t3 - t1); // Microsecond (as int)

  auto us = std::chrono::duration_cast<std::chrono::microseconds>( t3 - t1); // Microsecond (as int)
  // auto ms = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1); // Microsecond (as int)
  
  const float ms_fractional = static_cast<float>(us.count()) / 1000; // Milliseconds (as float)
  data_access_time += ms_fractional;

  //std::cout << "Kernel Time once: \t " << ms_fractional << std::endl;

  kernel_time += ms_fractional;
  total_access += num_index;

  return;
}

template <typename TYPE>
void BAM_Feature_Store<TYPE>::get_io_stat(uint64_t i_ptr){
  uint64_t *tensor_ptr = (uint64_t *)i_ptr; // 指向数据的指针
  
  range_d_t<TYPE> range_d;
  this->a->get_io_stat(range_d);

  tensor_ptr[0] = range_d.access_cnt;
  tensor_ptr[1] = range_d.miss_cnt;
  tensor_ptr[2] = range_d.hit_cnt;
}

template <typename TYPE>
void BAM_Feature_Store<TYPE>::read_feature_hetero(int num_iter, const std::vector<uint64_t>&  i_ptr_list, const std::vector<uint64_t>&  i_index_ptr_list,
                                     const std::vector<uint64_t>&   num_index, int dim, int cache_dim, const std::vector<uint64_t>& key_off) {

  cudaStream_t streams[num_iter];
  for (int i = 0; i < num_iter; i++) {
      cudaStreamCreate(&streams[i]);
  }

  cuda_err_chk(cudaDeviceSynchronize());
  auto t1 = Clock::now();

  for(uint64_t i = 0;  i < num_iter; i++){
    uint64_t i_ptr = i_ptr_list[i];
    uint64_t    i_index_ptr =  i_index_ptr_list[i];  
    TYPE *tensor_ptr = (TYPE *) i_ptr;
    int64_t *index_ptr = (int64_t *)i_index_ptr;

    uint64_t b_size = blkSize;
    uint64_t n_warp = b_size / 32;
    uint64_t g_size = (num_index[i]+n_warp - 1) / n_warp;

    if(cpu_buffer_flag == false){
      read_feature_kernel<TYPE><<<g_size, b_size, 0, streams[i] >>>(a->d_array_ptr, tensor_ptr,
                                                    index_ptr, dim, num_index[i], cache_dim, key_off[i], d_gpu_access);
    }
    else{
      read_feature_kernel_with_cpu_backing_memory<<<g_size, b_size, 0, streams[i] >>>(a->d_array_ptr, d_range ,tensor_ptr,
                                                    index_ptr, dim, num_index[i], cache_dim, CPU_buffer, seq_flag, 
                                                    d_cpu_access,  key_off[i], d_gpu_access, d_cpu_clock, d_gpu_clock);
    }
    total_access += num_index[i];
  }

  for (int i = 0; i < num_iter; i++) {
    cudaStreamSynchronize(streams[i]);
  }

  cuda_err_chk(cudaDeviceSynchronize());
  cuda_err_chk(cudaDeviceSynchronize());
  cudaMemcpy(&cpu_access_count, d_cpu_access, sizeof(unsigned int), cudaMemcpyDeviceToHost);

  auto t2 = Clock::now();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(
      t2 - t1); // Microsecond (as int)
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      t2 - t1); // Microsecond (as int)
  const float ms_fractional =
      static_cast<float>(us.count()) / 1000; // Milliseconds (as float)

  //std::cout << "Duration = " << us.count() << "µs (" << ms_fractional << "ms)"
    //        << std::endl;
 
  kernel_time += ms_fractional;

  for (int i = 0; i < num_iter; i++) {
      cudaStreamDestroy(streams[i]);
  }
  

  return;
}


// 同时读多个batch的feature
template <typename TYPE>
void BAM_Feature_Store<TYPE>::read_feature_merged(int num_iter, const std::vector<uint64_t>&  i_ptr_list, const std::vector<uint64_t>&  i_index_ptr_list,
                                     const std::vector<uint64_t>&   num_index, int dim, int cache_dim=1024) {

  cudaStream_t streams[num_iter];
  for (int i = 0; i < num_iter; i++) {
      cudaStreamCreate(&streams[i]);
  }
  unsigned long long int prev_cpu_access_count, prev_gpu_access_count;
  unsigned long long int prev_cpu_access_clock, prev_gpu_access_clock;

  cudaMemcpy(&prev_cpu_access_count, d_cpu_access, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&prev_gpu_access_count, d_gpu_access, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&prev_cpu_access_clock, d_cpu_clock, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&prev_gpu_access_clock, d_gpu_clock, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);

  cuda_err_chk(cudaDeviceSynchronize());
  auto t1 = Clock::now();

  for(uint64_t i = 0;  i < num_iter; i++){
    uint64_t i_ptr = i_ptr_list[i];
    uint64_t    i_index_ptr =  i_index_ptr_list[i];         
    TYPE *tensor_ptr = (TYPE *) i_ptr;
    int64_t *index_ptr = (int64_t *)i_index_ptr;

    uint64_t b_size = blkSize;
    uint64_t n_warp = b_size / 32;
    uint64_t g_size = (num_index[i]+n_warp - 1) / n_warp;
    

    if(cpu_buffer_flag == false){
      read_feature_kernel<TYPE><<<g_size, b_size, 0, streams[i] >>>(a->d_array_ptr, tensor_ptr,
                                                    index_ptr, dim, num_index[i], cache_dim, 0, d_gpu_access);
    }
    else{
      read_feature_kernel_with_cpu_backing_memory<<<g_size, b_size, 0, streams[i] >>>(a->d_array_ptr, d_range ,tensor_ptr,
                                                    index_ptr, dim, num_index[i], cache_dim, CPU_buffer, seq_flag, 
                                                    d_cpu_access, 0, d_gpu_access, d_cpu_clock, d_gpu_clock);
    }
    total_access += num_index[i];
  }

  for (int i = 0; i < num_iter; i++) {
    cudaStreamSynchronize(streams[i]);
  }

  cuda_err_chk(cudaDeviceSynchronize());
  cuda_err_chk(cudaDeviceSynchronize());
  cudaMemcpy(&cpu_access_count, d_cpu_access, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&gpu_access_count, d_gpu_access, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&cpu_access_time, d_cpu_clock, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&gpu_access_time, d_gpu_clock, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);

  cpu_access_count_total += cpu_access_count - prev_cpu_access_count;
  gpu_access_count_total += gpu_access_count - prev_gpu_access_count;
  cpu_access_time_total += cpu_access_time - prev_cpu_access_clock;
  gpu_access_time_total += gpu_access_time - prev_gpu_access_clock;

  auto t2 = Clock::now();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(
      t2 - t1); // Microsecond (as int)
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      t2 - t1); // Microsecond (as int)
  const float ms_fractional =
      static_cast<float>(us.count()) / 1000; // Milliseconds (as float)

  data_access_time += ms_fractional;

  //std::cout << "Duration = " << us.count() << "µs (" << ms_fractional << "ms)"
    //        << std::endl;
 
  kernel_time += ms_fractional;

  for (int i = 0; i < num_iter; i++) {
      cudaStreamDestroy(streams[i]);
  }
  return;
}





template <typename TYPE>
void BAM_Feature_Store<TYPE>::read_feature_merged_hetero(int num_iter, const std::vector<uint64_t>&  i_ptr_list, const std::vector<uint64_t>&  i_index_ptr_list,
                                     const std::vector<uint64_t>&   num_index, int dim, int cache_dim, const std::vector<uint64_t>& key_off) {

  cudaStream_t streams[num_iter];
  for (int i = 0; i < num_iter; i++) {
      cudaStreamCreate(&streams[i]);
  }

  cuda_err_chk(cudaDeviceSynchronize());
  auto t1 = Clock::now();

  for(uint64_t i = 0;  i < num_iter; i++){
    uint64_t i_ptr = i_ptr_list[i];
    uint64_t    i_index_ptr =  i_index_ptr_list[i];         
    TYPE *tensor_ptr = (TYPE *) i_ptr;
    int64_t *index_ptr = (int64_t *)i_index_ptr;

    uint64_t b_size = blkSize;
    uint64_t n_warp = b_size / 32;
    uint64_t g_size = (num_index[i]+n_warp - 1) / n_warp;
    

    if(cpu_buffer_flag == false){
      read_feature_kernel<TYPE><<<g_size, b_size, 0, streams[i] >>>(a->d_array_ptr, tensor_ptr,
                                                    index_ptr, dim, num_index[i], cache_dim, key_off[i], d_gpu_access);
    }
    else{
      read_feature_kernel_with_cpu_backing_memory<<<g_size, b_size, 0, streams[i] >>>(a->d_array_ptr, d_range ,tensor_ptr,
                                                    index_ptr, dim, num_index[i], cache_dim, CPU_buffer, seq_flag, 
                                                    d_cpu_access, key_off[i], d_gpu_access, d_cpu_clock, d_gpu_clock);
    }
    total_access += num_index[i];
  }

  for (int i = 0; i < num_iter; i++) {
    cudaStreamSynchronize(streams[i]);
  }

  cuda_err_chk(cudaDeviceSynchronize());
  cuda_err_chk(cudaDeviceSynchronize());
  cudaMemcpy(&cpu_access_count, d_cpu_access, sizeof(unsigned int), cudaMemcpyDeviceToHost);

  auto t2 = Clock::now();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(
      t2 - t1); // Microsecond (as int)
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      t2 - t1); // Microsecond (as int)
  const float ms_fractional =
      static_cast<float>(us.count()) / 1000; // Milliseconds (as float)

  //std::cout << "Duration = " << us.count() << "µs (" << ms_fractional << "ms)"
    //        << std::endl;
 
  kernel_time += ms_fractional;

  for (int i = 0; i < num_iter; i++) {
      cudaStreamDestroy(streams[i]);
  }
  return;
}







template <typename TYPE>
void  BAM_Feature_Store<TYPE>::store_tensor(uint64_t tensor_ptr, uint64_t num, uint64_t offset){


//__global__ void write_feature_kernel2(Controller** ctrls, page_cache_d_t* pc, array_d_t<T> *dr, T* in_tensor_ptr, uint64_t dim, uint32_t num_ctrls) {
	TYPE* t_ptr = (TYPE*) tensor_ptr;
	page_cache_d_t* d_pc = (page_cache_d_t*) (h_pc -> d_pc_ptr);
	size_t b_size = 128;
	printf("num of writing node data: %llu dim: %llu\n", num, dim);
	write_feature_kernel2<TYPE><<<num, b_size>>>(h_pc->pdt.d_ctrls, d_pc, a->d_array_ptr, t_ptr, dim,  n_ctrls, offset/sizeof(TYPE));
	cuda_err_chk(cudaDeviceSynchronize());
  	h_pc->flush_cache();
   	cuda_err_chk(cudaDeviceSynchronize());
/*
  uint64_t s_offset = 0; 
  
  uint64_t total_cache_size = (pageSize * numPages);
  uint64_t total_tensor_size = (sizeof(TYPE) * num);
  uint64_t num_pages = total_tensor_size / pageSize;

  uint32_t n_tsteps = ceil((float)(total_tensor_size)/(float)total_cache_size);  
  printf("total iter: %llu\n", (unsigned long long) n_tsteps);
  TYPE* t_ptr = (TYPE*) tensor_ptr;
  
  page_cache_d_t* d_pc = (page_cache_d_t*) (h_pc -> d_pc_ptr);
  size_t b_size = 128;
  size_t g_size = (((total_tensor_size + pageSize -1) / pageSize)  + b_size - 1)/b_size;

  for (uint32_t cstep =0; cstep < n_tsteps; cstep++) {
    uint64_t cpysize = std::min(total_cache_size, (total_tensor_size-s_offset));


   // printf("first ele:%f\n", t_ptr[0]);
    cuda_err_chk(cudaMemcpy(h_pc->pdt.base_addr, t_ptr+s_offset+offset, cpysize, cudaMemcpyHostToDevice));
    printf("g size: %i num: %llu\n", g_size, num);
    write_feature_kernel<TYPE><<<100, b_size>>>(h_pc->pdt.d_ctrls, d_pc, a->d_array_ptr, t_ptr, num_pages, pageSize, offset, s_offset, n_ctrls);
    cuda_err_chk(cudaDeviceSynchronize());
    
  // printf("CALLLING FLUSH\n");
  // h_pc->flush_cache();
    //cuda_err_chk(cudaDeviceSynchronize());
    s_offset = s_offset + cpysize; 

  }
*/
}


template <typename TYPE>
void  BAM_Feature_Store<TYPE>::flush_cache(){
  h_pc->flush_cache();
  cuda_err_chk(cudaDeviceSynchronize());
}



template <typename TYPE>
void  BAM_Feature_Store<TYPE>::set_cpu_buffer(uint64_t idx_buffer, int num){
  // GIDS_Loader.set_cpu_buffer(pr_ten, num_pinned_nodes)
  int bsize = 1024;
  int grid = (num + bsize - 1) / bsize;
  
  // idx_buffer 本质上是一个指向 uint64_t 数组的地址
  uint64_t* idx_ptr = (uint64_t* ) idx_buffer;
  set_cpu_buffer_kernel<TYPE><<<grid,bsize>>>(d_range, idx_ptr, num, pageSize);
  cuda_err_chk(cudaDeviceSynchronize());
  
  set_cpu_buffer_data_kernel<TYPE><<<num,32>>>(a->d_array_ptr, CPU_buffer.device_cpu_buffer, idx_ptr, dim, num);
  cuda_err_chk(cudaDeviceSynchronize());

  // 1) 每个节点 copy dim 个 TYPE；TYPE 一般是 float (4 字节) 或 double (8 字节)
  // 2) num 表示一共拷了多少个节点
  size_t bytes_copied = (size_t)num * (size_t)dim * sizeof(TYPE);

    // 3) 计算 GB 数 （1 GB = 1024^3 字节）
  double gb_copied = bytes_copied / (1024.0 * 1024.0 * 1024.0);

    // 4) 打印到标准输出
  std::cout << "===== Debug: 已向 CPU 页锁内存 写入 " 
            << bytes_copied << " 字节"
            << " （约 " << std::fixed << std::setprecision(3) 
            << gb_copied << " GB） ====="
            << std::endl;

  seq_flag = false;


}



template <typename TYPE>
void  BAM_Feature_Store<TYPE>::set_offsets(uint64_t in_off, uint64_t index_off, uint64_t data_off){

 offset_array = new uint64_t[3];
    printf("set offset: in_off: %llu index_off: %llu data_off: %llu offset_ptr:%llu\n", in_off, index_off, data_off, (uint64_t) offset_array);

  offset_array[0] = (in_off);
  offset_array[1] = (index_off);
  offset_array[2] = (data_off);

}


template <typename TYPE>
uint64_t BAM_Feature_Store<TYPE>::get_offset_array(){
  return ((uint64_t) offset_array);
}

template <typename TYPE>
uint64_t BAM_Feature_Store<TYPE>::get_array_ptr(){
	return ((uint64_t) (a->d_array_ptr));
}


template <typename TYPE>
void  BAM_Feature_Store<TYPE>::read_tensor(uint64_t num, uint64_t offset){
  printf("offset:%llu\n", (unsigned long long) offset);
  seq_read_kernel<TYPE><<<1, 1>>>(a->d_array_ptr, num, offset);
  cuda_err_chk(cudaDeviceSynchronize());

}


template <typename TYPE>
unsigned long long int BAM_Feature_Store<TYPE>::get_cpu_access_count(){
	return cpu_access_count;
}

template <typename TYPE>
void BAM_Feature_Store<TYPE>::flush_cpu_access_count(){
	cpu_access_count = 0;
  cudaMemset(d_cpu_access, 0 , sizeof(unsigned long long int));
}

template <typename T>
BAM_Feature_Store<T> create_BAM_Feature_Store() {
    return BAM_Feature_Store<T>();
}



PYBIND11_MODULE(BAM_Feature_Store, m) {
  m.doc() = "Python bindings for an example library";

  namespace py = pybind11;

  //py::class_<BAM_Feature_Store<>, std::unique_ptr<BAM_Feature_Store<float>, py::nodelete>>(m, "BAM_Feature_Store")
    py::class_<BAM_Feature_Store<float>>(m, "BAM_Feature_Store_float")
      .def(py::init<>())
      .def("init_controllers", &BAM_Feature_Store<float>::init_controllers)
      .def("get_io_stat", &BAM_Feature_Store<float>::get_io_stat)
      .def("read_feature", &BAM_Feature_Store<float>::read_feature)
      .def("read_feature_hetero", &BAM_Feature_Store<float>::read_feature_hetero)

      .def("read_feature_merged_hetero", &BAM_Feature_Store<float>::read_feature_merged_hetero)
      .def("read_feature_merged", &BAM_Feature_Store<float>::read_feature_merged)
      .def("set_window_buffering", &BAM_Feature_Store<float>::set_window_buffering)
      .def("cpu_backing_buffer", &BAM_Feature_Store<float>::cpu_backing_buffer)
      .def("set_cpu_buffer", &BAM_Feature_Store<float>::set_cpu_buffer)

      .def("flush_cache", &BAM_Feature_Store<float>::flush_cache)
      .def("store_tensor",  &BAM_Feature_Store<float>::store_tensor)
      .def("read_tensor",  &BAM_Feature_Store<float>::read_tensor)

      .def("get_array_ptr", &BAM_Feature_Store<float>::get_array_ptr)
      .def("get_offset_array", &BAM_Feature_Store<float>::get_offset_array)
      .def("set_offsets", &BAM_Feature_Store<float>::set_offsets)
      .def("get_cpu_access_count", &BAM_Feature_Store<float>::get_cpu_access_count)
      .def("flush_cpu_access_count", &BAM_Feature_Store<float>::flush_cpu_access_count)

      .def("print_stats", &BAM_Feature_Store<float>::print_stats);



    py::class_<BAM_Feature_Store<int64_t>>(m, "BAM_Feature_Store_long")
      .def(py::init<>())
      .def("init_controllers", &BAM_Feature_Store<int64_t>::init_controllers)
      .def("get_io_stat", &BAM_Feature_Store<int64_t>::get_io_stat)
      .def("read_feature", &BAM_Feature_Store<int64_t>::read_feature)
      .def("read_feature_hetero", &BAM_Feature_Store<int64_t>::read_feature_hetero)

      .def("read_feature_merged", &BAM_Feature_Store<int64_t>::read_feature_merged)
      .def("read_feature_merged_hetero", &BAM_Feature_Store<int64_t>::read_feature_merged_hetero)


      .def("set_window_buffering", &BAM_Feature_Store<int64_t>::set_window_buffering)
      .def("cpu_backing_buffer", &BAM_Feature_Store<int64_t>::cpu_backing_buffer)
      .def("set_cpu_buffer", &BAM_Feature_Store<int64_t>::set_cpu_buffer)

      .def("flush_cache", &BAM_Feature_Store<int64_t>::flush_cache)
      .def("store_tensor",  &BAM_Feature_Store<int64_t>::store_tensor)
      .def("read_tensor",  &BAM_Feature_Store<int64_t>::read_tensor)

      .def("get_array_ptr", &BAM_Feature_Store<int64_t>::get_array_ptr)
      .def("get_offset_array", &BAM_Feature_Store<int64_t>::get_offset_array)
      .def("set_offsets", &BAM_Feature_Store<int64_t>::set_offsets)
      .def("get_cpu_access_count", &BAM_Feature_Store<int64_t>::get_cpu_access_count)
      .def("flush_cpu_access_count", &BAM_Feature_Store<int64_t>::flush_cpu_access_count)


      .def("print_stats", &BAM_Feature_Store<int64_t>::print_stats);




      py::class_<GIDS_Controllers>(m, "GIDS_Controllers")
      .def(py::init<>())
      .def("init_GIDS_controllers", &GIDS_Controllers::init_GIDS_controllers);

}

//gids


