#include "cl_capacity.h"
// Development source; CUDA compilation and device acceptance remain required.
#include <limits>
#include <type_traits>

template <typename T>
void BAM_Feature_Store<T>::policy_check_geometry() {
    if (!std::is_same<T, float>::value || !h_pc || pageSize != 512 ||
        cache_entry_size != 512 || n_ctrls != 1 || mixed_split_cache_entries ||
        cpu_feature_path != 0 || numElems == 0 || numElems % 128 ||
        h_pc->pdt.replacement_policy > 1 || !mixed_io_enabled ||
        mixed_min_transfer_bytes != 512 || mixed_feature_row_bytes != 512 ||
        mixed_full_mask != 1 || numPages < digit_policy::scratch_pages)
        throw std::invalid_argument("Policy backend requires float128, 512B row I/O, mapped CPU, one SSD, legacy or FIFO transport");
}

template <typename T>
void BAM_Feature_Store<T>::policy_begin_cpu_cache(
    uint64_t host_rows, uint64_t count, uint64_t storage_rows) {
    policy_check_geometry();
    if (cpu_buffer_flag || policy_state || total_access || !count ||
        count > std::numeric_limits<uint32_t>::max() || storage_rows != numElems/128)
        throw std::invalid_argument("Exact CPU installation requires fresh store and matching extent");
    const auto cache_stats = get_gpu_cache_stats();
    if (cache_stats[2] || cache_stats[6]) throw std::runtime_error("Preload requires cold GPU cache");
    auto* rows = reinterpret_cast<const int64_t*>(host_rows);
    for (uint64_t i=0; i<count; ++i)
        if (!digit_policy::row_valid(rows[i], 0, storage_rows))
            throw std::invalid_argument("Primary CPU row outside payload");
    policy_state = 1;
    cuda_err_chk(cudaMalloc(&d_policy_counts, 2*sizeof(unsigned long long)));
    cuda_err_chk(cudaMemset(d_policy_counts, 0, 2*sizeof(unsigned long long)));
    cpu_backing_buffer(128, count);
    cuda_err_chk(cudaMalloc(&policy_row_map, storage_rows*sizeof(uint32_t)));
    cuda_err_chk(cudaMemset(policy_row_map, 0, storage_rows*sizeof(uint32_t)));
    CPU_buffer.row_map = policy_row_map;
    CPU_buffer.row_map_len = storage_rows;
    int64_t* chunk = nullptr;
    constexpr uint64_t chunk_rows = 65536;
    cuda_err_chk(cudaMalloc(&chunk, chunk_rows*sizeof(int64_t)));
    try {
        for (uint64_t lo=0; lo<count; lo+=chunk_rows) {
            const uint64_t n=std::min(chunk_rows, count-lo);
            cuda_err_chk(cudaMemcpy(chunk, rows+lo, n*sizeof(int64_t), cudaMemcpyHostToDevice));
            policy_read_rows<T><<<digit_policy::blocks_for(n, true),128>>>(
                a->d_array_ptr, d_range, CPU_buffer.device_cpu_buffer+lo*128,
                chunk, n, storage_rows, CPU_buffer, 0, 0,
                d_cpu_access, d_gpu_access, d_policy_counts, d_mixed_io_stats);
            cuda_err_chk(cudaGetLastError());
            cuda_err_chk(cudaDeviceSynchronize());
            policy_preload_rows += n;
        }
        cuda_err_chk(cudaFree(chunk));
    } catch (...) { cudaFree(chunk); throw; }
    seq_flag = false;
}

inline void ukl_segment_cuda_check(cudaError_t status) {
    if (status!=cudaSuccess) throw std::runtime_error(std::string("Segmented CUDA operation: ")+cudaGetErrorString(status));
}

// Independent bounded entry: external memory is owned by the Python lifecycle.
// Never allocate the entire pinned CPU cache or preload all rows inside one call.
template <typename T>
void BAM_Feature_Store<T>::policy_begin_external_cpu_cache(
    uint64_t host, uint64_t device, uint64_t count, uint64_t storage_rows) {
    policy_check_geometry();
    cl_capacity::validate(host, device, count, storage_rows, numElems,
                          cpu_buffer_flag, policy_state, total_access);
    const auto cache_stats=get_gpu_cache_stats();
    if (cache_stats[2] || cache_stats[6]) throw std::runtime_error("Preload requires cold GPU cache");
    policy_external_cpu=true;policy_state=1;
    CPU_buffer.cpu_buffer=reinterpret_cast<T*>(host);
    CPU_buffer.device_cpu_buffer=reinterpret_cast<T*>(device);
    CPU_buffer.cpu_buffer_dim=128;CPU_buffer.cpu_buffer_len=count;
    cpu_buffer_flag=true;
    ukl_segment_cuda_check(cudaMalloc(&d_policy_counts,2*sizeof(unsigned long long)));
    ukl_segment_cuda_check(cudaMemset(d_policy_counts,0,2*sizeof(unsigned long long)));
    ukl_segment_cuda_check(cudaMalloc(&policy_row_map,storage_rows*sizeof(uint32_t)));
    ukl_segment_cuda_check(cudaMemset(policy_row_map,0,storage_rows*sizeof(uint32_t)));
    CPU_buffer.row_map=policy_row_map;CPU_buffer.row_map_len=storage_rows;
    seq_flag=false;
}

template <typename T>
void BAM_Feature_Store<T>::policy_preload_chunk(uint64_t lo,uint64_t host_rows,uint64_t count) {
    if (!policy_external_cpu || policy_state!=1 || lo!=policy_preload_rows || !host_rows ||
        !count || count>32768 || lo>CPU_buffer.cpu_buffer_len || count>CPU_buffer.cpu_buffer_len-lo)
        throw std::invalid_argument("Consecutive bounded preload chunk required");
    auto* rows=reinterpret_cast<const int64_t*>(host_rows);
    for (uint64_t i=0;i<count;++i)
        if (!digit_policy::row_valid(rows[i],0,CPU_buffer.row_map_len))
            throw std::invalid_argument("Preload row outside payload");
    int64_t* chunk=nullptr;
    ukl_segment_cuda_check(cudaMalloc(&chunk,count*sizeof(int64_t)));
    try {
        ukl_segment_cuda_check(cudaMemcpy(chunk,rows,count*sizeof(int64_t),cudaMemcpyHostToDevice));
        policy_read_rows<T><<<digit_policy::blocks_for(count,true),128>>>(
            a->d_array_ptr,d_range,CPU_buffer.device_cpu_buffer+lo*128,
            chunk,count,CPU_buffer.row_map_len,CPU_buffer,0,0,
            d_cpu_access,d_gpu_access,d_policy_counts,d_mixed_io_stats);
        ukl_segment_cuda_check(cudaGetLastError());
        ukl_segment_cuda_check(cudaDeviceSynchronize());
        ukl_segment_cuda_check(cudaFree(chunk));
        policy_preload_rows+=count;
    } catch (...) {cudaFree(chunk);throw;}
}

template <typename T>
void BAM_Feature_Store<T>::policy_release_external_cpu_cache() {
    if (!policy_external_cpu)return;
    ukl_segment_cuda_check(cudaDeviceSynchronize());
    if (policy_row_map) {ukl_segment_cuda_check(cudaFree(policy_row_map));policy_row_map=nullptr;}
    if (d_policy_counts) {ukl_segment_cuda_check(cudaFree(d_policy_counts));d_policy_counts=nullptr;}
    CPU_buffer.cpu_buffer=nullptr;CPU_buffer.device_cpu_buffer=nullptr;
    CPU_buffer.row_map=nullptr;CPU_buffer.row_map_len=0;CPU_buffer.cpu_buffer_len=0;
    cpu_buffer_flag=false;policy_external_cpu=false;
    policy_state=policy_mode=0;policy_map_cursor=policy_preload_rows=0;
}

template <typename T>
void BAM_Feature_Store<T>::policy_write_cpu_map(
    uint64_t lo, uint64_t host_slots, uint64_t count) {
    if (policy_state != 1 || lo != policy_map_cursor || !count ||
        lo > CPU_buffer.row_map_len || count > CPU_buffer.row_map_len-lo)
        throw std::invalid_argument("CPU map must cover extent once, in consecutive chunks");
    auto* slots = reinterpret_cast<const uint32_t*>(host_slots);
    for (uint64_t i=0; i<count; ++i)
        if (!digit_policy::slot_valid(slots[i], CPU_buffer.cpu_buffer_len))
            throw std::invalid_argument("CPU slot exceeds selected hot-set budget");
    cuda_err_chk(cudaMemcpy(policy_row_map+lo, slots, count*sizeof(uint32_t), cudaMemcpyHostToDevice));
    policy_map_cursor += count;
}

template <typename T>
void BAM_Feature_Store<T>::policy_finish_cpu_cache() {
    if (policy_state != 1 || policy_map_cursor != CPU_buffer.row_map_len ||
        policy_preload_rows != CPU_buffer.cpu_buffer_len)
        throw std::runtime_error("Incomplete exact CPU cache installation");
    const auto stats=policy_stats();
    if (stats.back()) throw std::runtime_error("Invalid preload row");
    policy_state = 2;
}

template <typename T>
void BAM_Feature_Store<T>::policy_configure(uint32_t mode, uint64_t bytes) {
    policy_check_geometry();
    if (policy_state != 2 || policy_mode || total_access ||
        (mode != 1 && mode != 2 && mode != 3) ||
        (mode == 1 && (bytes || numPages*512 != digit_policy::scratch_bytes)) ||
        ((mode == 2 || mode == 3) && (!bytes || bytes != numPages*512)) ||
        (h_pc->pdt.replacement_policy != (mode == 3 ? 0u : 1u)))
        throw std::invalid_argument("Invalid policy capacity/state; no policy switches within a run");
    const auto stats=get_gpu_cache_stats();
    if (stats[2] || stats[6]) throw std::runtime_error("CPU preload polluted persistent GPU cache");
    policy_mode=mode;
    policy_feature_bytes=bytes;
    policy_state=3;
}

template <typename T>
std::vector<uint64_t> BAM_Feature_Store<T>::policy_stats() {
    uint64_t counts[2] = {0, 0};
    cuda_err_chk(cudaDeviceSynchronize());
    if (d_policy_counts)
        cuda_err_chk(cudaMemcpy(counts,d_policy_counts,sizeof(counts),cudaMemcpyDeviceToHost));
    return {1, policy_mode, policy_state, CPU_buffer.row_map_len,
            cpu_buffer_flag ? CPU_buffer.cpu_buffer_len : 0,
            policy_feature_bytes, digit_policy::scratch_bytes,
            policy_preload_rows, counts[0], counts[1]};
}

template <typename T>
std::vector<uint64_t> BAM_Feature_Store<T>::request_size_stats(){
 cuda_err_chk(cudaDeviceSynchronize());
 std::vector<uint64_t> v(5);
 cuda_err_chk(cudaMemcpy(v.data(),h_pc->pdt.request_size_counts,5*sizeof(uint64_t),cudaMemcpyDeviceToHost));
 return v;
}
