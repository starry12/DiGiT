// Development source; CUDA compilation and device acceptance remain required.
#include <limits>
#include <type_traits>

template <typename T>
void BAM_Feature_Store<T>::policy_check_geometry() {
    if (!std::is_same<T, float>::value || !h_pc || pageSize != 4096 ||
        cache_entry_size != 4096 || n_ctrls != 1 || mixed_split_cache_entries ||
        cpu_feature_path != 0 || numElems == 0 || numElems % 1024 ||
        h_pc->pdt.replacement_policy != 1 || !mixed_io_enabled ||
        mixed_min_transfer_bytes != 4096 || mixed_feature_row_bytes != 512 ||
        mixed_full_mask != 1 || numPages < digit_policy::scratch_pages)
        throw std::invalid_argument("Policy backend requires float128, full 4KiB I/O, mapped CPU, one SSD, FIFO transport");
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
        (mode != 1 && mode != 2) ||
        (mode == 1 && (bytes || numPages*4096 != digit_policy::scratch_bytes)) ||
        (mode == 2 && (!bytes || bytes != numPages*4096)))
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
