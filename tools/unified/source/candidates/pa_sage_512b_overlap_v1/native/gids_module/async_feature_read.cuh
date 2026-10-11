// Uses the existing production feature kernels, with request-local CPU counts.
template <typename TYPE>
void BAM_Feature_Store<TYPE>::prepare_async_reads(uint64_t stream_ptr) {
  if (async_read) throw std::runtime_error("Async reads are already prepared");
  if (!async_controllers_ready || sizeof(TYPE) != sizeof(float) || pageSize != 512 ||
      cache_entry_size != 512 || !cpu_buffer_flag || seq_flag || cpu_feature_path || mixed_io_enabled)
    throw std::runtime_error("Async reads require initialized mapped CPU cache, FP32, unmixed 512-B backend");
  async_read = std::make_shared<AsyncFeatureRead>(stream_ptr);
}

template <typename TYPE>
void BAM_Feature_Store<TYPE>::set_window_buffering_async(
    uint64_t index_ptr, int64_t count, int hash_off, uint64_t stream_ptr) {
  if (!async_read) throw std::runtime_error("Prepare async reads first");
  async_read->check_stream(stream_ptr);
  if (!index_ptr || count <= 0 || count > INT32_MAX || hash_off != 0)
    throw std::invalid_argument("Invalid bounded homogeneous async window hint");
  // The caller supplies input-ready ordering on the IO stream. Hints and the
  // subsequent ready marker use that same stream; reads cannot overtake hints.
  set_window_buffering_kernel<TYPE><<<count, 32, 0, async_read->io>>>(
      a->d_array_ptr, reinterpret_cast<uint64_t*>(index_ptr), pageSize, hash_off);
  try { async_cuda_check(cudaGetLastError()); }
  catch (...) { async_read->fail(); throw; }
}

template <typename TYPE>
uint64_t BAM_Feature_Store<TYPE>::read_feature_merged_async(int count,
    const std::vector<uint64_t>& outputs, const std::vector<uint64_t>& indices,
    const std::vector<uint64_t>& rows, int dim, int cache_dim, uint64_t stream_ptr) {
  if (!async_read) throw std::runtime_error("Prepare async reads first");
  async_read->check_stream(stream_ptr); async_read->require_idle();
  uint64_t total = validate_async_vectors(count, outputs, indices, rows, dim, cache_dim);
  try {
    uint64_t ticket = async_read->begin(count, total, stream_ptr);
    for (int i = 0; i < count; ++i) {
      uint64_t warps = blkSize / 32;
      read_feature_kernel_with_cpu_backing_memory<TYPE>
          <<<(rows[i] + warps - 1) / warps, blkSize, 0, async_read->children[i]>>>(
              a->d_array_ptr, d_range, reinterpret_cast<TYPE*>(outputs[i]),
              reinterpret_cast<int64_t*>(indices[i]), dim, rows[i], cache_dim,
              CPU_buffer, seq_flag, async_read->device_cpu_count, 0, d_gpu_access, pageSize);
      async_cuda_check(cudaGetLastError());
    }
    async_read->finish(count);
    return ticket;
  } catch (...) { async_read->fail(); throw; }
}

template <typename TYPE>
uint64_t BAM_Feature_Store<TYPE>::collect_feature_read(uint64_t ticket) {
  if (!async_read) throw std::runtime_error("Prepare async reads first");
  uint64_t cpu = async_read->collect(ticket);
  // The global synchronous counter is never reset or reused by an async read.
  cpu_access_count_total += cpu;
  total_access += async_read->rows;
  data_access_time += async_read->last_elapsed_ms;
  kernel_time += async_read->last_elapsed_ms;
  return cpu;
}

template <typename TYPE>
void BAM_Feature_Store<TYPE>::wait_feature_read(uint64_t ticket, uint64_t stream_ptr) {
  if (!async_read) throw std::runtime_error("Prepare async reads first");
  async_read->wait(ticket, stream_ptr);
}

template <typename TYPE>
bool BAM_Feature_Store<TYPE>::query_feature_read(uint64_t ticket) {
  if (!async_read) throw std::runtime_error("Prepare async reads first");
  return async_read->query(ticket);
}

template <typename TYPE>
void BAM_Feature_Store<TYPE>::synchronize_async_reads() {
  if (!async_read) throw std::runtime_error("Prepare async reads first");
  async_read->drain();
}
