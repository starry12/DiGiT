// No controller or page cache exists in this fixture. It tests scheduling,
// lifetime/error guards and request-local counts, not SSD feature correctness.
__global__ void async_probe_hint_kernel(uint64_t* value, uint64_t requested) {
  if (!threadIdx.x && !blockIdx.x) *value = requested;
}
__global__ void async_probe_spin_kernel(uint64_t cycles) {
  uint64_t begin = clock64();
  while (clock64() - begin < cycles) asm volatile("");
}
__global__ void async_probe_gather_kernel(float* output, const int64_t* indices,
    uint64_t count, unsigned long long* cpu, const uint64_t* hint, uint64_t delay_cycles) {
  // A single block deliberately leaves GPU capacity for the independent model
  // fixture. Never interpret this synthetic overlap as a performance result.
  uint64_t begin = clock64();
  while (clock64() - begin < delay_cycles) asm volatile("");
  __syncthreads();
  for (uint64_t i = threadIdx.x; i < count * 128; i += blockDim.x) {
    uint64_t row = i / 128, column = i % 128;
    output[i] = static_cast<float>(indices[row] * 128 + column + (hint ? *hint : 0));
    if (!column && indices[row] % 3 == 0) atomicAdd(cpu, 1ULL);
  }
}
void async_probe_busy_wait(uint64_t stream_ptr, uint64_t cycles) {
  if (!stream_ptr || cycles > 1000000000ULL) throw std::invalid_argument("Invalid bounded probe delay");
  async_probe_spin_kernel<<<1, 32, 0, reinterpret_cast<cudaStream_t>(stream_ptr)>>>(cycles);
  async_cuda_check(cudaGetLastError());
}

struct AsyncFeatureProbe {
  AsyncFeatureRead read;
  uint64_t cpu_total = 0, requests = 0;
  explicit AsyncFeatureProbe(uint64_t stream_ptr) : read(stream_ptr) {}
  void hint(uint64_t pointer, uint64_t value, uint64_t stream_ptr) {
    read.check_stream(stream_ptr);
    if (!pointer) throw std::invalid_argument("Null probe hint pointer");
    async_probe_hint_kernel<<<1, 1, 0, read.io>>>(reinterpret_cast<uint64_t*>(pointer), value);
    async_cuda_check(cudaGetLastError());
  }
  uint64_t enqueue(const std::vector<uint64_t>& outputs,
      const std::vector<uint64_t>& indices, const std::vector<uint64_t>& rows,
      uint64_t hint_ptr, uint64_t cycles, uint64_t stream_ptr) {
    read.check_stream(stream_ptr); read.require_idle();
    if (cycles > 1000000000ULL) throw std::invalid_argument("Probe delay exceeds bound");
    uint64_t count = validate_async_vectors(outputs.size(), outputs, indices, rows, 128, 128);
    try {
      uint64_t ticket = read.begin(outputs.size(), count, stream_ptr);
      for (unsigned i = 0; i < outputs.size(); ++i) {
        async_probe_gather_kernel<<<1, 128, 0, read.children[i]>>>(
            reinterpret_cast<float*>(outputs[i]), reinterpret_cast<int64_t*>(indices[i]),
            rows[i], read.device_cpu_count, reinterpret_cast<uint64_t*>(hint_ptr), cycles);
        async_cuda_check(cudaGetLastError());
      }
      read.finish(outputs.size());
      return ticket;
    } catch (...) { read.fail(); throw; }
  }
  uint64_t collect(uint64_t ticket) {
    uint64_t cpu = read.collect(ticket); cpu_total += cpu; ++requests; return cpu;
  }
  bool query(uint64_t ticket) { return read.query(ticket); }
  void wait(uint64_t ticket, uint64_t stream_ptr) { read.wait(ticket, stream_ptr); }
  void require_idle() { read.require_idle(); }
  void drain() { read.drain(); }
};
