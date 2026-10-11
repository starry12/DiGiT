#ifndef DIGIT_ASYNC_FEATURE_READ_H
#define DIGIT_ASYNC_FEATURE_READ_H

#include <cuda_runtime.h>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

inline void async_cuda_check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

// Shared by the real BaM read and the no-SSD test fixture. One host owner,
// one request awaiting collect, four persistent child streams, borrowed IO
// stream. Pointer lifetimes remain the caller's responsibility until completion.
struct AsyncFeatureRead {
  cudaStream_t io = nullptr;
  std::array<cudaStream_t, 4> children = {{nullptr, nullptr, nullptr, nullptr}};
  std::array<cudaEvent_t, 4> done = {{nullptr, nullptr, nullptr, nullptr}};
  cudaEvent_t ready = nullptr, complete = nullptr;
  unsigned long long* device_cpu_count = nullptr;
  unsigned long long* host_cpu_count = nullptr;
  uint64_t next_ticket = 1, active_ticket = 0, rows = 0;
  bool pending = false, poisoned = false;
  int device = -1;
  float last_elapsed_ms = 0;
  std::thread::id owner;

  explicit AsyncFeatureRead(uint64_t stream_ptr) : owner(std::this_thread::get_id()) {
    if (stream_ptr <= 2) throw std::invalid_argument("Explicit nonblocking IO stream required");
    io = reinterpret_cast<cudaStream_t>(stream_ptr);
    unsigned flags = 0;
    async_cuda_check(cudaStreamGetFlags(io, &flags));
    if (!(flags & cudaStreamNonBlocking)) throw std::invalid_argument("IO stream must be nonblocking");
    async_cuda_check(cudaGetDevice(&device));
    try {
      async_cuda_check(cudaEventCreate(&ready));
      async_cuda_check(cudaEventCreate(&complete));
      for (unsigned i = 0; i != 4; ++i) {
        async_cuda_check(cudaStreamCreateWithFlags(&children[i], cudaStreamNonBlocking));
        async_cuda_check(cudaEventCreateWithFlags(&done[i], cudaEventDisableTiming));
      }
      async_cuda_check(cudaMalloc(&device_cpu_count, sizeof(*device_cpu_count)));
      async_cuda_check(cudaHostAlloc(&host_cpu_count, sizeof(*host_cpu_count), cudaHostAllocDefault));
      *host_cpu_count = 0;
    } catch (...) { cleanup(); throw; }
  }
  AsyncFeatureRead(const AsyncFeatureRead&) = delete;
  AsyncFeatureRead& operator=(const AsyncFeatureRead&) = delete;
  ~AsyncFeatureRead() { cleanup(); }

  void cleanup() noexcept {
    // Never free an event/counter while a launched operation could still use it.
    // All waits are stream-local. The caller must retain the borrowed IO stream.
    int old_device = -1;
    cudaGetDevice(&old_device);
    if (device >= 0) cudaSetDevice(device);
    for (auto stream : children) if (stream) cudaStreamSynchronize(stream);
    if (io) cudaStreamSynchronize(io);
    if (device_cpu_count) cudaFree(device_cpu_count);
    if (host_cpu_count) cudaFreeHost(host_cpu_count);
    for (auto event : done) if (event) cudaEventDestroy(event);
    for (auto stream : children) if (stream) cudaStreamDestroy(stream);
    if (ready) cudaEventDestroy(ready);
    if (complete) cudaEventDestroy(complete);
    device_cpu_count = nullptr; host_cpu_count = nullptr;
    ready = complete = nullptr;
    done.fill(nullptr); children.fill(nullptr); io = nullptr;
    if (old_device >= 0 && old_device != device) cudaSetDevice(old_device);
  }
  void check_owner() const {
    if (owner != std::this_thread::get_id()) throw std::runtime_error("Async store has one host owner");
    int current = -1; async_cuda_check(cudaGetDevice(&current));
    if (current != device) throw std::runtime_error("Async store CUDA device changed");
    if (poisoned) throw std::runtime_error("Async store is poisoned after an enqueue failure");
  }
  void check_stream(uint64_t stream_ptr) const {
    check_owner();
    if (reinterpret_cast<cudaStream_t>(stream_ptr) != io)
      throw std::invalid_argument("Async operation must use its prepared IO stream");
  }
  void require_idle() const {
    check_owner();
    if (pending) throw std::runtime_error("Collect the pending feature request first");
  }
  void check_ticket(uint64_t ticket) const {
    check_owner();
    if (!pending || ticket != active_ticket) throw std::invalid_argument("Unknown or already collected feature ticket");
  }
  uint64_t begin(unsigned count, uint64_t requested_rows, uint64_t stream_ptr) {
    check_stream(stream_ptr); require_idle();
    if (!count || count > 4 || !requested_rows || next_ticket == std::numeric_limits<uint64_t>::max())
      throw std::invalid_argument("Invalid bounded async feature request");
    pending = true; active_ticket = next_ticket++; rows = requested_rows;
    async_cuda_check(cudaMemsetAsync(device_cpu_count, 0, sizeof(*device_cpu_count), io));
    async_cuda_check(cudaEventRecord(ready, io));
    for (unsigned i = 0; i < count; ++i)
      async_cuda_check(cudaStreamWaitEvent(children[i], ready, 0));
    return active_ticket;
  }
  void finish(unsigned count) {
    for (unsigned i = 0; i < count; ++i) {
      async_cuda_check(cudaEventRecord(done[i], children[i]));
      async_cuda_check(cudaStreamWaitEvent(io, done[i], 0));
    }
    async_cuda_check(cudaMemcpyAsync(host_cpu_count, device_cpu_count, sizeof(*host_cpu_count), cudaMemcpyDeviceToHost, io));
    async_cuda_check(cudaEventRecord(complete, io));
  }
  void fail() noexcept {
    // Drain only this API's streams before unwinding a partially launched read.
    for (auto stream : children) if (stream) cudaStreamSynchronize(stream);
    if (io) cudaStreamSynchronize(io);
    poisoned = true;
  }
  bool query(uint64_t ticket) const {
    check_ticket(ticket);
    auto status = cudaEventQuery(complete);
    if (status == cudaErrorNotReady) return false;
    async_cuda_check(status); return true;
  }
  void wait(uint64_t ticket, uint64_t consumer_stream_ptr) const {
    check_ticket(ticket);
    async_cuda_check(cudaStreamWaitEvent(reinterpret_cast<cudaStream_t>(consumer_stream_ptr), complete, 0));
  }
  uint64_t collect(uint64_t ticket) {
    check_ticket(ticket);
    try {
      async_cuda_check(cudaEventSynchronize(complete));
      async_cuda_check(cudaEventElapsedTime(&last_elapsed_ms, ready, complete));
      if (*host_cpu_count > rows) throw std::runtime_error("Async CPU count exceeds request rows");
      uint64_t count = *host_cpu_count;
      pending = false; active_ticket = 0;
      return count;
    } catch (...) { fail(); throw; }
  }
  void drain() {
    require_idle();
    async_cuda_check(cudaStreamSynchronize(io));
  }
};

inline uint64_t validate_async_vectors(int count,
    const std::vector<uint64_t>& outputs, const std::vector<uint64_t>& indices,
    const std::vector<uint64_t>& rows, int dim, int cache_dim) {
  if (count < 1 || count > 4 || outputs.size() != static_cast<size_t>(count) ||
      indices.size() != outputs.size() || rows.size() != outputs.size() || dim != 128 || cache_dim != 128)
    throw std::invalid_argument("Async reads require 1..4 FP32/128-column batches");
  uint64_t total = 0;
  for (int i = 0; i < count; ++i) {
    if (!outputs[i] || !indices[i] || !rows[i] || rows[i] > static_cast<uint64_t>(INT32_MAX) ||
        rows[i] > std::numeric_limits<uint64_t>::max() - total)
      throw std::invalid_argument("Invalid async pointer or row count");
    total += rows[i];
  }
  return total;
}
#endif
