// Included after the frozen parent's kernels. Never changes BaM page-cache ABI.
#include "include/policy_math.h"

// mode: 0 = CPU preload, 1 = static bypass, 2 = FIFO with exact CPU fallback.
// One scratch page per launched warp; strided iterations reuse only that page.
// read_data submits actual NVMe READs and retains BaM's visibility replay.
// Bypass NEVER calls the cache lookup, insertion, eviction or resident probe.
template <typename T>
__global__ void policy_read_rows(
    array_d_t<T>* array, range_d_t<T>* range, T* output,
    const int64_t* indices, uint64_t count, uint64_t storage_rows,
    GIDS_CPU_buffer<T> cpu, uint32_t mode, uint64_t key,
    unsigned long long* cpu_count, unsigned long long* route_count,
    unsigned long long* policy_counts, unsigned long long* mixed_counts) {
    const uint64_t lane = threadIdx.x & 31;
    const uint64_t warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const uint64_t stride = gridDim.x * blockDim.x / 32;
    for (uint64_t i = warp; i < count; i += stride) {
        if (!digit_policy::row_valid(indices[i], key, storage_rows)) {
            if (!lane) atomicAdd(policy_counts + 1, 1ULL);
            continue; // No out-of-range DMA; host fails the batch after sync.
        }
        const uint64_t row = static_cast<uint64_t>(indices[i]) + key;
        const uint64_t page = digit_policy::page_of(row);
        const uint32_t slot = mode == 0 ? 0 : cpu.row_map[row];
        if (!digit_policy::slot_valid(slot, cpu.cpu_buffer_len)) {
            if (!lane) atomicAdd(policy_counts + 1, 1ULL);
            continue;
        }
        int resident = 0;
        if ((mode == 2 || mode == 3) && slot) {
            if (!lane) resident = full_try_pin_resident(range, page);
            resident = __shfl_sync(0xffffffff, resident, 0);
        }
        if (slot && !resident) {
            if (!lane) atomicAdd(cpu_count, 1ULL);
            for (uint64_t f = lane; f < 256; f += 32)
                output[i * 256 + f] = cpu.device_cpu_buffer[(uint64_t(slot)-1) * 256 + f];
        } else if ((mode == 2 || mode == 3)) {
            if (!lane) atomicAdd(route_count, 1ULL);
            // Fixed g2/r20 experiment always transfers full 4096-byte pages.
            read_mixed_feature_row(array, output, i, row, 256, 256, true,
                                  4096, 4096, 1, mixed_counts);
            if (resident) full_unpin_resident(range, page);
        } else {
            auto& cache = range->cache;
            if (!lane) {
                Controller* controller = cache.d_ctrls[0];
                const uint32_t queue = get_smid() % controller->n_qps;
                read_data(&cache, controller->d_qps + queue,
                          range->get_backing_page(page) * cache.n_blocks_per_page,
                          cache.n_blocks_per_page, warp);
                if (mode == 1) {
                    atomicAdd(route_count, 1ULL);
                    atomicAdd(policy_counts, 1ULL);
                    cache.useful_fill(warp, 0, 4096, true);
                    cache.useful_consume(warp, digit_policy::byte_in_page(row), 1024);
                }
            }
            __syncwarp();
            // Volatile reads avoid retaining stale values across DMA slot reuse.
            const volatile T* payload = reinterpret_cast<volatile T*>(
                cache.base_addr + warp * 4096 + digit_policy::byte_in_page(row));
            for (uint64_t f = lane; f < 256; f += 32)
                output[i * 256 + f] = payload[f];
        }
        __syncwarp(); // all lanes finish before this warp reuses its DMA page
    }
}
