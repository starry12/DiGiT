#pragma once
#include <cstdint>

#ifdef __CUDACC__
#define POLICY_HD __host__ __device__
#else
#define POLICY_HD
#endif
namespace digit_policy {
constexpr uint64_t row_bytes = 1024;
constexpr uint64_t page_bytes = 4096;
constexpr uint64_t scratch_pages = 4096; // 16 MiB DMA transport, no resident tags.
constexpr uint64_t scratch_bytes = scratch_pages * page_bytes;

// Subtraction avoids wrapping a signed ID or overflowing base + offset.
POLICY_HD inline bool row_valid(int64_t index, uint64_t key, uint64_t rows) {
    return index >= 0 && key < rows && static_cast<uint64_t>(index) < rows-key;
}
POLICY_HD inline uint64_t page_of(uint64_t row) { return row / 4; }
POLICY_HD inline uint64_t byte_in_page(uint64_t row) { return (row % 4) * row_bytes; }
POLICY_HD inline uint64_t blocks_for(uint64_t rows, bool direct) {
    uint64_t blocks = rows / 4 + (rows % 4 != 0);
    return direct && blocks > scratch_pages / 4 ? scratch_pages / 4 : blocks;
}
POLICY_HD inline bool slot_valid(uint32_t slot, uint64_t cpu_rows) {
    return slot == 0 || slot <= cpu_rows;
}
}
#undef POLICY_HD
