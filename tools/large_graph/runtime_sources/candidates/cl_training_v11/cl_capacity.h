#pragma once
#include <cstdint>
#include <stdexcept>

#ifndef CL_STORAGE_ROWS_MAX
#error CL_STORAGE_ROWS_MAX must identify the arm at build time
#endif

namespace cl_capacity {
constexpr uint64_t hot_max = 97840809ULL;
constexpr uint64_t storage_max = CL_STORAGE_ROWS_MAX;
static_assert(storage_max == 978408104ULL || storage_max == 1174089720ULL,
              "Only padded CL GIDS or DiGiT capacity supported");

// Shared by the actual installation path and allocation-free boundary tests.
// All inputs use the production uint64_t ABI; no pointers are dereferenced.
inline void validate(uint64_t host, uint64_t device, uint64_t count,
                     uint64_t storage_rows, uint64_t num_elements,
                     bool cpu_buffer, uint64_t state, uint64_t accesses) {
    if (cpu_buffer || state || accesses || !host || !device || !count ||
        count > hot_max || count > storage_rows || !storage_rows ||
        storage_rows > storage_max || storage_rows % 8 ||
        !num_elements || num_elements % 128 || storage_rows != num_elements / 128)
        throw std::invalid_argument("CL external cache capacity, geometry or state mismatch");
}
}
