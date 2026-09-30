#include "gids_module/include/policy_math.h"
#include <cassert>
#include <cstdint>
#include <limits>
#include <set>
int main() {
    using namespace digit_policy;
    assert(!row_valid(-1, 0, 24));
    assert(!row_valid(24, 0, 24));
    assert(!row_valid(1, std::numeric_limits<uint64_t>::max(), 24));
    assert(row_valid(23, 0, 24));
    assert(row_valid(3, 20, 24));
    assert(!row_valid(4, 20, 24));
    assert(slot_valid(0, 2) && slot_valid(2, 2) && !slot_valid(3, 2));
    for (uint64_t n : {1ULL, 7ULL, 4095ULL, 4096ULL, 4097ULL, 10003ULL}) {
        const uint64_t warps = blocks_for(n, true)*4;
        assert(warps && warps <= scratch_pages);
        std::set<uint64_t> requests;
        for (uint64_t warp=0; warp<warps; ++warp) {
            assert((warp+1)*page_bytes <= scratch_bytes);
            for (uint64_t i=warp; i<n; i+=warps) assert(requests.insert(i).second);
        }
        assert(requests.size()==n);
    }
    for (uint64_t row=0; row<10003; ++row) {
        assert(page_of(row)*4096 + byte_in_page(row) == row*512);
        assert(byte_in_page(row)+512 <= page_bytes);
    }
}
