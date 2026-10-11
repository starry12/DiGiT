#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <sstream>
#include <stdexcept>

namespace py = pybind11;
#include "profile_events.h"
#include "optimized_helpers.h"

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;
constexpr int kThreadsPerBlock = kWarpSize * kWarpsPerBlock;
constexpr int kMaxFanout = 128;

__device__ __forceinline__ uint64_t splitmix64(uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

__device__ __forceinline__ int64_t warp_sum(int64_t value) {
  constexpr unsigned mask = 0xffffffffu;
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(mask, value, offset);
  }
  return __shfl_sync(mask, value, 0);
}

template <typename IdT>
__global__ void group_sample_kernel(
    const int64_t* __restrict__ reorganized_indptr,
    const IdT* __restrict__ reorganized_indices,
    const IdT* __restrict__ group_members,
    const IdT* __restrict__ group_storage_base,
    const IdT* __restrict__ supernode_to_group,
    const IdT* __restrict__ node_to_primary_row,
    const int64_t* __restrict__ seeds,
    int64_t num_seeds,
    int64_t num_nodes,
    int64_t num_groups,
    int group_size,
    int fanout,
    uint64_t random_seed,
    int64_t* __restrict__ output_sources,
    int64_t* __restrict__ output_storage_rows,
    uint8_t* __restrict__ output_is_group,
    int64_t* __restrict__ output_group_counts,
    int64_t* __restrict__ output_node_counts) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp_in_block = threadIdx.x / kWarpSize;
  const int64_t seed_position =
      static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp_in_block;
  if (seed_position >= num_seeds) {
    return;
  }

  __shared__ int64_t selected_units[kWarpsPerBlock][kMaxFanout];
  const int64_t output_base = seed_position * fanout;
  if (lane == 0) {
    for (int slot = 0; slot < fanout; ++slot) {
      output_sources[output_base + slot] = -1;
      output_storage_rows[output_base + slot] = -1;
      output_is_group[output_base + slot] = 0;
    }
    output_group_counts[seed_position] = 0;
    output_node_counts[seed_position] = 0;
  }
  __syncwarp();

  const int64_t owner = seeds[seed_position];
  const int64_t unit_start = reorganized_indptr[owner];
  const int64_t unit_end = reorganized_indptr[owner + 1];
  const int64_t num_units = unit_end - unit_start;
  int selected_count = 0;
  int remaining = fanout;
  int output_count = 0;
  int selected_groups = 0;

  while (remaining > 0 && selected_count < kMaxFanout) {
    int64_t lane_weight = 0;
    for (int64_t local = lane; local < num_units; local += kWarpSize) {
      bool already_selected = false;
      for (int selected = 0; selected < selected_count; ++selected) {
        if (selected_units[warp_in_block][selected] == local) {
          already_selected = true;
          break;
        }
      }
      if (already_selected) {
        continue;
      }
      const int64_t graph_id = reorganized_indices[unit_start + local];
      const int cost = graph_id < num_nodes ? 1 : group_size;
      if (cost <= remaining) {
        lane_weight += cost;
      }
    }

    const int64_t total_weight = warp_sum(lane_weight);
    if (total_weight == 0) {
      break;
    }

    int64_t threshold = 0;
    if (lane == 0) {
      const uint64_t counter = random_seed
          ^ (static_cast<uint64_t>(owner) * 0xd6e8feb86659fd93ULL)
          ^ (static_cast<uint64_t>(seed_position) * 0xa0761d6478bd642fULL)
          ^ static_cast<uint64_t>(selected_count);
      threshold = static_cast<int64_t>(splitmix64(counter) % total_weight);
    }
    threshold = __shfl_sync(0xffffffffu, threshold, 0);

    int64_t inclusive = lane_weight;
    for (int offset = 1; offset < kWarpSize; offset <<= 1) {
      const int64_t previous = __shfl_up_sync(0xffffffffu, inclusive, offset);
      if (lane >= offset) {
        inclusive += previous;
      }
    }
    const int64_t exclusive = inclusive - lane_weight;
    const unsigned owner_mask = __ballot_sync(
        0xffffffffu, lane_weight > 0 && threshold >= exclusive && threshold < inclusive);
    const int selected_lane = __ffs(owner_mask) - 1;
    int64_t chosen_local = -1;
    if (lane == selected_lane) {
      int64_t local_threshold = threshold - exclusive;
      int64_t cumulative = 0;
      for (int64_t local = lane; local < num_units; local += kWarpSize) {
        bool already_selected = false;
        for (int selected = 0; selected < selected_count; ++selected) {
          if (selected_units[warp_in_block][selected] == local) {
            already_selected = true;
            break;
          }
        }
        if (already_selected) {
          continue;
        }
        const int64_t graph_id = reorganized_indices[unit_start + local];
        const int cost = graph_id < num_nodes ? 1 : group_size;
        if (cost > remaining) {
          continue;
        }
        if (local_threshold < cumulative + cost) {
          chosen_local = local;
          break;
        }
        cumulative += cost;
      }
    }
    chosen_local = __shfl_sync(0xffffffffu, chosen_local, selected_lane);
    if (chosen_local < 0) {
      break;
    }

    if (lane == 0) {
      selected_units[warp_in_block][selected_count] = chosen_local;
      ++selected_count;
      const int64_t graph_id = reorganized_indices[unit_start + chosen_local];
      if (graph_id < num_nodes) {
        output_sources[output_base + output_count] = graph_id;
        output_storage_rows[output_base + output_count] =
            node_to_primary_row[graph_id];
        output_is_group[output_base + output_count] = 0;
        ++output_count;
        --remaining;
      } else {
        const int64_t supernode_slot = graph_id - num_nodes;
        if (supernode_slot >= 0 && supernode_slot < num_groups) {
          const int64_t group_id = supernode_to_group[supernode_slot];
          const int64_t storage_base = group_storage_base[group_id];
          for (int member_offset = 0; member_offset < group_size; ++member_offset) {
            const int64_t output_slot = output_base + output_count;
            output_sources[output_slot] =
                group_members[group_id * group_size + member_offset];
            output_storage_rows[output_slot] = storage_base + member_offset;
            output_is_group[output_slot] = 1;
            ++output_count;
          }
          remaining -= group_size;
          ++selected_groups;
        }
      }
    }
    selected_count = __shfl_sync(0xffffffffu, selected_count, 0);
    remaining = __shfl_sync(0xffffffffu, remaining, 0);
    output_count = __shfl_sync(0xffffffffu, output_count, 0);
    selected_groups = __shfl_sync(0xffffffffu, selected_groups, 0);
    __syncwarp();
  }

  if (lane == 0) {
    output_group_counts[seed_position] = selected_groups;
    output_node_counts[seed_position] = output_count;
  }
}

template <typename IdT>
__global__ void group_sample_optimized_kernel(
    const int64_t* __restrict__ reorganized_indptr,
    const IdT* __restrict__ reorganized_indices,
    const IdT* __restrict__ group_members,
    const IdT* __restrict__ group_storage_base,
    const IdT* __restrict__ supernode_to_group,
    const IdT* __restrict__ node_to_primary_row,
    const int64_t* __restrict__ seeds,
    int64_t num_seeds,
    int64_t num_nodes,
    int64_t num_groups,
    int group_size,
    int fanout,
    uint64_t random_seed,
    int64_t* __restrict__ output_sources,
    int64_t* __restrict__ output_storage_rows,
    uint8_t* __restrict__ output_is_group,
    int64_t* __restrict__ output_group_counts,
    int64_t* __restrict__ output_node_counts) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp_in_block = threadIdx.x / kWarpSize;
  const int64_t seed_position =
      static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp_in_block;
  if (seed_position >= num_seeds) {
    return;
  }

  __shared__ int64_t selected_units[kWarpsPerBlock][kMaxFanout];
  const int64_t output_base = seed_position * fanout;
  if (lane == 0) {
    for (int slot = 0; slot < fanout; ++slot) {
      output_sources[output_base + slot] = -1;
      output_storage_rows[output_base + slot] = -1;
      output_is_group[output_base + slot] = 0;
    }
    output_group_counts[seed_position] = 0;
    output_node_counts[seed_position] = 0;
  }
  __syncwarp();

  const int64_t owner = seeds[seed_position];
  const int64_t unit_start = reorganized_indptr[owner];
  const int64_t unit_end = reorganized_indptr[owner + 1];
  const int64_t num_units = unit_end - unit_start;
  int selected_count = 0;
  int remaining = fanout;
  int output_count = 0;
  int selected_groups = 0;
  const bool masked = num_units <= kMaskDegreeCap;
  uint32_t live_masks[kMaskWords];
  uint32_t group_masks[kMaskWords];
  int64_t raw_count=0, group_count=0;
  // One coalesced UVA pass: remember only live/group bits, not all graph IDs.
  #pragma unroll
  for (int word=0; word<kMaskWords; ++word) {
    uint32_t live=0, groups=0;
    if (masked) {
      for (int bit=0;bit<32;++bit) {
        const int64_t local=lane+32*(word*32+bit);
        if (local>=num_units) break;
        const int64_t graph_id=reorganized_indices[unit_start+local];
        live|=1u<<bit;
        if (graph_id>=num_nodes) groups|=1u<<bit;
      }
      group_count+=__popc(groups);raw_count+=__popc(live)-__popc(groups);
    }
    live_masks[word]=live;group_masks[word]=groups;
  }
  if (!masked) {
    for (int64_t local=lane;local<num_units;local+=kWarpSize) {
      if (reorganized_indices[unit_start+local]<num_nodes) ++raw_count;
      else ++group_count;
    }
  }

  while (remaining > 0 && selected_count < kMaxFanout) {
    const bool allow_group=group_size<=remaining;
    const int64_t lane_weight=raw_count+(allow_group?group_count*group_size:0);

    const int64_t total_weight = warp_sum(lane_weight);
    if (total_weight == 0) {
      break;
    }

    int64_t threshold = 0;
    if (lane == 0) {
      const uint64_t counter = random_seed
          ^ (static_cast<uint64_t>(owner) * 0xd6e8feb86659fd93ULL)
          ^ (static_cast<uint64_t>(seed_position) * 0xa0761d6478bd642fULL)
          ^ static_cast<uint64_t>(selected_count);
      threshold = static_cast<int64_t>(splitmix64(counter) % total_weight);
    }
    threshold = __shfl_sync(0xffffffffu, threshold, 0);

    int64_t inclusive = lane_weight;
    for (int offset = 1; offset < kWarpSize; offset <<= 1) {
      const int64_t previous = __shfl_up_sync(0xffffffffu, inclusive, offset);
      if (lane >= offset) {
        inclusive += previous;
      }
    }
    const int64_t exclusive = inclusive - lane_weight;
    const unsigned owner_mask = __ballot_sync(
        0xffffffffu, lane_weight > 0 && threshold >= exclusive && threshold < inclusive);
    const int selected_lane = __ffs(owner_mask) - 1;
    int64_t chosen_local = -1;
    int64_t chosen_graph = -1;
    if (lane == selected_lane) {
      int64_t local_threshold = threshold - exclusive;
      if (masked) {
        #pragma unroll
        for (int word=0;word<kMaskWords;++word) {
          const uint32_t groups=group_masks[word];
          const uint32_t eligible=allow_group?live_masks[word]:(live_masks[word]&~groups);
          const int weight=mask_weight(eligible,allow_group?groups:0,group_size);
          if (chosen_local<0 && local_threshold<weight) {
            const int bit=weighted_bit(eligible,allow_group?groups:0,group_size,local_threshold);
            chosen_local=lane+32*(word*32+bit);
            live_masks[word]&=~(1u<<bit);
            const bool is_group=(groups&(1u<<bit))!=0;
            if (is_group) --group_count; else --raw_count;
          } else if (chosen_local<0) local_threshold-=weight;
        }
        if (chosen_local>=0) chosen_graph=reorganized_indices[unit_start+chosen_local];
      } else {
        // Bounded fallback for arbitrarily high degree: weights remain incremental.
        int64_t cumulative=0;
        for (int64_t local=lane;local<num_units;local+=kWarpSize) {
          bool selected=false;
          for (int j=0;j<selected_count;++j) {
            if (selected_units[warp_in_block][j]==local) {selected=true;break;}
          }
          if (selected) continue;
          const int64_t graph_id=reorganized_indices[unit_start+local];
          const int cost=graph_id<num_nodes?1:group_size;
          if (cost>remaining) continue;
          if (local_threshold<cumulative+cost) {
            chosen_local=local;chosen_graph=graph_id;
            if (graph_id<num_nodes) --raw_count;else --group_count;
            break;
          }
          cumulative+=cost;
        }
      }
    }
    chosen_local = __shfl_sync(0xffffffffu, chosen_local, selected_lane);
    chosen_graph = __shfl_sync(0xffffffffu, chosen_graph, selected_lane);
    if (chosen_local < 0) {
      break;
    }

    if (lane == 0) {
      selected_units[warp_in_block][selected_count] = chosen_local;
      ++selected_count;
      const int64_t graph_id = chosen_graph;
      if (graph_id < num_nodes) {
        output_sources[output_base + output_count] = graph_id;
        output_storage_rows[output_base + output_count] =
            node_to_primary_row[graph_id];
        output_is_group[output_base + output_count] = 0;
        ++output_count;
        --remaining;
      } else {
        const int64_t supernode_slot = graph_id - num_nodes;
        if (supernode_slot >= 0 && supernode_slot < num_groups) {
          const int64_t group_id = supernode_to_group[supernode_slot];
          const int64_t storage_base = group_storage_base[group_id];
          for (int member_offset = 0; member_offset < group_size; ++member_offset) {
            const int64_t output_slot = output_base + output_count;
            output_sources[output_slot] =
                group_members[group_id * group_size + member_offset];
            output_storage_rows[output_slot] = storage_base + member_offset;
            output_is_group[output_slot] = 1;
            ++output_count;
          }
          remaining -= group_size;
          ++selected_groups;
        }
      }
    }
    selected_count = __shfl_sync(0xffffffffu, selected_count, 0);
    remaining = __shfl_sync(0xffffffffu, remaining, 0);
    output_count = __shfl_sync(0xffffffffu, output_count, 0);
    selected_groups = __shfl_sync(0xffffffffu, selected_groups, 0);
    __syncwarp();
  }

  if (lane == 0) {
    output_group_counts[seed_position] = selected_groups;
    output_node_counts[seed_position] = output_count;
  }
}

template <typename IdT>
__global__ void group_sample_filtered_kernel(
    const int64_t* __restrict__ reorganized_indptr,
    const IdT* __restrict__ reorganized_indices,
    const IdT* __restrict__ group_members,
    const IdT* __restrict__ group_storage_base,
    const IdT* __restrict__ supernode_to_group,
    const IdT* __restrict__ node_to_primary_row,
    const int64_t* __restrict__ seeds,
    int64_t num_seeds,
    int64_t num_nodes,
    int64_t num_groups,
    int group_size,
    int fanout,
    uint64_t random_seed,
    int64_t* __restrict__ output_sources,
    int64_t* __restrict__ output_storage_rows,
    uint8_t* __restrict__ output_is_group,
    int64_t* __restrict__ output_group_counts,
    int64_t* __restrict__ output_node_counts,
    int64_t degree_min, int64_t degree_max) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp_in_block = threadIdx.x / kWarpSize;
  const int64_t seed_position =
      static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp_in_block;
  if (seed_position >= num_seeds) {
    return;
  }

  __shared__ int64_t selected_units[kWarpsPerBlock][kMaxFanout];
  const int64_t output_base = seed_position * fanout;
  if (lane == 0) {
    for (int slot = 0; slot < fanout; ++slot) {
      output_sources[output_base + slot] = -1;
      output_storage_rows[output_base + slot] = -1;
      output_is_group[output_base + slot] = 0;
    }
    output_group_counts[seed_position] = 0;
    output_node_counts[seed_position] = 0;
  }
  __syncwarp();

  const int64_t owner = seeds[seed_position];
  const int64_t unit_start = reorganized_indptr[owner];
  const int64_t unit_end = reorganized_indptr[owner + 1];
  const int64_t num_units = unit_end - unit_start;
  if (num_units<=degree_min || num_units>degree_max) return;
  int selected_count = 0;
  int remaining = fanout;
  int output_count = 0;
  int selected_groups = 0;
  const bool masked = num_units <= kMaskDegreeCap;
  uint32_t live_masks[kMaskWords];
  uint32_t group_masks[kMaskWords];
  int64_t raw_count=0, group_count=0;
  // One coalesced UVA pass: remember only live/group bits, not all graph IDs.
  #pragma unroll
  for (int word=0; word<kMaskWords; ++word) {
    uint32_t live=0, groups=0;
    if (masked) {
      for (int bit=0;bit<32;++bit) {
        const int64_t local=lane+32*(word*32+bit);
        if (local>=num_units) break;
        const int64_t graph_id=reorganized_indices[unit_start+local];
        live|=1u<<bit;
        if (graph_id>=num_nodes) groups|=1u<<bit;
      }
      group_count+=__popc(groups);raw_count+=__popc(live)-__popc(groups);
    }
    live_masks[word]=live;group_masks[word]=groups;
  }
  if (!masked) {
    for (int64_t local=lane;local<num_units;local+=kWarpSize) {
      if (reorganized_indices[unit_start+local]<num_nodes) ++raw_count;
      else ++group_count;
    }
  }

  while (remaining > 0 && selected_count < kMaxFanout) {
    const bool allow_group=group_size<=remaining;
    const int64_t lane_weight=raw_count+(allow_group?group_count*group_size:0);

    const int64_t total_weight = warp_sum(lane_weight);
    if (total_weight == 0) {
      break;
    }

    int64_t threshold = 0;
    if (lane == 0) {
      const uint64_t counter = random_seed
          ^ (static_cast<uint64_t>(owner) * 0xd6e8feb86659fd93ULL)
          ^ (static_cast<uint64_t>(seed_position) * 0xa0761d6478bd642fULL)
          ^ static_cast<uint64_t>(selected_count);
      threshold = static_cast<int64_t>(splitmix64(counter) % total_weight);
    }
    threshold = __shfl_sync(0xffffffffu, threshold, 0);

    int64_t inclusive = lane_weight;
    for (int offset = 1; offset < kWarpSize; offset <<= 1) {
      const int64_t previous = __shfl_up_sync(0xffffffffu, inclusive, offset);
      if (lane >= offset) {
        inclusive += previous;
      }
    }
    const int64_t exclusive = inclusive - lane_weight;
    const unsigned owner_mask = __ballot_sync(
        0xffffffffu, lane_weight > 0 && threshold >= exclusive && threshold < inclusive);
    const int selected_lane = __ffs(owner_mask) - 1;
    int64_t chosen_local = -1;
    int64_t chosen_graph = -1;
    if (lane == selected_lane) {
      int64_t local_threshold = threshold - exclusive;
      if (masked) {
        #pragma unroll
        for (int word=0;word<kMaskWords;++word) {
          const uint32_t groups=group_masks[word];
          const uint32_t eligible=allow_group?live_masks[word]:(live_masks[word]&~groups);
          const int weight=mask_weight(eligible,allow_group?groups:0,group_size);
          if (chosen_local<0 && local_threshold<weight) {
            const int bit=weighted_bit(eligible,allow_group?groups:0,group_size,local_threshold);
            chosen_local=lane+32*(word*32+bit);
            live_masks[word]&=~(1u<<bit);
            const bool is_group=(groups&(1u<<bit))!=0;
            if (is_group) --group_count; else --raw_count;
          } else if (chosen_local<0) local_threshold-=weight;
        }
        if (chosen_local>=0) chosen_graph=reorganized_indices[unit_start+chosen_local];
      } else {
        // Bounded fallback for arbitrarily high degree: weights remain incremental.
        int64_t cumulative=0;
        for (int64_t local=lane;local<num_units;local+=kWarpSize) {
          bool selected=false;
          for (int j=0;j<selected_count;++j) {
            if (selected_units[warp_in_block][j]==local) {selected=true;break;}
          }
          if (selected) continue;
          const int64_t graph_id=reorganized_indices[unit_start+local];
          const int cost=graph_id<num_nodes?1:group_size;
          if (cost>remaining) continue;
          if (local_threshold<cumulative+cost) {
            chosen_local=local;chosen_graph=graph_id;
            if (graph_id<num_nodes) --raw_count;else --group_count;
            break;
          }
          cumulative+=cost;
        }
      }
    }
    chosen_local = __shfl_sync(0xffffffffu, chosen_local, selected_lane);
    chosen_graph = __shfl_sync(0xffffffffu, chosen_graph, selected_lane);
    if (chosen_local < 0) {
      break;
    }

    if (lane == 0) {
      selected_units[warp_in_block][selected_count] = chosen_local;
      ++selected_count;
      const int64_t graph_id = chosen_graph;
      if (graph_id < num_nodes) {
        output_sources[output_base + output_count] = graph_id;
        output_storage_rows[output_base + output_count] =
            node_to_primary_row[graph_id];
        output_is_group[output_base + output_count] = 0;
        ++output_count;
        --remaining;
      } else {
        const int64_t supernode_slot = graph_id - num_nodes;
        if (supernode_slot >= 0 && supernode_slot < num_groups) {
          const int64_t group_id = supernode_to_group[supernode_slot];
          const int64_t storage_base = group_storage_base[group_id];
          for (int member_offset = 0; member_offset < group_size; ++member_offset) {
            const int64_t output_slot = output_base + output_count;
            output_sources[output_slot] =
                group_members[group_id * group_size + member_offset];
            output_storage_rows[output_slot] = storage_base + member_offset;
            output_is_group[output_slot] = 1;
            ++output_count;
          }
          remaining -= group_size;
          ++selected_groups;
        }
      }
    }
    selected_count = __shfl_sync(0xffffffffu, selected_count, 0);
    remaining = __shfl_sync(0xffffffffu, remaining, 0);
    output_count = __shfl_sync(0xffffffffu, output_count, 0);
    selected_groups = __shfl_sync(0xffffffffu, selected_groups, 0);
    __syncwarp();
  }

  if (lane == 0) {
    output_group_counts[seed_position] = selected_groups;
    output_node_counts[seed_position] = output_count;
  }
}

template <typename IdT, bool Filtered>
__global__ void group_sample_cooperative_kernel(
    const int64_t* __restrict__ reorganized_indptr,
    const IdT* __restrict__ reorganized_indices,
    const IdT* __restrict__ group_members,
    const IdT* __restrict__ group_storage_base,
    const IdT* __restrict__ supernode_to_group,
    const IdT* __restrict__ node_to_primary_row,
    const int64_t* __restrict__ seeds,
    int64_t num_seeds,
    int64_t num_nodes,
    int64_t num_groups,
    int group_size,
    int fanout,
    uint64_t random_seed,
    int64_t* __restrict__ output_sources,
    int64_t* __restrict__ output_storage_rows,
    uint8_t* __restrict__ output_is_group,
    int64_t* __restrict__ output_group_counts,
    int64_t* __restrict__ output_node_counts,
    int64_t degree_min, int64_t degree_max) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp_in_block = threadIdx.x / kWarpSize;
  const int64_t seed_position =
      static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp_in_block;
  if (seed_position >= num_seeds) {
    return;
  }

  __shared__ int64_t selected_units[kWarpsPerBlock][kMaxFanout];
  const int64_t output_base = seed_position * fanout;
  if (lane == 0) {
    for (int slot = 0; slot < fanout; ++slot) {
      output_sources[output_base + slot] = -1;
      output_storage_rows[output_base + slot] = -1;
      output_is_group[output_base + slot] = 0;
    }
    output_group_counts[seed_position] = 0;
    output_node_counts[seed_position] = 0;
  }
  __syncwarp();

  const int64_t owner = seeds[seed_position];
  const int64_t unit_start = reorganized_indptr[owner];
  const int64_t unit_end = reorganized_indptr[owner + 1];
  const int64_t num_units = unit_end - unit_start;
  if (Filtered && (num_units<=degree_min || num_units>degree_max)) return;
  int selected_count = 0;
  int remaining = fanout;
  int output_count = 0;
  int selected_groups = 0;
  const bool masked = num_units <= kMaskDegreeCap;
  uint32_t live_masks[kMaskWords];
  uint32_t group_masks[kMaskWords];
  int64_t raw_count=0, group_count=0;
  // One coalesced UVA pass: remember only live/group bits, not all graph IDs.
  #pragma unroll
  for (int word=0; word<kMaskWords; ++word) {
    uint32_t live=0, groups=0;
    if (masked) {
      for (int bit=0;bit<32;++bit) {
        const int64_t local=lane+32*(word*32+bit);
        if (local>=num_units) break;
        const int64_t graph_id=reorganized_indices[unit_start+local];
        live|=1u<<bit;
        if (graph_id>=num_nodes) groups|=1u<<bit;
      }
      group_count+=__popc(groups);raw_count+=__popc(live)-__popc(groups);
    }
    live_masks[word]=live;group_masks[word]=groups;
  }
  if (!masked) {
    for (int64_t local=lane;local<num_units;local+=kWarpSize) {
      if (reorganized_indices[unit_start+local]<num_nodes) ++raw_count;
      else ++group_count;
    }
  }

  while (remaining > 0 && selected_count < kMaxFanout) {
    const bool allow_group=group_size<=remaining;
    const int64_t lane_weight=raw_count+(allow_group?group_count*group_size:0);

    const int64_t total_weight = warp_sum(lane_weight);
    if (total_weight == 0) {
      break;
    }

    int64_t threshold = 0;
    if (lane == 0) {
      const uint64_t counter = random_seed
          ^ (static_cast<uint64_t>(owner) * 0xd6e8feb86659fd93ULL)
          ^ (static_cast<uint64_t>(seed_position) * 0xa0761d6478bd642fULL)
          ^ static_cast<uint64_t>(selected_count);
      threshold = static_cast<int64_t>(splitmix64(counter) % total_weight);
    }
    threshold = __shfl_sync(0xffffffffu, threshold, 0);

    int64_t inclusive = lane_weight;
    for (int offset = 1; offset < kWarpSize; offset <<= 1) {
      const int64_t previous = __shfl_up_sync(0xffffffffu, inclusive, offset);
      if (lane >= offset) {
        inclusive += previous;
      }
    }
    const int64_t exclusive = inclusive - lane_weight;
    const unsigned owner_mask = __ballot_sync(
        0xffffffffu, lane_weight > 0 && threshold >= exclusive && threshold < inclusive);
    const int selected_lane = __ffs(owner_mask) - 1;
    int64_t chosen_local = -1;
    int64_t chosen_graph = -1;
    if (masked) {
      if (lane == selected_lane) {
        int64_t local_threshold = threshold - exclusive;
        #pragma unroll
        for (int word=0;word<kMaskWords;++word) {
          const uint32_t groups=group_masks[word];
          const uint32_t eligible=allow_group?live_masks[word]:(live_masks[word]&~groups);
          const int weight=mask_weight(eligible,allow_group?groups:0,group_size);
          if (chosen_local<0 && local_threshold<weight) {
            const int bit=weighted_bit(eligible,allow_group?groups:0,group_size,local_threshold);
            chosen_local=lane+32*(word*32+bit);
            live_masks[word]&=~(1u<<bit);
            const bool is_group=(groups&(1u<<bit))!=0;
            if (is_group) --group_count; else --raw_count;
          } else if (chosen_local<0) local_threshold-=weight;
        }
        if (chosen_local>=0) chosen_graph=reorganized_indices[unit_start+chosen_local];
      }
    } else {
      // Preserve the selected original lane's order; all32 lanes help scan it.
      int64_t local_threshold = __shfl_sync(0xffffffffu, threshold-exclusive, selected_lane);
      for (int64_t tile=0; selected_lane+32*tile<num_units; tile+=32) {
        const int64_t local=selected_lane+32*(tile+lane);
        int64_t graph_id=-1, weight=0;
        if (local<num_units) {
          bool selected=false;
          for (int j=0; j<selected_count; ++j) {
            if (selected_units[warp_in_block][j]==local) {selected=true;break;}
          }
          if (!selected) {
            graph_id=reorganized_indices[unit_start+local];
            const int cost=graph_id<num_nodes?1:group_size;
            if (cost<=remaining) weight=cost;
          }
        }
        int64_t prefix=weight;
        for (int offset=1; offset<kWarpSize; offset<<=1) {
          const int64_t previous=__shfl_up_sync(0xffffffffu,prefix,offset);
          if (lane>=offset) prefix+=previous;
        }
        const int64_t tile_weight=__shfl_sync(0xffffffffu,prefix,31);
        if (local_threshold<tile_weight) {
          const unsigned selected_mask=__ballot_sync(0xffffffffu,weight>0 && local_threshold>=prefix-weight && local_threshold<prefix);
          const int helper_lane=__ffs(selected_mask)-1;
          chosen_local=__shfl_sync(0xffffffffu,local,helper_lane);
          chosen_graph=__shfl_sync(0xffffffffu,graph_id,helper_lane);
          if (lane==selected_lane) {
            if (chosen_graph<num_nodes) --raw_count;else --group_count;
          }
          break;
        }
        local_threshold-=tile_weight;
      }
    }
    chosen_local = __shfl_sync(0xffffffffu, chosen_local, selected_lane);
    chosen_graph = __shfl_sync(0xffffffffu, chosen_graph, selected_lane);
    if (chosen_local < 0) {
      break;
    }

    if (lane == 0) {
      selected_units[warp_in_block][selected_count] = chosen_local;
      ++selected_count;
      const int64_t graph_id = chosen_graph;
      if (graph_id < num_nodes) {
        output_sources[output_base + output_count] = graph_id;
        output_storage_rows[output_base + output_count] =
            node_to_primary_row[graph_id];
        output_is_group[output_base + output_count] = 0;
        ++output_count;
        --remaining;
      } else {
        const int64_t supernode_slot = graph_id - num_nodes;
        if (supernode_slot >= 0 && supernode_slot < num_groups) {
          const int64_t group_id = supernode_to_group[supernode_slot];
          const int64_t storage_base = group_storage_base[group_id];
          for (int member_offset = 0; member_offset < group_size; ++member_offset) {
            const int64_t output_slot = output_base + output_count;
            output_sources[output_slot] =
                group_members[group_id * group_size + member_offset];
            output_storage_rows[output_slot] = storage_base + member_offset;
            output_is_group[output_slot] = 1;
            ++output_count;
          }
          remaining -= group_size;
          ++selected_groups;
        }
      }
    }
    selected_count = __shfl_sync(0xffffffffu, selected_count, 0);
    remaining = __shfl_sync(0xffffffffu, remaining, 0);
    output_count = __shfl_sync(0xffffffffu, output_count, 0);
    selected_groups = __shfl_sync(0xffffffffu, selected_groups, 0);
    __syncwarp();
  }

  if (lane == 0) {
    output_group_counts[seed_position] = selected_groups;
    output_node_counts[seed_position] = output_count;
  }
}

template <typename IdT, typename EidT>
__global__ void resolve_eids_kernel(
    const int64_t* __restrict__ original_indptr,
    const IdT* __restrict__ original_indices,
    const EidT* __restrict__ original_eids,
    const int64_t* __restrict__ seeds,
    int64_t num_seeds,
    int fanout,
    const int64_t* __restrict__ sampled_sources,
    int64_t* __restrict__ sampled_eids) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int64_t slot =
      (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / kWarpSize;
  const int64_t total_slots = num_seeds * fanout;
  if (slot >= total_slots) {
    return;
  }
  const int64_t source = sampled_sources[slot];
  if (source < 0) {
    if (lane == 0) {
      sampled_eids[slot] = -1;
    }
    return;
  }
  const int64_t owner = seeds[slot / fanout];
  const int64_t start = original_indptr[owner];
  const int64_t end = original_indptr[owner + 1];
  int64_t best_position = end;
  for (int64_t position = start + lane; position < end; position += kWarpSize) {
    if (original_indices[position] == source) {
      best_position = position;
      break;
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    best_position = min(best_position, __shfl_down_sync(0xffffffffu, best_position, offset));
  }
  if (lane == 0) {
    sampled_eids[slot] =
        best_position < end ? original_eids[best_position] : static_cast<int64_t>(-2);
  }
}

void check_cuda_launch(const char* kernel_name) {
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::ostringstream message;
    message << kernel_name << " launch failed: " << cudaGetErrorString(error);
    throw std::runtime_error(message.str());
  }
}

template <typename IdT, typename OriginalIdT, typename EidT>
void sample_group_aware(
    uintptr_t reorganized_indptr_ptr,
    uintptr_t reorganized_indices_ptr,
    uintptr_t group_members_ptr,
    uintptr_t group_storage_base_ptr,
    uintptr_t supernode_to_group_ptr,
    uintptr_t node_to_primary_row_ptr,
    uintptr_t original_indptr_ptr,
    uintptr_t original_indices_ptr,
    uintptr_t original_eids_ptr,
    uintptr_t seeds_ptr,
    int64_t num_seeds,
    int64_t num_nodes,
    int64_t num_groups,
    int group_size,
    int fanout,
    uint64_t random_seed,
    uintptr_t output_sources_ptr,
    uintptr_t output_storage_rows_ptr,
    uintptr_t output_is_group_ptr,
    uintptr_t output_eids_ptr,
    uintptr_t output_group_counts_ptr,
    uintptr_t output_node_counts_ptr,
    uintptr_t stream_ptr) {
  const bool selection_only = original_indptr_ptr == 0 &&
      original_indices_ptr == 0 && original_eids_ptr == 0;
  if (!selection_only && (!original_indptr_ptr || !original_indices_ptr || !original_eids_ptr)) {
    throw std::invalid_argument("original CSC pointers must be all present or all null");
  }
  if (fanout <= 0 || fanout > kMaxFanout) {
    throw std::invalid_argument("CUDA fanout must be in [1, 128]");
  }
  if (group_size <= 0 || group_size > fanout) {
    throw std::invalid_argument("group_size must be positive and no larger than fanout");
  }
  if (num_seeds < 0 || num_nodes <= 0 || num_groups < 0) {
    throw std::invalid_argument("invalid sampler dimensions");
  }
  if (num_seeds == 0) {
    return;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  const int sample_blocks =
      static_cast<int>((num_seeds + kWarpsPerBlock - 1) / kWarpsPerBlock);
  if (kernel_profile.armed && selection_only) throw std::invalid_argument("Profile requires both kernels");
  profile_start(stream,num_seeds,fanout,random_seed);
  if (use_optimized_group) {
    if (use_cooperative_group) {
      if (bucket_enabled) {
  group_sample_cooperative_kernel<IdT,true><<<sample_blocks, kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const int64_t*>(reorganized_indptr_ptr),
      reinterpret_cast<const IdT*>(reorganized_indices_ptr),
      reinterpret_cast<const IdT*>(group_members_ptr),
      reinterpret_cast<const IdT*>(group_storage_base_ptr),
      reinterpret_cast<const IdT*>(supernode_to_group_ptr),
      reinterpret_cast<const IdT*>(node_to_primary_row_ptr),
      reinterpret_cast<const int64_t*>(seeds_ptr),
      num_seeds,
      num_nodes,
      num_groups,
      group_size,
      fanout,
      random_seed,
      reinterpret_cast<int64_t*>(output_sources_ptr),
      reinterpret_cast<int64_t*>(output_storage_rows_ptr),
      reinterpret_cast<uint8_t*>(output_is_group_ptr),
      reinterpret_cast<int64_t*>(output_group_counts_ptr),
      reinterpret_cast<int64_t*>(output_node_counts_ptr), bucket_min, bucket_max);
      } else {
  group_sample_cooperative_kernel<IdT,false><<<sample_blocks, kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const int64_t*>(reorganized_indptr_ptr),
      reinterpret_cast<const IdT*>(reorganized_indices_ptr),
      reinterpret_cast<const IdT*>(group_members_ptr),
      reinterpret_cast<const IdT*>(group_storage_base_ptr),
      reinterpret_cast<const IdT*>(supernode_to_group_ptr),
      reinterpret_cast<const IdT*>(node_to_primary_row_ptr),
      reinterpret_cast<const int64_t*>(seeds_ptr),
      num_seeds,
      num_nodes,
      num_groups,
      group_size,
      fanout,
      random_seed,
      reinterpret_cast<int64_t*>(output_sources_ptr),
      reinterpret_cast<int64_t*>(output_storage_rows_ptr),
      reinterpret_cast<uint8_t*>(output_is_group_ptr),
      reinterpret_cast<int64_t*>(output_group_counts_ptr),
      reinterpret_cast<int64_t*>(output_node_counts_ptr), bucket_min, bucket_max);
      }
    } else if (bucket_enabled) {
  group_sample_filtered_kernel<IdT><<<sample_blocks, kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const int64_t*>(reorganized_indptr_ptr),
      reinterpret_cast<const IdT*>(reorganized_indices_ptr),
      reinterpret_cast<const IdT*>(group_members_ptr),
      reinterpret_cast<const IdT*>(group_storage_base_ptr),
      reinterpret_cast<const IdT*>(supernode_to_group_ptr),
      reinterpret_cast<const IdT*>(node_to_primary_row_ptr),
      reinterpret_cast<const int64_t*>(seeds_ptr),
      num_seeds,
      num_nodes,
      num_groups,
      group_size,
      fanout,
      random_seed,
      reinterpret_cast<int64_t*>(output_sources_ptr),
      reinterpret_cast<int64_t*>(output_storage_rows_ptr),
      reinterpret_cast<uint8_t*>(output_is_group_ptr),
      reinterpret_cast<int64_t*>(output_group_counts_ptr),
      reinterpret_cast<int64_t*>(output_node_counts_ptr), bucket_min, bucket_max);
    } else {
  group_sample_optimized_kernel<IdT><<<sample_blocks, kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const int64_t*>(reorganized_indptr_ptr),
      reinterpret_cast<const IdT*>(reorganized_indices_ptr),
      reinterpret_cast<const IdT*>(group_members_ptr),
      reinterpret_cast<const IdT*>(group_storage_base_ptr),
      reinterpret_cast<const IdT*>(supernode_to_group_ptr),
      reinterpret_cast<const IdT*>(node_to_primary_row_ptr),
      reinterpret_cast<const int64_t*>(seeds_ptr),
      num_seeds,
      num_nodes,
      num_groups,
      group_size,
      fanout,
      random_seed,
      reinterpret_cast<int64_t*>(output_sources_ptr),
      reinterpret_cast<int64_t*>(output_storage_rows_ptr),
      reinterpret_cast<uint8_t*>(output_is_group_ptr),
      reinterpret_cast<int64_t*>(output_group_counts_ptr),
      reinterpret_cast<int64_t*>(output_node_counts_ptr));
    }
  } else {
  group_sample_kernel<IdT><<<sample_blocks, kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const int64_t*>(reorganized_indptr_ptr),
      reinterpret_cast<const IdT*>(reorganized_indices_ptr),
      reinterpret_cast<const IdT*>(group_members_ptr),
      reinterpret_cast<const IdT*>(group_storage_base_ptr),
      reinterpret_cast<const IdT*>(supernode_to_group_ptr),
      reinterpret_cast<const IdT*>(node_to_primary_row_ptr),
      reinterpret_cast<const int64_t*>(seeds_ptr),
      num_seeds,
      num_nodes,
      num_groups,
      group_size,
      fanout,
      random_seed,
      reinterpret_cast<int64_t*>(output_sources_ptr),
      reinterpret_cast<int64_t*>(output_storage_rows_ptr),
      reinterpret_cast<uint8_t*>(output_is_group_ptr),
      reinterpret_cast<int64_t*>(output_group_counts_ptr),
      reinterpret_cast<int64_t*>(output_node_counts_ptr));
  }
  check_cuda_launch("group_sample_kernel");
  profile_boundary(1,stream);

  // Optional capacity path: caller resolves first-occurrence EIDs on CPU.
  // No original CSC pointer is dereferenced in this mode.
  if (selection_only) return;

  const int64_t total_slots = num_seeds * fanout;
  const int resolve_threads = 256;
  const int64_t warps_per_resolve_block = resolve_threads / kWarpSize;
  const int resolve_blocks = static_cast<int>(
      (total_slots + warps_per_resolve_block - 1) / warps_per_resolve_block);
  resolve_eids_kernel<OriginalIdT, EidT><<<resolve_blocks, resolve_threads, 0, stream>>>(
      reinterpret_cast<const int64_t*>(original_indptr_ptr),
      reinterpret_cast<const OriginalIdT*>(original_indices_ptr),
      reinterpret_cast<const EidT*>(original_eids_ptr),
      reinterpret_cast<const int64_t*>(seeds_ptr),
      num_seeds,
      fanout,
      reinterpret_cast<const int64_t*>(output_sources_ptr),
      reinterpret_cast<int64_t*>(output_eids_ptr));
  check_cuda_launch("resolve_eids_kernel");
  profile_boundary(2,stream);
}

// The sampler kernels can dereference mapped host memory without changing
// selection, RNG, expansion or first-occurrence EID semantics.
py::dict mapped_pointer(uintptr_t pointer) {
  if (!pointer) throw std::invalid_argument("non-null pinned allocation required");
  cudaPointerAttributes attr{};
  cudaError_t error = cudaPointerGetAttributes(&attr, reinterpret_cast<void*>(pointer));
  if (error != cudaSuccess) {
    cudaGetLastError();
    throw std::invalid_argument("pointer is not a registered CUDA allocation");
  }
  if (attr.type != cudaMemoryTypeHost) {
    throw std::invalid_argument("UVA metadata must be CUDA registered host memory");
  }
  void* device_pointer = nullptr;
  error = cudaHostGetDevicePointer(&device_pointer, reinterpret_cast<void*>(pointer), 0);
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
  int device = -1;
  error = cudaGetDevice(&device);
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
  cudaDeviceProp prop{};
  error = cudaGetDeviceProperties(&prop, device);
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
  if (!prop.unifiedAddressing || !prop.canMapHostMemory)
    throw std::runtime_error("device does not support mapped UVA host memory");
  py::dict result;
  result["host_pointer"] = pointer;
  result["device_pointer"] = reinterpret_cast<uintptr_t>(device_pointer);
  result["memory_type"] = "host";
  result["cuda_device"] = device;
  result["unified_addressing"] = true;
  result["can_map_host_memory"] = true;
  return result;
}

template <typename IdT>
__global__ void gather_mapping_kernel(const IdT* values, const int64_t* ids,
                                     int64_t* output, int64_t count) {
  const int64_t at = static_cast<int64_t>(blockIdx.x)*blockDim.x + threadIdx.x;
  if (at < count) output[at] = static_cast<int64_t>(values[ids[at]]);
}

template <typename IdT>
void gather_mapping(uintptr_t values, uintptr_t ids, uintptr_t output,
                    int64_t count, uintptr_t stream) {
  if (count < 0 || count > (1LL<<30)) throw std::invalid_argument("invalid gather count");
  if (!count) return;
  if (!values || !ids || !output) throw std::invalid_argument("null gather pointer");
  gather_mapping_kernel<IdT><<<static_cast<int>((count+255)/256),256,0,
      reinterpret_cast<cudaStream_t>(stream)>>>(reinterpret_cast<const IdT*>(values),
      reinterpret_cast<const int64_t*>(ids),reinterpret_cast<int64_t*>(output),count);
  check_cuda_launch("gather_mapping_kernel");
}

}  // namespace

PYBIND11_MODULE(IGGroupWarpCUDA, module) {
  module.def("set_cooperative", [](bool value){use_cooperative_group=value;});
  module.def("set_bucket", [](bool enabled,int64_t lo,int64_t hi){
    if (lo < -1 || hi<=lo) throw std::invalid_argument("Invalid degree bucket");
    bucket_enabled=enabled;bucket_min=lo;bucket_max=hi;
  });
  module.attr("WARP_FALLBACK_API")=1;
  module.def("set_optimized", [](bool value){use_optimized_group=value;});
  module.def("get_optimized", [](){return use_optimized_group;});
  module.attr("MASK_DEGREE_CAP")=kMaskDegreeCap;
  module.def("profile_prepare", &profile_prepare);
  module.def("profile_arm", &profile_arm);
  module.def("profile_pending", [](){return kernel_profile.armed;});
  module.def("profile_collect", &profile_collect);
  module.def("profile_release", &profile_release);
  module.attr("KERNEL_SPLIT_API") = 1;
  module.doc() = "Isolated IG prototype: mapped host metadata with signed 64-bit EIDs";
  module.def("sample_group_aware", &sample_group_aware<int64_t, int64_t, int64_t>, py::call_guard<py::gil_scoped_release>());
  module.def("sample_group_aware_i32_eid64", &sample_group_aware<int32_t, int32_t, int64_t>, py::call_guard<py::gil_scoped_release>());
  module.def("sample_group_aware_shared_i64_csc", &sample_group_aware<int32_t, int64_t, int64_t>, py::call_guard<py::gil_scoped_release>());
  module.def("mapped_pointer", &mapped_pointer);
  module.def("gather_i32", &gather_mapping<int32_t>, py::call_guard<py::gil_scoped_release>());
  module.def("gather_i64", &gather_mapping<int64_t>, py::call_guard<py::gil_scoped_release>());
  module.attr("UVA_EID64_API") = 1;
  module.attr("SHARED_CSC_API") = 1;
  module.attr("INT32_METADATA_API") = 1;
  module.attr("MAX_FANOUT") = kMaxFanout;
  module.attr("CPU_EID_SELECTION_API") = 1;
}
