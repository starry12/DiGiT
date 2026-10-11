#pragma once
#include <cstdint>
#ifdef __CUDACC__
#define PRIORITY_HD __host__ __device__
#else
#define PRIORITY_HD
#endif
PRIORITY_HD inline bool prefer_groups(int remaining,int group_size,int64_t groups) {
 return remaining>=group_size && groups>0;
}
PRIORITY_HD inline int64_t priority_weight(int64_t nodes,int64_t groups,int size,bool first) {
 return first ? groups*size : nodes;
}
#undef PRIORITY_HD
