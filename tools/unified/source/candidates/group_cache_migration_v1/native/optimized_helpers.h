// Select in the SAME lane-local index order as the reference weighted scan.
constexpr int kMaskWords = 8;
constexpr int kMaskDegreeCap = kMaskWords * 32 * 32;
static thread_local bool use_optimized_group = false;

__device__ __forceinline__ int mask_weight(uint32_t live,uint32_t groups,int g) {
  return __popc(live)+(g-1)*__popc(live&groups);
}
__device__ __forceinline__ int weighted_bit(uint32_t live,uint32_t groups,
                                           int g,int64_t threshold) {
  int base=0;
  #pragma unroll
  for (int half=16;half>=1;half>>=1) {
    const uint32_t lower=(1u<<half)-1;
    const int weight=mask_weight(live&lower,groups&lower,g);
    if (threshold<weight) {live&=lower;groups&=lower;}
    else {threshold-=weight;live>>=half;groups>>=half;base+=half;}
  }
  return base;
}

static thread_local bool use_cooperative_group=false;
static thread_local bool bucket_enabled=false;
static thread_local int64_t bucket_min=-1, bucket_max=INT64_MAX;
