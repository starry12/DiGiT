#include "core.h"
#include <cuda_runtime.h>

__global__ void baseline_kernel(View v,const int32_t*s,int count,int f,uint64_t seed,Out o){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)sample_one(v,s,i,f,0,seed,o);
}

// One warp per owner. RNG and selection run in lane 0 without changes in order.
// EIDs retain the old earliest-unused-occurrence semantics, including duplicates.
__global__ void grouped_kernel(View v,const int32_t*s,int count,int f,uint64_t seed,Out o){
 int lane=threadIdx.x&31;int item=(blockIdx.x*blockDim.x+threadIdx.x)/32;
 if(item>=count)return;
 if(lane==0)sample_one(v,s,item,f,1,seed,o,false);
 __syncwarp();
 int n=__shfl_sync(0xffffffff,lane==0?o.counts[item]:0,0);
 int err=__shfl_sync(0xffffffff,lane==0?o.errors[item]:0,0);
 if(err||!n)return;
 int owner=s[item],offset=item*f;
 int target=lane<n?o.src[offset+lane]:-1;
 int64_t found=-1,begin=v.ptr[owner],end=v.ptr[owner+1];int missing=n;
 for(int64_t j=begin;j<end&&missing;j+=32){
  int64_t eid=j+lane;int node=eid<end?v.idx[eid-v.origin]:-1;
  unsigned used=0;
  for(int i=0;i<n;i++){
   int src=__shfl_sync(0xffffffff,target,i);
   int64_t prior=__shfl_sync(0xffffffff,(long long)found,i);
   unsigned matches=__ballot_sync(0xffffffff,eid<end&&node==src)&~used;
   if(prior<0&&matches){
    int take=__ffs(matches)-1;used|=1u<<take;missing--;
    if(lane==i)found=j+take;
   }
  }
 }
 if(lane<n)o.eid[offset+lane]=found;
 if(lane==0&&missing){o.errors[item]=8;o.counts[item]=0;}
}

extern "C" int sample_cuda(View v,const int32_t*s,int count,int f,int grouped,uint64_t seed,Out o){
 if(count<=0||f<=0||f>MAX_F)return -1;
 if(grouped)grouped_kernel<<<(count+3)/4,128>>>(v,s,count,f,seed,o);
 else baseline_kernel<<<(count+63)/64,64>>>(v,s,count,f,seed,o);
 auto rc=cudaGetLastError();if(rc!=cudaSuccess)return int(rc);return int(cudaDeviceSynchronize());
}
