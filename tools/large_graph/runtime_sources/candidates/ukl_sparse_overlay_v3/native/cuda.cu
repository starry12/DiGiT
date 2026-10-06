#include "core.h"
#include <cuda_runtime.h>
// Caller owns mapped host arrays and device output buffers; no hidden full-graph allocations.
__global__ void sample_kernel(View v,const int32_t*seeds,int count,int f,int grouped,uint64_t seed,Out o){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)sample_one(v,seeds,i,f,grouped,seed,o);
}
extern "C" int sample_cuda(View v,const int32_t*seeds,int count,int fanout,int grouped,uint64_t seed,Out o){
 if(count<=0||fanout<=0||fanout>MAX_F)return -1;
 sample_kernel<<<(count+63)/64,64>>>(v,seeds,count,fanout,grouped,seed,o);
 auto rc=cudaGetLastError();if(rc!=cudaSuccess)return int(rc);return int(cudaDeviceSynchronize());
}
