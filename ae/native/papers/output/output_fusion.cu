#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <climits>

// Group rows are nonnegative int64 IDs; LLONG_MAX is the neutral amin value.
__global__ void init_rows(int64_t* out, int64_t n) {
    for (int64_t i=blockIdx.x*blockDim.x+threadIdx.x; i<n; i+=gridDim.x*blockDim.x)
        out[i]=LLONG_MAX;
}
__global__ void reduce_groups(const int64_t* sources, const int64_t* rows,
                              const bool* groups, int64_t* out, int64_t edges) {
    for (int64_t i=blockIdx.x*blockDim.x+threadIdx.x; i<edges; i+=gridDim.x*blockDim.x)
        if (groups[i]) atomicMin(reinterpret_cast<unsigned long long*>(out+sources[i]),
                                static_cast<unsigned long long>(rows[i]));
}
template<class ID>
__global__ void finish_rows(const int64_t* nodes,const ID* primary,int64_t* out,bool* flags,int64_t n) {
    for (int64_t i=blockIdx.x*blockDim.x+threadIdx.x; i<n; i+=gridDim.x*blockDim.x) {
        const bool grouped=out[i]!=LLONG_MAX;
        flags[i]=grouped;
        if (!grouped) out[i]=static_cast<int64_t>(primary[nodes[i]]);
    }
}
int blocks(int64_t n) { return static_cast<int>(n>1048576 ? 4096 : (n+255)/256); }
void check_launch() {
    auto e=cudaGetLastError();
    if(e!=cudaSuccess) throw std::runtime_error(std::string("output fusion launch: ")+cudaGetErrorString(e));
}
void annotate(uint64_t nodes,uint64_t primary,int bits,uint64_t sources,uint64_t rows,uint64_t groups,
              uint64_t output,uint64_t flags,int64_t n,int64_t edges,uint64_t stream_ptr) {
    if(n<0 || edges<0 || (bits!=32 && bits!=64) || (!n && edges))
        throw std::invalid_argument("invalid output fusion geometry");
    if(!n) return;
    auto stream=reinterpret_cast<cudaStream_t>(stream_ptr);
    auto out=reinterpret_cast<int64_t*>(output);
    init_rows<<<blocks(n),256,0,stream>>>(out,n);check_launch();
    if(edges) {
        reduce_groups<<<blocks(edges),256,0,stream>>>(reinterpret_cast<const int64_t*>(sources),
            reinterpret_cast<const int64_t*>(rows),reinterpret_cast<const bool*>(groups),out,edges);
        check_launch();
    }
    if(bits==32) finish_rows<<<blocks(n),256,0,stream>>>(reinterpret_cast<const int64_t*>(nodes),
        reinterpret_cast<const int32_t*>(primary),out,reinterpret_cast<bool*>(flags),n);
    else finish_rows<<<blocks(n),256,0,stream>>>(reinterpret_cast<const int64_t*>(nodes),
        reinterpret_cast<const int64_t*>(primary),out,reinterpret_cast<bool*>(flags),n);
    check_launch();
}
PYBIND11_MODULE(DiGiTOutputCUDA,m) {
    m.attr("API_VERSION")=1;
    m.def("annotate",&annotate);
}
