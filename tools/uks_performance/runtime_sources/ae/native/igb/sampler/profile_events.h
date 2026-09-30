// Host-only instrumentation. No kernel body or launch geometry changes.
#include <array>
#include <vector>
#include <cmath>
struct KernelTiming {
  std::array<cudaEvent_t,3> events{};
  int batch=0; int fanout=0; int64_t seeds=0; uint64_t random_seed=0;
};
struct KernelProfileState {
  std::vector<KernelTiming> pool;
  size_t used=0;
  bool armed=false;
  int batch=0;
};
static thread_local KernelProfileState kernel_profile;
static void profile_check(cudaError_t error) {
  if (error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
static void profile_prepare(int capacity,uintptr_t stream_ptr) {
  if (capacity<1 || capacity>1024 || !kernel_profile.pool.empty())
    throw std::invalid_argument("Bad capacity or existing profile");
  kernel_profile.pool.resize(capacity);
  auto stream=reinterpret_cast<cudaStream_t>(stream_ptr);
  for (auto& row:kernel_profile.pool) for (auto& e:row.events) {
    profile_check(cudaEventCreate(&e));profile_check(cudaEventRecord(e,stream));
  }
  // Initialization only, before warmup and any measured iteration.
  profile_check(cudaEventSynchronize(kernel_profile.pool.back().events[2]));
}
static void profile_arm(int batch) {
  if (kernel_profile.armed || kernel_profile.used>=kernel_profile.pool.size() || batch<=0 ||
      (kernel_profile.used && batch<=kernel_profile.pool[kernel_profile.used-1].batch))
    throw std::invalid_argument("Invalid or overlapping profile batch");
  kernel_profile.batch=batch;kernel_profile.armed=true;
}
static void profile_start(cudaStream_t stream,int64_t seeds,int fanout,uint64_t random_seed) {
  if (!kernel_profile.armed) return;
  auto& row=kernel_profile.pool[kernel_profile.used];
  row.batch=kernel_profile.batch;row.seeds=seeds;row.fanout=fanout;row.random_seed=random_seed;
  profile_check(cudaEventRecord(row.events[0],stream));
}
static void profile_boundary(int index,cudaStream_t stream) {
  if (!kernel_profile.armed) return;
  profile_check(cudaEventRecord(kernel_profile.pool[kernel_profile.used].events[index],stream));
  if (index==2) {kernel_profile.used++;kernel_profile.armed=false;}
}
static py::list profile_collect() {
  if (kernel_profile.armed || kernel_profile.pool.empty() || kernel_profile.used!=kernel_profile.pool.size())
    throw std::runtime_error("Incomplete kernel profile");
  py::list result;
  for (const auto& row:kernel_profile.pool) {
    for (const auto& e:row.events) profile_check(cudaEventQuery(e));
    float group=0,resolve=0,total=0;
    profile_check(cudaEventElapsedTime(&group,row.events[0],row.events[1]));
    profile_check(cudaEventElapsedTime(&resolve,row.events[1],row.events[2]));
    profile_check(cudaEventElapsedTime(&total,row.events[0],row.events[2]));
    py::dict d;d["batch"]=row.batch;d["seeds"]=row.seeds;d["fanout"]=row.fanout;
    d["random_seed"]=row.random_seed;d["group_sample_ms"]=group;
    d["resolve_eids_ms"]=resolve;d["total_ms"]=total;result.append(d);
  }
  return result;
}
static void profile_release() {
  if (kernel_profile.armed) throw std::runtime_error("Profile still armed");
  for (auto& row:kernel_profile.pool) for (auto& e:row.events) {
    profile_check(cudaEventQuery(e));profile_check(cudaEventDestroy(e));
  }
  kernel_profile.pool.clear();kernel_profile.used=0;kernel_profile.batch=0;
}
