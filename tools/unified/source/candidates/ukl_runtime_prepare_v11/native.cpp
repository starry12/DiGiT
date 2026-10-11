// CPU-only bounded preparation over caller-owned anonymous arrays.
// No mmap, CUDA, subprocess, raw device, or file I/O.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <vector>
using Tick = int(*)(uint64_t,uint64_t,int);
static void check(bool b,const char* m){if(!b)throw std::runtime_error(m);}
static void tick(Tick f,uint64_t a,uint64_t b,int stage){
  if(f && f(a,b,stage))throw std::runtime_error("preparation cancelled");
}
extern "C" int bfs(const int64_t* p,const int32_t* idx,int64_t n,int64_t edges,
 const int64_t* train,int64_t nt,int64_t* output,uint64_t edge_cap,Tick cb,
 uint64_t* stats,char* error,size_t error_len){try{
 check(n>0&&n<INT32_MAX&&nt>0&&nt<=n&&edges>0,"dimensions");
 check(p&&idx&&train&&output&&stats&&p[0]==0&&p[n]==edges,"pointers/endpoints");
 check(edge_cap<=((64ULL<<30)/4),"edge budget");
 tick(cb,0,nt,0);std::vector<int32_t> local(n,-1);
 for(int64_t i=0;i<nt;i++){
  check(train[i]>=0&&train[i]<n&&(i==0||train[i]>train[i-1]),"train sorted unique range");
  local[train[i]]=i;if(i%65536==0)tick(cb,i,nt,0);
 }
 std::vector<uint64_t> ptr(nt+1,0);uint64_t total=0;
 auto visit=[&](auto consume,int stage){uint64_t work=0;
  for(int64_t i=0;i<nt;i++){
   int64_t owner=train[i],lo=p[owner],hi=p[owner+1];
   check(lo>=0&&hi>=lo&&hi<=edges,"CSC offsets");
   for(int64_t e=lo;e<hi;e++){
    int64_t src=idx[e];check(src>=0&&src<n,"source range");int32_t j=local[src];
    if(j>=0&&j!=i)consume(i,j);
    if(++work%1048576==0)tick(cb,i,nt,stage);
   }
   if(i%65536==0)tick(cb,i,nt,stage);
  }
 };
 visit([&](int64_t i,int32_t j){check(total+2<=edge_cap,"BFS edge cap exceeded");
     total+=2;ptr[i+1]++;ptr[j+1]++;},1);
 std::partial_sum(ptr.begin(),ptr.end(),ptr.begin());
 tick(cb,nt,nt,1);std::vector<int32_t> neighbors(total);std::vector<uint64_t> cursor(ptr);
 visit([&](int64_t i,int32_t j){neighbors[cursor[i]++]=j;neighbors[cursor[j]++]=i;},2);
 std::vector<int32_t> priority(nt);std::iota(priority.begin(),priority.end(),0);
 uint64_t comparisons=0;
 for(int64_t i=0;i<nt;i++){
  std::sort(neighbors.begin()+ptr[i],neighbors.begin()+ptr[i+1],
   [&](int32_t a,int32_t b){if(++comparisons%1048576==0)tick(cb,i,nt,3);return a<b;});
  if(i%65536==0)tick(cb,i,nt,3);
 }
 std::sort(priority.begin(),priority.end(),[&](int32_t a,int32_t b){
   if(++comparisons%1048576==0)tick(cb,comparisons,0,4);
   auto da=ptr[a+1]-ptr[a],db=ptr[b+1]-ptr[b];return da!=db?da>db:train[a]<train[b];});
 std::vector<uint8_t> seen(nt,0);std::vector<int32_t> queue;queue.reserve(nt);
 uint64_t components=0,done=0,work=0;
 for(auto root:priority){if(seen[root])continue;components++;queue.clear();queue.push_back(root);seen[root]=1;
  for(size_t h=0;h<queue.size();h++){int32_t u=queue[h];output[done++]=train[u];
   for(uint64_t k=ptr[u];k<ptr[u+1];k++){auto v=neighbors[k];
    if(!seen[v]){seen[v]=1;queue.push_back(v);}if(++work%1048576==0)tick(cb,done,nt,5);
   }
   if(done%65536==0)tick(cb,done,nt,5);
  }
 }
 check(done==uint64_t(nt),"BFS incomplete");stats[0]=total;stats[1]=components;tick(cb,done,nt,5);return 0;
 }catch(const std::exception& e){snprintf(error,error_len,"%s",e.what());return 1;}}

extern "C" int topk(const uint64_t* counts,int64_t n,int64_t* output,Tick cb,char* error,size_t error_len){try{
 check(n>=10&&n<INT32_MAX&&counts&&output,"topk dimensions");
 std::vector<uint32_t> ids(n);std::iota(ids.begin(),ids.end(),0);size_t k=n/10;uint64_t work=0;
 auto less=[&](uint32_t a,uint32_t b){if(++work%1048576==0)tick(cb,work,0,6);
  return counts[a]!=counts[b]?counts[a]>counts[b]:a<b;};
 std::nth_element(ids.begin(),ids.begin()+k,ids.end(),less);ids.resize(k);
 std::sort(ids.begin(),ids.end(),[&](uint32_t a,uint32_t b){if(++work%1048576==0)tick(cb,work,0,6);return a<b;});
 for(size_t i=0;i<k;i++)output[i]=ids[i];tick(cb,k,k,6);return 0;
 }catch(const std::exception& e){snprintf(error,error_len,"%s",e.what());return 1;}}
