// CPU-only canonical BFS over incoming CSC edges restricted to the train split.
// Traversal never modifies graph topology or registers file-backed CUDA memory.
#include <cstdint>
#include <vector>
template<class Index> int traverse(const int64_t* ptr,const Index* idx,int64_t nodes,
                    int64_t edges,int64_t train,int64_t* out){
 if(nodes<=0 || train<=0 || train>nodes || nodes>INT32_MAX || ptr[0]!=0 || ptr[nodes]!=edges)return 1;
 std::vector<uint64_t> seen((train+63)/64,0);int64_t tail=0,head=0;
 auto insert=[&](int64_t u){uint64_t mask=1ULL<<(u%64);if(!(seen[u/64]&mask)){seen[u/64]|=mask;out[tail++]=u;}};
 for(int64_t start=0;start<train;start++){
  insert(start);
  while(head<tail){int64_t u=out[head++],lo=ptr[u],hi=ptr[u+1];
   if(lo<0 || hi<lo || hi>edges)return 2;
   for(int64_t e=lo;e<hi;e++){int64_t v=idx[e];if(v<0 || v>=nodes)return 3;if(v<train)insert(v);}
  }
 }
 return tail==train?0:4;
}
extern "C" int bfs(const int64_t*p,const int32_t*i,int64_t n,int64_t e,int64_t t,int64_t*o){return traverse(p,i,n,e,t,o);}
extern "C" int bfs64(const int64_t*p,const int64_t*i,int64_t n,int64_t e,int64_t t,int64_t*o){return traverse(p,i,n,e,t,o);}
