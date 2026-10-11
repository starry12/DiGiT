// Independent filesystem preparation. No CUDA, raw devices, or graph-sized edge heap.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
using namespace std;
void check(bool b,const string&s){if(!b)throw runtime_error(s);}
struct Map {int fd; size_t bytes;void*base; int64_t*data;
 Map(string p,bool npy=true){fd=open(p.c_str(),O_RDONLY);check(fd>=0,"open "+p);struct stat st;fstat(fd,&st);bytes=st.st_size;
 base=mmap(nullptr,bytes,PROT_READ,MAP_SHARED,fd,0);check(base!=MAP_FAILED,"mmap");size_t off=0;
 if(npy){auto b=(unsigned char*)base;check(bytes>=10&&!memcmp(b,"\x93NUMPY",6),"NPY magic");unsigned len=b[8]|b[9]<<8;off=10;if(b[6]>1){len|=b[10]<<16|b[11]<<24;off=12;}string h((char*)b+off,len);check(h.find("'<i8'")!=string::npos,"expected int64 little endian");off+=len;}
 data=(int64_t*)((char*)base+off);}
 ~Map(){munmap(base,bytes);close(fd);}
};
template<class T>void save(const string&p,const vector<T>&v){ofstream f(p,ios::binary);check(bool(f),"create "+p);f.write((char*)v.data(),v.size()*sizeof(T));check(bool(f),"write "+p);}
// Canonical training-induced undirected BFS, preserving multiedge degree.
// Neighbors ordered by global ID; components seeded by descending degree / ID.
int main(int argc,char**argv){try{
 check(argc==7,"bfs indptr.npy indices.npy train.npy nodes train_count output.bin");
 Map p(argv[1]),idx(argv[2]),t(argv[3]);uint64_t n=stoull(argv[4]),nt=stoull(argv[5]);
 check(nt<(1ULL<<31)&&n<(1ULL<<31),"int32 local IDs");
 vector<int32_t> local(n,-1);vector<int64_t> train(t.data,t.data+nt);
 sort(train.begin(),train.end());check(adjacent_find(train.begin(),train.end())==train.end(),"duplicate train");
 for(uint64_t i=0;i<nt;i++){check(train[i]>=0&&uint64_t(train[i])<n,"train range");local[train[i]]=i;}
 vector<uint64_t> ptr(nt+1,0);
 auto visit=[&](auto consume){for(uint64_t i=0;i<nt;i++){
  uint64_t owner=train[i];for(int64_t e=p.data[owner];e<p.data[owner+1];e++){
   int64_t src=idx.data[e];check(src>=0&&uint64_t(src)<n,"source range");int32_t j=local[src];
   if(j>=0&&uint64_t(j)!=i)consume(i,uint64_t(j));
  }
  if(i%1000000==0)cerr<<"training_rows "<<i<<"/"<<nt<<endl;
 }};
 visit([&](uint64_t i,uint64_t j){ptr[i+1]++;ptr[j+1]++;});
 partial_sum(ptr.begin(),ptr.end(),ptr.begin());vector<int32_t> neighbors(ptr.back());vector<uint64_t> cursor(ptr);
 visit([&](uint64_t i,uint64_t j){neighbors[cursor[i]++]=j;neighbors[cursor[j]++]=i;});
 vector<int32_t> priority(nt);iota(priority.begin(),priority.end(),0);
 for(uint64_t i=0;i<nt;i++)sort(neighbors.begin()+ptr[i],neighbors.begin()+ptr[i+1]);
 sort(priority.begin(),priority.end(),[&](int32_t a,int32_t b){uint64_t da=ptr[a+1]-ptr[a],db=ptr[b+1]-ptr[b];return da!=db?da>db:train[a]<train[b];});
 vector<uint8_t> seen(nt,0);vector<int32_t> queue;queue.reserve(nt);vector<int64_t> order;order.reserve(nt);uint64_t components=0;
 for(auto root:priority){if(seen[root])continue;components++;queue.clear();queue.push_back(root);seen[root]=1;
  for(uint64_t head=0;head<queue.size();head++){int32_t u=queue[head];order.push_back(train[u]);
   for(uint64_t k=ptr[u];k<ptr[u+1];k++){int32_t v=neighbors[k];if(!seen[v]){seen[v]=1;queue.push_back(v);}}
  }
 }
 check(order.size()==nt,"BFS incomplete");save(string(argv[6]),order);
 cout<<"{\"train_nodes\":"<<nt<<",\"undirected_induced_edges\":"<<ptr.back()<<",\"components\":"<<components<<"}"<<endl;
}catch(const exception&e){cerr<<e.what()<<endl;return 1;}}
