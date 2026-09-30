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
void csc(string src,uint64_t n,uint64_t e,bool rows,string out,uint64_t width){
 check(n<(1ULL<<31)&&width>0,"node/partition bound");Map a(src);size_t nb=(n+width-1)/width;
 struct Pair{uint32_t src,dst;};vector<ofstream> streams(nb);vector<vector<Pair>> buffers(nb);vector<uint64_t> sizes(nb,0);uint64_t removed=0;
 for(size_t b=0;b<nb;b++){streams[b].open(out+"/bucket_"+to_string(b)+".bin",ios::binary);check(bool(streams[b]),"bucket create");buffers[b].reserve(8192);}
 auto flush=[&](size_t b){auto&v=buffers[b];streams[b].write((char*)v.data(),v.size()*sizeof(Pair));check(bool(streams[b]),"bucket write");v.clear();};
 for(uint64_t k=0;k<e;k++){int64_t s=a.data[rows?k:2*k],d=a.data[rows?e+k:2*k+1];check(s>=0&&d>=0&&uint64_t(s)<n&&uint64_t(d)<n,"endpoint range");if(s==d){removed++;continue;}
 size_t b=d/width;buffers[b].push_back({uint32_t(s),uint32_t(d)});sizes[b]++;if(buffers[b].size()==8192)flush(b);
 if(k%100000000==0)cout<<"partition "<<k<<"/"<<e<<endl;}
 for(size_t b=0;b<nb;b++){flush(b);streams[b].close();}
 vector<int64_t> ptr(n+1,0);ofstream idx(out+"/indices.bin",ios::binary);check(bool(idx),"indices output");uint64_t total=0;
 for(size_t b=0;b<nb;b++){uint64_t lo=b*width,hi=min(n,lo+width);vector<uint64_t> count(hi-lo,1),cursor(hi-lo);string path=out+"/bucket_"+to_string(b)+".bin";
 ifstream f(path,ios::binary);vector<Pair> chunk(65536);while(f){f.read((char*)chunk.data(),chunk.size()*sizeof(Pair));auto got=f.gcount()/sizeof(Pair);for(size_t j=0;j<got;j++)count[chunk[j].dst-lo]++;}
 uint64_t length=0;for(uint64_t j=lo;j<hi;j++){ptr[j]=total+length;cursor[j-lo]=length;length+=count[j-lo];}check(length<=uint64_t(4)*1024*1024*1024,"bucket exceeds 32 GiB; reduce bucket width");vector<int64_t> values(length);
 f.clear();f.seekg(0);while(f){f.read((char*)chunk.data(),chunk.size()*sizeof(Pair));auto got=f.gcount()/sizeof(Pair);for(size_t j=0;j<got;j++){auto z=chunk[j];values[cursor[z.dst-lo]++]=z.src;}}
 for(uint64_t j=lo;j<hi;j++)values[cursor[j-lo]++]=j;
 idx.write((char*)values.data(),length*8);check(bool(idx),"indices write");total+=length;cout<<"csc_bucket "<<b+1<<"/"<<nb<<" edges "<<total<<endl;
 }
 ptr[n]=total;save(out+"/indptr.bin",ptr);ofstream meta(out+"/counts.txt");meta<<total<<" "<<removed<<endl;
}
void rank_rev(string ptrfile,string idxfile,uint64_t n,string out){
 Map p(ptrfile),idx(idxfile);vector<double> rank(n,1./n),next(n); // PageRank of reverse graph: outgoing degree = original in-degree.
 for(int it=0;it<20;it++){double dangling=0;for(uint64_t d=0;d<n;d++)if(p.data[d]==p.data[d+1])dangling+=rank[d];fill(next.begin(),next.end(),(.15+.85*dangling)/n);
 for(uint64_t d=0;d<n;d++){int64_t a=p.data[d],b=p.data[d+1];if(a==b)continue;double value=.85*rank[d]/(b-a);for(auto j=a;j<b;j++)next[idx.data[j]]+=value;}
 rank.swap(next);cout<<"revpr_iteration "<<it+1<<"/20"<<endl;}
 save(out+"/revpr.bin",rank);vector<uint32_t> order(n);iota(order.begin(),order.end(),0);sort(order.begin(),order.end(),[&](uint32_t a,uint32_t b){return rank[a]!=rank[b]?rank[a]>rank[b]:a<b;});
 vector<int64_t> hot(n/10);for(size_t i=0;i<hot.size();i++)hot[i]=order[i];sort(hot.begin(),hot.end());save(out+"/hot_nodes.bin",hot);
}
void group(string ptrfile,string idxfile,string hotfile,uint64_t n,string out){
 Map p(ptrfile),idx(idxfile),h(hotfile);uint64_t e=p.data[n];check(e<(1ULL<<63),"EID overflow");vector<uint8_t> hot(n,0),used(n,0),covered(e,0);
 for(uint64_t i=0;i<n/10;i++){check(h.data[i]>=0&&uint64_t(h.data[i])<n,"hot range");hot[h.data[i]]=1;}
 vector<uint32_t> order(n);iota(order.begin(),order.end(),0);sort(order.begin(),order.end(),[&](uint32_t a,uint32_t b){auto da=p.data[a+1]-p.data[a],db=p.data[b+1]-p.data[b];return da!=db?da>db:a<b;});
 struct G{uint32_t a,b,owner;};vector<G> groups;vector<int64_t> counts(n,0);mt19937_64 rng(0);uint64_t primary=0,replica=0,cap=(n/5)/2;
 for(int phase=0;phase<2;phase++)for(uint64_t oi=0;oi<n;oi++){uint32_t owner=order[oi];auto a=p.data[owner],b=p.data[owner+1];if(b-a<2||(phase&&replica>=cap))break;
 vector<pair<uint32_t,uint64_t>> candidates;
 for(auto j=a;j<b;j++){auto v=idx.data[j];check(v>=0&&uint64_t(v)<n,"CSC node range");if(!hot[v]&&(!phase?!used[v]:(used[v]&&!covered[j])))candidates.emplace_back(v,j);}
 sort(candidates.begin(),candidates.end());candidates.erase(unique(candidates.begin(),candidates.end(),[](auto a,auto b){return a.first==b.first;}),candidates.end());
 // Frozen unbiased Fisher-Yates; independent of std::shuffle implementation.
 for(uint64_t k=candidates.size();k>1;k--){uint64_t bound=uint64_t(0)-k;bound%=k;uint64_t r;do{r=rng();}while(r<bound);swap(candidates[k-1],candidates[r%k]);}
 uint64_t take=candidates.size()/2;if(phase)take=min(take,cap-replica);
 for(uint64_t k=0;k<take;k++){auto x=candidates[2*k],y=candidates[2*k+1];groups.push_back({x.first,y.first,owner});covered[x.second]=covered[y.second]=1;counts[owner]++;if(!phase)used[x.first]=used[y.first]=1;}
 if(phase)replica+=take;else primary+=take;if(oi%1000000==0)cout<<"g2_phase "<<phase<<" owners "<<oi<<" groups "<<groups.size()<<endl;}
 uint64_t g=groups.size();check(n+g<(1ULL<<31),"supernode int32 overflow");vector<int64_t> storage(g*4,-1),node(n,-1),members,owners,bases(g),super(g);
 members.reserve(g*2);owners.reserve(g);
 for(uint64_t i=0;i<g;i++){auto v=groups[i];storage[4*i]=v.a;storage[4*i+1]=v.b;members.push_back(v.a);members.push_back(v.b);owners.push_back(v.owner);bases[i]=i*4;super[i]=i;if(i<primary){node[v.a]=4*i;node[v.b]=4*i+1;}}
 for(uint64_t i=0;i<n;i++)if(hot[i]){node[i]=storage.size();storage.push_back(i);}while(storage.size()%4)storage.push_back(-1);
 for(uint64_t i=0;i<n;i++)if(!hot[i]&&!used[i]){node[i]=storage.size();storage.push_back(i);}while(storage.size()%4)storage.push_back(-1);
 check(storage.size()<(1ULL<<31),"storage int32 overflow");vector<int64_t> rp(n+g+1,0),prefix(n+1,0);
 for(uint64_t i=0;i<n;i++){rp[i+1]=rp[i]+p.data[i+1]-p.data[i]-counts[i];prefix[i+1]=prefix[i]+counts[i];}fill(rp.begin()+n+1,rp.end(),rp[n]);
 vector<int64_t> go(g),cursor(prefix);for(uint64_t i=0;i<g;i++)go[cursor[groups[i].owner]++]=i;
 ofstream ri(out+"/reordered_indices.bin",ios::binary);vector<int64_t> buffer;buffer.reserve(65536);
 auto emit=[&](int64_t v){buffer.push_back(v);if(buffer.size()==65536){ri.write((char*)buffer.data(),buffer.size()*8);check(bool(ri),"rewrite output");buffer.clear();}};
 for(uint64_t owner=0;owner<n;owner++){for(int64_t j=p.data[owner];j<p.data[owner+1];j++)if(!covered[j])emit(idx.data[j]);for(auto j=prefix[owner];j<prefix[owner+1];j++)emit(n+go[j]);}
 ri.write((char*)buffer.data(),buffer.size()*8);check(bool(ri),"rewrite output");
 save(out+"/reordered_indptr.bin",rp);save(out+"/group_members.bin",members);save(out+"/group_owner.bin",owners);save(out+"/group_storage_base.bin",bases);save(out+"/supernode_to_group.bin",super);save(out+"/node_to_primary_row.bin",node);save(out+"/storage_to_node.bin",storage);
 ofstream meta(out+"/counts.txt");meta<<primary<<" "<<replica<<" "<<storage.size()<<" "<<rp[n]<<endl;
}
void validate_g2(string ptrfile,string idxfile,uint64_t n,string out){
 Map p(ptrfile),idx(idxfile),rp(out+"/reordered_indptr.npy"),ri(out+"/reordered_indices.npy"),members(out+"/group_members.npy"),owners(out+"/group_owner.npy"),node(out+"/node_to_primary_row.npy"),storage(out+"/storage_to_node.npy");
 uint64_t primary,replica,rows,units;ifstream meta(out+"/counts.txt");meta>>primary>>replica>>rows>>units;uint64_t g=primary+replica;vector<uint8_t> seen(g,0);
 check(rp.data[0]==0&&rp.data[n]==int64_t(units),"rewrite pointers");
 for(uint64_t v=0;v<n;v++) {check(node.data[v]>=0&&uint64_t(node.data[v])<rows&&storage.data[node.data[v]]==int64_t(v),"primary inverse");
 vector<int64_t> expected(idx.data+p.data[v],idx.data+p.data[v+1]),got;
 for(auto j=rp.data[v];j<rp.data[v+1];j++){auto x=ri.data[j];check(x>=0&&uint64_t(x)<n+g,"rewrite range");if(uint64_t(x)<n)got.push_back(x);else{uint64_t id=x-n;check(!seen[id]&&owners.data[id]==int64_t(v),"group owner/reuse");seen[id]=1;got.push_back(members.data[2*id]);got.push_back(members.data[2*id+1]);}}
 sort(expected.begin(),expected.end());sort(got.begin(),got.end());check(expected==got,"expanded adjacency multiset");
 if(v%5000000==0)cout<<"validate_owner "<<v<<"/"<<n<<endl;}
 for(uint64_t i=0;i<g;i++){check(seen[i]&&members.data[2*i]!=members.data[2*i+1],"group coverage");check(storage.data[4*i]==members.data[2*i]&&storage.data[4*i+1]==members.data[2*i+1]&&storage.data[4*i+2]==-1&&storage.data[4*i+3]==-1,"group feature/padding");check(rp.data[n+i+1]==int64_t(units),"supernode degree");}
 cout<<"expanded_adjacency_and_primary_inverse_passed"<<endl;
}
int main(int argc,char**argv){try{check(argc>=2,"missing mode");string mode=argv[1];if(mode=="csc"){check(argc==8,"csc args");csc(argv[2],stoull(argv[3]),stoull(argv[4]),stoi(argv[5]),argv[6],stoull(argv[7]));}
 else if(mode=="validate"){check(argc==6,"validate args");validate_g2(argv[2],argv[3],stoull(argv[4]),argv[5]);}
 else if(mode=="rank"){check(argc==6,"rank args");rank_rev(argv[2],argv[3],stoull(argv[4]),argv[5]);}
 else if(mode=="g2"){check(argc==7,"g2 args");group(argv[2],argv[3],argv[4],stoull(argv[5]),argv[6]);}else throw runtime_error("unknown mode");return 0;}catch(exception&e){cerr<<e.what()<<endl;return 1;}}
