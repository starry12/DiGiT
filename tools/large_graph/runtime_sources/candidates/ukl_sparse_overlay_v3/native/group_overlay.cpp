// Original CSC + sparse coverage. Never allocates an E-sized mark or reordered adjacency.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
using namespace std;
void req(bool b,const string&s){if(!b)throw runtime_error(s);}
struct Map {int fd;size_t size;void*data;Map(string path){fd=open(path.c_str(),O_RDONLY);req(fd>=0,"open "+path);struct stat st;req(fstat(fd,&st)==0,"stat");size=st.st_size;data=size?mmap(nullptr,size,PROT_READ,MAP_SHARED,fd,0):nullptr;req(!size||data!=MAP_FAILED,"mmap");}~Map(){if(size)munmap(data,size);close(fd);}};
struct File {FILE*f;File(string p){f=fopen(p.c_str(),"wbx");req(f,"preserve output "+p);}void put(const void*p,size_t n){req(fwrite(p,1,n,f)==n,"write");}void done(){req(fflush(f)==0&&fsync(fileno(f))==0,"flush");req(fclose(f)==0,"close");f=nullptr;}~File(){if(f)fclose(f);}};
struct Group {uint32_t a,b;uint64_t ea,eb;};
int main(int argc,char**argv){try{
 req(argc==6,"group_overlay indptr.i64 indices.i32 hot_nodes.i64 output replica-cap");Map pm(argv[1]),xm(argv[2]),hm(argv[3]);string out=argv[4];uint64_t cap=stoull(argv[5]);
 req(pm.size>=16&&pm.size%8==0&&xm.size%4==0&&hm.size%8==0,"input widths");uint64_t n=pm.size/8-1,e=xm.size/4;req(n<2147483647&&cap<=n/2,"extent");auto p=(int64_t*)pm.data;auto idx=(int32_t*)xm.data;auto hotids=(int64_t*)hm.data;req(p[0]==0&&p[n]>=0&&uint64_t(p[n])==e,"CSC extent");
 vector<uint8_t> hot(n,0),used(n,0);for(uint64_t i=0;i<hm.size/8;i++){req(hotids[i]>=0&&uint64_t(hotids[i])<n&&!hot[hotids[i]],"hot IDs");hot[hotids[i]]=1;}
 vector<uint32_t> order(n);iota(order.begin(),order.end(),0);for(uint64_t i=0;i<n;i++)req(p[i]>=0&&p[i]<=p[i+1],"offset monotonicity");
 sort(order.begin(),order.end(),[&](uint32_t a,uint32_t b){auto da=p[a+1]-p[a],db=p[b+1]-p[b];return da!=db?da>db:a<b;});
 vector<uint64_t> begin[2];vector<uint32_t> count[2];for(int t=0;t<2;t++){begin[t].resize(n);count[t].resize(n);}
 uint64_t upper=(n-hm.size/8)/2+cap;req(n+upper<2147483647,"group ID range");vector<Group> groups;groups.reserve(upper);mt19937_64 rng(0);uint64_t primary=0,replica=0;
 for(int phase=0;phase<2;phase++)for(uint64_t oi=0;oi<n;oi++){
  uint32_t owner=order[oi];int64_t a=p[owner],b=p[owner+1];if(b-a<2||(phase&&replica>=cap))break;
  // Per-owner candidate scratch is bounded separately; never silently OOM on hubs.
  req(uint64_t(b-a)<=((8ULL<<30)/32),"owner scratch exceeds 8 GiB planning cap");
  vector<uint64_t> covered;if(phase){covered.reserve(2*uint64_t(count[0][owner]));for(uint64_t j=begin[0][owner];j<begin[0][owner]+count[0][owner];j++){covered.push_back(groups[j].ea);covered.push_back(groups[j].eb);}sort(covered.begin(),covered.end());}
  vector<pair<uint32_t,uint64_t>> candidates;
  for(int64_t j=a;j<b;j++){int32_t v=idx[j];req(v>=0&&uint64_t(v)<n,"node range");if(!hot[v]&&(!phase?!used[v]:(used[v]&&!binary_search(covered.begin(),covered.end(),uint64_t(j)))))candidates.emplace_back(v,j);}
  sort(candidates.begin(),candidates.end());candidates.erase(unique(candidates.begin(),candidates.end(),[](auto a,auto b){return a.first==b.first;}),candidates.end());
  for(uint64_t k=candidates.size();k>1;k--){uint64_t bound=(uint64_t(0)-k)%k,r;do{r=rng();}while(r<bound);swap(candidates[k-1],candidates[r%k]);}
  uint64_t take=candidates.size()/2;if(phase)take=min(take,cap-replica);req(take<=UINT32_MAX&&groups.size()+take<=upper,"group count");begin[phase][owner]=groups.size();count[phase][owner]=take;
  for(uint64_t k=0;k<take;k++){auto x=candidates[2*k],y=candidates[2*k+1];groups.push_back({x.first,y.first,x.second,y.second});if(!phase)used[x.first]=used[y.first]=1;}
  if(phase)replica+=take;else primary+=take;
  if(oi%1000000==0)cout<<"phase "<<phase<<" owners "<<oi<<" groups "<<groups.size()<<endl;
 }
 vector<uint32_t>().swap(order);vector<uint8_t>().swap(hot);vector<uint8_t>().swap(used);
 File mf(out+"/members.i32"),bf(out+"/bases.i64"),nf(out+"/primary.i64");vector<int64_t> primary_rows(n,-1);
 for(uint64_t g=0;g<groups.size();g++){auto z=groups[g];uint32_t pair[2]={z.a,z.b};mf.put(pair,8);uint64_t base=g<primary?2*g:n+2*(g-primary);bf.put(&base,8);if(g<primary){req(primary_rows[z.a]<0&&primary_rows[z.b]<0,"duplicate primary");primary_rows[z.a]=base;primary_rows[z.b]=base+1;}}
 uint64_t next=2*primary;for(uint64_t i=0;i<n;i++)if(primary_rows[i]<0)primary_rows[i]=next++;req(next==n,"primary layout extent");nf.put(primary_rows.data(),8*n);nf.done();mf.done();bf.done();vector<int64_t>().swap(primary_rows);
 File pf(out+"/group_ptr.i64"),gf(out+"/group_ids.i32"),cf(out+"/covered.i64");uint64_t prefix=0;pf.put(&prefix,8);
 for(uint64_t owner=0;owner<n;owner++){
  vector<uint64_t> covers;
  for(int phase=0;phase<2;phase++)for(uint64_t g=begin[phase][owner];g<begin[phase][owner]+count[phase][owner];g++){uint32_t id=g;gf.put(&id,4);covers.push_back(groups[g].ea);covers.push_back(groups[g].eb);prefix++;}
  sort(covers.begin(),covers.end());for(size_t j=0;j<covers.size();j++){req(covers[j]>=uint64_t(p[owner])&&covers[j]<uint64_t(p[owner+1]),"cover range");req(!j||covers[j]!=covers[j-1],"cover overlap");}
  if(!covers.empty())cf.put(covers.data(),8*covers.size());pf.put(&prefix,8);
 }
 req(prefix==groups.size(),"group ownership");pf.done();gf.done();cf.done();
 File meta(out+"/overlay.json");string s="{\"complete\":true,\"nodes\":"+to_string(n)+",\"edges\":"+to_string(e)+",\"primary_groups\":"+to_string(primary)+",\"replica_groups\":"+to_string(replica)+",\"groups\":"+to_string(groups.size())+",\"storage_rows\":"+to_string(n+2*replica)+",\"second_adjacency\":false,\"edge_sized_marker\":false,\"layout\":\"packed512-g2\"}\n";meta.put(s.data(),s.size());meta.done();return 0;
 }catch(const exception&e){cerr<<"ERROR: "<<e.what()<<endl;return 1;}}
