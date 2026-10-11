// CPU filesystem preparation only. Original graph and feature files are read-only.
#include "../../uks_group_reuse_v2/matching.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <map>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
using I=int64_t; namespace fs=std::filesystem;
void need(bool p,const std::string&s){if(!p)throw std::runtime_error(s);}
struct Array {
 int fd;void* mm;size_t bytes,offset,count;int width;
 Array(const std::string&path,int w,size_t o,size_t c):offset(o),count(c),width(w){
  fd=open(path.c_str(),O_RDONLY|O_NOFOLLOW);need(fd>=0,"open "+path);struct stat st{};
  need(!fstat(fd,&st)&&S_ISREG(st.st_mode),"regular source required");bytes=st.st_size;
  need((w==4||w==8)&&o%w==0&&o<=bytes&&c==(bytes-o)/w&&(bytes-o)%w==0,"array extent");
  mm=mmap(nullptr,bytes,PROT_READ,MAP_PRIVATE,fd,0);need(mm!=MAP_FAILED,"CPU mmap failed");
 }
 ~Array(){munmap(mm,bytes);close(fd);}
 I operator[](size_t i)const{need(i<count,"array index");auto p=(const char*)mm+offset;return width==8?((const I*)p)[i]:((const int32_t*)p)[i];}
};
struct Input {
 I n,g,e;bool dense;std::map<std::string,std::unique_ptr<Array>> a;
 explicit Input(const fs::path&spec){std::ifstream in(spec);std::string mode;in>>mode>>n>>g>>e;dense=mode=="dense";
  need((dense||mode=="sparse")&&n>0&&g>=0&&n+g<INT32_MAX&&e>=0,"metadata");
  std::string key,path;int width;size_t offset,count;
  while(in>>key>>width>>offset>>count>>path){need(!a.count(key),"duplicate spec");a[key].reset(new Array(path,width,offset,count));}
  need(in.eof(),"malformed spec");need(at("ptr").count==size_t(n+1)&&at("idx").count==size_t(e)&&at("members").count==size_t(2*g),"input extent");
  need(at("ptr")[0]==0&&at("ptr")[n]==e,"CSC boundaries");
  need(at("gp").count==size_t(dense?n+g+1:n+1)&&at("gp")[0]==0&&at("gp")[n]==I(at("gi").count),"overlay boundaries");
  if(dense){need(at("super").count==size_t(g),"super extent");for(I i=0;i<g;++i){need(at("super")[i]==i,"nonidentity super mapping");need(at("gp")[n+i+1]==at("gp")[n],"nonempty super row");}}
  else need(at("covered").count==2*at("gi").count,"covered extent");
  for(I i=0;i<g;++i){I x=at("members")[2*i],y=at("members")[2*i+1];need(x>=0&&x<n&&y>=0&&y<n&&x!=y,"invalid physical group");}
 }
 const Array&at(const std::string&k)const{return *a.at(k);}
 std::vector<I> units(I owner)const{
  I lo=at("ptr")[owner],hi=at("ptr")[owner+1],a=at("gp")[owner],b=at("gp")[owner+1];
  need(0<=lo&&lo<=hi&&hi<=e&&0<=a&&a<=b&&b<=I(at("gi").count),"nonmonotone row");
  std::vector<I> u,expanded;u.reserve(hi-lo);
  if(dense){for(I j=a;j<b;++j){I v=at("gi")[j];need(v>=0&&v<n+g,"bad unit");u.push_back(v);}}
  else{
   I c=2*a,prev=lo-1;for(I j=2*a;j<2*b;++j){I v=at("covered")[j];need(v>prev&&v>=lo&&v<hi,"bad covered EID");prev=v;}
   for(I j=lo;j<hi;++j){if(c<2*b&&at("covered")[c]==j){++c;continue;}u.push_back(at("idx")[j]);}
   for(I j=a;j<b;++j){I gid=at("gi")[j];need(gid>=0&&gid<g,"bad group ID");u.push_back(n+gid);}
  }
  for(I v:u){need(v>=0&&v<n+g,"unit bounds");if(v<n)expanded.push_back(v);else{expanded.push_back(at("members")[2*(v-n)]);expanded.push_back(at("members")[2*(v-n)+1]);}}
  std::vector<I> original;for(I j=lo;j<hi;++j){I v=at("idx")[j];need(v>=0&&v<n,"node bounds");original.push_back(v);}
  std::sort(original.begin(),original.end());std::sort(expanded.begin(),expanded.end());need(original==expanded,"expanded multigraph mismatch at "+std::to_string(owner));return u;
 }
};
struct Index {
 std::vector<uint32_t> offsets,ids;
 explicit Index(const Input&x):offsets(x.n+1),ids(x.g){
  const auto&m=x.at("members");for(I j=0;j<x.g;++j)++offsets[std::min(m[2*j],m[2*j+1])+1];
  for(I i=0;i<x.n;++i)offsets[i+1]+=offsets[i];auto cursor=offsets;
  for(I j=0;j<x.g;++j)ids[cursor[std::min(m[2*j],m[2*j+1])]++]=j;
 }
};
struct Row {
 std::vector<I> existing,nodes;std::vector<size_t> counts;std::vector<PairEdge> edges;
 Row(const Input&x,const Index&idx,const std::vector<I>&u){
  std::vector<I> raw;for(I v:u)(v<x.n?raw:existing).push_back(v<x.n?v:v-x.n);
  std::sort(raw.begin(),raw.end());std::sort(existing.begin(),existing.end());need(std::adjacent_find(existing.begin(),existing.end())==existing.end(),"duplicate original reference");
  std::vector<uint32_t> candidates;
  for(size_t i=0;i<raw.size();){size_t j=i+1;while(j<raw.size()&&raw[j]==raw[i])++j;
   I v=raw[i];nodes.push_back(v);counts.push_back(j-i);for(uint32_t k=idx.offsets[v];k<idx.offsets[v+1];++k)candidates.push_back(idx.ids[k]);i=j;}
  std::sort(candidates.begin(),candidates.end());
  for(I g:candidates){if(std::binary_search(existing.begin(),existing.end(),g))continue;I a=x.at("members")[2*g],b=x.at("members")[2*g+1];
   auto ia=std::lower_bound(nodes.begin(),nodes.end(),a),ib=std::lower_bound(nodes.begin(),nodes.end(),b);
   if(ia!=nodes.end()&&*ia==a&&ib!=nodes.end()&&*ib==b)edges.push_back({size_t(ia-nodes.begin()),size_t(ib-nodes.begin()),g});}
 }
 std::vector<I> rewrite(const Input&x,const std::vector<I>&u,const Selection&s)const{
  std::vector<size_t> used(nodes.size());std::vector<I> added,out;
  for(size_t i=0;i<edges.size();++i)if(s.chosen[i]){++used[edges[i].a];++used[edges[i].b];added.push_back(x.n+edges[i].gid);}
  for(I v:u){if(v<x.n){auto j=std::lower_bound(nodes.begin(),nodes.end(),v)-nodes.begin();if(used[j]){--used[j];continue;}}out.push_back(v);}
  need(std::all_of(used.begin(),used.end(),[](size_t v){return !v;}),"overconsumed occurrences");out.insert(out.end(),added.begin(),added.end());return out;
 }
};
struct Writer {
 int fd,width;uint64_t count=0,flushed=0,synced=0;std::vector<char> buffer;
 Writer(fs::path p,int w):width(w){fd=open(p.c_str(),O_CREAT|O_EXCL|O_WRONLY,0644);need(fd>=0,"refuse existing output "+p.string());buffer.reserve(8<<20);}
 void flush(){size_t done=0;while(done<buffer.size()){auto n=write(fd,buffer.data()+done,buffer.size()-done);need(n>0,"write failed");done+=n;}flushed+=done;buffer.clear();
  if(flushed-synced>=(64<<20)){need(!fdatasync(fd),"bounded output sync failed");posix_fadvise(fd,synced,flushed-synced,POSIX_FADV_DONTNEED);synced=flushed;}}
 void push(I v){need(width==8||(v>=0&&v<=INT32_MAX),"i32 overflow");auto p=(char*)&v;buffer.insert(buffer.end(),p,p+width);++count;if(buffer.size()>=(8<<20))flush();}
 void finish(){flush();need(!fsync(fd),"fsync failed");posix_fadvise(fd,0,0,POSIX_FADV_DONTNEED);close(fd);fd=-1;}
 ~Writer(){if(fd>=0)close(fd);}
};
struct Result {std::vector<I> values,covered;uint64_t old=0,added=0,greedy=0,improved=0;};
Result rewrite(const Input&x,const Index&idx,I owner){
 auto u=x.units(owner);Row row(x,idx,u);auto selected=select_pairs(row.counts,row.edges);auto result=row.rewrite(x,u,selected);
 Result r;r.old=row.existing.size();r.greedy=selected.greedy;r.added=selected.greedy+selected.swaps;r.improved=selected.swaps>0;
 if(x.dense){r.values=std::move(result);return r;}
 // Preserve original covered occurrences; consume added groups only from the remainder.
 I a=x.at("gp")[owner],b=x.at("gp")[owner+1];for(I j=2*a;j<2*b;++j)r.covered.push_back(x.at("covered")[j]);
 std::map<I,std::vector<I>> available;I c=2*a;
 for(I j=x.at("ptr")[owner];j<x.at("ptr")[owner+1];++j){if(c<2*b&&x.at("covered")[c]==j){++c;continue;}available[x.at("idx")[j]].push_back(j);}
 for(I v:result)if(v>=x.n){I g=v-x.n;r.values.push_back(g);if(std::binary_search(row.existing.begin(),row.existing.end(),g))continue;
  for(int k=0;k<2;++k){auto&slots=available[x.at("members")[2*g+k]];need(!slots.empty(),"missing group occurrence");r.covered.push_back(slots.back());slots.pop_back();}}
 std::sort(r.covered.begin(),r.covered.end());need(r.covered.size()==2*r.values.size(),"covered output count");return r;
}
void independent_output_check(const Input&x,I owner,const Array&p,const Array&u,const Array*c){
 I a=p[owner],b=p[owner+1];need(0<=a&&a<=b&&b<=I(u.count),"output pointers");
 std::vector<I> groups,actual,expected,old;
 for(I j=a;j<b;++j){I v=u[j];if(x.dense&&v<x.n){need(v>=0,"negative raw output");actual.push_back(v);continue;}
  I gid=x.dense?v-x.n:v;need(gid>=0&&gid<x.g,"group output range");groups.push_back(gid);actual.push_back(x.at("members")[2*gid]);actual.push_back(x.at("members")[2*gid+1]);}
 std::sort(groups.begin(),groups.end());need(std::adjacent_find(groups.begin(),groups.end())==groups.end(),"duplicate output reference");
 for(I j=x.at("gp")[owner];j<x.at("gp")[owner+1];++j){I v=x.at("gi")[j];if(!x.dense||v>=x.n)old.push_back(x.dense?v-x.n:v);}
 std::sort(old.begin(),old.end());need(std::includes(groups.begin(),groups.end(),old.begin(),old.end()),"removed original reference");
 I lo=x.at("ptr")[owner],hi=x.at("ptr")[owner+1];
 if(x.dense){for(I j=lo;j<hi;++j)expected.push_back(x.at("idx")[j]);}
 else{need(c&&c->count==2*u.count,"covered output extent");I prev=lo-1;
  for(I j=2*a;j<2*b;++j){I eid=(*c)[j];need(eid>prev&&eid>=lo&&eid<hi,"invalid covered occurrence");prev=eid;expected.push_back(x.at("idx")[eid]);}
  std::vector<I> old_eids;for(I j=2*x.at("gp")[owner];j<2*x.at("gp")[owner+1];++j)old_eids.push_back(x.at("covered")[j]);
  std::vector<I> new_eids;for(I j=2*a;j<2*b;++j)new_eids.push_back((*c)[j]);need(std::includes(new_eids.begin(),new_eids.end(),old_eids.begin(),old_eids.end()),"lost original covered occurrence");}
 std::sort(expected.begin(),expected.end());std::sort(actual.begin(),actual.end());need(actual==expected,"output multigraph expansion mismatch");
}
int main(int argc,char**argv){try{
 need(argc==5,"usage: reuse build|validate SPEC OUTPUT THREADS");std::string mode=argv[1];need(mode=="build"||mode=="validate","mode");Input x(argv[2]);Index idx(x);fs::path out=argv[3];
 int threads=std::stoi(argv[4]);need(threads>0&&threads<=16,"thread limit");
 std::unique_ptr<Writer> ptr,units,covered;std::unique_ptr<Array> op,oi,oc;
 if(mode=="build"){
  need(fs::create_directory(out),"preserve existing output");ptr.reset(new Writer(out/"ptr.i64",8));units.reset(new Writer(out/(x.dense?"units.i64":"units.i32"),x.dense?8:4));ptr->push(0);
  if(!x.dense)covered.reset(new Writer(out/"covered.i64",8));
 }else{
  op.reset(new Array((out/"ptr.i64").string(),8,0,x.dense?x.n+x.g+1:x.n+1));auto p=out/(x.dense?"units.i64":"units.i32");oi.reset(new Array(p.string(),x.dense?8:4,0,fs::file_size(p)/(x.dense?8:4)));
  if(!x.dense)oc.reset(new Array((out/"covered.i64").string(),8,0,fs::file_size(out/"covered.i64")/8));need((*op)[0]==0,"output pointer start");
 }
 uint64_t old=0,added=0,greedy=0,improved=0,owners=0,total=0;
 constexpr I chunk=4096;
 for(I start=0;start<x.n;start+=chunk){I end=std::min(x.n,start+chunk);std::vector<Result> rows(end-start);std::vector<std::string> errors(end-start);
  #pragma omp parallel for num_threads(threads) schedule(dynamic,16)
  for(I i=start;i<end;++i){try{rows[i-start]=rewrite(x,idx,i);}catch(const std::exception&e){errors[i-start]=e.what();}}
  for(I i=start;i<end;++i){need(errors[i-start].empty(),"owner "+std::to_string(i)+": "+errors[i-start]);auto&r=rows[i-start];
   if(mode=="build"){for(I v:r.values)units->push(v);if(covered)for(I v:r.covered)covered->push(v);ptr->push(units->count);}
   else{need((*op)[i]==I(total)&&(*op)[i+1]==I(total+r.values.size()),"output pointer mismatch");
    independent_output_check(x,i,*op,*oi,oc.get());
    // Also require exact deterministic v2 output, independently of thread count.
    for(size_t j=0;j<r.values.size();++j)need((*oi)[total+j]==r.values[j],"output value mismatch");
    if(oc)for(size_t j=0;j<r.covered.size();++j)need((*oc)[2*total+j]==r.covered[j],"output occurrence mismatch");
   }
   old+=r.old;added+=r.added;greedy+=r.greedy;improved+=r.improved;owners+=r.added>0;total+=r.values.size();
  }
  if(start% (chunk*256)==0)std::cerr<<mode<<" owners "<<end<<"/"<<x.n<<" added "<<added<<" extra "<<added-greedy<<std::endl;
 }
 if(mode=="build"){if(x.dense)for(I i=0;i<x.g;++i)ptr->push(total);ptr->finish();units->finish();if(covered)covered->finish();}
 else{need(oi->count==total,"output terminal extent");if(oc)need(oc->count==2*total,"covered terminal extent");if(x.dense)for(I i=x.n;i<=x.n+x.g;++i)need((*op)[i]==I(total),"output super row");}
 need(x.dense?total+added==x.at("gi").count:total==old+added,"unit conservation");
 std::cout<<"{\"passed\":true,\"nodes\":"<<x.n<<",\"edges\":"<<x.e<<",\"groups\":"<<x.g<<",\"old_group_references\":"<<old<<",\"added_group_references\":"<<added<<",\"v1_greedy_added_references\":"<<greedy<<",\"extra_references_over_v1\":"<<added-greedy<<",\"owners_improved_over_v1\":"<<improved<<",\"owners_with_added_groups\":"<<owners<<",\"output_units\":"<<total<<",\"group_covered_edges\":"<<2*(old+added)<<",\"feature_layout_changed\":false}"<<std::endl;
 return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
