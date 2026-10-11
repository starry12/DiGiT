// Filesystem-only streaming rewrite. No CUDA, raw devices or feature writes.
#include <algorithm>
#include "matching.h"
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
namespace fs=std::filesystem;
using I=int64_t;
void need(bool ok,const std::string& why){if(!ok)throw std::runtime_error(why);}

struct Npy {
 int fd=-1;size_t bytes=0;void* address=MAP_FAILED;const I* p=nullptr;size_t count=0;
 std::vector<size_t> shape;
 explicit Npy(const fs::path& file){
  fd=open(file.c_str(),O_RDONLY|O_NOFOLLOW);need(fd>=0,"open "+file.string());
  struct stat st{};need(fstat(fd,&st)==0&&S_ISREG(st.st_mode)&&st.st_size>=12,"invalid input file");bytes=st.st_size;
  address=mmap(nullptr,bytes,PROT_READ,MAP_PRIVATE,fd,0);need(address!=MAP_FAILED,"input mmap");
  auto b=static_cast<const unsigned char*>(address);need(!memcmp(b,"\x93NUMPY",6),"NPY magic");
  need((b[6]==1||b[6]==2)&&b[7]==0,"NPY version");size_t offset=b[6]==1?10:12;
  size_t length=size_t(b[8])|(size_t(b[9])<<8);
  if(offset==12)length|=(size_t(b[10])<<16)|(size_t(b[11])<<24);
  need(length<=bytes-offset,"NPY header length");std::string header((const char*)b+offset,length);offset+=length;
  need(header.find("'<i8'")!=std::string::npos&&header.find("False")!=std::string::npos,"expected little-endian C-order int64");
  std::smatch match;need(std::regex_search(header,match,std::regex("'shape'\\s*:\\s*\\(([^)]*)\\)")),"NPY shape");
  std::string dims=match[1];std::regex digits("[0-9]+");size_t product=1;
  for(auto it=std::sregex_iterator(dims.begin(),dims.end(),digits);it!=std::sregex_iterator();++it){
   size_t v=std::stoull(it->str());need(!v||product<=SIZE_MAX/v,"NPY extent overflow");shape.push_back(v);product*=v;
  }
  need(!shape.empty()&&offset%8==0&&(bytes-offset)%8==0&&product==(bytes-offset)/8,"NPY payload extent");
  count=product;p=(const I*)(b+offset);
 }
 ~Npy(){if(address!=MAP_FAILED)munmap(address,bytes);if(fd>=0)close(fd);}
 Npy(const Npy&)=delete;
};

struct Writer {
 std::fstream stream;std::vector<I> buffer;uint64_t count=0;
 explicit Writer(const fs::path& path){
  int fd=open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL,0644);need(fd>=0,"refuse existing output");close(fd);
  stream.open(path,std::ios::in|std::ios::out|std::ios::binary);need(bool(stream),"output open");header(0);buffer.reserve(65536);
 }
 void header(uint64_t n){
  std::string h="{'descr': '<i8', 'fortran_order': False, 'shape': ("+std::to_string(n)+",), }";
  need(h.size()<245,"NPY output header");h.resize(245,' ');h+='\n';
  const unsigned char prefix[]={0x93,'N','U','M','P','Y',1,0,246,0};
  stream.seekp(0);stream.write((const char*)prefix,10);stream.write(h.data(),h.size());
 }
 void flush(){stream.write((const char*)buffer.data(),buffer.size()*sizeof(I));need(bool(stream),"output write");buffer.clear();}
 void push(I v){buffer.push_back(v);++count;if(buffer.size()==65536)flush();}
 void finish(){flush();header(count);stream.flush();need(bool(stream),"output finish");stream.close();}
};

struct Inputs {
 Npy cp,ci,rp,ri,m,super;I n,g;
 Inputs(const fs::path& data,I nodes):cp(data/"csc/indptr.npy"),ci(data/"csc/indices.npy"),
  rp(data/"g2/reordered_indptr.npy"),ri(data/"g2/reordered_indices.npy"),m(data/"g2/group_members.npy"),super(data/"g2/supernode_to_group.npy"),n(nodes),g(m.count/2){
  need(n>0&&n+g<INT32_MAX&&m.shape==std::vector<size_t>{size_t(g),2},"g2/int32 extent");
  pointers(cp,ci,n+1,n);pointers(rp,ri,n+g+1,n);
  for(I i=n+1;i<=n+g;++i)need(rp.p[i]==rp.p[n],"nonempty supernode row");
  need(super.shape==std::vector<size_t>{size_t(g)},"supernode mapping extent");
  for(I j=0;j<g;++j)need(super.p[j]==j,"UKS builder requires its frozen identity supernode mapping");
  for(I j=0;j<g;++j)need(m.p[2*j]>=0&&m.p[2*j]<n&&m.p[2*j+1]>=0&&m.p[2*j+1]<n&&m.p[2*j]!=m.p[2*j+1],"bad group members");
 }
 static void pointers(const Npy& p,const Npy& idx,I size,I n){
  need(p.shape==std::vector<size_t>{size_t(size)}&&idx.shape.size()==1&&p.p[0]==0&&p.p[n]==I(idx.count),"pointer extent");
  for(I i=0;i+1<size;++i)need(p.p[i]>=0&&p.p[i]<=p.p[i+1]&&p.p[i+1]<=I(idx.count),"nonmonotone pointer");
 }
 std::vector<I> expanded(const I* a,const I* b)const{
  std::vector<I> out;out.reserve(2*(b-a));
  for(;a!=b;++a){need(*a>=0&&*a<n+g,"unit range");if(*a<n)out.push_back(*a);else{out.push_back(m.p[2*(*a-n)]);out.push_back(m.p[2*(*a-n)+1]);}}
  std::sort(out.begin(),out.end());return out;
 }
 void exact(I owner,const I* a,const I* b)const{
  std::vector<I> expected(ci.p+cp.p[owner],ci.p+cp.p[owner+1]);std::sort(expected.begin(),expected.end());
  need(expanded(a,b)==expected,"expanded multigraph mismatch at owner "+std::to_string(owner));
 }
};

struct Stats {uint64_t old_refs=0,added=0,owners=0,units=0,greedy_added=0,improved_owners=0;
 void print(const Inputs& x)const{
  std::cout<<"{\"passed\":true,\"nodes\":"<<x.n<<",\"edges\":"<<x.ci.count<<",\"groups\":"<<x.g
   <<",\"old_group_references\":"<<old_refs<<",\"added_group_references\":"<<added
   <<",\"v1_greedy_added_references\":"<<greedy_added<<",\"extra_references_over_v1\":"<<added-greedy_added
   <<",\"owners_improved_over_v1\":"<<improved_owners
   <<",\"owners_with_added_groups\":"<<owners<<",\"units\":"<<units
   <<",\"old_group_covered_edges\":"<<2*old_refs<<",\"new_group_covered_edges\":"<<2*(old_refs+added)
   <<",\"expanded_multigraph_exact\":true,\"original_group_references_preserved\":true,\"feature_layout_changed\":false}"<<std::endl;
 }
};

struct GroupIndex {
 std::vector<uint32_t> offsets,ids;
 explicit GroupIndex(const Inputs& x):offsets(x.n+1,0),ids(x.g){
  for(I j=0;j<x.g;++j)++offsets[std::min(x.m.p[2*j],x.m.p[2*j+1])+1];
  for(I i=0;i<x.n;++i)offsets[i+1]+=offsets[i];
  auto cursor=offsets;
  for(I j=0;j<x.g;++j)ids[cursor[std::min(x.m.p[2*j],x.m.p[2*j+1])]++]=j;
 }
};
struct Row {
 std::vector<std::pair<I,size_t>> raw;
 std::vector<I> existing,nodes;
 std::vector<size_t> counts;
 std::vector<PairEdge> edges;
 Row(const Inputs& x,const GroupIndex& index,const I* begin,const I* end){
  for(const I* q=begin;q!=end;++q){if(*q<x.n)raw.emplace_back(*q,q-begin);else existing.push_back(*q-x.n);}
  std::sort(raw.begin(),raw.end());std::sort(existing.begin(),existing.end());
  need(std::adjacent_find(existing.begin(),existing.end())==existing.end(),"duplicate base group reference");
  std::vector<uint32_t> candidates;
  for(size_t i=0;i<raw.size();){size_t j=i+1;while(j<raw.size()&&raw[j].first==raw[i].first)++j;
   I v=raw[i].first;nodes.push_back(v);counts.push_back(j-i);
   for(uint32_t z=index.offsets[v];z<index.offsets[v+1];++z)candidates.push_back(index.ids[z]);
   i=j;
  }
  std::sort(candidates.begin(),candidates.end());
  for(uint32_t gid:candidates){
   if(std::binary_search(existing.begin(),existing.end(),I(gid)))continue;
   I a=x.m.p[2*gid],b=x.m.p[2*gid+1];
   auto ia=std::lower_bound(nodes.begin(),nodes.end(),a),ib=std::lower_bound(nodes.begin(),nodes.end(),b);
   if(ia==nodes.end()||*ia!=a||ib==nodes.end()||*ib!=b)continue;
   edges.push_back({size_t(ia-nodes.begin()),size_t(ib-nodes.begin()),I(gid)});
  }
 }
 std::vector<I> rewrite(const I* begin,const I* end,I n,const Selection& selected)const{
  std::vector<size_t> consume(nodes.size(),0);std::vector<I> added;
  for(size_t i=0;i<edges.size();++i)if(selected.chosen[i]){
   ++consume[edges[i].a];++consume[edges[i].b];added.push_back(n+edges[i].gid);
  }
  std::vector<unsigned char> removed(end-begin,0);
  size_t cursor=0;
  for(size_t i=0;i<nodes.size();++i){
   need(consume[i]<=counts[i],"overconsumed raw occurrences");
   for(size_t j=0;j<consume[i];++j)removed[raw[cursor+j].second]=1;
   cursor+=counts[i];
  }
  std::vector<I> result;
  for(auto q=begin;q!=end;++q)if(!removed[q-begin])result.push_back(*q);
  result.insert(result.end(),added.begin(),added.end());return result;
 }
};

void build(const fs::path& data,const fs::path& output,I n){
 need(!fs::exists(output),"output directory exists; preserve previous work");Inputs x(data,n);GroupIndex index(x);
 need(fs::create_directory(output),"cannot create output directory");Writer ptr(output/"reordered_indptr.npy"),idx(output/"reordered_indices.npy");ptr.push(0);Stats s;
 for(I owner=0;owner<n;++owner){
  const I* begin=x.ri.p+x.rp.p[owner];const I* end=x.ri.p+x.rp.p[owner+1];
  x.exact(owner,begin,end);Row row(x,index,begin,end);
  auto selected=select_pairs(row.counts,row.edges);auto result=row.rewrite(begin,end,n,selected);
  size_t added=std::count(selected.chosen.begin(),selected.chosen.end(),1);
  need(added==selected.greedy+selected.swaps,"nonmonotone matching improvement");
  s.old_refs+=row.existing.size();s.greedy_added+=selected.greedy;s.improved_owners+=selected.swaps>0;
  for(I v:result)idx.push(v);
  s.added+=added;s.owners+=added>0;ptr.push(idx.count);
  if(owner%1000000==0)std::cerr<<"reuse v2 owners "<<owner<<"/"<<n<<" added "<<s.added<<" extra_over_v1 "<<s.added-s.greedy_added<<std::endl;
 }
 for(I i=0;i<x.g;++i)ptr.push(idx.count);
 s.units=idx.count;need(s.units+s.added==x.ri.count,"unit reduction mismatch");ptr.finish();idx.finish();s.print(x);
}

void validate(const fs::path& data,const fs::path& output,I n){
 Inputs x(data,n);GroupIndex index(x);Npy rp(output/"reordered_indptr.npy"),ri(output/"reordered_indices.npy");
 Inputs::pointers(rp,ri,n+x.g+1,n);for(I i=n+1;i<=n+x.g;++i)need(rp.p[i]==rp.p[n],"nonempty output supernode");Stats s;
 for(I owner=0;owner<n;++owner){
  auto a=x.ri.p+x.rp.p[owner],b=x.ri.p+x.rp.p[owner+1],c=ri.p+rp.p[owner],d=ri.p+rp.p[owner+1];x.exact(owner,a,b);x.exact(owner,c,d);
  std::vector<I> oldg,newg,oldraw,newraw;
  for(auto q=a;q!=b;++q)(*q<n?oldraw:oldg).push_back(*q);
  for(auto q=c;q!=d;++q)(*q<n?newraw:newg).push_back(*q);
  std::sort(oldg.begin(),oldg.end());std::sort(newg.begin(),newg.end());std::sort(oldraw.begin(),oldraw.end());std::sort(newraw.begin(),newraw.end());
  need(std::includes(newg.begin(),newg.end(),oldg.begin(),oldg.end()),"removed original group");
  need(std::adjacent_find(newg.begin(),newg.end())==newg.end(),"duplicate group within owner");
  need(std::includes(oldraw.begin(),oldraw.end(),newraw.begin(),newraw.end()),"introduced raw edge");
  auto added=newg.size()-oldg.size();Row row(x,index,a,b);auto baseline=select_pairs(row.counts,row.edges,false);
  need(added>=baseline.greedy,"coverage regressed below v1 at owner "+std::to_string(owner));
  s.greedy_added+=baseline.greedy;s.improved_owners+=added>baseline.greedy;s.old_refs+=oldg.size();s.added+=added;s.owners+=added>0;
 }
 s.units=ri.count;need(s.units+s.added==x.ri.count,"unit count mismatch");s.print(x);
}
int main(int argc,char** argv){try{
 need(argc==5,"usage: reuse build|validate INPUT_DATA OUTPUT NODES");std::string mode=argv[1];
 if(mode=="build")build(argv[2],argv[3],std::stoll(argv[4]));else if(mode=="validate")validate(argv[2],argv[3],std::stoll(argv[4]));else throw std::runtime_error("unknown mode");return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
