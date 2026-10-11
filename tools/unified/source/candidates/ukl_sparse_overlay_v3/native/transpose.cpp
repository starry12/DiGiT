// Bounded external transpose. No full edge marker, COO, DGL graph or EID array in RAM.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>
using namespace std;
void req(bool x,const string&s){if(!x)throw runtime_error(s);}
uint32_t be32(){unsigned char a[4];req(fread(a,1,4,stdin)==4,"truncated stream");return uint32_t(a[0])<<24|uint32_t(a[1])<<16|uint32_t(a[2])<<8|a[3];}
uint64_t be64(){uint64_t a=be32();return (a<<32)|be32();}
void writeall(FILE*f,const void*p,size_t n){req(fwrite(p,1,n,f)==n,"write failed");}
void finish(FILE*f){req(fflush(f)==0,"flush failed");req(fsync(fileno(f))==0,"fsync failed");req(fclose(f)==0,"close failed");}
void jsonfile(const string&p,const string&v){FILE*f=fopen((p+".next").c_str(),"wb");req(f,"json open");writeall(f,v.data(),v.size());finish(f);req(rename((p+".next").c_str(),p.c_str())==0,"json rename");}
string bucket(const string&d,int b){return d+"/bucket_"+to_string(b)+".u32";}
uint64_t bytes(const string&p){struct stat s;req(stat(p.c_str(),&s)==0,"stat "+p);return s.st_size;}
void scatter(const string&dir,int bins){
 req(be32()==0x554b4c33,"header magic");uint32_t n=be32();uint64_t expected=be64();uint32_t limit=be32();req(n>0&&n<2147483647&&limit>0&&limit<=n,"header extent");req(bins>0&&bins<=4096,"bins");
 uint64_t width=(uint64_t(n)+bins-1)/bins;vector<FILE*>files(bins,nullptr);uint64_t seen=0,removed=0;auto t=chrono::steady_clock::now();
 for(uint32_t u=0;u<limit;u++){
  req(be32()==u,"owner gap");uint32_t degree=be32();req(degree<=expected-seen,"too many edges");
  for(uint32_t j=0;j<degree;j++){
   uint32_t v=be32();req(v<n,"invalid neighbor");if(u==v){removed++;continue;}
   int b=v/width;if(!files[b]){files[b]=fopen(bucket(dir,b).c_str(),"wbx");req(files[b],"bucket exists or cannot open");setvbuf(files[b],nullptr,_IOFBF,65536);}
   uint32_t pair[2]={u,v};writeall(files[b],pair,8);
  }
  seen+=degree;
  if((u+1)%100000==0||u+1==limit){double secs=chrono::duration<double>(chrono::steady_clock::now()-t).count();jsonfile(dir+"/progress.json","{\"stage\":\"scatter\",\"owners\":"+to_string(u+1)+",\"owner_limit\":"+to_string(limit)+",\"edges\":"+to_string(seen)+",\"seconds\":"+to_string(secs)+"}\n");}
 }
 req(be32()==0x454e4433&&be64()==seen,"footer count");req(fgetc(stdin)==EOF&&!ferror(stdin),"trailing stream");req(limit<n||seen==expected,"incomplete full source");
 for(int b=0;b<bins;b++){if(!files[b]){files[b]=fopen(bucket(dir,b).c_str(),"wbx");req(files[b],"bucket open");}finish(files[b]);}
 jsonfile(dir+"/scatter.json","{\"nodes\":"+to_string(n)+",\"source_edges\":"+to_string(expected)+",\"owners\":"+to_string(limit)+",\"edges_seen\":"+to_string(seen)+",\"removed_self\":"+to_string(removed)+",\"bins\":"+to_string(bins)+",\"complete_source\":"+(limit==n?"true":"false")+"}\n");
}
void gather(const string&dir,uint32_t n,int bins,uint64_t expected,uint64_t cap){
 req(n>0&&bins>0&&bins<=4096,"extent");uint64_t width=(uint64_t(n)+bins-1)/bins,total=0;
 FILE*pf=fopen((dir+"/indptr.i64.partial").c_str(),"wbx");req(pf,"preserve pointer output");FILE*xf=fopen((dir+"/indices.i32.partial").c_str(),"wbx");req(xf,"preserve index output");writeall(pf,&total,8);
 for(int b=0;b<bins;b++){
  uint64_t start=uint64_t(b)*width,end=min(uint64_t(n),start+width);if(start>=n){req(bytes(bucket(dir,b))==0,"extra bucket");continue;}
  uint64_t size=bytes(bucket(dir,b));req(size%8==0,"partial pair");uint64_t m=size/8,k=end-start;
  req(4*m+24*(k+1)<=cap,"bucket exceeds explicit RAM cap; increase bins in new build");vector<uint64_t>ptr(k+1,0),cursor;FILE*f=fopen(bucket(dir,b).c_str(),"rb");req(f,"read bucket");uint32_t pair[2];
  for(uint64_t j=0;j<m;j++){req(fread(pair,8,1,f)==1,"short bucket");req(pair[0]<n&&pair[1]>=start&&pair[1]<end&&pair[0]!=pair[1],"bucket range/self");ptr[pair[1]-start+1]++;}
  for(uint64_t i=0;i<k;i++)ptr[i+1]+=ptr[i];cursor=ptr;vector<uint32_t>idx(m);rewind(f);
  for(uint64_t j=0;j<m;j++){req(fread(pair,8,1,f)==1,"reread bucket");idx[cursor[pair[1]-start]++]=pair[0];}req(fclose(f)==0,"close read");
  for(uint64_t i=0;i<k;i++){uint64_t amount=ptr[i+1]-ptr[i];if(amount)writeall(xf,idx.data()+ptr[i],4*amount);uint32_t self=start+i;writeall(xf,&self,4);total+=amount+1;writeall(pf,&total,8);}
  jsonfile(dir+"/progress.json","{\"stage\":\"gather\",\"buckets_complete\":"+to_string(b+1)+",\"buckets\":"+to_string(bins)+",\"normalized_edges\":"+to_string(total)+"}\n");
 }
 req(total==expected,"normalized edge count mismatch");finish(pf);finish(xf);
 req(rename((dir+"/indptr.i64.partial").c_str(),(dir+"/indptr.i64").c_str())==0,"rename ptr");req(rename((dir+"/indices.i32.partial").c_str(),(dir+"/indices.i32").c_str())==0,"rename idx");
 jsonfile(dir+"/csc.json","{\"complete\":true,\"orientation\":\"incoming\",\"nodes\":"+to_string(n)+",\"edges\":"+to_string(total)+",\"second_adjacency\":false,\"eid_array\":false}\n");
}
int main(int argc,char**argv){try{
 req(argc>=4,"scatter dir bins | gather dir nodes bins normalized-edges ram-cap-bytes");string action=argv[1],dir=argv[2];
 if(action=="scatter"){req(argc==4,"scatter args");scatter(dir,stoi(argv[3]));}
 else{req(action=="gather"&&argc==7,"gather args");gather(dir,stoul(argv[3]),stoi(argv[4]),stoull(argv[5]),stoull(argv[6]));}return 0;
 }catch(const exception&e){cerr<<"ERROR: "<<e.what()<<endl;return 1;}}
