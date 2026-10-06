// Bounded CPU direct writer and cooperative cancellation, no CUDA or file maps.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
using Tick=int(*)(uint64_t,uint64_t,int);
static thread_local Tick tick_fn=nullptr;
static thread_local uint64_t write_rate=512ULL<<20;
inline void poll(uint64_t done=0,uint64_t total=0,int phase=0){
 if(tick_fn&&tick_fn(done,total,phase))throw std::runtime_error("cooperative cancellation");
}
struct File {
 int fd=-1;void*mem=nullptr;size_t used=0;uint64_t logical=0,written=0;
 std::string final,partial;std::chrono::steady_clock::time_point start;
 static constexpr size_t chunk=1<<20;
 File(std::string p):final(p),partial(p+".partial"),start(std::chrono::steady_clock::now()){
  poll();if(access(p.c_str(),F_OK)==0)throw std::runtime_error("preserve output "+p);
  if(posix_memalign(&mem,4096,chunk))throw std::runtime_error("aligned buffer allocation");
  fd=open(partial.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_DIRECT|O_NOFOLLOW|O_CLOEXEC,0644);
  if(fd<0){free(mem);mem=nullptr;throw std::runtime_error("direct open "+partial);}
 }
 void flush(){
  if(!used)return;poll();size_t size=(used+4095)/4096*4096;
  memset((char*)mem+used,0,size-used);
  auto batch_start=std::chrono::steady_clock::now();
  auto got=pwrite(fd,mem,size,written);
  if(got!=ssize_t(size))throw std::runtime_error("short direct write");
  written+=size;used=0;
  while(true){poll();double delay=double(size)/write_rate-std::chrono::duration<double>(std::chrono::steady_clock::now()-batch_start).count();
   if(delay<=0)break;std::this_thread::sleep_for(std::chrono::duration<double>(std::min(delay,.1)));}
 }
 void put(const void*p,size_t n){
  logical+=n;const char*src=(const char*)p;
  while(n){size_t take=std::min(n,chunk-used);memcpy((char*)mem+used,src,take);used+=take;src+=take;n-=take;if(used==chunk)flush();}
 }
 void done(){
  flush();poll();if(ftruncate(fd,logical)||fsync(fd))throw std::runtime_error("direct truncate/fsync");
  if(close(fd))throw std::runtime_error("close output");fd=-1;
  if(link(partial.c_str(),final.c_str()))throw std::runtime_error("exclusive publication");
  if(unlink(partial.c_str()))throw std::runtime_error("unlink published partial");
  auto parent=final.substr(0,final.find_last_of('/'));int d=open(parent.c_str(),O_DIRECTORY|O_RDONLY);
  if(d<0)throw std::runtime_error("open parent");int rc=fsync(d);close(d);if(rc)throw std::runtime_error("directory fsync");
 }
 ~File(){if(fd>=0)close(fd);free(mem);}
};
inline void save(std::string p,const void*data,size_t n){File f(p);f.put(data,n);f.done();}
