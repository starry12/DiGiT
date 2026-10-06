// One outstanding command, owned bounded anonymous DMA, no CUDA/file registration.
#include <nvm_types.h>
#include <nvm_ctrl.h>
#include <nvm_dma.h>
#include <nvm_aq.h>
#include <nvm_admin.h>
#include <nvm_cmd.h>
#include <nvm_queue.h>
#include <nvm_error.h>
#include <nvm_util.h>
#include <algorithm>
#include <cstring>
#include <string>
#include <stdexcept>
#include <chrono>
#include <thread>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
static thread_local std::string last_error;
static void require(bool b,const char*m){if(!b)throw std::runtime_error(m);}
static void ok(int s){if(s)throw std::runtime_error(nvm_strerror(s));}
struct Session {
 nvm_ctrl_t*ctrl=nullptr;nvm_aq_ref aq=nullptr;nvm_dma_t *admin=nullptr,*queues=nullptr,*data=nullptr,*prp=nullptr;
 nvm_queue_t cq={},sq={};bool have_cq=false,have_sq=false,poison=false,live=false;
 uint64_t capacity=0,begin[2]={},end[2]={};uint32_t ns=1;size_t chunk=0;
 void alloc(nvm_dma_t**p,size_t size){ok(nvm_dma_create(p,ctrl,size));require(madvise((void*)(*p)->vaddr,(*p)->page_size*(*p)->n_ioaddrs,MADV_DONTFORK)==0,"DMA DONTFORK failed");memset((void*)(*p)->vaddr,0,(*p)->page_size*(*p)->n_ioaddrs);}
 void open_device(const char*serial,uint64_t cap,const uint64_t*ranges){
  int fd=::open("/dev/libnvm0",O_RDWR|O_NONBLOCK|O_CLOEXEC|O_NOFOLLOW);require(fd>=0,"open fixed libnvm0");int rc=nvm_ctrl_init(&ctrl,fd);::close(fd);ok(rc);
  require(ctrl->page_size==4096,"controller page size");alloc(&admin,8192);ok(nvm_aq_create(&aq,ctrl,admin));alloc(&data,524288);alloc(&prp,4096);
  nvm_ctrl_info ci={};nvm_ns_info ni={};ok(nvm_admin_ctrl_info(aq,&ci,(void*)data->vaddr,data->ioaddrs[0]));ok(nvm_admin_ns_info(aq,&ni,ns,(void*)data->vaddr,data->ioaddrs[0]));
  std::string actual(ci.serial_no,20);while(!actual.empty()&&(actual.back()==' '||actual.back()==0))actual.pop_back();
  require(actual==serial&&ni.ns_id==1&&ni.lba_data_size==512&&ni.metadata_size==0&&ni.capacity==cap/512,"actual device identity/geometry mismatch");capacity=cap;
  chunk=std::min(size_t(524288),ci.max_data_size?ci.max_data_size:size_t(524288));chunk=chunk/4096*4096;require(chunk>=4096,"MDTS too small");
  for(int i=0;i<2;i++){begin[i]=ranges[2*i];end[i]=ranges[2*i+1];require(begin[i]>=4096&&begin[i]%4096==0&&end[i]%4096==0&&end[i]>begin[i]&&end[i]<=cap-4096,"device range bounds");}
  require(end[0]+4096<=begin[1]-4096,"overlapping intervals");
  uint16_t nc=1,nsq=1;ok(nvm_admin_request_num_queues(aq,&nc,&nsq));require(nc&&nsq,"no I/O queues");alloc(&queues,8192);
  ok(nvm_admin_cq_create(aq,&cq,1,queues,0,16));have_cq=true;ok(nvm_admin_sq_create(aq,&sq,&cq,1,queues,1,16));have_sq=true;live=true;
 }
 void command(uint8_t op,uint64_t offset,size_t size){
  require(live&&!poison,"transport not live");nvm_cmd_t*cmd=nvm_sq_enqueue(&sq);require(cmd,"SQ unexpectedly full");memset(cmd,0,sizeof(*cmd));
  uint16_t cid=NVM_DEFAULT_CID(&sq);nvm_cmd_header(cmd,cid,op,1);
  if(op!=NVM_IO_FLUSH){nvm_prp_list_t list=NVM_PRP_LIST(prp,0);require(nvm_cmd_data(cmd,1,&list,size/4096,data->ioaddrs)==size/4096,"PRP coverage");nvm_cmd_rw_blks(cmd,offset/512,size/512);}
  nvm_sq_submit(&sq);auto start=std::chrono::steady_clock::now();nvm_cpl_t*c=nullptr;
  while(!(c=nvm_cq_dequeue(&cq))){if(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()>5){poison=true;throw std::runtime_error("NVMe completion deadline 5s");}std::this_thread::sleep_for(std::chrono::microseconds(20));}
  int status=NVM_ERR_STATUS(c);bool valid=*NVM_CPL_SQID(c)==1&&*NVM_CPL_CID(c)==cid;nvm_sq_update(&sq);nvm_cq_update(&cq);
  if(!valid||status){poison=true;throw std::runtime_error("NVMe completion status/CID mismatch");}
 }
 void transfer(bool write,uint64_t off,void*buffer,size_t bytes){
  require(bytes>0&&bytes<=8*1024*1024&&off%4096==0&&bytes%4096==0&&off<=capacity&&bytes<=capacity-off,"unaligned or oversized transfer");bool allowed=false;
  for(int i=0;i<2;i++)if(off>=begin[i]-(write?0:4096)&&off+bytes<=end[i]+(write?0:4096))allowed=true;
  require(allowed,"transfer outside admitted extents");
  for(size_t done=0;done<bytes;){size_t n=std::min(bytes-done,chunk);if(write)memcpy((void*)data->vaddr,(char*)buffer+done,n);command(write?NVM_IO_WRITE:NVM_IO_READ,off+done,n);if(!write)memcpy((char*)buffer+done,(void*)data->vaddr,n);done+=n;}
 }
 void close_device(){
  if(poison){require(ctrl&&admin,"poisoned session incomplete");memset((void*)admin->vaddr,0,8192);ok(nvm_raw_ctrl_reset(ctrl,admin->ioaddrs[0],admin->ioaddrs[1]));have_sq=have_cq=false;poison=false;}
  if(have_sq){ok(nvm_admin_sq_delete(aq,&sq,&cq));have_sq=false;}
  if(have_cq){ok(nvm_admin_cq_delete(aq,&cq));have_cq=false;}
  live=false;if(aq){nvm_aq_destroy(aq);aq=nullptr;}
  for(auto p:{&prp,&data,&queues,&admin})if(*p){nvm_dma_unmap(*p);*p=nullptr;}
  if(ctrl){nvm_ctrl_free(ctrl);ctrl=nullptr;}
 }
};
extern "C" const char* ukl_transport_error(){return last_error.c_str();}
extern "C" void* ukl_transport_open(const char*serial,uint64_t capacity,const uint64_t*ranges){
 Session*s=new Session;try{s->open_device(serial,capacity,ranges);return s;}catch(const std::exception&e){last_error=e.what();try{s->close_device();delete s;}catch(...){last_error+="; cleanup failed, process quarantine required";}return nullptr;}}
extern "C" int ukl_transport_io(void*p,int write,uint64_t off,void*buf,size_t bytes){try{((Session*)p)->transfer(write!=0,off,buf,bytes);return 0;}catch(const std::exception&e){last_error=e.what();return -1;}}
extern "C" int ukl_transport_flush(void*p){try{((Session*)p)->command(NVM_IO_FLUSH,0,0);return 0;}catch(const std::exception&e){last_error=e.what();return -1;}}
extern "C" int ukl_transport_close(void*p){try{((Session*)p)->close_device();delete (Session*)p;return 0;}catch(const std::exception&e){last_error=e.what();return -1;}}
