#include "core.h"
extern "C" int sample_cpu(View v,const int32_t*seeds,int count,int fanout,int grouped,uint64_t seed,Out o){
 if(count<=0||fanout<=0||fanout>MAX_F)return -1;
 for(int i=0;i<count;i++)sample_one(v,seeds,i,fanout,grouped,seed,o);
 return 0;
}
