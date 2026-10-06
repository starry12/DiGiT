#include "core.h"

// Same chunk-level rejection rule as the old NumPy validator. Each owner/chunk
// is read once for all non-primary sampled rows. No global map or allocation.
extern "C" int validate_group_rows(View v,const int32_t*seeds,int count,int f,
 const int32_t*src,const int64_t*rows,const int32_t*counts){
 if(count<=0||f<=0||f>MAX_F||int64_t(count)*f>65536)return 1;
 for(int i=0;i<count;i++){
  int owner=seeds[i],n=counts[i];if(owner<0||owner>=v.nodes||n<0||n>f)return 2;
  bool pending[MAX_F];int remaining=0;
  for(int j=0;j<n;j++){
   int u=src[i*f+j];if(u<0||u>=v.nodes)return 2;
   pending[j]=rows[i*f+j]!=v.primary[u];remaining+=pending[j];
  }
  if(!remaining)continue;
  int64_t a=v.gptr[owner],b=v.gptr[owner+1];if(a<0||b<a||b>v.units)return 3;
  for(int64_t start=a;start<b&&remaining;start+=65536){
   int64_t end=start+65536;if(end>b)end=b;
   // Validate the whole chunk before any early success, exactly as np.any did.
   for(int64_t k=start;k<end;k++){int64_t g=v.gidx[k];if(g<0||g>=v.groups)return 4;}
   for(int64_t k=start;k<end&&remaining;k++){
    int64_t g=v.gidx[k],base=v.bases[g];
    for(int j=0;j<n;j++)if(pending[j]){
     int64_t row=rows[i*f+j];
     if((row==base||row-1==base)&&row>=base){
      int slot=int(row-base);
      if(v.members[2*g+slot]==src[i*f+j]){pending[j]=false;remaining--;}
     }
    }
   }
  }
  if(remaining)return 5;
 }
 return 0;
}
