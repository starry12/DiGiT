// Existing sampling/RNG semantics, on the caller's stream, without bulk D2H.
__global__ void checked_owners(View v,int32_t*s,int n,int grouped){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 int owner=s[i];bool ok=owner>=0&&owner<v.nodes;
 if(ok){int64_t a=v.ptr[owner],b=v.ptr[owner+1];ok=a>=v.origin&&b>=a&&b-v.origin<=v.edges;
  if(ok&&grouped){int64_t lo=v.gptr[owner],hi=v.gptr[owner+1];ok=lo>=v.unit_origin&&hi>=lo&&hi-v.unit_origin<=v.units&&2*(hi-lo)<=b-a;
   int64_t last=-1;
   if(ok)for(int64_t j=2*(lo-v.unit_origin);j<2*(hi-v.unit_origin);j++){
    int64_t x=int64_t(v.covered[j]);
#ifdef COVERED_ABSOLUTE
    x-=a;
#endif
    if(x<=last||x>=b-a){ok=false;break;}last=x;
   }
  }
 }
 if(!ok)s[i]=-1; // Existing kernels reject this owner before dereferencing it.
}
__global__ void check_outputs(View v,const int32_t*s,int n,int f,int grouped,int64_t rows,Out o,int64_t*groups){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 groups[i]=0;if(o.errors[i])return;
 int owner=s[i],count=o.counts[i];if(count<0||count>f){o.errors[i]=20;return;}
 if(grouped){int64_t g=v.gptr[owner+1]-v.gptr[owner];groups[i]=2*(g<f/2?g:f/2);}
 for(int j=0;j<count;j++){
  int at=i*f+j;int src=o.src[at];int64_t eid=o.eid[at],row=o.rows[at];
  if(src<0||src>=v.nodes||o.dst[at]!=owner||eid<v.ptr[owner]||eid>=v.ptr[owner+1]||v.idx[eid-v.origin]!=src||row<0||row>=rows){o.errors[i]=21;return;}
  int64_t primary=v.primary?v.primary[src]:src;
  if(row!=primary){bool found=false;
   if(grouped)for(int64_t k=v.gptr[owner];k<v.gptr[owner+1];k++){
    int g=v.gidx[k-v.unit_origin];if(g<0||g>=v.groups){o.errors[i]=22;return;}
    if((v.bases[g]==row&&v.members[2*g]==src)||(v.bases[g]+1==row&&v.members[2*g+1]==src)){found=true;break;}
   }
   if(!found){o.errors[i]=23;return;}
  }
 }
}
extern "C" int sample_stream(View v,int32_t*s,int n,int f,int grouped,uint64_t seed,int64_t rows,Out o,int64_t*groups,uint64_t stream){
 if(n<=0||n>36864||f<=0||f>32||int64_t(n)*f>1048576)return -1;
 auto st=reinterpret_cast<cudaStream_t>(stream);
 checked_owners<<<(n+127)/128,128,0,st>>>(v,s,n,grouped);
 if(grouped)grouped_kernel<<<(n+3)/4,128,0,st>>>(v,s,n,f,seed,o);
 else baseline_kernel<<<(n+63)/64,64,0,st>>>(v,s,n,f,seed,o);
 check_outputs<<<(n+127)/128,128,0,st>>>(v,s,n,f,grouped,rows,o,groups);
 return int(cudaGetLastError());
}
__global__ void addresses(View v,const int64_t*ids,const int64_t*first,const int64_t*edge_rows,int64_t edges,int n,int64_t extent,int grouped,int64_t*out,int64_t*heads,int*errors){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 int64_t id=ids[i];heads[i]=-1;
 if(id<0||id>=v.nodes){atomicAdd(errors,1);out[i]=-1;return;}
 int64_t row=grouped?v.primary[id]:id;
 if(grouped&&first[i]<edges)row=edge_rows[first[i]];
 out[i]=row;
 if(row<0||row>=extent){atomicAdd(errors,1);return;}
 if(!grouped)return;
 int64_t primary_groups=v.groups-(extent-v.nodes)/2,head=-1,g=-1;
 if(row<2*primary_groups){head=(row/2)*2;g=head/2;}
 else if(row>=v.nodes){head=v.nodes+((row-v.nodes)/2)*2;g=primary_groups+(head-v.nodes)/2;}
 if(g>=0){
  if(g>=v.groups||v.bases[g]!=head||v.members[2*g+(row-head)]!=id){atomicAdd(errors,1);return;}
  heads[i]=head;
 }else if(row!=v.primary[id])atomicAdd(errors,1);
}
extern "C" int annotate_stream(View v,const int64_t*ids,const int64_t*first,const int64_t*rows,int64_t edges,int n,int64_t extent,int grouped,int64_t*out,int64_t*heads,int*errors,uint64_t stream){
 if(n<=0||n>405504||edges<0||edges>1048576||extent<v.nodes)return -1;
 addresses<<<(n+255)/256,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(v,ids,first,rows,edges,n,extent,grouped,out,heads,errors);
 return int(cudaGetLastError());
}
