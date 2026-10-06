#pragma once
#include <stdint.h>
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif
constexpr int MAX_F=32;
struct View {
 int32_t nodes; int64_t edges,origin,units,unit_origin,groups;
 const int64_t *ptr,*gptr,*bases,*primary,*covered;
 const int32_t *idx,*gidx,*members;
};
struct Out {int32_t *src,*dst,*counts,*errors; int64_t *eid,*rows;};
HD uint64_t next64(uint64_t &s){s+=0x9e3779b97f4a7c15ULL;uint64_t z=s;z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;return z^(z>>31);}
HD uint64_t bounded(uint64_t &s,uint64_t n){uint64_t t=(-n)%n,x;do{x=next64(s);}while(x<t);return x%n;}
HD bool contains(const int64_t*a,int n,int64_t x){for(int i=0;i<n;i++)if(a[i]==x)return true;return false;}
HD int64_t lookup(const int64_t*k,const int64_t*v,int n,int64_t x){for(int i=0;i<n;i++)if(k[i]==x)return v[i];return x;}
HD void sample_one(View v,const int32_t*seeds,int batch,int f,int grouped,uint64_t seed,Out o,bool resolve_eids=true){
 int32_t owner=seeds[batch];int count=0;int offset=batch*f;o.counts[batch]=0;o.errors[batch]=0;
 if(owner<0||owner>=v.nodes){o.errors[batch]=1;return;}
 int64_t begin=v.ptr[owner],end=v.ptr[owner+1];
 if(begin<v.origin||end<begin||end-v.origin>v.edges){o.errors[batch]=2;return;}
 uint64_t rng=seed^(uint64_t(owner)*0xd6e8feb86659fd93ULL)^uint64_t(batch);
 int64_t picked[MAX_F],keys[MAX_F],vals[MAX_F];int selected=0,nkeys=0;
 if(!grouped){
  int64_t degree=end-begin;int amount=degree<f?int(degree):f;
  for(int i=0;i<amount;i++){
   int64_t remaining=degree-i,j=bounded(rng,remaining),slot=lookup(keys,vals,nkeys,j),replacement=lookup(keys,vals,nkeys,remaining-1);
   int at=0;while(at<nkeys&&keys[at]!=j)at++;if(at==nkeys)nkeys++;keys[at]=j;vals[at]=replacement;
   int32_t src=v.idx[begin-v.origin+slot];if(src<0||src>=v.nodes){o.errors[batch]=3;return;}
   o.src[offset+count]=src;o.dst[offset+count]=owner;o.eid[offset+count]=begin+slot;o.rows[offset+count]=v.primary?v.primary[src]:src;count++;
  }
 }else{
  // Sparse overlay: original CSC plus sorted covered EIDs (two per group).
  int64_t a=v.gptr[owner],b=v.gptr[owner+1];
  if(a<0||b<a||b>v.units){o.errors[batch]=4;return;}
  int64_t raw_initial=end-begin-2*(b-a),raw=raw_initial,group=b-a;
  if(raw<0){o.errors[batch]=4;return;}
  int remaining=f;
  while(remaining){
   uint64_t total=uint64_t(raw)+(remaining>=2?2*uint64_t(group):0);if(!total)break;
   uint64_t rank=bounded(rng,total);int64_t chosen;
   if(rank<uint64_t(raw)){
    // kth remaining raw ordinal: remove only <= MAX_F previously selected units.
    chosen=int64_t(rank);
    for(;;){int64_t next=int64_t(rank);for(int j=0;j<selected;j++)if(picked[j]<raw_initial&&picked[j]<=chosen)next++;if(next==chosen)break;chosen=next;}
    // Count covered positions whose uncovered rank is <= chosen.
    // covered[k] - begin - k is monotone even for consecutive covers.
    int64_t left=2*a,right=2*b;
    while(left<right){int64_t mid=left+(right-left)/2;
     if(v.covered[mid]-begin-(mid-2*a)<=chosen)left=mid+1;else right=mid;}
    int64_t lo=begin+chosen+(left-2*a);
    int32_t u=v.idx[lo-v.origin];if(u<0||u>=v.nodes){o.errors[batch]=3;return;}
    o.src[offset+count]=u;o.rows[offset+count]=v.primary[u];count++;raw--;remaining--;
   }else{
    int64_t target=(int64_t(rank)-raw)/2,slot=target;
    for(;;){int64_t next=target;for(int j=0;j<selected;j++)if(picked[j]>=raw_initial&&picked[j]-raw_initial<=slot)next++;if(next==slot)break;slot=next;}
    chosen=raw_initial+slot;int64_t g=v.gidx[a+slot];
    if(g<0||g>=v.groups){o.errors[batch]=5;return;}
    for(int j=0;j<2;j++){int32_t n=v.members[2*g+j];if(n<0||n>=v.nodes){o.errors[batch]=7;return;}o.src[offset+count]=n;o.rows[offset+count]=v.bases[g]+j;count++;}
    group--;remaining-=2;
   }
   picked[selected++]=chosen;
  }
  // Resolve selected occurrences in one pass, without a full EID array.
  for(int i=0;i<count;i++){o.eid[offset+i]=-1;o.dst[offset+i]=owner;}
  if(resolve_eids){
  int missing=count;
  for(int64_t j=begin;j<end&&missing;j++)for(int i=0;i<count;i++)if(o.eid[offset+i]<0&&o.src[offset+i]==v.idx[j-v.origin]){o.eid[offset+i]=j;missing--;break;}
  if(missing){o.errors[batch]=8;return;}
  }
 }
 o.counts[batch]=count;
}
