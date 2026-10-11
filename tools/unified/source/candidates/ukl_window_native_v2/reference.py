"""Independent Python reference for the prototype's seeded sampling decisions."""
import numpy as np
MASK=(1<<64)-1
class RNG:
 def __init__(self,s):self.s=s&MASK
 def next(self):
  self.s=(self.s+0x9e3779b97f4a7c15)&MASK;z=self.s;z=((z^(z>>30))*0xbf58476d1ce4e5b9)&MASK;z=((z^(z>>27))*0x94d049bb133111eb)&MASK
  return z^(z>>31)
 def bounded(self,n):
  threshold=((1<<64)-n)%n
  while True:
   x=self.next()
   if x>=threshold:return x%n

def sample(a,seeds,f,grouped,seed):
 result=[];n=len(a['ptr'])-1
 for batch,owner in enumerate(map(int,seeds)):
  rng=RNG(seed^((owner*0xd6e8feb86659fd93)&MASK)^batch);start,end=map(int,a['ptr'][owner:owner+2]);sources=[];rows=[];eids=[]
  if not grouped:
   swaps={};degree=end-start
   for i in range(min(f,degree)):
    left=degree-i;j=rng.bounded(left);slot=swaps.get(j,j);swaps[j]=swaps.get(left-1,left-1)
    src=int(a['idx'][start-int(a['ptr'][0])+slot]);sources.append(src);rows.append(int(a['primary'][src]));eids.append(start+slot)
  else:
   lo,hi=map(int,a['gptr'][owner:owner+2]);units=list(map(int,a['gidx'][lo-int(a['gptr'][0]):hi-int(a['gptr'][0])]))
   available=list(range(len(units)));remaining=f
   while remaining:
    eligible=[i for i in available if (1 if units[i]<n else 2)<=remaining];weights=[1 if units[i]<n else 2 for i in eligible]
    if not eligible:break
    rank=rng.bounded(sum(weights))
    for i,w in zip(eligible,weights):
     if rank<w:chosen=i;break
     rank-=w
    u=units[chosen];available.remove(chosen);cost=1 if u<n else 2;remaining-=cost
    if cost==1:sources.append(u);rows.append(int(a['primary'][u]))
    else:sources.extend(map(int,a['members'][u-n]));rows.extend([int(a['bases'][u-n]),int(a['bases'][u-n])+1])
   adjacency=list(map(int,a['idx'][start-int(a['ptr'][0]):end-int(a['ptr'][0])]))
   used=set()
   for source in sources:
    slot=next(j for j,v in enumerate(adjacency) if v==source and j not in used);used.add(slot);eids.append(start+slot)
  result.append(dict(src=sources,dst=[owner]*len(sources),eid=eids,rows=rows))
 return result
