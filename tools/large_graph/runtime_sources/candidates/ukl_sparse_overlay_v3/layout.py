"""Canonical raw-first/group-last overlay. No second adjacency or edge-sized marker.
Validation/materialization helpers are bounded fixtures; production loader is separate.
"""
import numpy as np

def validate(a):
 a=dict(a);p=a['ptr'];x=a['idx'];n=len(p)-1
 if not 0<n<=4096 or len(x)>131072:raise ValueError('Fixture validation bound')
 if p.dtype!=np.int64 or p.shape!=(n+1,) or p[0]<0 or p[-1]-p[0]!=len(x) or np.any(p[1:]<p[:-1]):raise ValueError('Bad CSC offsets')
 if x.dtype!=np.int32 or x.ndim!=1 or (len(x) and (x.min()<0 or x.max()>=n)):raise ValueError('Bad CSC nodes')
 if a['gptr'] is None:
  if any(a[k] is not None for k in ('gidx','members','bases','covered')):raise ValueError('Partial overlay')
  a.update(gptr=np.zeros(n+1,np.int64),gidx=np.empty(0,np.int32),members=np.empty((0,2),np.int32),bases=np.empty(0,np.int64),covered=np.empty(0,np.int64))
 if a['primary'] is None:a['primary']=np.arange(n,dtype=np.int64)
 gp=a['gptr'];gi=a['gidx'];m=a['members'];c=a['covered'];g=len(gi)
 if gp.dtype!=np.int64 or gp.shape!=(n+1,) or gp[0]!=0 or gp[-1]!=g or np.any(gp[1:]<gp[:-1]):raise ValueError('Bad overlay offsets')
 if gi.dtype!=np.int32 or gi.ndim!=1 or not np.array_equal(np.sort(gi),np.arange(g)):raise ValueError('Group IDs must be a permutation')
 if m.dtype!=np.int32 or m.shape!=(g,2) or (m.size and (m.min()<0 or m.max()>=n or np.any(m[:,0]==m[:,1]))):raise ValueError('Bad members')
 if c.dtype!=np.int64 or c.shape!=(2*g,):raise ValueError('Bad cover width')
 for k,shape in [('bases',(g,)),('primary',(n,))]:
  v=a[k]
  if v.dtype!=np.int64 or v.shape!=shape or np.any(v<0) or np.any(v>=(2**63-1)//512-1):raise ValueError('Bad storage rows')
 for owner in range(n):
  lo,hi=map(int,gp[owner:owner+2]);cv=c[2*lo:2*hi]
  if len(cv) and (cv[0]<p[owner] or cv[-1]>=p[owner+1] or np.any(cv[1:]<=cv[:-1])):raise ValueError('Overlapping/out-of-owner covers')
  if not np.array_equal(np.sort(x[cv-p[0]]),np.sort(m[gi[lo:hi]].ravel())):raise ValueError('Covered occurrences differ from members')
 return {k:np.ascontiguousarray(v) for k,v in a.items()}

def materialize(a):
 """Only independent fixture oracle; forbidden for full UKL."""
 from candidates.ukl_window_native_v2.native import graph
 a=validate(a);n=len(a['ptr'])-1;units=[];ptr=[0]
 for owner in range(n):
  lo,hi=map(int,a['gptr'][owner:owner+2]);cover=set(map(int,a['covered'][2*lo:2*hi]))
  units.extend(int(a['idx'][e-int(a['ptr'][0])]) for e in range(int(a['ptr'][owner]),int(a['ptr'][owner+1])) if e not in cover)
  units.extend(n+int(g) for g in a['gidx'][lo:hi]);ptr.append(len(units))
 return graph(a['ptr'],a['idx'],gptr=np.array(ptr,np.int64),gidx=np.array(units,np.int32),members=a['members'],bases=a['bases'],primary=a['primary'])
