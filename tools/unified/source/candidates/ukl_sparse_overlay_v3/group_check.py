"""Production sparse builder vs frozen explicit UKS grouping, on small CSCs."""
import json,subprocess
import numpy as np
from .transpose import OUT
from .layout import validate,materialize
from .tests import compare

def main():
 cases=[]
 for seed in (0,7,31):
  f=OUT/'checks'/('group_'+str(seed));f.mkdir();old=f/'old';old.mkdir();new=f/'overlay';new.mkdir();rng=np.random.RandomState(seed);n=67;ptr=[0];idx=[]
  for v in range(n):
   row=rng.randint(0,n,size=1003 if v==0 else 23 if v!=1 else 0).tolist();idx.extend(row);ptr.append(len(idx))
  p=np.array(ptr,np.int64);x=np.array(idx,np.int32);hot=np.arange(n//10,dtype=np.int64)
  for name,arr in [('ptr',p),('idx',x.astype(np.int64)),('hot',hot)]:np.save(str(f/(name+'.npy')),arr)
  p.tofile(f/'ptr.i64');x.tofile(f/'idx.i32');hot.tofile(f/'hot.i64')
  with (f/'build.log').open('w') as log:
   subprocess.run([str(OUT/'build/legacy_group_reference'),'g2',str(f/'ptr.npy'),str(f/'idx.npy'),str(f/'hot.npy'),str(n),str(old)],check=True,stdout=log)
   subprocess.run([str(OUT/'build/group_overlay'),str(f/'ptr.i64'),str(f/'idx.i32'),str(f/'hot.i64'),str(new),str((n//5)//2)],check=True,stdout=log)
  a=dict(ptr=p,idx=x,gptr=np.fromfile(new/'group_ptr.i64',np.int64),gidx=np.fromfile(new/'group_ids.i32',np.int32),covered=np.fromfile(new/'covered.i64',np.int64),members=np.fromfile(new/'members.i32',np.int32).reshape(-1,2),bases=np.fromfile(new/'bases.i64',np.int64),primary=np.fromfile(new/'primary.i64',np.int64));a=validate(a);mat=materialize(a)
  np.testing.assert_array_equal(a['members'].ravel(),np.fromfile(old/'group_members.bin',np.int64))
  np.testing.assert_array_equal(mat['gidx'],np.fromfile(old/'reordered_indices.bin',np.int64))
  np.testing.assert_array_equal(mat['gptr'],np.fromfile(old/'reordered_indptr.bin',np.int64)[:n+1])
  meta=json.loads((new/'overlay.json').read_text());pg=meta['primary_groups'];rg=meta['replica_groups'];assert meta['storage_rows']==n+2*rg
  # Every original logical node has exactly one primary row; replicas preserve identity.
  np.testing.assert_array_equal(np.sort(a['primary']),np.arange(n))
  for g,(left,right) in enumerate(a['members']):
   if g<pg:assert (a['primary'][left],a['primary'][right])==(a['bases'][g],a['bases'][g]+1)
   else:assert a['bases'][g]==n+2*(g-pg)
  for fanout in (1,2,5,10,32):compare(a,np.array([0,1,2,3,66],np.int32),fanout,True,23)
  cases.append(dict(seed=seed,groups=len(a['members']),primary_groups=pg,replica_groups=rg))
 receipt=dict(passed=True,cases=cases,exact_same_group_selection_and_virtual_adjacency_as_legacy=True,feature_layout='packed 512 B rows, primary groups adjacent, replicas appended',full_graph=False)
 (OUT/'checks/group.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
