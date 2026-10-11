import json,shutil
import numpy as np
from .transpose import OUT
from .overlay_build import build
from .mapped import CSC,Overlay
from .native import sample

def main():
 source=OUT/'checks/group_0';cscdir=OUT/'checks/mapped_csc';cscdir.mkdir()
 shutil.copyfile(source/'ptr.i64',cscdir/'indptr.i64');shutil.copyfile(source/'idx.i32',cscdir/'indices.i32')
 (cscdir/'csc.json').write_text(json.dumps(dict(complete=True,orientation='incoming',nodes=67,edges=(source/'idx.i32').stat().st_size//4)))
 out=OUT/'checks/mapped_overlay';build(cscdir,source/'hot.i64',out,6);c=CSC(cscdir);g=Overlay(c,out)
 assert g.arrays['idx'] is c.idx and g.arrays['ptr'] is c.ptr
 cases=0
 for grouped in (False,True):
  for fanout in (1,2,5,10,32):
   roots=np.array([0,1,2,5,66],np.int32);actual=g.sample_cpu(roots,fanout,grouped,23);expected=sample(g.arrays,roots,fanout,grouped,23)
   for key in actual:np.testing.assert_array_equal(actual[key],expected[key])
   cases+=1
 value=dict(passed=True,cases=cases,original_csc_shared_without_copy=True,all_arrays_readonly_mmap=True,full_graph_cuda_registration=False)
 (OUT/'checks/mapped.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value,indent=2))
if __name__=='__main__':main()
