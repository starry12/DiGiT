"""Small CUDA/UVA correctness probe; shares controller lock, never opens SSD."""
import fcntl,json,os,sys,time
from pathlib import Path
import numpy as np
from .tests import fixture,compare
from .native import sample
sys.path.insert(0,'/srv/digit-ae/admin/gpu_selection_v1')
import gpu_selection
OUT=Path(__file__).resolve().parents[2]/'results/ukl_sparse_overlay_20260930_v3/checks'
def main():
 receipt=OUT/'gpu.json'
 if receipt.exists():raise RuntimeError('Preserve prior GPU evidence')
 with open('/tmp/digit-pa-bidir-controller.lock','rb') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  locks=OUT/'gpu_locks';locks.mkdir(exist_ok=True)
  with gpu_selection.reserve(lock_root=locks) as assignment:
   gpu_selection.install_assignment(assignment);gpu_selection.idle();started=time.monotonic()
   roots=np.array([0,1,2,5,16,66],np.int32);cases=0
   for high in (False,True):
    g=fixture(high)
    for grouped in (False,True):
     for f in (1,2,5,10,32):
      for seed in (0,23):
       actual=compare(g,roots,f,grouped,seed,gpu=True);cpu=sample(g,roots,f,grouped,seed)
       for key in actual:np.testing.assert_array_equal(actual[key],cpu[key])
       cases+=1
   value=dict(passed=True,cases=cases,gpu=assignment['selected'],selection_evidence=assignment,
    graph_location='mapped pinned host arena; fixture copy only',full_graph_gpu_bytes=0,
    node_dtype='int32',offset_eid_storage_row_dtype='int64',above_2pow32_eid_and_rows_tested=True,
    grouped_and_uniform=True,python_cpu_cuda_exact=True,ssd_access=False,full_ukl=False,
    seconds=time.monotonic()-started,performance_claim=False)
   receipt.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value,indent=2))
if __name__=='__main__':main()
