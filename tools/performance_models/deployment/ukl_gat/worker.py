"""Fresh restricted worker: immutable author computation, AE output transport."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import model_context
import os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
sys.path.insert(0,str(ROOT))
if __name__=='__main__':
 require(os.geteuid()==0 and os.environ.get('UKL_V27_BOUNDED_WORKER')=='1','Bounded root worker required')
 out=Path(os.environ['UKL_V27_OUTPUT']);request_path(out.parent.parent)
 require(out.parent.name=='experiment' and out.resolve()==out,'Invalid worker output')
 from candidates.ukl_common_gpu_five_v27.phase import await_phase
 try:
  await_phase(out,'init')
  model_context.activate()
  identity();configure(out.parent)
  from candidates.ukl_common_gpu_five_v27.worker import main
  main()
 except BaseException as e:
  from candidates.ukl_common_gpu_five_v27.worker import persist_failure
  persist_failure(e);raise
