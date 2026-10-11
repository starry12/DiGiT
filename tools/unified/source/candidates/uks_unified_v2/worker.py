import argparse,os,json
from pathlib import Path
from candidates.uks_pa_fast_v1.identity import install
install()
from .common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['short1','short5','full20'],required=True)
    p.add_argument('--arm',choices=['gids','digit'],required=True);p.add_argument('--repetition',type=int,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--lock-fds',required=True);p.add_argument('--fixtures',action='store_true');a=p.parse_args()
    from candidates.uks_pa_fast_v1.common import validate_inherited_locks,inputs
    require(os.geteuid()==0 and __debug__ and os.environ.get('CUDA_VISIBLE_DEVICES')=='2','Root GPU2 worker required')
    validate_inherited_locks(a.lock_fds);source=verify()
    require(a.output.resolve().parent.parent==OUT/'runs' and not a.output.exists(),'Unexpected output')
    a.output.mkdir()
    if a.fixtures:
        from candidates.uks_pa_fast_v1.checks import gpu_check
        write(a.output/'report.json',dict(gpu_check(),source_sha256=source));return
    from candidates.uks_pa_fast_v1.engine import admission
    from candidates.uks_group_first_cpu2_v1.affinity import install as affinity
    from candidates.uks_group_first_cpu2_v1.common import check_prepared,DEFAULT_OUTPUT,prepared_hashes
    from candidates.uks_group_first_v1.common import OUT as FIRST_OUT,verify as first_verify
    from candidates.cache_epoch_v1.core import validate_order
    import numpy as np
    admission();binding=inputs();prepared=check_prepared(DEFAULT_OUTPUT);before=prepared_hashes()
    accepted=read(FIRST_OUT/'gpu_checks.json');require(accepted['passed'] and accepted['source_sha256']==first_verify(),'Group sampler not accepted')
    canonical=np.load(binding['canonical']['path']);training=np.load(binding['training']['path'])
    validate_order(canonical,training,binding['nodes']);del training
    from ae.common import host
    from candidates.unified_training_v2.order import order_memory_bytes
    require(host()>=256*2**30+order_memory_bytes(len(canonical),a.arm),'Insufficient host memory for root order')
    from .resources import create
    from .adapter import UKSAdapter
    from candidates.unified_training_v2.loop import run_arm
    with affinity():
        r=create(a.arm,a.output,canonical,source,binding,a.lock_fds)
        def emit(v):write(a.output/'progress.json',dict(v,updated_unix=__import__('time').time()))
        report=run_arm(config(a.mode),a.arm,a.repetition,canonical,UKSAdapter(r),emit)
    require(report['initial']['initial_model_sha256']==binding['initial_model_sha256'],'UKS initial model drift')
    require(check_prepared(DEFAULT_OUTPUT)==prepared and prepared_hashes()==before,'Prepared graph changed')
    require(verify()==source,'Source drift');inputs()
    write(a.output/'report.json',report)
if __name__=='__main__':main()
