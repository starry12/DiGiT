
import argparse,fcntl,json,os,time
from functools import partial
from pathlib import Path
import numpy as np
from .common import *

def execute(a):
    require(os.geteuid()==0 and __debug__,'Admitted native root worker required')
    launch=read(a.launch);require(launch['controller_pid']==os.getppid() and launch['source_sha256']==verify(),'Wrong controller/source')
    require(sha(launch['reference_input_snapshot'])==launch['reference_input_snapshot_sha256'],'Changed source snapshot')
    reference=read(launch['reference_input_snapshot']);check_inputs(reference)
    setup()
    import torch
    from .admission import estimate
    from ae.pa_sage.observations import CheckpointObservations
    from ae.common import check_device
    from candidates.unified_training_v2.config import load_config
    from candidates.unified_training_v2.order import arm_batches
    from candidates.unified_training_v2.loop import run_arm
    from .adapter import IGAdapter
    from .resources import create
    a.output.mkdir(parents=True,exist_ok=False);observer=CheckpointObservations(a.output/'resources.json')
    budget=estimate();parent=os.getppid()
    def check():
        require(os.getppid()==parent and parent==launch['controller_pid'],'Controller disappeared')
        check_inputs(reference)
        require(host()>=cfg()['runtime_host_floor_bytes'],'Runtime host floor')
        if torch.cuda.is_initialized():
            free,total=torch.cuda.mem_get_info()
            require(free>=cfg()['runtime_gpu_floor_bytes'] and total-free<=budget['required_bytes'],'Runtime GPU budget')
    def mark(stage,**kw):
        check();observer.mark(stage);progress(a.output,stage,**kw)
    check_device();bfs=read(launch['bfs_receipt']);canonical=np.load(bfs['output'],mmap_mode='r')
    c=load_config('IG',a.scope);arm='digit' if a.arm=='digit_full' else 'gids'
    factory=partial(create,output=a.output,canonical=canonical,bfs_receipt=bfs,check=check,mark=mark,
                    profile_directory=Path(launch['profile_directory']),smoke=a.smoke)
    adapter=IGAdapter(factory)
    mark('start')
    if a.smoke:
        try:
            initial=adapter.initialize(c,arm);iterator=arm_batches(canonical,c,0,arm)
            row=adapter.run_window([next(iterator),next(iterator)],0,'smoke');final=adapter.finish()
            require(row['finite'] and final['finite'] and len(adapter.r.audits)==2,'Missing native smoke evidence')
            report=dict(passed=True,complete=True,smoke=True,initial=initial,final=final,window=row,audits=adapter.r.audits)
        finally:adapter.close()
    else:
        def emit(row):append_sync(a.output/'events.jsonl',row);mark(row['stage'],epoch=row.get('epoch'),window=row.get('window'))
        report=run_arm(c,arm,a.repetition,canonical,adapter,emit)
    require(verify()==launch['source_sha256'],'Source changed during IG worker')
    check();mark('training_complete')
    report.update(resource_observations=observer.result(),input_binding_sha256=sha(a.launch),model_name='sage')
    write(a.output/'report.json',report);write(a.output/'worker_ready.json',dict(passed=True,report_sha256=sha(a.output/'report.json')))
    deadline=time.monotonic()+120
    while not (a.output/'release_worker.json').exists():
        require(time.monotonic()<deadline,'Monitor shutdown handshake timeout');time.sleep(.25)

def main():
    p=argparse.ArgumentParser();p.add_argument('--scope',choices=['short1','short5','full20'],required=True)
    p.add_argument('--arm',choices=['gids','digit_full'],required=True);p.add_argument('--repetition',type=int,default=1)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--launch',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    with open('/tmp/digit-pa-sage-libnvm0.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);execute(a)
if __name__=='__main__':main()
