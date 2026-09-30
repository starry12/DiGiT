"""Read-only pool validation, ABBA calibration, then the gated 15-point grid."""
import argparse
import csv
import fcntl
import os
import signal
import subprocess
import sys
import time
from .common import ROOT,HERE,Path,read,write,sha,require,verify
from .pool import OUT,check_receipt
from ae.common import host
from candidates.pa_sage_layout_native_v2.common import GPU_UUID
PY='/home/embed/miniconda3/envs/gids/bin/python'
PLAN=OUT/'layout_plan.json'
INDEX=OUT/'protocols/index.json'
TMP=Path('/mnt/n0/digit/pa_sage_layout_shared_resume_20260925_v3/.tmp')

def available(required_host=192*2**30,required_gpu=0):
    while True:
        values=subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=uuid,memory.free,memory.total',
                '--format=csv,noheader,nounits'],text=True,timeout=30).strip().split(',')
        uuid,free,total=values[0].strip(),int(values[1])*2**20,int(values[2])*2**20
        require(uuid==GPU_UUID and required_gpu<=total,'Wrong GPU or point exceeds total GPU capacity')
        ram=host()
        if ram>=required_host and free>=max(total-2**30,required_gpu):return
        write(OUT/'resource_wait.json',dict(host_available=ram,host_required=required_host,
             gpu_free=free,gpu_required=max(total-2**30,required_gpu),updated_unix=time.time()))
        time.sleep(30)

def overlay(key):
    from .overlay import build_overlay
    plan=read(PLAN);p=read(OUT/'protocols'/(key+'.json'))
    build_overlay(p['base_layout'],plan['inputs']['indptr']['path'],ROOT/p['data'],
            p['graph']['edges']-plan['dataset']['directed_edges'],Path(p['overlay']))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--overlay');a=parser.parse_args()
    require(os.geteuid()==0 and os.environ.get('CUDA_VISIBLE_DEVICES')=='2','Root / physical GPU 2 required')
    if a.overlay:verify();overlay(a.overlay);return
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    require(not (OUT/'status.json').exists(),'Preserve existing run; continuation needs evidence review')
    execution=verify();state=dict(schema='digit-shared-pool-grid-v1',passed=False,complete=False,stage='starting',
        candidate_sha256=execution,pid=os.getpid(),started_unix=time.time(),steps=[],completed=[],raw_ssd_writes=False)
    def save(stage,**kw):
        state.update(stage=stage,updated_unix=time.time(),**kw);write(OUT/'status.json',state);print(stage,kw,flush=True)
    def child(name,command):
        verify();save(name);folder=OUT/'logs';folder.mkdir(exist_ok=True)
        row=dict(name=name,command=command,started_unix=time.time());state['steps'].append(row);write(OUT/'status.json',state)
        with (folder/(name+'.log')).open('x') as log:
            try:subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True);row['returncode']=0
            except subprocess.CalledProcessError as e:row['returncode']=e.returncode;raise
            finally:row['finished_unix']=time.time();write(OUT/'status.json',state)
    def point(name,protocol,output):
        budget_path=OUT/'preflight'/(name+'.json')
        child('budget_'+name,[PY,'-B','-u','-m','candidates.pa_sage_layout_shared_resume_v3.cli',
                             '--protocol',str(protocol),'--budget-output',str(budget_path)])
        budget=read(budget_path);save('waiting_'+name);available(budget['host_required_bytes'],budget['required_bytes'])
        child(name,[PY,'-B','-u','-m','candidates.pa_sage_layout_shared_resume_v3.cli',
                    '--protocol',str(protocol),'--output',str(output),'--execute'])
    try:
        TMP.mkdir(parents=True,exist_ok=True);os.environ['TMPDIR']=str(TMP)
        save('reusing_verified_pools')
        from .resume import reuse
        reuse()
        shared=OUT/'protocols/g2_r20.json';real=OUT/'protocols/real_g2_r20.json'
        for name,protocol,output in [('real_first',real,OUT/'calibration/real_first'),
              ('shared_first',shared,OUT/'native/g2_r20'),('shared_repeat',shared,OUT/'calibration/shared_repeat'),
              ('real_second',real,OUT/'calibration/real_second')]:point(name,protocol,output)
        from .compare import collect
        save('calibration_review');collect(OUT);state['completed'].append('g2_r20');save('calibration_passed')
        from candidates.pa_sage_layout_native_v2.common import ORDER
        for key in ORDER:
            if key=='g2_r20':continue
            p=read(OUT/'protocols'/(key+'.json'));base=Path(p['base_layout'])
            available(100*2**30)
            child('layout_'+key,[PY,'-B','-u','-m','candidates.pa_sage_layout_shared_resume_v3.build','--plan',str(PLAN),
                                  '--point',key,'--execute-filesystem-build'])
            child('overlay_'+key,[PY,'-B','-u','-m','candidates.pa_sage_layout_shared_resume_v3.queue','--overlay',key])
            point('native_'+key,OUT/'protocols'/(key+'.json'),OUT/'native'/key)
            state['completed'].append(key);save('point_complete',point=key)
        from .aggregate import collect as aggregate
        result=aggregate(read(INDEX),OUT/'native');write(OUT/'summary.json',result)
        fields=['point','group_size','replica_percent','achieved_replica_fraction','mean_training_epoch_seconds',
            'order_excluded_mean_seconds','speedup_vs_fresh_g2_r20','order_excluded_speedup_vs_g2_r20',
            'layout_address_span_bytes','groups','padding_rows','group_sample_fraction']
        with (OUT/'summary.csv').open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for r in result['points']:
                row={k:r[k] for k in fields if k not in ('point','group_size','replica_percent')}
                row.update(point=r['point']['id'],group_size=r['point']['group_size'],replica_percent=r['point']['replica_percent']);writer.writerow(row)
        verify();save('complete',passed=True,complete=True,finished_unix=time.time())
    except BaseException as exc:save('failed',error=type(exc).__name__+': '+str(exc));raise
if __name__=='__main__':main()
