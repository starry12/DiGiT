"""Serial filesystem preparation -> verified raw payload -> smoke -> full epoch."""
import argparse
import csv
import shutil
import subprocess
import sys
from .common import *
LARGE_TMP=Path('/mnt/n0/digit/pa_sage_layout_sweep_20260924_v1/.tmp')

def available(required_host=192*2**30,required_gpu=0):
    while True:
        line=subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=uuid,memory.free,memory.total',
                    '--format=csv,noheader,nounits'],text=True,timeout=30).strip().split(',')
        uuid,free,total=line[0].strip(),int(line[1])*2**20,int(line[2])*2**20
        require(uuid==GPU_UUID,'Physical GPU 2 identity changed')
        require(required_gpu<=total,'Point exceeds physical GPU capacity; do not reduce the protocol')
        ram=host()
        if free>=max(total-2**30,required_gpu) and ram>=required_host:return
        progress(OUT,'waiting_resources',host_available_bytes=ram,host_required_bytes=required_host,
                 gpu_free_bytes=free,gpu_required_bytes=max(total-2**30,required_gpu))
        time.sleep(30)

def execute(command,log,env=None):
    require(not log.exists(),'Preserve earlier stage log: '+str(log))
    with log.open('x') as stream:
        subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)

def overlay(key):
    from candidates.pa_sage_layout_sweep_v1.overlay import build_overlay
    c=next(x for x in cells() if x['point']==key);p=read(c['protocol']);plan=read(PLAN)
    build_overlay(c['layout'],plan['inputs']['indptr']['path'],ROOT/p['data'],
                  p['graph']['edges']-plan['dataset']['directed_edges'],Path(c['layout'])/'overlay')

def select_point(key):
    # common.P is captured at import time, including imports inside verify().
    # Resolve the selected point BEFORE any training-adapter import.
    cell=next(x for x in cells() if x['point']==key)
    os.environ['DIGIT_LAYOUT_PROTOCOL']=cell['protocol']

def budget(key):
    select_point(key)
    from candidates.pa_sage_layout_perf_v3.common import P
    require(P==Path(os.environ['DIGIT_LAYOUT_PROTOCOL']).resolve(),'Protocol selected after adapter import; use a fresh process')
    from candidates.pa_sage_layout_perf_v3.admission import estimate
    write(OUT/'preflight'/key/'admission_plan.json',estimate())

def pipeline():
    require(os.geteuid()==0 and os.environ.get('CUDA_VISIBLE_DEVICES')=='2','Root on physical GPU 2 required')
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    execution=verify();path=OUT/'status.json'
    state=read(path) if path.exists() else dict(schema='digit-layout-grid-execution-v1',passed=False,
          complete=False,candidate_sha256=execution,order=ORDER,completed=[],stages=[],started_unix=time.time())
    require(state['candidate_sha256']==execution,'Do not resume with changed execution code')
    require(not state['complete'],'Grid already complete')
    state.pop('error',None)
    def save(stage,**kw):
        state.update(stage=stage,pid=os.getpid(),updated_unix=time.time(),**kw);write(path,state)
        progress(OUT,stage,**kw)
    def stage(name,cell,command,env=None):
        verify();save(name,point=cell['point'])
        log=OUT/'logs'/(cell['point']+'_'+name+'.log');log.parent.mkdir(exist_ok=True)
        row=dict(point=cell['point'],stage=name,command=command,log=str(log),started_unix=time.time());state['stages'].append(row);write(path,state)
        try:execute(command,log,env);row['returncode']=0
        except subprocess.CalledProcessError as exc:row['returncode']=exc.returncode;raise
        finally:row['finished_unix']=time.time();write(path,state)
    try:
        save('preflight')
        require(shutil.disk_usage('/mnt/n0').free>2**40,'Insufficient initial /mnt/n0 free space')
        LARGE_TMP.mkdir(parents=True,exist_ok=True);os.environ['TMPDIR']=str(LARGE_TMP)
        for cell in cells():
            key=cell['point'];base=Path(cell['layout']);native=OUT/'native'/key
            if key in state['completed']:
                done=read(native/'status.json');require(done['passed'] and done['complete'],'Completed point changed');continue
            if not (base/'build_receipt.json').exists():
                available(required_host=100*2**30)
                stage('layout',cell,[PY,'-B','-u','-m','candidates.pa_sage_layout_sweep_v1.build',
                      '--plan',str(PLAN),'--point',key,'--execute-filesystem-build'])
            require(read(base/'build_receipt.json')['passed'],'Layout build failed')
            if not (base/'overlay/overlay_receipt.json').exists():
                available(required_host=100*2**30)
                stage('overlay',cell,[PY,'-B','-u','-m','candidates.pa_sage_layout_native_v2.run','--stage','overlay','--point',key])
            require(read(base/'overlay/overlay_receipt.json')['passed'],'Overlay failed')
            pre=OUT/'preflight'/key;pre.mkdir(parents=True,exist_ok=True)
            if not (pre/'admission_plan.json').exists():
                stage('budget',cell,[PY,'-B','-u','-m','candidates.pa_sage_layout_native_v2.run','--stage','budget','--point',key])
            plan=read(pre/'admission_plan.json');available(plan['host_required_bytes'],plan['required_bytes'])
            if not (base/'ssd_ready.json').exists():
                stage('ssd',cell,[PY,'-B','-u','-m','candidates.pa_sage_layout_native_v2.storage','provision','--point',key])
            if not (native/'status.json').exists():
                available(plan['host_required_bytes'],plan['required_bytes'])
                stage('native',cell,[PY,'-B','-u','-m','candidates.pa_sage_layout_perf_v3.cli',
                      '--protocol',cell['protocol'],'--output',str(native),'--execute'])
            completed=read(native/'status.json')
            require(completed['passed'] and completed['complete'],'Incomplete native attempt; preserve evidence and investigate')
            state['completed'].append(key);save('point_complete',point=key)
        from candidates.pa_sage_layout_perf_v3.aggregate import collect
        summary=collect(read(INDEX),OUT/'native')
        for row in summary['points']:
            base=Path(next(c['layout'] for c in cells() if c['point']==row['point']['id']))
            build=read(base/'build_receipt.json');m=read(base/'final/bundle/manifest.json')
            row.update(payload_bytes=build['payload_bytes'],padding_rows=build['padding_rows'],
                achieved_replica_fraction=build['achieved_replica_fraction'],groups=m['grouping']['num_groups'],
                grouped_directed_edges=m['grouping']['num_groups']*m['grouping']['group_size'],
                preparation=build['preparation'])
        write(OUT/'summary.json',summary)
        with (OUT/'summary.csv').open('x',newline='') as f:
            fields=['point','group_size','replica_percent','achieved_replica_fraction','mean_training_epoch_seconds',
                    'order_excluded_mean_seconds','speedup_vs_fresh_g2_r20','payload_bytes','groups','grouped_directed_edges','padding_rows']
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for r in summary['points']:
                row={k:r[k] for k in fields if k not in ('point','group_size','replica_percent')}
                row.update(point=r['point']['id'],group_size=r['point']['group_size'],replica_percent=r['point']['replica_percent']);writer.writerow(row)
        verify();save('complete',passed=True,complete=True,finished_unix=time.time())
    except BaseException as exc:
        save('failed',error=type(exc).__name__+': '+str(exc));raise

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('pipeline','overlay','budget'),default='pipeline');p.add_argument('--point',choices=ORDER);a=p.parse_args()
    if a.stage!='pipeline':
        require(a.point is not None,'Point required');select_point(a.point)
    if a.stage!='budget':require(os.geteuid()==0,'Large preparation/native run requires root')
    verify()
    if a.stage=='pipeline':pipeline()
    else:
        require(a.point is not None,'Point required')
        if a.stage=='overlay':overlay(a.point)
        else:budget(a.point)
if __name__=='__main__':main()
