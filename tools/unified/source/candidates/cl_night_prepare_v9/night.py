"""Serial CPU queue: useful small-memory work first, optional full graph last."""
import argparse,fcntl,json,os,time
from contextlib import ExitStack
from pathlib import Path
from . import protocol as P
from .worker_json import write

def available():
    return next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))

def plan():
    specs=P.binding();meta=P.metadata()['metadata'];n=meta['nodes'];rows=meta['storage_rows']
    graph=sum((v['length']+4095)//4096*4096 for v in specs.values())
    training={}
    for arm,count in [('gids',n),('digit',rows)]:
        extent=(count+7)//8*8
        host=graph+(n//10)*512+(n//10)*16+(0 if arm=='gids' else rows*4)+16*P.MIB+n*4+64*P.MIB+16*P.GIB
        training[arm]=dict(estimated_host_bytes=host,feature_bytes=count*512,cpu_feature_cache_bytes=(n//10)*512,
            page_bytes=512,page_state_bytes=extent*32,
            gpu_estimated_bytes=4*P.GIB+extent*4+extent*32+2*P.GIB+256*P.MIB+4*P.GIB+20*P.MIB,
            note='Prospective UKL v25 512 B page ABI; not a measured CL peak or GPU admission')
    return dict(schema='cl-night-v9',nodes=n,edges=meta['edges'],storage_rows=rows,graph_arena_bytes=graph,
        stages=[dict(stage=s,budget=P.budget(s),output_bytes=sum(P.expected_files(s).values()),depends=list(P.dependencies(s))) for s in P.STAGES],
        feature_protocol=dict(dim=128,dtype='float32',row_bytes=512,generator='UKL splitmix64 logical ID generator',feature_seed=20260922,label_seed=20260923,classes=19,replicas='bit-identical logical values',supersedes='CL protocol v1 block PCG64 synthetic generator; new explicit inputs only'),
        model=dict(type='sage',layers=3,hidden=128,fanouts=[10,5,5],batch=1024),
        window=dict(warmup_batches=20,measured_batches=300,repeats=5,summary='maximum_same_round_speedup'),
        training_budget=training,training_budget_formally_admitted=False,
        gpu_enabled=False,raw_ssd_access=False,raw_ssd_writes=False,
        future=['CL native GPU/SSD training adapter and bounded smoke','SSD region allocation/write/readback','four-batch GIDS/DiGiT acceptance','five paired performance rounds'])

def decision(stage,completed,free_bytes):
    missing=[s for s in P.dependencies(stage) if s not in completed]
    if missing:return dict(run=False,reason='missing_dependencies',missing=missing)
    minimum=P.budget(stage)['host_min']
    if free_bytes<minimum:return dict(run=False,reason='insufficient_memory',available=free_bytes,required=minimum)
    return dict(run=True,available=free_bytes,required=minimum)

def run_queue(run_stage,status,wait_seconds=0,poll_seconds=30):
    """Shared production/test state machine; no retry of a failed worker."""
    states=[];completed=set()
    for stage in P.STAGES:
        P.configure_stage(stage)
        try:
            old=P.accepted_stage(stage);completed.add(stage)
            states.append(dict(stage=stage,state='reused',evidence=old['path']))
            print(stage+': reuse accepted files',flush=True);continue
        except (OSError,RuntimeError,ValueError,KeyError):pass
        choice=decision(stage,completed,available())
        deadline=time.monotonic()+wait_seconds
        while not choice['run'] and choice['reason']=='insufficient_memory' and time.monotonic()<deadline:
            status(stage,completed,state='waiting_memory',decision=choice,
                wait_remaining_seconds=max(0,deadline-time.monotonic()))
            time.sleep(min(poll_seconds,max(0,deadline-time.monotonic())))
            choice=decision(stage,completed,available())
        if not choice['run']:
            states.append(dict(stage=stage,state='skipped',**choice))
            print(json.dumps(states[-1]),flush=True);continue
        status(stage,completed)
        try:
            out,result=run_stage()
            if result.get('passed') is not True:raise RuntimeError('Unaccepted stage result')
            completed.add(stage);states.append(dict(stage=stage,state='passed',output=str(out)))
        except BaseException as error:
            states.append(dict(stage=stage,state='failed',error=repr(error)))
            return states,error
    return states,None

def finalize(run_id,states):
    accepted={}
    for stage in P.STAGES:
        try:accepted[stage]=P.accepted_stage(stage)
        except (OSError,ValueError,RuntimeError,KeyError):pass
    complete=set(accepted)==set(P.STAGES)
    result=dict(queue_id=run_id,time=time.time(),all_preparation_complete=complete,
        outcome='complete' if complete else ('failed' if any(x['state']=='failed' for x in states) else 'partial'),
        stages=states,accepted=accepted,pending=[s for s in P.STAGES if s not in accepted],
        raw_ssd_written=False,gpu_called=False,training_completed=False,
        manifest_sha256=P.verify_manifest())
    write(P.OUT/('report_'+run_id+'.json'),result);write(P.OUT/'latest_report.json',result)
    if complete:
        write(P.OUT/'runtime_inputs.json',dict(passed=True,dataset='CL',manifest_sha256=P.verify_manifest(),
            source=P.metadata(),stages=accepted,protocol=plan(),raw_ssd_bound=False,gpu_training_accepted=False))
    feature_sources={}
    for arm in ('gids','digit'):
        s='features-'+arm
        if s in accepted:feature_sources[arm]=accepted[s]['files'][arm+'_features.f32']
    write(P.OUT/'ssd_source_requirements.json',dict(dataset='CL',sources=feature_sources,
        both_sources_accepted=len(feature_sources)==2,device_region_allocated=False,raw_ssd_bound=False,
        required_payload_bytes={arm:((size+4095)//4096)*4096 for arm,size in [('gids',P.N*512),('digit',P.ROWS*512)]},
        note='Reserve disjoint regions from current AE, PA, IG, UKS and UKL allocations before any write; this file does not authorize or perform raw-device writes.'))
    return result

def execute(wait_seconds=8*3600):
    if os.geteuid()!=0:raise RuntimeError('sudo required for per-stage cgroup limits and kernel checks')
    from .start import execute as execute_stage
    P.verify_manifest();P.binding()
    for stage in ('mapping','split'):P.accepted_stage(stage)
    run_id=time.strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid());states=[]
    with ExitStack() as stack:
        # Queue identity only while waiting. Each actual stage also takes the
        # established large-load lock. No GPU or raw SSD device is accessed.
        for path in (P.OUT/'queue.lock',):
            handle=stack.enter_context(path.open('r+' if path.exists() else 'x+'))
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(P.OUT/'runtime_plan.json',plan())
        try:
            def status(stage,completed,state='running',**extra):
                write(P.OUT/'queue_status.json',dict(queue_id=run_id,stage=stage,state=state,completed=sorted(completed),time=time.time(),**extra))
            def run_stage():
                # Waiting for free memory holds no large-load lock. Acquire
                # it only for actual admission/work/exit checks, never steal it.
                with P.BASE.LOCK.open('r+' if P.BASE.LOCK.exists() else 'x+') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    return execute_stage()
            states,error=run_queue(run_stage,status,wait_seconds=wait_seconds)
            if error:raise error
        finally:
            result=finalize(run_id,states)
            write(P.OUT/'queue_status.json',dict(queue_id=run_id,state=result['outcome'],pending=result['pending'],time=time.time()))
            print(json.dumps(dict(report=str(P.OUT/'latest_report.json'),outcome=result['outcome'],pending=result['pending'])),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--wait-hours',type=float,default=8);a=p.parse_args()
    if not 0<=a.wait_hours<=24:p.error('--wait-hours must be between 0 and 24')
    if a.execute:execute(wait_seconds=a.wait_hours*3600)
    else:print(json.dumps(plan(),indent=2))
if __name__=='__main__':main()
