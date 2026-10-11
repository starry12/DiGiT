"""Run UKL five pairs, then CL five pairs; fail closed and preserve all outputs."""
import argparse,fcntl,importlib,json,os,subprocess,time
from .freeze import ROOT,OUT,verify,digest
from .review import review
DATASETS=(('UKL','ukl_common_gpu_five_v27'),('CL','cl_common_gpu_five_v17'))

def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)

def plan():
    return dict(order=['UKL five rounds','CL five rounds'],arms=['gids','digit'],rounds_per_dataset=5,
                warmup_batches=20,measured_batches=300,stage_profiling=False,raw_ssd_writes=False,
                selection='maximum same-round GIDS_seconds / DiGiT_seconds',reuse_accepted_micro=True)

def run_sequence(run_one,record):
    results={}
    for dataset,module in DATASETS:
        record(dataset,results)
        results[dataset]=run_one(dataset,module)
        if not results[dataset]['passed']:raise RuntimeError(dataset+' not accepted')
    record('complete',results)
    return results

def execute():
    if os.geteuid()!=0:raise RuntimeError('sudo required for bounded native services')
    with (OUT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run_id=time.strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid())
        work=OUT/('run_'+run_id);work.mkdir()
        state=dict(passed=False,run_id=run_id,started=time.time(),plan=plan(),results={},stage='preflight')
        def record(stage,results):
            state.update(stage=stage,results=dict(results),updated=time.time())
            write(work/'status.json',state);write(OUT/'latest_status.json',state)
        record('preflight',{})
        try:
            verify(archive=True)
            # Verify both frozen versions and storage bindings before the first heavy job.
            for _,module in DATASETS:
                p=importlib.import_module('candidates.'+module+'.protocol')
                p.verify_manifest();p.require_micro()
                importlib.import_module('candidates.'+module+'.storage').accepted_storage()
            def run_one(dataset,module):
                cmd=['/usr/bin/python3','-B','-u','-m','candidates.'+module+'.cli','--execute']
                env=dict(os.environ,UKL_V27_VARIANT='gpu',CL_V17_VARIANT='gpu',PYTHONDONTWRITEBYTECODE='1')
                p=importlib.import_module('candidates.'+module+'.protocol')
                previous=(p.OUT/'latest_status.json').read_bytes() if (p.OUT/'latest_status.json').exists() else None
                print('Starting '+dataset+' five rounds; log '+str(work/(dataset+'.log')),flush=True)
                with (work/(dataset+'.log')).open('x') as log:
                    subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                current=(p.OUT/'latest_status.json').read_bytes()
                if current==previous:raise RuntimeError('Stale result: '+dataset)
                series=json.loads(current);result=review(series)
                # Revalidate the full original worker acceptance, including memory/NUMA/I/O checks.
                for run in series['runs']:
                    a=json.loads(Path(run['path']).read_text())
                    p.SELECTED=tuple(a['selected_gpu']);p.PAIR_ID=a['pair_id'];p.configure_stage(run['arm'])
                    if not p.worker_passed(a['worker_state'],a['worker']):raise RuntimeError('Worker revalidation failed')
                p.verify_manifest();verify()
                write(work/(dataset+'_series.json'),series);write(work/(dataset+'_review.json'),result)
                result.update(series_path=str(work/(dataset+'_series.json')),series_sha256=digest(work/(dataset+'_series.json')))
                return result
            results=run_sequence(run_one,record)
            state['passed']=True;record('complete',results)
            write(work/'completion_review.json',state)
            print(json.dumps({k:dict(speedup=v['speedup'],best_round=v['best_round']) for k,v in results.items()}),flush=True)
        except BaseException as e:
            state.update(error=repr(e),failed_stage=state['stage'],passed=False)
            record('failed',state['results']);raise

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.execute:execute()
    else:print(json.dumps(plan(),indent=2))

from pathlib import Path
if __name__=='__main__':main()
