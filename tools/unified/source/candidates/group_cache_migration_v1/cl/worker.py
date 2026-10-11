"""One fresh bounded process per arm; no direct unguarded execution."""
import argparse,json,os,resource,signal,time,traceback
from pathlib import Path
from . import protocol as P

def write(path,v):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');os.replace(str(tmp),str(path))

def persist_failure(error):
    """Preserve the first worker failure even if the service is later unloaded."""
    if os.environ.get('CL_V17_BOUNDED_WORKER')!='1':return
    try:
        out=Path(os.environ['CL_V17_OUTPUT']).resolve(strict=True)
        if (out/'failure.json').exists():return
        before=json.loads((out/'before.json').read_text())
        status=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else None
        r=dict(passed=False,scope='worker_failure',time=time.time(),pid=os.getpid(),
               stage=before.get('stage'),pair_id=before.get('pair_id'),selected_gpu=before.get('selected_gpu'),
               error_type=type(error).__name__,error=str(error),
               traceback=''.join(traceback.format_exception(type(error),error,error.__traceback__)),
               original_traceback=getattr(error,'original_traceback',None),
               cleanup_traceback=getattr(error,'cleanup_traceback',None),
               original_error=getattr(error,'original_error',None),last_progress=status,
               training_checkpoint_available=(out/'training_checkpoint.json').exists(),
               manifest_sha256=before.get('manifest_sha256'),raw_ssd_writes=False)
        write(out/'failure.json',r);write(out/'worker.json',r)
    except BaseException as report_error:
        import sys
        print('Unable to persist worker failure: '+repr(report_error),file=sys.stderr,flush=True)

def main():
    try:return run()
    except BaseException as error:
        persist_failure(error)
        raise

def run():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=P.STAGES,required=True);a=p.parse_args();P.configure_stage(a.stage)
    if os.environ.get('CL_V17_BOUNDED_WORKER')!='1':raise RuntimeError('Use the bounded performance controller')
    out=Path(os.environ['CL_V17_OUTPUT']).resolve(strict=True);before=json.loads((out/'before.json').read_text());lease=json.loads((out/'controller_lease.json').read_text())
    P.PAIR_ID=before['pair_id'];P.SELECTED=tuple(before['selected_gpu']);P.require_admitted_worker(P.arm());digest=P.verify_manifest()
    from candidates.ukl_training_native_v15r11.gpu_identity import selection_from_env
    selection_from_env(P.SELECTED)
    if P.require_predecessor()!=before['predecessor']:raise RuntimeError('Worker admission changed')
    stopped=[False];events=[]
    def stop(*args):stopped[0]=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def check():
        if stopped[0] or (out/'STOP').exists():raise RuntimeError('Controller requested cooperative stop')
        if Path('/proc/%d/stat'%lease['pid']).read_text().rsplit(')',1)[1].split()[19]!=lease['start_ticks']:raise RuntimeError('Controller lease expired')
    last_event=[0.0,None]
    def event(stage,**kw):
        if stage not in ('cache_release_begin','release_chunk','cache_release_complete','lifecycle_cleanup_failed','training_window_complete'):check()
        now=time.monotonic()
        if stage.endswith('_chunk') and last_event[1]==stage and now-last_event[0]<1:return
        last_event[:]=[now,stage]
        v=dict(stage=stage,arm=P.STAGE,time=time.time(),**kw)
        if stage=='training_window_complete':write(out/'training_checkpoint.json',dict(v,accepted=False,lifecycle_complete=False))
        write(out/'status.json',v)
        with (out/'lifecycle.jsonl').open('a') as stream:stream.write(json.dumps(v)+'\n')
    from .affinity import verify as verify_affinity
    affinity_before=verify_affinity(P.arm())
    check();event('initializing')
    from .session import run_arm
    if P.small():
        from .preload import run
        r=run(P.arm(),check,event)
    else:r=run_arm(P.arm(),'performance',check=check,event=event)
    check();P.require_admitted_worker(P.arm())
    from .sampling import effective_variant
    r.update(sampler_variant=effective_variant(P.arm()),experiment_variant=P.VARIANT,cpu_affinity=verify_affinity(P.arm()),cpu_affinity_initial=affinity_before,pair_id=P.PAIR_ID,selected_gpu=list(P.SELECTED),maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,effective_limits_verified=True,raw_ssd_writes=False)
    if digest!=P.verify_manifest() or not P.validate_worker_report(r):raise RuntimeError('Performance receipt rejected')
    write(out/'worker.json',r);event('complete',updates=r['updates'],passed=True)
if __name__=='__main__':main()
