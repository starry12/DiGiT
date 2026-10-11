"""Dispatch in a fresh service; underlying adapters retain their own guards."""
import argparse,os,subprocess,sys,time
from importlib import import_module
from .common import *
from .registry import plan

def wait_resources(folder):
    import fcntl
    from contextlib import ExitStack
    paths=('/tmp/digit-pa-bidir-controller.lock','/tmp/digit-pa-sage-experiment.lock',
        '/tmp/digit-pa-sage-libnvm0.lock',str(ROOT/'ssd_state/libnvm0.prepare.lock'),'/run/digit-ae-selfservice/exclusive.lock')
    while True:
        try:
            with ExitStack() as stack:
                for path in paths:
                    if not Path(path).is_file() or Path(path).is_symlink():raise RuntimeError('Unsafe shared lock')
                    f=stack.enter_context(open(path,'rb'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
                return
        except BlockingIOError:
            write(folder/'status.json',dict(stage='waiting',passed=False,complete=False,reason='shared experiment lock busy',poll_seconds=60,updated_unix=time.time()))
        time.sleep(60)

def execute(dataset,mode,folder):
    from candidates.unified_results_v2.__main__ import export_csv
    source=verify();p=plan(dataset,mode)
    if p['missing_preparation']:raise RuntimeError('Complete preparation first: '+', '.join(p['missing_preparation']))
    write(folder/'config.json',p['config']);write(folder/'status.json',dict(stage='dispatch',complete=False,passed=False,updated_unix=time.time()))
    stamp=folder.name
    if dataset=='PA':
        module='candidates.pa_sage_bfs256_full20_v1.common' if mode=='full20' else 'candidates.pa_sage_bfs256_short_v1.common'
        write(folder/'config.json',dict(p['config'],actual_protocol=import_module(module).cfg()))
        if mode=='full20':
            from candidates.pa_sage_bfs256_full20_v1.common import OUT as backend_out
            control=import_module(p['runtime_module']).control
            run=backend_out/'runs'/('unified-'+stamp);write(folder/'backend.json',dict(output=str(run)))
            control(run)
            from candidates.unified_results_v2.pa import load
            result=load(run)
        else:
            from candidates.pa_sage_bfs256_short_v1.common import OUT as backend_out
            control=import_module(p['runtime_module']).control
            from .pa_report import summarize
            runs=[]
            for rep in range(1,6 if mode=='short5' else 2):
                run=backend_out/'runs'/('unified-'+stamp+'-r'+str(rep))
                write(folder/'backend.json',dict(output=str(run),completed_rounds=[str(p) for p in runs]))
                control(run,('gids','digit') if rep%2 else ('digit','gids'));runs.append(run)
            result=summarize(mode,runs)
    elif dataset=='UKS':
        from candidates.uks_pa_fast_v1.identity import install
        install()
        from candidates.uks_unified_v2.common import OUT as backend_out
        run=import_module(p['runtime_module']).execute
        dest=backend_out/'runs'/stamp;write(folder/'backend.json',dict(output=str(dest)))
        result=read(run(mode,dest))
    elif dataset=='IG':
        from candidates.ig_unified_v2.common import OUT as backend_out
        dest=backend_out/'runs'/stamp;write(folder/'backend.json',dict(output=str(dest)))
        subprocess.run([sys.executable,'-B','-u','-m',p['runtime_module'],'--output',str(dest),'--mode',mode],check=True,cwd=ROOT)
        # The IG launcher creates a guarded child directory; follow its receipt.
        matches=list(dest.rglob('summary.json'))
        matches=[x for x in matches if read(x).get('schema')=='digit-unified-results-v2']
        if len(matches)!=1:raise RuntimeError('Expected one accepted IG summary')
        result=read(matches[0])
    else:
        from candidates.group_cache_migration_v1.common import stable
        stable(dataset);wait_resources(folder)
        subprocess.run([sys.executable,'-B','-m','candidates.large_graph_fast_v1.gpu_check'],check=True,cwd=ROOT)
        os.environ[dataset+'_UNIFIED_MODE']=mode
        if dataset=='UKL':
            subprocess.run([sys.executable,'-B','-m','candidates.ukl_compact_cover_v1.gpu_check'],check=True,cwd=ROOT)
        module=import_module(p['runtime_module'])
        if module.P.SCOPE!=mode:
            raise RuntimeError('Imported backend execution mode differs from requested mode')
        write(folder/'backend.json',dict(output=str(module.P.OUT)))
        result=module.execute()
    if source!=verify():raise RuntimeError('Entry source changed')
    if not result['passed'] or not result['complete']:raise RuntimeError('Incomplete backend result')
    write(folder/'summary.json',result);export_csv(folder/'summary.csv',result)
    write(folder/'status.json',dict(stage='complete',passed=True,complete=True,source_sha256=source,updated_unix=time.time()))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--mode',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if os.geteuid()!=0 or not __debug__:raise RuntimeError('Use the guarded root service')
    try:execute(a.dataset,a.mode,a.output)
    except BaseException as e:
        write(a.output/'status.json',dict(stage='failed',passed=False,complete=False,error=repr(e),updated_unix=time.time()));raise
if __name__=='__main__':main()
