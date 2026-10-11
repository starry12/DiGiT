import argparse,contextlib,fcntl,os,subprocess,time
from .common import *
JOBS=[('merge_audit','merge_audit','diagnostic'),('digit_profile','digit_profile','diagnostic'),('gids_profile','gids_profile','diagnostic')]
def main():
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);args=a.parse_args()
    output=args.output.resolve();require(output.parent==(OUT/'runs').resolve(),'Invalid retry output')
    run(output)

def run(OUT):
    require(os.geteuid()==0,'Root required');check_ready();OUT.mkdir(exist_ok=True,parents=True)
    with contextlib.ExitStack() as stack:
        for path in [OUT/'controller.lock',Path('/tmp/digit-pa-sage-experiment.lock'),Path('/tmp/digit-pa-bidir-controller.lock'),Path('/tmp/digit-pa-sage-libnvm0.lock'),ROOT/'ssd_state/libnvm0.prepare.lock',Path('/run/digit-ae-selfservice/exclusive.lock')]:
            require(not path.is_symlink(),'Symlink lock');f=stack.enter_context(path.open('rb') if path.exists() else path.open('x+'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not (OUT/'status.json').exists(),'Preserve previous run')
        state=dict(source_sha256=verify(),complete=False,passed=False,started_unix=time.time(),completed=[]);reports={}
        try:
            schedule=JOBS
            for name,variant,probe in schedule:
                state.update(stage=name,updated_unix=time.time());write(OUT/'status.json',state);folder=OUT/name;folder.mkdir()
                cmd=[PYTHON,'-B','-u','-m','candidates.uks_merge_audit_v1.'+('gpu_check' if name=='gpu_check' else 'worker')]
                if variant:cmd+=['--job',variant,'--output',str(folder)]
                with (folder/'worker.log').open('x') as log:r=subprocess.run(cmd,cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT)
                write(folder/'exit.json',dict(returncode=r.returncode));require(r.returncode==0,'Worker failed: '+name)
                report=read(folder/'report.json');require(report['passed'] and report['source_sha256']==verify(),'Invalid report');report.update(normal_exit=True,report_sha256=sha(folder/'report.json'));write(folder/'accepted.json',report);reports[name]=report;state['completed'].append(name)
            measured=[reports[n] for n,_,mode in JOBS if mode=='diagnostic']
            for key in ('roots_sha256','initial_model_sha256','hot_file_sha256'):require(len({r[key] for r in measured})==1,'Unpaired '+key)
            require(all(r['updates']==320 and r['training']['reconciled'] for r in measured),'Incomplete diagnosis')
            write(OUT/'summary.json',dict(passed=True,source_sha256=verify(),merge_audit=reports['merge_audit']['diagnostic']['totals'],native_requests=reports['merge_audit']['training']['request_sizes'],performance_result=False))
            state.update(stage='complete',complete=True,passed=True,updated_unix=time.time());write(OUT/'status.json',state)
        except BaseException as e:state.update(stage='failed',error=repr(e),updated_unix=time.time());write(OUT/'status.json',state);raise
if __name__=='__main__':main()
