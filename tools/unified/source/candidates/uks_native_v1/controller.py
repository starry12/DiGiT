"""Sequential isolated workers; acceptance requires normal process teardown."""
import contextlib,fcntl,os,subprocess,time,traceback
from pathlib import Path
from .common import *
from .binding import bind,check
PYTHON='/home/embed/miniconda3/envs/gids/bin/python'
STAGES=('profile','write_gids','verify_gids','write_digit','verify_digit','smoke_gids','smoke_digit')

def accept(stage,report,returncode,source):
    require(returncode==0,'Worker failed or crashed during teardown: '+stage)
    require(report.get('passed') is True and report.get('source_sha256')==source,'Unaccepted worker report')
    if stage=='profile':
        require(report['batches']==100 and report['seed']==23 and report['feature_reads']==report['optimizer_updates']==0,'Invalid independent profile')
    if stage.startswith('smoke_'):
        require(report['native'] and report['updates']==4 and report['examples']==4096 and report['feature_bit_exact'] and report['sample_edges_verified'] and report['region']['reconciled'],'Incomplete short acceptance')
    return report

def main():
    require(os.geteuid()==0,'Root controller required');heavy_gate();source=verify();binary_receipt();OUT.mkdir(exist_ok=True,parents=True)
    with contextlib.ExitStack() as stack:
        paths=[OUT/'controller.lock',Path('/tmp/digit-pa-sage-experiment.lock'),Path('/tmp/digit-pa-bidir-controller.lock'),Path('/tmp/digit-pa-sage-libnvm0.lock'),ROOT/'ssd_state/libnvm0.prepare.lock']
        ae_lock=Path('/run/digit-ae-selfservice/exclusive.lock')
        require(ae_lock.exists(),'Missing AE exclusion lock');paths.append(ae_lock)
        for path in paths:
            require(not path.is_symlink(),'Symlink lock rejected');f=stack.enter_context(path.open('rb') if path.exists() else path.open('x+'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=dict(source_sha256=source,complete=False,passed=False,started_unix=time.time(),pid=os.getpid(),completed={})
        def status(stage,**kw):
            state.update(stage=stage,updated_unix=time.time(),**kw);write(OUT/'status.json',state)
        try:
            status('binding')
            if (OUT/'binding.json').exists():check()
            else:bind(OUT)
            for stage in STAGES:
                status(stage);folder=OUT/stage;folder.mkdir(exist_ok=True)
                if (folder/'accepted.json').exists():
                    report=accept(stage,read(folder/'accepted.json'),0,source)
                    require(sha(folder/'report.json')==report['report_sha256'],'Archived report changed')
                else:
                    require(not (folder/'worker.log').exists(),'Preserve failed worker evidence before retry: '+stage)
                    cmd=[PYTHON,'-B','-u','-m','candidates.uks_native_v1.worker',stage,'--output',str(folder)]
                    write(folder/'command.json',dict(command=cmd,source_sha256=source))
                    with (folder/'worker.log').open('x') as log:
                        r=subprocess.run(cmd,cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT)
                    write(folder/'exit.json',dict(returncode=r.returncode,finished_unix=time.time()))
                    require(r.returncode==0,'Worker failed, see '+str(folder/'worker.log'))
                    report=accept(stage,read(folder/'report.json'),r.returncode,source)
                    check();report=dict(report,report_sha256=sha(folder/'report.json'),normal_exit=True)
                    write(folder/'accepted.json',report)
                state['completed'][stage]=sha(folder/'accepted.json')
            a=read(OUT/'smoke_gids/accepted.json');b=read(OUT/'smoke_digit/accepted.json')
            for k in ('initial_model_sha256','roots_sha256'):require(a[k]==b[k],'Unpaired short: '+k)
            write(OUT/'summary.json',dict(passed=True,native_short_accepted=True,full_epoch=False,accuracy_claim=False,source_sha256=source,completed=state['completed']))
            status('native_short_complete',complete=True,passed=True)
        except BaseException as e:
            status('failed',error=repr(e));raise
if __name__=='__main__':main()
