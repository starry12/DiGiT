import contextlib,fcntl,os,subprocess,time
from .common import *
from .statistics import jobs,summarize

def main():
    require(os.geteuid()==0,'Root required');check_ready();OUT.mkdir(exist_ok=True,parents=True)
    with contextlib.ExitStack() as stack:
        for path in [OUT/'controller.lock',native.OUT/'controller.lock',repair.OUT/'controller.lock',Path('/tmp/digit-pa-sage-experiment.lock'),Path('/tmp/digit-pa-bidir-controller.lock'),Path('/tmp/digit-pa-sage-libnvm0.lock'),ROOT/'ssd_state/libnvm0.prepare.lock',Path('/run/digit-ae-selfservice/exclusive.lock')]:
            require(not path.is_symlink(),'Symlink lock');f=stack.enter_context(path.open('rb') if path.exists() else path.open('x+'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not (OUT/'status.json').exists(),'Preserve existing run; no automatic retry')
        source=verify();state=dict(source_sha256=source,complete=False,passed=False,started_unix=time.time(),completed=[]);rows=[]
        def mark(stage,**kw):state.update(stage=stage,updated_unix=time.time(),**kw);write(OUT/'status.json',state)
        try:
            for job in jobs():
                mark(job['mode']);folder=OUT/job['mode'];folder.mkdir(exist_ok=False)
                cmd=[PYTHON,'-B','-u','-m','candidates.uks_revpr_diagnostic_v1.worker','--variant',job['variant'],'--profile-mode',job['profile_mode'],'--round',str(job['repetition']),'--output',str(folder)]
                write(folder/'command.json',dict(command=cmd))
                with (folder/'worker.log').open('x') as log:r=subprocess.run(cmd,cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT)
                write(folder/'exit.json',dict(returncode=r.returncode,finished_unix=time.time()));require(r.returncode==0,'Worker failed: '+job['mode'])
                report=read(folder/'report.json');require(report['passed'] and report['source_sha256']==source and report['variant']==job['variant'] and report['profile_mode']==job['profile_mode'] and report['repetition']==job['repetition'],'Wrong worker report')
                check_ready();report.update(mode=job['mode'],normal_exit=True,report_sha256=sha(folder/'report.json'))
                write(folder/'accepted.json',report);rows.append(report);state['completed'].append(job['mode']);write(OUT/'runs.json',rows)
            result=summarize(rows);result['source_sha256']=source;write(OUT/'summary.json',result)
            lines=['UKS/SAGE，同一 RevPR 热集，五轮最大同轮观察值：','']
            for variant,c in result['comparisons'].items():lines.append('%s：**%.2f×**'%(variant,c['max_observed_speedup']))
            (OUT/'RESULT.md').write_text('\n'.join(lines)+'\n')
            mark('complete',passed=True,complete=True)
        except BaseException as e:mark('failed',error=repr(e));raise
if __name__=='__main__':main()
