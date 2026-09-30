"""Fresh native processes, symmetric off/host controls, stop on first failure."""
import argparse
import fcntl
import subprocess
import traceback
from .common import *
from .analyze import analyze, markdown
from candidates.pa_sage_512b_pair_v4.control import gpu_admission


def stages():
    return [('smoke_'+a+'_host',True,a,'host') for a in ('gids','digit')]+[
        ('full_%02d_%s_%s'%(i+1,a,m),False,a,m) for i,(a,m) in enumerate(full_schedule())]


def run(folder):
    source=verify()
    require(os.geteuid()==0, 'Native controller requires local administrator execution')
    require(not folder.exists(), 'Preserve previous run output')
    folder.mkdir(parents=True)
    state=dict(passed=False,complete=False,directory=str(folder),candidate_sha256=source,
               started_unix=time.time(),completed_stages=[])
    def status(stage,**extra):
        state.update(stage=stage,updated_unix=time.time(),**extra)
        write(OUT/'status.json',state);write(folder/'status.json',state)
        print(json.dumps(state),flush=True)
    reports=[];evidence=[]
    try:
        check_inputs(read(OUT/'inputs.json'))
        for stage,smoke,arm,mode in stages():
            status(stage,arm=arm,profile_mode=mode)
            gpu_admission(folder,stage)
            dest=folder/stage
            cmd=[PY,'-B','-u','-m',MODULE+'.worker','--output',str(dest),'--arm',arm,'--profile-mode',mode]
            if smoke:cmd.append('--smoke')
            write(folder/(stage+'_command.json'),cmd)
            with (folder/(stage+'.log')).open('x') as log:
                worker=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            write(folder/(stage+'_exit.json'),dict(returncode=worker.returncode,finished_unix=time.time()))
            require(worker.returncode==0, stage+' failed; see '+str(folder/(stage+'.log')))
            report=read(dest/'report.json');receipt=read(dest/'accepted.json')
            require(receipt['passed'] and receipt['report_sha256']==sha(dest/'report.json') and
                    receipt['candidate_sha256']==source==report['candidate_sha256'] and
                    receipt['binary_sha256']==sha(BINARY),'Acceptance identity mismatch')
            validate_completion(report)
            require(report['smoke']==smoke and report['arm']==arm and report['sampling_profile']['mode']==mode,
                    'Worker mode differs')
            evidence.append(dict(stage=stage,report=str(dest/'report.json'),report_sha256=sha(dest/'report.json'),
                                 receipt_sha256=sha(dest/'accepted.json'),returncode=worker.returncode))
            if not smoke:reports.append(report)
            state['completed_stages'].append(stage);status(stage+'_accepted')
        require(verify()==source,'Candidate changed during diagnostic')
        check_inputs(read(OUT/'inputs.json'))
        result=analyze(reports);result.update(run_directory=str(folder),evidence=evidence)
        for target in (folder,OUT):
            write(target/'profile_analysis.json',result)
            (target/'ANALYSIS.md').write_text(markdown(result))
        status('complete',passed=True,complete=True,finished_unix=time.time())
    except BaseException as error:
        status('failed',error=str(error),finished_unix=time.time())
        write(folder/'failure.json',dict(error=str(error),traceback=traceback.format_exc()))
        raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with open('/tmp/digit-pa-sage-512b-pair-controller.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args.output)


if __name__=='__main__':main()
