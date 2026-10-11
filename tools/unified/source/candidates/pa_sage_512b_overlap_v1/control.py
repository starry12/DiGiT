"""Validate each independent change before the combined native experiment."""
import argparse
import fcntl
import subprocess
import traceback
from .common import *
from .analyze import analyze, markdown, compare_workload
from candidates.pa_sage_512b_pair_v4.control import gpu_admission


def stages():
    result=[]
    for i,variant in enumerate(full_schedule()):
        if i<3:result.append(('smoke_'+variant,True,variant))
        result.append(('full_%02d_%s'%(i+1,variant),False,variant))
    result.append(('trace_overlap',False,'overlap'))
    return result


def run(folder):
    source=verify()
    require(os.geteuid()==0,'Native controller requires local sudo')
    require(not folder.exists(),'Preserve previous evidence');folder.mkdir(parents=True)
    state=dict(passed=False,complete=False,directory=str(folder),candidate_sha256=source,
               started_unix=time.time(),completed_stages=[])
    def status(stage,**extra):
        state.update(stage=stage,updated_unix=time.time(),**extra)
        write(OUT/'status.json',state);write(folder/'status.json',state)
        print(json.dumps(state),flush=True)
    reports=[];evidence=[];reference={}
    try:
        check_inputs(read(OUT/'inputs.json'))
        for stage,smoke,variant in stages():
            status(stage,variant=variant)
            gpu_admission(folder,stage)
            dest=folder/stage
            cmd=[PY,'-B','-u','-m',MODULE+'.worker','--output',str(dest),'--variant',variant]
            if smoke:cmd.append('--smoke')
            diagnostic=stage.startswith('trace_')
            if diagnostic:cmd.append('--trace')
            write(folder/(stage+'_command.json'),cmd)
            with (folder/(stage+'.log')).open('x') as log:
                worker=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            write(folder/(stage+'_exit.json'),dict(returncode=worker.returncode,finished_unix=time.time()))
            require(worker.returncode==0,stage+' failed; see '+str(folder/(stage+'.log')))
            r=read(dest/'report.json');a=read(dest/'accepted.json')
            require(a['passed'] and a['report_sha256']==sha(dest/'report.json') and
                    a['candidate_sha256']==source==r['candidate_sha256'] and
                    a['binary_sha256']==r['binary_sha256']==sha(BINARY),'Acceptance identity differs')
            require(r['input_binding_sha256']==sha(OUT/'inputs.json'),'Worker input identity differs')
            validate_completion(r)
            require(r['smoke']==smoke and r['variant']==variant,'Worker mode differs')
            require(r['diagnostic_trace']==diagnostic,'Trace mode differs')
            if smoke not in reference:reference[smoke]=r
            parity=compare_workload(reference[smoke],r)
            write(folder/(stage+'_parity.json'),parity)
            evidence.append(dict(stage=stage,report=str(dest/'report.json'),report_sha256=sha(dest/'report.json'),
                receipt_sha256=sha(dest/'accepted.json'),parity_sha256=sha(folder/(stage+'_parity.json')),returncode=0))
            if not smoke and not diagnostic:reports.append(r)
            if diagnostic:
                from .trace import overlap_evidence
                overlap=overlap_evidence(dest/'cuda_trace.json')
                write(folder/'overlap_trace_review.json',overlap)
                write(OUT/'overlap_trace_review.json',overlap)
                require(overlap['read_kernels']>0 and overlap['model_kernels']>0,'Trace missed read/model kernels')
            state['completed_stages'].append(stage);status(stage+'_accepted')
        require(verify()==source,'Source changed during execution');check_inputs(read(OUT/'inputs.json'))
        result=analyze(reports);result.update(run_directory=str(folder),evidence=evidence,overlap_trace=overlap)
        for target in (OUT,folder):
            write(target/'summary.json',result);(target/'ANALYSIS.md').write_text(markdown(result))
        status('complete',passed=True,complete=True,finished_unix=time.time())
    except BaseException as error:
        status('failed',error=str(error),finished_unix=time.time())
        write(folder/'failure.json',dict(error=str(error),traceback=traceback.format_exc()));raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    with open('/tmp/digit-pa-sage-512b-pair-controller.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(args.output)


if __name__=='__main__':main()
