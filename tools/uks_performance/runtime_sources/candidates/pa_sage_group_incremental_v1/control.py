"""Fresh legacy/incremental group processes on the accepted dense-graph pipeline."""
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
        if i<2:result.append(('smoke_'+variant,True,variant))
        result.append(('full_%02d_%s'%(i+1,variant),False,variant))
    for mode in VARIANTS:result.append(('trace_'+mode,False,mode))
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
    reports=[];evidence=[];reference={};traces={}
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
            require(a['sampler_binary_sha256']==r['sampler_binary_sha256']==sha(SAMPLER_BINARY),'Group binary identity differs')
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
                from .trace import kernel_evidence
                traces[variant]=kernel_evidence(dest/'cuda_trace.json',variant)
                write(folder/(stage+'_review.json'),traces[variant])
            state['completed_stages'].append(stage);status(stage+'_accepted')
        require(verify()==source,'Source changed during execution');check_inputs(read(OUT/'inputs.json'))
        from .trace import compare_traces
        result=analyze(reports);result.update(run_directory=str(folder),evidence=evidence,
            trace_comparison=compare_traces(traces['legacy'],traces['incremental']))
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
