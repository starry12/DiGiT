"""Wait for shared-device owners, then run gated single/mixed comparisons."""
import argparse,fcntl,signal,subprocess,traceback,statistics
from .common import *
from .queue import exclusion
from candidates.pa_sage_512b_pair_v4.control import gpu_admission


def run(folder):
    require(os.geteuid()==0,'Native controller requires local sudo')
    singleton=open('/tmp/digit-pa-mixed-group-controller.lock','a+')
    fcntl.flock(singleton,fcntl.LOCK_EX|fcntl.LOCK_NB)
    folder.mkdir(parents=True,exist_ok=False)
    state=dict(passed=False,complete=False,pid=os.getpid(),output=str(folder),completed_stages=[],started_unix=time.time())
    def save(**kw):
        if kw.get('stage')!='waiting_for_reviewers':state.pop('reason',None)
        state.update(kw,updated_unix=time.time());write(folder/'status.json',state);write(OUT/'latest.json',state)
        print(json.dumps(state),flush=True)
    def stop(signum,frame):raise KeyboardInterrupt('Controller signaled')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    child=None
    try:
        save(stage='queued')
        with exclusion(save) as ssd_fd:
            source=verify();check_inputs(read(OUT/'inputs.json'))
            require(read(OUT/'cpu_checks.json')['source_sha256']==source and read(OUT/'cpu_checks.json')['passed'],'Missing/stale CPU checks')
            save(stage='gpu_only_checks');gpu_admission(folder,'gpu_checks')
            with (folder/'gpu_checks.log').open('x') as log:
                subprocess.run([PY,'-B','-u','-m',MODULE+'.gpu_check','--output',str(folder/'gpu_checks.json')],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
            require(read(folder/'gpu_checks.json')['passed'],'GPU fixtures failed')
            stages=[('smoke_'+v,True,v) for v in VARIANTS]+[('full_%02d_%s'%(i+1,v),False,v) for i,v in enumerate(full_schedule())]
            references={};reports=[];evidence=[]
            for stage,smoke,variant in stages:
                gpu_admission(folder,stage);dest=folder/stage
                cmd=[PY,'-B','-u','-m',MODULE+'.worker','--output',str(dest),'--variant',variant,'--lock-fd',str(ssd_fd)]
                if smoke:cmd.append('--smoke')
                write(folder/(stage+'_command.json'),dict(command=cmd,started_unix=time.time()))
                with (folder/(stage+'.log')).open('x') as log:
                    child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,pass_fds=(ssd_fd,))
                    save(stage=stage,worker_pid=child.pid,worker_output=str(dest))
                    code=child.wait();child=None
                write(folder/(stage+'_exit.json'),dict(returncode=code,finished_unix=time.time()))
                require(code==0,'Worker failed: '+stage)
                r=read(dest/'report.json');a=read(dest/'accepted.json')
                require(a['passed'] and a['report_sha256']==sha(dest/'report.json'),'Missing acceptance or altered report')
                require(r['candidate_sha256']==source and r['binary_sha256']==sha(BINARY),'Wrong source/backend')
                require(r['smoke']==smoke and r['variant']==variant and not r['diagnostic_trace'],'Wrong worker mode')
                validate_completion(r)
                if smoke not in references:references[smoke]=r
                parity=compare_workload(references[smoke],r);write(folder/(stage+'_parity.json'),parity)
                evidence.append(dict(stage=stage,report=str(dest/'report.json'),sha256=sha(dest/'report.json'),returncode=code))
                if not smoke:reports.append(r)
                state['completed_stages'].append(stage);save(stage=stage+'_accepted')
            rows={}
            for variant in VARIANTS:
                selected=[r for r in reports if r['variant']==variant]
                commands=sum(r['device_io']['completed_commands'] for r in selected)
                active=sum(r['timing']['ssd_active_seconds'] for r in selected)
                elapsed=sum(r['timing']['e2e_seconds'] for r in selected)
                rows[variant]=dict(repeats=len(selected),epoch_seconds=[r['timing']['e2e_seconds'] for r in selected],
                    mean_epoch_seconds=elapsed/len(selected),mean_ssd_active_seconds=active/len(selected),
                    mean_read_commands=commands/len(selected),mean_read_bytes=statistics.mean(r['device_io']['completed_bytes'] for r in selected),
                    active_iops=commands/active,epoch_average_iops=commands/elapsed,
                    request_histograms=[r['pair_counts'] for r in selected])
            require(verify()==source,'Source changed during run');check_inputs(read(OUT/'inputs.json'))
            result=dict(passed=True,complete=True,rows=rows,evidence=evidence,
                training_speedup=rows['single']['mean_epoch_seconds']/rows['mixed']['mean_epoch_seconds'],
                ssd_active_speedup=rows['single']['mean_ssd_active_seconds']/rows['mixed']['mean_ssd_active_seconds'],
                limits=['DiGiT only; two fresh single-seed full epochs per mode','No validation/test accuracy evaluation',
                        'Time includes pair planning and scatter; setup separate','Original 512B control uses same instrumented backend'])
            write(folder/'summary.json',result);write(OUT/'summary.json',result)
            save(stage='complete',passed=True,complete=True,summary_sha256=sha(folder/'summary.json'),finished_unix=time.time())
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=60)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        save(stage='failed',error=str(error),finished_unix=time.time())
        write(folder/'failure.json',dict(error=str(error),traceback=traceback.format_exc()));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output.resolve())
