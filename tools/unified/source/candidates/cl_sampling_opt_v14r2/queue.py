"""Tiny CUDA parity then one bounded GIDS/optimized-DiGiT diagnostic pair."""
import argparse,fcntl,json,os,subprocess,time
from contextlib import ExitStack
from pathlib import Path
from . import protocol as P,gpu as GPU

def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(str(tmp),str(path))

def micro_command(selected,out,run_id):
    return ['/usr/bin/systemd-run','--wait','--pipe','--collect','--unit=digit-cl-opt-micro-'+run_id,
        '--property=Type=exec','--property=WorkingDirectory='+str(P.ROOT),
        '--property=MemoryMax=2G','--property=MemorySwapMax=0','--property=LimitMEMLOCK=64M',
        '--property=TasksMax=64','--property=CPUQuota=100%','--property=RuntimeMaxSec=240','--property=TimeoutStopSec=30',
        '--property=DevicePolicy=closed','--property=DeviceAllow=/dev/nvidia'+str(selected[0])+' rw',
        '--property=DeviceAllow=/dev/nvidiactl rw','--property=DeviceAllow=/dev/nvidia-uvm rw',
        '--property=InaccessiblePaths=-/dev/libnvm0',
        '--setenv=CUDA_VISIBLE_DEVICES='+selected[1],'--setenv=CL_OPT_MICRO=1',
        '--setenv=OMP_NUM_THREADS=1','--setenv=OPENBLAS_NUM_THREADS=1','--setenv=MKL_NUM_THREADS=1',
        P.PYTHON,'-B','-u','-m','candidates.cl_sampling_opt_v14r2.gpu_micro_worker','--out',str(out)]

def run_micro_process(command,out,kernel,selected):
    """Keep the worker error and inspect cleanup even when systemd-run fails."""
    evidence=dict(passed=False,command=command,started=time.time(),gpu_release_observations=[])
    failure=None
    try:
        with out.with_suffix('.log').open('x') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=280)
        evidence['returncode']=result.returncode
        if result.returncode:
            raise RuntimeError('Micro service failed (exit %s); see %s' % (result.returncode,out.with_suffix('.log')))
    except BaseException as exc:
        failure=exc;evidence['error']=repr(exc)
    finally:
        def observe(value):
            evidence['gpu_release_observations'].append(value)
            write(out.with_suffix('.process.json'),evidence)
        for name,check in (('gpu_idle',lambda:GPU.wait_released(selected,record=observe)),('kernel',kernel.poll)):
            try:evidence[name]=check()
            except BaseException as exc:
                evidence[name+'_error']=repr(exc)
                if failure is None:failure=exc
        if not evidence.get('kernel',{}).get('ok') and failure is None:
            failure=RuntimeError('Micro kernel post-check failed')
        evidence.update(passed=failure is None,finished=time.time())
        write(out.with_suffix('.process.json'),evidence)
    if failure is not None:raise failure
    return evidence['kernel']

def micro():
    from .kernel_scope import KernelCursor
    with ExitStack() as stack:
        for path in (Path('/tmp/digit-pa-sage-experiment.lock'),Path('/tmp/digit-pa-bidir-controller.lock'),Path('/run/digit-ae-selfservice/exclusive.lock')):
            f=stack.enter_context(path.open('r+' if path.exists() else 'x+'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        selected=GPU.select(GPU.snapshot());identity=GPU.kernel_identity(selected)
        kernel=KernelCursor(selected_uuid=identity['uuid'],selected_pci=identity['pci'])
        if not kernel.establish()['ok']:raise RuntimeError('Micro kernel baseline failed')
        run_id=time.strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid());out=P.BASE_OUT/('micro_'+run_id+'.json')
        GPU.check_idle(selected)
        observation=run_micro_process(micro_command(selected,out,run_id),out,kernel,selected)
        write(P.BASE_OUT/('micro_kernel_'+run_id+'.json'),observation)
        r=json.loads(out.read_text())
        if not (observation['ok'] and r['passed'] and r['graph_released'] and r['cases']==210
                and r['block_cases']==18 and r['training_steps']==9 and r['anonymous_graph_bytes']<=2*2**20
                and r['manifest_sha256']==P.verify_manifest() and r['selected_uuid']==selected[1]
                and r['raw_ssd_access'] is False):raise RuntimeError('CUDA micro parity incomplete')
        r.update(normal_exit=True,gpu_released=True,kernel_ok=True,selected_gpu=list(selected))
        write(P.BASE_OUT/'optimized_micro_acceptance.json',r)
        return r

def execute():
    if os.geteuid()!=0:raise RuntimeError('sudo required for bounded GPU services')
    P.verify_manifest()
    from .qualification import prior_smoke
    from .storage import accepted_storage
    prior_smoke();accepted_storage()
    status=dict(started=time.time(),passed=False,stage='micro',raw_ssd_writes=False)
    with (P.BASE_OUT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            write(P.BASE_OUT/'queue_status.json',status);status['micro']=micro()
            status['stage']='single_pair';write(P.BASE_OUT/'queue_status.json',status)
            from .start import execute as run
            status['pair']=run();status.update(passed=status['pair']['passed'],stage='complete')
        except BaseException as e:
            status.update(error=repr(e),failed_stage=status['stage'],stage='failed');raise
        finally:
            status['updated']=time.time();write(P.BASE_OUT/'queue_status.json',status)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');a=ap.parse_args()
    if a.execute:execute()
    else:print(json.dumps(dict(execute=False,stages=['bounded_cuda_parity','gids_20_plus_300','optimized_digit_20_plus_300'],raw_ssd_writes=False,rounds=1),indent=2))
if __name__=='__main__':main()
