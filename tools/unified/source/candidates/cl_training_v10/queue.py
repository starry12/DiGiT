"""Bounded micro -> accepted SSD write/readback -> GIDS/DiGiT four-batch smoke."""
import argparse,fcntl,json,os,subprocess,time
from contextlib import ExitStack
from pathlib import Path
from . import protocol as P,gpu as GPU

def write(path,value):
    p=path.with_suffix('.tmp');p.write_text(json.dumps(value,indent=2)+'\n');os.replace(str(p),str(path))

def micro():
    from .kernel_scope import KernelCursor
    with ExitStack() as stack:
        for path in (Path('/tmp/digit-pa-sage-experiment.lock'),Path('/tmp/digit-pa-bidir-controller.lock'),Path('/run/digit-ae-selfservice/exclusive.lock')):
            f=stack.enter_context(path.open('r+' if path.exists() else 'x+'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        selected=GPU.select(GPU.snapshot());identity=GPU.kernel_identity(selected)
        kernel=KernelCursor(selected_uuid=identity['uuid'],selected_pci=identity['pci'])
        initial=kernel.establish()
        if not initial['ok']:raise RuntimeError('Kernel micro baseline failed')
        run_id=time.strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid())
        reports={}
        for arm in ('gids','digit'):
            GPU.check_idle(selected)
            out=P.BASE_OUT/('micro_'+run_id+'_'+arm+'.json')
            cmd=['/usr/bin/systemd-run','--wait','--pipe','--collect','--unit=digit-cl-micro-'+run_id+'-'+arm,
                '--property=Type=exec','--property=WorkingDirectory='+str(P.ROOT),'--property=MemoryMax=2G','--property=MemorySwapMax=0','--property=TasksMax=64','--property=CPUQuota=100%','--property=RuntimeMaxSec=120','--property=TimeoutStopSec=30',
                '--property=DevicePolicy=closed','--property=DeviceAllow=/dev/nvidia'+str(selected[0])+' rw','--property=DeviceAllow=/dev/nvidiactl rw','--property=DeviceAllow=/dev/nvidia-uvm rw','--property=InaccessiblePaths=-/dev/libnvm0',
                '--setenv=CUDA_VISIBLE_DEVICES='+selected[1],'--setenv=CL_MICRO_WORKER=1','--setenv=OMP_NUM_THREADS=1','--setenv=OPENBLAS_NUM_THREADS=1','--setenv=MKL_NUM_THREADS=1',
                P.PYTHON,'-B','-m','candidates.cl_training_v10.micro_worker','--arm',arm,'--out',str(out)]
            subprocess.run(cmd,check=True,timeout=160)
            reports[arm]=json.loads(out.read_text())
            observation=kernel.poll()
            write(P.BASE_OUT/('micro_kernel_'+run_id+'_'+arm+'.json'),observation)
            if not observation['ok']:raise RuntimeError('New relevant kernel alert during micro')
            GPU.check_idle(selected)
        if reports['gids']['resident']!=reports['digit']['resident'] or not all(v['passed'] for v in reports.values()):
            raise RuntimeError('Page-state CUDA behavior differs')
        result=dict(passed=True,selected_gpu=list(selected),reports=reports,manifest_sha256=P.verify_manifest(),raw_ssd_access=False)
        write(P.BASE_OUT/'micro_acceptance.json',result)
        return result

def execute():
    if os.geteuid()!=0:raise RuntimeError('sudo required for bounded CL write and GPU services')
    P.verify_manifest()
    status=dict(started=time.time(),passed=False,stage='micro',raw_ssd_write_accepted=False,gpu_smoke_accepted=False)
    with (P.BASE_OUT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            write(P.BASE_OUT/'queue_status.json',status);micro()
            status['stage']='storage';write(P.BASE_OUT/'queue_status.json',status)
            from candidates.cl_ssd_writer_v10.binding import accepted_storage
            from candidates.cl_ssd_writer_v10 import start as writer
            try:storage=accepted_storage()
            except (OSError,RuntimeError,ValueError,KeyError):
                writer.execute();storage=accepted_storage()
            status.update(raw_ssd_write_accepted=True,storage=storage['acceptance_path'],stage='smoke')
            write(P.BASE_OUT/'queue_status.json',status)
            from .start import execute as smoke
            result=smoke()
            status.update(passed=result['passed'],gpu_smoke_accepted=result['passed'],stage='complete',smoke=result)
        except BaseException as error:
            status.update(error=repr(error),failed_stage=status['stage']);status['stage']='failed';raise
        finally:
            status['updated']=time.time();write(P.BASE_OUT/'queue_status.json',status)

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.execute:execute()
    else:
        from .budget import budget
        print(json.dumps(dict(execute=False,stages=['micro','storage','gids_four_batches','digit_four_batches'],budgets={a:budget(a) for a in ('gids','digit')}),indent=2))
if __name__=='__main__':main()
