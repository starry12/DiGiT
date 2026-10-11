"""Launch the authorized serial grid; default is a side-effect-free preview."""
import argparse
import subprocess
from .common import *

def command():
    return ['/usr/bin/systemd-run','--unit='+UNIT,'--property=Type=exec',
        '--property=WorkingDirectory='+str(ROOT),'--property=UMask=0022',
        '--property=Restart=no','--property=KillMode=control-group','--property=TimeoutStopSec=90',
        '--property=LimitMEMLOCK=infinity','--setenv=CUDA_VISIBLE_DEVICES=2',
        '--setenv=PYTHONDONTWRITEBYTECODE=1','--setenv=OMP_NUM_THREADS=1',
        '--setenv=MKL_NUM_THREADS=1','--setenv=OPENBLAS_NUM_THREADS=1',
        '--setenv=LD_LIBRARY_PATH='+str(ROOT/'bam/build/lib'),
        PY,'-B','-u','-m','candidates.pa_sage_layout_native_v1.run']

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');a=p.parse_args()
    execution=verify();grid=cells();cmd=command()
    preview=dict(command=cmd,execute=a.execute,candidate_sha256=execution,order=ORDER,
        points=len(grid),epochs_per_point=1,evaluation='disabled',gpu=2,
        account='root author native backend; this grid is not an AE self-service command',
        large_files='/mnt/n0/digit/pa_sage_layout_sweep_20260924_v1',
        ssd=dict(device=DEVICE,offset=OFFSET,max_bytes=SLOT_BYTES,blank_check_before_first_write=True,
                 reuse_only_after_previous_point_passed=True),output=str(OUT))
    print(json.dumps(preview,indent=2),flush=True)
    if not a.execute:return
    require(os.geteuid()==0,'Start from administrator terminal with sudo')
    require(not (OUT/'status.json').exists() or not read(OUT/'status.json').get('complete'),'Already complete')
    subprocess.run(cmd,check=True)
if __name__=='__main__':main()
