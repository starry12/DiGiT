"""Preview or launch the read-only shared-pool calibration and gated grid."""
import argparse
import subprocess
from .common import *
from .pool import OUT,spec

def command():
    return ['/usr/bin/systemd-run','--unit=digit-pa-layout-shared-resume-20260925-v3',
        '--property=Type=exec','--property=WorkingDirectory='+str(ROOT),'--property=UMask=0022',
        '--property=Restart=no','--property=KillMode=control-group','--property=TimeoutStopSec=90',
        '--property=LimitMEMLOCK=infinity','--setenv=CUDA_VISIBLE_DEVICES=2',
        '--setenv=PYTHONDONTWRITEBYTECODE=1','--setenv=OPENBLAS_NUM_THREADS=1',
        '--setenv=LD_LIBRARY_PATH='+str(ROOT/'bam/build/lib'),
        '/home/embed/miniconda3/envs/gids/bin/python','-B','-u','-m','candidates.pa_sage_layout_shared_resume_v3.queue']

def main():
    import json
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');a=parser.parse_args()
    execution=verify();pools={mode:spec(mode) for mode in ('real','shared')}
    cmd=command();print(json.dumps(dict(command=cmd,execute=a.execute,candidate_sha256=execution,raw_ssd_writes=False,
         calibration='g2/r20 ABBA pilot with independent smoke; prior attempts started no native workers',
         remaining_points_verification_policy=POLICY,pool_readback='Reuse completed v1 receipts; do not read back SSD extents again',
         grid='15 points only after calibration passes; each one full epoch, no evaluation',
         pools={k:dict(offset=v['offset'],verify_bytes=v['verify_bytes']) for k,v in pools.items()}),indent=2),flush=True)
    if a.execute:
        require(os.geteuid()==0,'Native read access needs administrator launch')
        from .resume import prerequisites
        prerequisites()
        require(not (OUT/'status.json').exists(),'Keep prior run evidence');subprocess.run(cmd,check=True)
if __name__=='__main__':main()
