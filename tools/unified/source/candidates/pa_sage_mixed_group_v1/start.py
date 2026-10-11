"""Preview by default; --execute queues a persistent single/mixed comparison."""
import argparse
import datetime
import subprocess
from .common import *


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    source=verify();check_inputs(read(OUT/'inputs.json'))
    checks=read(OUT/'cpu_checks.json')
    require(checks['passed'] and checks['source_sha256']==source,'Missing/stale CPU checks')
    stamp=datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+str(os.getpid())
    unit='digit-pa-mixed-group-'+stamp
    folder=OUT/'runs'/stamp
    command=['/usr/bin/systemd-run','--unit='+unit,
        '--property=Type=exec','--property=WorkingDirectory='+str(ROOT),
        '--property=UMask=0022','--property=Restart=no','--property=KillMode=control-group',
        '--property=TimeoutStopSec=90','--property=LimitMEMLOCK=infinity',
        '--setenv=CUDA_VISIBLE_DEVICES=2','--setenv=PYTHONDONTWRITEBYTECODE=1',
        '--setenv=OPENBLAS_NUM_THREADS=1','--setenv=OMP_NUM_THREADS=16','--setenv=MKL_NUM_THREADS=16',
        '--setenv=LD_LIBRARY_PATH='+str(ROOT/'bam/build/lib'),PY,'-B','-u','-m',
        MODULE+'.control','--output',str(folder)]
    plan=dict(command=command,unit=unit+'.service',output=str(folder),execute=args.execute,
        candidate_sha256=source,full_order=full_schedule(),epochs_per_worker=1,
        wait_for_reviewer_queue=True,poll_seconds=60,gpu_checks_after_admission=True,
        smoke_workers=2,full_workers=4,raw_ssd_writes=False,evaluation=False,
        cpu_cache_rows=11105995,gpu_feature_cache_bytes=4*2**30,
        note='Same optimized DiGiT in both arms; only 512B versus paired 512B/1KiB read path differs')
    print(json.dumps(plan,indent=2),flush=True)
    if args.execute:
        require(os.geteuid()==0,'Run this launcher using local sudo')
        subprocess.run(command,check=True)
        write(OUT/'launch.json',dict(plan,started_unix=time.time()))


if __name__=='__main__':main()
