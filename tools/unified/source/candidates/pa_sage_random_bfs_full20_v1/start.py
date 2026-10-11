"""Preview by default; explicit start queues the entire preparation and pair."""
import argparse
import datetime
import subprocess
from .common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args()
    source=verify();binary_receipt();checks=read(OUT/'cpu_checks.json')
    require(checks['passed'] and checks['source_sha256']==source,'Missing matching CPU/build checks')
    stamp=datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+str(os.getpid());folder=OUT/'runs'/stamp;unit='digit-pa-random-bfs-full20-'+stamp
    cmd=['systemd-run','--unit='+unit,'--service-type=exec','--property=WorkingDirectory='+str(ROOT),
         '--property=MemoryMax=352G','--property=MemorySwapMax=0','--property=Restart=no','--property=KillMode=control-group',
         '--property=TimeoutStopSec=90','--property=LimitMEMLOCK=infinity','--property=UMask=0022',
         '--setenv=CUDA_VISIBLE_DEVICES=2','--setenv=OMP_NUM_THREADS=1','--setenv=MKL_NUM_THREADS=1',
         '--setenv=OPENBLAS_NUM_THREADS=1','--setenv=LD_LIBRARY_PATH='+str(ROOT/'bam/build/lib'),
         PY,'-B','-u','-m',MODULE+'.controller','--output',str(folder)]
    plan=dict(command=cmd,output=str(folder),unit=unit,source_sha256=source,execute=a.execute,
       stages=['reuse verified parent data and unchanged BFS profile','GPU fixtures',
               'real feature smoke per arm','GIDS full20 + checkpoint validation/final test','DiGiT full20 + checkpoint validation/final test'],
       epochs_per_arm=20,batches_per_epoch=1179,tail_nodes=907,updates_per_arm=23580,
       validation_calls_per_arm=20,test_calls_per_arm=1,wait_for_existing_queue=True,raw_ssd_writes=False,installed_ae_changed=False)
    if a.execute:
        require(os.geteuid()==0,'Run with local sudo');subprocess.run(cmd,check=True);write(OUT/'launch.json',plan)
    print(json.dumps(plan,indent=2))

if __name__=='__main__':main()
