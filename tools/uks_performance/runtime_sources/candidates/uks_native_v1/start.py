"""Preview or launch isolated UKS preparation and four-update native checks."""
import argparse,json,os,subprocess
from .common import *
from .controller import PYTHON

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args()
    verify();binary_receipt();heavy_gate()
    cmd=['/usr/bin/systemd-run','--unit=digit-uks-native-20260929-v1','--property=Type=exec','--property=WorkingDirectory='+str(ROOT),'--property=UMask=0022','--property=Restart=no','--property=KillMode=control-group','--property=TimeoutStopSec=90','--property=LimitMEMLOCK=infinity','--setenv=CUDA_VISIBLE_DEVICES=2','--setenv=PYTHONDONTWRITEBYTECODE=1','--setenv=PYTHONFAULTHANDLER=1','--setenv=OMP_NUM_THREADS=16','--setenv=MKL_NUM_THREADS=16','--setenv=OPENBLAS_NUM_THREADS=1','--setenv=LD_LIBRARY_PATH='+str(ROOT/'bam/build/lib'),PYTHON,'-B','-u','-m','candidates.uks_native_v1.controller']
    print(json.dumps(dict(command=cmd,execute=a.execute,profile_batches=100,short_updates_per_arm=4,full_epoch=False,raw_ssd_writes=True,ssd_offsets_bytes={'gids':4*2**40,'digit':int(4.25*2**40)},continuous_gpu_monitoring=False),indent=2),flush=True)
    if a.execute:
        require(os.geteuid()==0,'sudo required for SSD binding');subprocess.run(cmd,check=True)
if __name__=='__main__':main()
