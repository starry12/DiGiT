"""Preview a persistent service launch; explicit execution is gated by grid completion."""
import argparse
import json
import subprocess
from .common import ROOT,OUT,PY,MODULE,heavy_gate,verify


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    command=['/usr/bin/systemd-run','--unit=digit-pa-cpu-capacity-20260925-v1',
        '--property=Type=exec','--property=WorkingDirectory='+str(ROOT),'--property=UMask=0022',
        '--property=Restart=no','--property=KillMode=control-group','--property=TimeoutStopSec=90',
        '--property=LimitMEMLOCK=infinity','--setenv=CUDA_VISIBLE_DEVICES=2',
        '--setenv=PYTHONDONTWRITEBYTECODE=1','--setenv=OPENBLAS_NUM_THREADS=1',
        '--setenv=OMP_NUM_THREADS=16','--setenv=MKL_NUM_THREADS=16',
        '--setenv=LD_LIBRARY_PATH='+str(ROOT/'bam/build/lib'),
        PY,'-B','-u','-m',MODULE+'.cli','run','--protocol',str(OUT/'plan/protocol.json'),
        '--output',str(OUT/'native'),'--execute']
    if a.resume:command.append('--resume')
    print(json.dumps(dict(execute=a.execute,command=command,automatic_start=False),indent=2),flush=True)
    if a.execute:
        heavy_gate();verify()
        subprocess.run(command,check=True)


if __name__=='__main__':main()
