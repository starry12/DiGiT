"""Persistent isolated v6 service; preview unless explicitly executed as admin."""
import argparse
import datetime
import json
import os
import subprocess
from .common import ROOT,OUT,PY,MODULE,heavy_gate,verify,require


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    unit='digit-pa-cache-policy-20260929-v6'
    if a.resume:unit+='-r'+datetime.datetime.now().strftime('%Y%m%d%H%M%S')+'-'+str(os.getpid())
    command=['/usr/bin/systemd-run','--unit='+unit,'--property=Type=exec',
        '--property=WorkingDirectory='+str(ROOT),'--property=UMask=0022','--property=Restart=no',
        '--property=KillMode=control-group','--property=TimeoutStopSec=90','--property=LimitMEMLOCK=infinity',
        '--setenv=CUDA_VISIBLE_DEVICES=2','--setenv=PYTHONDONTWRITEBYTECODE=1',
        '--setenv=OPENBLAS_NUM_THREADS=1','--setenv=OMP_NUM_THREADS=16','--setenv=MKL_NUM_THREADS=16',
        '--setenv=LD_LIBRARY_PATH='+str(ROOT/'bam/build/lib'),PY,'-B','-u','-m',MODULE+'.cli',
        'run','--protocol',str(OUT/'plan/protocol.json'),'--output',str(OUT/'native'),'--execute']
    if a.resume:command.append('--resume')
    print(json.dumps(dict(execute=a.execute,command=command,continuous_gpu_monitoring=False,
        reused_preparation=True,smoke_workers='four fresh native checks before any formal epoch',
        full_workers='four fresh complete epochs'),indent=2),flush=True)
    if a.execute:
        os.environ['CUDA_VISIBLE_DEVICES']='2'
        heavy_gate();verify()
        from .common import binary_receipt
        binary_receipt()
        require((OUT/'native/status.json').exists() if a.resume else not (OUT/'native').exists(),
                'Use --resume only for an existing interrupted v6 run')
        subprocess.run(command,check=True)


if __name__=='__main__':main()
