"""Persistent, low-priority filesystem preparation; no native training."""
import argparse,json,os,subprocess,time
from .common import ROOT,OUT,verify
PY='/home/embed/miniconda3/envs/gids/bin/python'
def main():
 p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
 cmd=['/usr/bin/systemd-run','--unit=digit-uks-filesystem-v3-'+time.strftime('%Y%m%d%H%M%S'),
      '--property=Type=exec','--property=WorkingDirectory='+str(ROOT),'--property=Nice=10',
      '--property=CPUQuota=200%','--property=IOSchedulingClass=idle','--property=KillMode=control-group',
      '--property=TimeoutStopSec=90','--setenv=CUDA_VISIBLE_DEVICES=',
      '--setenv=OPENBLAS_NUM_THREADS=1','--setenv=OMP_NUM_THREADS=1','--setenv=MKL_NUM_THREADS=1',
      '--setenv=PYTHONDONTWRITEBYTECODE=1',PY,'-B','-u','-m','candidates.uks_prepare_v3.prepare','--execute']
 if a.resume:cmd.append('--resume')
 print(json.dumps(dict(execute=a.execute,command=cmd,native_training=False,raw_ssd_writes=False),indent=2),flush=True)
 if a.execute:
  assert os.geteuid()==0,'Local sudo needed for administrator-owned shared locks'
  verify();subprocess.run(cmd,check=True)
if __name__=='__main__':main()
