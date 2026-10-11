"""Launch one isolated pair after current AE/author work releases shared locks."""
import argparse
import datetime
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args()
    from candidates.ig_demand_pair_v1.common import verify,OUT,write,require
    from candidates.ig_demand_pair_v1.gpu_check import require_accepted
    candidate=verify();require_accepted()
    stamp=datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+str(os.getpid())
    output=OUT/'runs'/stamp;unit='digit-ig-demand-pair-'+stamp
    command=['/home/embed/miniconda3/envs/gids/bin/python','-B','-u','-m',
             'candidates.ig_demand_pair_v1.launcher','--output',str(output),'--wait']
    plan=dict(candidate_sha256=candidate,unit=unit,output=str(output),command=command,
              schedule='both 2-batch smokes, then GIDS and DiGiT each 20+300 batches',poll_seconds=60)
    if a.execute:
        require(os.geteuid()==0,'Native launch needs sudo')
        subprocess.run(['systemd-run','--unit='+unit,'--property=Type=exec','--property=Restart=no',
                        '--property=WorkingDirectory='+str(ROOT),'--property=KillMode=control-group']+command,check=True)
        write(OUT/'latest.json',plan)
    print(plan)

if __name__=='__main__':main()
