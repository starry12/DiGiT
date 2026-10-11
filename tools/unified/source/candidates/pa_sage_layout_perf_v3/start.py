"""Preview a per-point systemd launch; execution remains explicitly opt-in."""
import argparse,os,subprocess
from .common import ROOT,Path,read,sha,require
from .protocol import validate

def command(protocol,output):
    p=validate(read(protocol));key=p['point']['id']
    return ['/usr/bin/systemd-run','--unit=digit-pa-layout-'+key+'-perf-v3',
        '--property=Type=exec','--property=WorkingDirectory='+str(ROOT),'--property=UMask=0022',
        '--property=Restart=no','--property=KillMode=control-group','--property=TimeoutStopSec=90',
        '--property=LimitMEMLOCK=infinity','--setenv=CUDA_VISIBLE_DEVICES=2',
        '--setenv=PYTHONDONTWRITEBYTECODE=1','--setenv=LD_LIBRARY_PATH='+str(ROOT/'bam/build/lib'),
        '/home/embed/miniconda3/envs/gids/bin/python','-B','-u','-m','candidates.pa_sage_layout_perf_v3.cli',
        '--protocol',str(Path(protocol).resolve()),'--output',str(Path(output).resolve()),'--execute']

def main():
    import json
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    cmd=command(a.protocol,a.output);print(json.dumps(dict(command=cmd,execute=a.execute,protocol_sha256=sha(a.protocol)),indent=2))
    if not a.execute:return
    require(os.geteuid()==0,'Administrator launch required');require(not a.output.exists(),'Output already exists')
    # CLI and controller perform readiness, immutable-code and global-lock checks before any worker.
    subprocess.run(cmd,check=True)
if __name__=='__main__':main()
