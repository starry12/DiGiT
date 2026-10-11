import argparse,datetime,json,os,subprocess,uuid
from .common import *
def plan(run_id):
    require(run_id and all(c.isalnum() or c in '-_' for c in run_id),'Invalid run id')
    output=OUT/'runs'/run_id;unit='digit-uks-group-incremental-'+run_id
    cmd=['/usr/bin/systemd-run','--unit='+unit,'--property=Type=exec','--property=WorkingDirectory='+str(ROOT),'--property=UMask=0022','--property=Restart=no','--property=KillMode=control-group','--property=TimeoutStopSec=90','--property=LimitMEMLOCK=infinity','--setenv=CUDA_VISIBLE_DEVICES=2','--setenv=PYTHONDONTWRITEBYTECODE=1','--setenv=PYTHONFAULTHANDLER=1','--setenv=OMP_NUM_THREADS=16','--setenv=MKL_NUM_THREADS=16','--setenv=OPENBLAS_NUM_THREADS=1','--setenv=LD_LIBRARY_PATH='+str(ROOT/'bam/build/lib'),PYTHON,'-B','-u','-m','candidates.uks_group_incremental_v1.controller','--output',str(output)]
    return dict(command=cmd,unit=unit+'.service',output=str(output),source_sha256=verify(),reused_gpu_check=True,performance_workers=10,diagnostic_workers=2,raw_ssd_writes=False)
def main():
    a=argparse.ArgumentParser();a.add_argument('--execute',action='store_true');args=a.parse_args();check_ready()
    parity=read(OUT/'gpu_check/report.json');require(parity['passed'] and parity['source_sha256']==verify(),'GPU parity not accepted')
    require(read(OUT/'gpu_check/exit.json')['returncode']==0,'GPU parity exit failed')
    run_id=datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:10];p=plan(run_id)
    print(json.dumps(dict(p,execute=args.execute),indent=2),flush=True)
    if args.execute:
        require(os.geteuid()==0,'Local sudo required')
        from candidates.uks_revpr_diagnostic_v1.worker import admission
        admission()  # Do not create another service if GPU/host headroom is insufficient.
        Path(p['output']).mkdir(parents=True,exist_ok=False)
        write(Path(p['output'])/'launch.json',p)
        subprocess.run(p['command'],check=True)
        write(OUT/'latest.json',p)
if __name__=='__main__':main()
