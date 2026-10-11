import argparse,datetime,json,os,subprocess
from .common import *
from .registry import DATASETS,plan
from candidates.unified_training_v2.config import MODES

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=DATASETS,required=True)
    p.add_argument('--mode',choices=tuple(MODES),default='short1');p.add_argument('--execute',action='store_true');p.add_argument('--status',action='store_true');a=p.parse_args()
    latest=OUT/a.dataset/'latest.json'
    if a.status:
        v=read(latest);folder=Path(v['output']);v['status']=read(folder/'status.json') if (folder/'status.json').exists() else {'stage':'starting'}
        if (folder/'backend.json').exists():
            v['backend']=read(folder/'backend.json');q=Path(v['backend']['output'])/'status.json'
            if q.exists():v['backend_status']=read(q)
        print(json.dumps(v,indent=2));return
    source=verify();v=plan(a.dataset,a.mode);stamp=datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+str(os.getpid())
    folder=OUT/a.dataset/'runs'/stamp;unit='digit-unified-v2-'+a.dataset.lower()+'-'+stamp
    cmd=['systemd-run','--unit='+unit,'--service-type=exec','--property=WorkingDirectory='+str(ROOT),
        '--property=Restart=no','--property=KillMode=control-group','--property=LimitMEMLOCK=infinity',
        '--property=MemorySwapMax=0','--setenv=CUDA_VISIBLE_DEVICES=2','--setenv=OMP_NUM_THREADS=1',
        '--setenv=MKL_NUM_THREADS=1','--setenv=OPENBLAS_NUM_THREADS=1','--setenv=PYTHONDONTWRITEBYTECODE=1',
        '--setenv=LD_LIBRARY_PATH='+str(ROOT/'bam/build/lib'),PY,'-B','-u','-m','candidates.unified_experiments_v2.run',
        '--dataset',a.dataset,'--mode',a.mode,'--output',str(folder)]
    if a.dataset in ('PA','UKS'):cmd.insert(2,'--property=MemoryMax=384G')
    v.update(source_sha256=source,unit=unit,output=str(folder),command=cmd,execute=a.execute)
    if a.execute:
        if os.geteuid()!=0:raise RuntimeError('Use sudo for the bounded service')
        if v['missing_preparation']:raise RuntimeError('Missing accepted preparation: '+', '.join(v['missing_preparation']))
        folder.mkdir(parents=True,exist_ok=False);write(folder/'plan.json',v)
        subprocess.run(cmd,check=True);write(latest,v)
    print(json.dumps(v,indent=2))
if __name__=='__main__':main()
