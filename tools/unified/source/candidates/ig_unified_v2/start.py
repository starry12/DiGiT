import argparse,datetime,os,subprocess,json
from .common import ROOT,OUT,verify,write
def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['short1','short5','full20'],default='short1');p.add_argument('--execute',action='store_true');a=p.parse_args()
    stamp=datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+str(os.getpid())
    unit='digit-ig-unified-'+stamp;output=OUT/'runs'/stamp
    cmd=['/home/embed/miniconda3/envs/gids/bin/python','-B','-u','-m','candidates.ig_unified_v2.launcher','--output',str(output),'--mode',a.mode]
    plan=dict(source_sha256=verify(),unit=unit,output=str(output),mode=a.mode,command=cmd,poll_seconds=60,
              schedule='tiny GPU checks, new BFS preparation, source-checked smokes, then requested paired mode')
    if a.execute:
        if os.geteuid()!=0:raise RuntimeError('Local sudo required for isolated native service')
        subprocess.run(['systemd-run','--unit='+unit,'--property=Type=exec','--property=Restart=no','--property=WorkingDirectory='+str(ROOT),
                        '--property=KillMode=control-group']+cmd,check=True)
        write(OUT/'latest.json',plan)
    print(json.dumps(plan,indent=2))
if __name__=='__main__':main()
