"""Prepare only after the existing shared experiments release their locks."""
import argparse,datetime,json,os,subprocess,sys,time
from pathlib import Path
from .build import ROOT,OUT
def run():
    from .provenance import verify
    from candidates.ig_demand_pair_v1.launcher import admission
    from candidates.group_cache_migration_v1.prepare import check_prepared
    from .convert import main as unused
    source=verify();base=OUT.parent;state=base/'preparation_status.json'
    def save(**kw):state.write_text(json.dumps(dict(pid=os.getpid(),source_sha256=source,time=time.time(),**kw),indent=2))
    while True:
        with admission() as reason:
            if reason is None:
                save(stage='converting',passed=False)
                check_prepared('UKL')
                try:
                    subprocess.run([sys.executable,'-B','-u','-m','candidates.ukl_compact_cover_v1.convert',
                                    '--prepared','/mnt/n0/digit/group_cache_migration_v1/UKL/prepared.json',
                                    '--output','/mnt/n0/digit/ukl_compact_cover_20261011_v1/covered.u32','--execute'],check=True,cwd=ROOT)
                    from .binding import binding
                    binding();verify();save(stage='complete',passed=True,full_graph_gpu_acceptance=False)
                except BaseException as e:save(stage='failed',passed=False,error=repr(e));raise
                return
            save(stage='waiting',passed=False,reason=reason,poll_seconds=60)
        time.sleep(60)
def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--worker',action='store_true');a=p.parse_args()
    if a.worker:run();return
    from .provenance import verify
    verify();stamp=datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+str(os.getpid())
    unit='digit-ukl-compact-prepare-'+stamp
    cmd=['/home/embed/miniconda3/envs/gids/bin/python','-B','-u','-m','candidates.ukl_compact_cover_v1.prepare_start','--worker']
    if a.execute:
        if os.geteuid()!=0:raise RuntimeError('Local sudo required for bounded background service')
        subprocess.run(['systemd-run','--unit='+unit,'--property=Type=exec','--property=Restart=no',
                        '--property=WorkingDirectory='+str(ROOT),'--property=KillMode=control-group',
                        '--property=MemoryMax=8G','--property=MemorySwapMax=0','--property=CPUQuota=200%',
                        '--property=Nice=10','--property=IOWeight=10']+cmd,check=True)
    print(json.dumps(dict(unit=unit,command=cmd,poll_seconds=60,starts_after_shared_locks_release=True,original_files_preserved=True),indent=2))
if __name__=='__main__':main()
