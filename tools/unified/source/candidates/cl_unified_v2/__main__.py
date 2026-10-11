import argparse,json,os
from .common import *
from candidates.unified_training_v2.config import MODES

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=tuple(MODES),default='short1');p.add_argument('--execute',action='store_true');a=p.parse_args()
    source=verify();ready=Path('/mnt/n0/digit/group_cache_migration_v1/CL/prepared.json').is_file()
    plan=dict(candidate='CL unified',config=config(a.mode),source_sha256=source,
        jobs=list(MODES[a.mode].schedule()),prepared_receipt_present=ready,native_execution_accepted=False,
        preserves_numa_balance=True,anonymous_graph_only=True,accuracy_evaluated=False,
        raw_ssd_writes=False,execute=a.execute)
    if not a.execute:print(json.dumps(plan,indent=2));return
    require(os.geteuid()==0,'Bounded controller requires sudo')
    require(ready,'CL v2 adjacency is not prepared; complete the existing CL preparation first')
    from candidates.group_cache_migration_v1.common import stable
    stable('CL')
    os.environ['CL_UNIFIED_MODE']=a.mode
    from .runtime.start import execute
    execute()

if __name__=='__main__':main()
