"""Plan by default; execute only after accepted raw SSD preparation."""
import argparse,json,os
from . import protocol as P
from .phase import INIT_LIMIT,DATA_LIMIT
def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args()
    digest=P.verify_manifest();P.configure_stage('gids')
    try:prior=dict(ready=True,evidence=P.require_predecessor())
    except (RuntimeError,OSError,ValueError,KeyError) as e:prior=dict(ready=False,error=str(e))
    if not a.execute:
        print(json.dumps(dict(execute=False,manifest_sha256=digest,batches_per_arm=4,arms=list(P.STAGES),budgets={s:P.budget(s) for s in P.STAGES},phase_accounting=dict(initialization_file_limit=INIT_LIMIT,data_file_limit=DATA_LIMIT,aggregate_file_limit=INIT_LIMIT+DATA_LIMIT,probe_bytes_each=4*P.MIB,kernel_charge_handoff_required=True),predecessor=prior,raw_ssd_writes=False,performance_run=False,monitoring=dict(gpu_interval_seconds=10,gpu_query_timeout_is_fatal=False,attribution=False,separate_qualification=False,successful_post_seconds=30,failure_post_seconds=180)),indent=2));return
    if not prior['ready']:raise RuntimeError('SSD and initialization acceptance are required: '+prior['error'])
    if os.geteuid()!=0:raise RuntimeError('sudo required for bounded GPU/SSD smoke')
    from .start import execute
    execute()
if __name__=='__main__':main()
