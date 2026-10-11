"""Plan by default; execute only after accepted raw SSD preparation."""
import argparse,json,os
from . import protocol as P
from .phase import INIT_LIMIT,DATA_LIMIT
def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args()
    if P.VARIANT!='gpu':raise RuntimeError('The accepted CL implementation uses GPU postprocessing for DiGiT')
    digest=P.verify_manifest();P.configure_stage('gids')
    try:prior=dict(ready=True,evidence=P.require_predecessor())
    except (RuntimeError,OSError,ValueError,KeyError) as e:prior=dict(ready=False,error=str(e))
    if not a.execute:
        print(json.dumps(dict(execute=False,numa_placement=dict(gids='unchanged',digit='VMA interleave physical nodes 0,1 before first touch',digit_regions=['graph','auxiliary','cpu_features'],digit_cpu_affinity=[2],migrate_existing_pages=False,pressure_thresholds_changed=False),sampler_variant=P.VARIANT,gids_postprocessing="gpu_common_v16",digit_postprocessing=P.VARIANT,manifest_sha256=digest,rounds=P.ROUNDS,warmup_batches=P.WARMUP,measured_batches=P.MEASURED,arms=list(P.STAGES),budgets={s:P.budget(s) for s in P.STAGES},io_geometry=dict(row_bytes=512,cache_page_bytes=512,gids_reads=[512],digit_reads=[512,1024],gids_policy='legacy',digit_policy='fifo',digit_cpu=2,sampler_changed=False),gpu_budgets={s:__import__('candidates.cl_common_gpu_five_v17.budget',fromlist=['budget']).budget(s) for s in P.STAGES},phase_accounting=dict(initialization_file_limit=INIT_LIMIT,data_file_limit=DATA_LIMIT,aggregate_file_limit=INIT_LIMIT+DATA_LIMIT,probe_bytes_each=4*P.MIB,kernel_charge_handoff_required=True),predecessor=prior,raw_ssd_writes=False,performance_run=True,profiling_run=False,monitoring=dict(gpu_interval_seconds=10,gpu_query_timeout_is_fatal=False,attribution="selected_gpu_uuid_pci",unattributed_thread_timeout="record_only",system_faults="fatal",separate_qualification=False,successful_post_seconds=30,failure_post_seconds=180)),indent=2));return
    if not prior['ready']:raise RuntimeError('SSD write/readback acceptance are required: '+prior['error'])
    if os.geteuid()!=0:raise RuntimeError('sudo required for bounded GPU/SSD performance')
    from .start import execute
    execute()
if __name__=='__main__':main()
