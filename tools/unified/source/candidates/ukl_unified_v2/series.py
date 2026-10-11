"""Same-round accepted workers, five-pair median or 20-epoch total ratio."""
import os
from pathlib import Path
from .common import config,read,write,sha,require
from candidates.unified_training_v2.series import run_series

def execute_series(protocol,run_id,run_one):
    P=protocol;c=config(P.SCOPE);receipts=[];folder=P.OUT/run_id;folder.mkdir(parents=True,exist_ok=False)
    write(folder/'config.json',c)
    os.environ.pop('UKL_UNIFIED_PROFILE',None)
    def launch(config,arm,repetition):
        P.REPETITION=repetition;P.PAIR_ID=run_id+'_r'+str(repetition);P.configure_stage(arm)
        entry=run_one();a=read(entry['path'])
        require(sha(entry['path'])==entry['sha256'] and a['passed'] and a['pair_id']==P.PAIR_ID,'Wrong native acceptance')
        require(a['gpu_released'] and a['kernel_monitor_ok'] and a['post_guard_passed'],'Incomplete cleanup/guards')
        state=a['worker_state'];require(state['Result']=='success' and state['ExecMainStatus']=='0' and state['MainPID']=='0','Worker exit failed')
        r=a['worker']['common_training']
        if arm=='digit':os.environ['UKL_UNIFIED_PROFILE']=str(Path(entry['path']).parent/'profile_binding.json')
        receipt=dict(report=r,returncode=0,cleanup_verified=True,admission_verified=True,correctness_verified=True,
                     acceptance=entry)
        receipts.append(receipt);return receipt
    result=run_series(c,launch,lambda state:write(folder/'series.json',state))
    # Enriched common summary keeps all rounds and denominators, not only speedup.
    from candidates.unified_results_v2.report import summarize
    result=summarize(c,receipts);write(folder/'summary.json',result);write(P.OUT/'latest_summary.json',result)
    from candidates.unified_results_v2.__main__ import export_csv
    export_csv(folder/'summary.csv',result)
    return result
