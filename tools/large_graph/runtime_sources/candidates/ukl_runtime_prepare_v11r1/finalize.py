"""Join accepted stage receipts; never touch a raw SSD or launch training."""
import hashlib
import json
from . import protocol as P

def collect_inputs():
    saved=P.STAGE;stages={};digest=P.verify_manifest()
    try:
        for stage in P.STAGES:
            P.configure_stage(stage);runs=sorted(P.OUT.glob('stage_'+stage+'_run_*'))
            if not runs:raise RuntimeError('Missing completed stage '+stage)
            path=runs[-1]/'acceptance.json';a=json.loads(path.read_text())
            if (not a['passed'] or a['manifest_sha256']!=digest or not a['kernel_monitor_ok']
                    or not a['post_guard_passed'] or not P.worker_passed(a['worker_state'],a['worker'])):
                raise RuntimeError('Stage not accepted: '+stage)
            stages[stage]=dict(run=str(runs[-1]),acceptance_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                               files=a['worker']['files'],predecessor=a['worker']['predecessor'])
        for stage in P.STAGES[1:]:
            if stages[stage]['predecessor']['graph_receipt_sha256']!=stages['graph']['acceptance_sha256']:
                raise RuntimeError('Feature file belongs to different graph preparation')
        return dict(prepared=True,manifest_sha256=digest,stages=stages,nodes=P.N,storage_rows=P.ROWS,
                    feature_dim=128,feature_dtype='float32',row_bytes=512,classes=19,
                    gpu_cache_bytes=4*P.GIB,cpu_cache_rows=P.N//10,
                    gids_cpu_hot='gids_hot.i64 (RevPR)',digit_cpu_hot='freq_hot.i64',
                    roots_format='raw int64, 320 batches x 1024, first 20 warmup then 300 measured',
                    rounds=5,selection='maximum same-round speedup',accuracy=False,
                    raw_ssd_written=False,raw_ssd_bound=False,native_training_ready=False)
    finally:P.configure_stage(saved)

def main():
    from .start import write
    result=collect_inputs();write(P.OUT/'runtime_inputs.json',result)
    print(json.dumps(dict(prepared=True,output=str(P.OUT/'runtime_inputs.json'),raw_ssd_bound=False)))

if __name__=='__main__':main()
