"""IG math/CUDA/memory admission and data bindings; no raw SSD access."""
import argparse,subprocess
from candidates.ig_perf_v5.common import *
from candidates.ig_perf_v5.inputs import bind
def checked_child(module,output):
    log=output/((module.replace('.','_') if module.startswith('candidates.ig_monitor_v1.') else module.rsplit('.',1)[-1])+'.log')
    with log.open('x') as f:code=subprocess.call([sys.executable,'-u','-m',module],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=environment())
    require(code==0,'Check failed: '+str(log));text=log.read_text();at=text.rfind('\n{')+1;value=json.JSONDecoder().raw_decode(text[at:])[0];require(value['passed'],'Check did not pass');return value
def check(output,gpu):
    from submission.v9.common import verify as verify_entry
    execution=verify();entry=verify_entry();output=output_path(output);output.mkdir(parents=True);start=time.time()
    state=dict(schema='digit-ig-perf-preflight-v1',passed=False,complete=False,candidate_sha256=execution,entry_manifest_sha256=entry,gpu=gpu,raw_ssd_access=False,started_unix=start)
    def save(**kw):state.update(kw,updated_unix=time.time());write(output/'status.json',state);print(json.dumps(dict(stage=state.get('stage'),passed=state['passed'])),flush=True)
    try:
        save(stage='monitor_cpu_checks');write(output/'monitor_checks.json',checked_child('candidates.ig_monitor_v1.tests',output));write(output/'monitor_review_checks.json',checked_child('candidates.ig_monitor_v1.review_tests',output))
        save(stage='model_cpu_checks');write(output/'model_checks.json',checked_child('candidates.ig_perf_v5.tests',output))
        save(stage='cuda_sampler_and_4k_io_checks');write(output/'cuda_checks.json',checked_child('candidates.ig_perf_v5.cuda_checks',output))
        save(stage='maximum_batch_memory');write(output/'memory_checks.json',checked_child('candidates.ig_perf_v5.memory_checks',output))
        setup();import torch,dgl
        from candidates.ig_perf_v5.admission import check_live
        plan=check_live();require(plan['passed'] and plan['free_bytes']>plan['total_bytes']-2**30,'GPU/host unavailable')
        write(output/'admission.json',plan)
        write(output/'environment.json',dict(python=sys.version,executable=sys.executable,torch=torch.__version__,dgl=dgl.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),native_sha256=sha(HERE/'runtime/IGPerfNative.so')))
        save(stage='prepare_roots_and_hash_ig_inputs',live_admission=plan);bind(output,execution,preflight=True)
        require(verify()==execution and verify_entry()==entry,'Code changed during preflight')
        evidence=['monitor_checks.json','monitor_review_checks.json','candidates_ig_monitor_v1_tests.log','candidates_ig_monitor_v1_review_tests.log','model_checks.json','cuda_checks.json','memory_checks.json','admission.json','environment.json','inputs.json','tests.log','cuda_checks.log','memory_checks.log']
        save(stage='complete',passed=True,complete=True,evidence_sha256={n:sha(output/n) for n in evidence},seconds=time.time()-start,finished_unix=time.time(),scope='Persistent NVML monitor lifecycle/strict-review regressions, model math, native 4KiB CPU routing/useful counters, real CUDA/UVA sampler parity, max-batch allocation and metadata/CSC hashes. TB feature/SSD readback receipts inherited; CPU cache fully hashed during each worker. Native paired smoke and bounded timed windows not yet run.')
    except BaseException as exc:save(stage='failed',error=type(exc).__name__+': '+str(exc));raise
    return state
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args();require(a.gpu>=0,'Invalid GPU');os.environ.update(environment(a.gpu));print(json.dumps(check(a.output,a.gpu),indent=2))
if __name__=='__main__':main()
