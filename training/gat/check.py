"""GAT model/CUDA admission and complete input hashes, before any native SSD run."""
import argparse,os,time,subprocess,sys
from training.gat.common import *
from training.gat.inputs import input_binding,compatibility

def checked_child(module,output):
    log=output/(module.rsplit('.',1)[-1]+'.log')
    with log.open('x') as f:
        code=subprocess.call([sys.executable,'-u','-m',module],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=environment())
    require(code==0,'Check failed; inspect '+str(log))
    # All check modules print one final JSON object after diagnostics.
    value=json.JSONDecoder().raw_decode(log.read_text()[log.read_text().rfind('\n{')+1:])[0]
    require(value['passed'],'Model check did not pass')
    return value

def check(output,gpu):
    from evaluation.gat.common import verify as verify_entry
    candidate=verify();entry=verify_entry();output=output_path(output);output.mkdir(parents=True)
    start=time.time();state=dict(schema='digit-pa-gat-preflight-v1',passed=False,complete=False,candidate_sha256=candidate,
        entry_manifest_sha256=entry,gpu=gpu,stage='model_cpu_checks',raw_ssd_access=False,started_unix=start)
    def save(**kw):state.update(kw,updated_unix=time.time());write(output/'status.json',state)
    save()
    try:
        model_checks=checked_child('training.gat.tests',output);write(output/'model_checks.json',model_checks)
        save(stage='cuda_model_sampler_checks')
        cuda_checks=checked_child('training.gat.cuda_checks',output);write(output/'cuda_checks.json',cuda_checks)
        save(stage='maximum_batch_attention_memory')
        memory_checks=checked_child('training.gat.memory_checks',output);write(output/'memory_checks.json',memory_checks)
        save(stage='model_budget');setup()
        from training.gat.admission import estimate
        from digit.gpu_admission import check_live
        import torch
        require(torch.cuda.device_count()==1 and torch.cuda.get_device_capability()==(8,9),'Expected one visible sm89 GPU')
        plan=check_live(estimate());require(plan['passed'],'Live GAT GPU/host budget failed')
        require(plan['free_bytes']>plan['total_bytes']-2**30,'Selected GPU already in use')
        write(output/'admission.json',plan)
        save(stage='hashing_unchanged_pa_inputs',live_admission=plan,model_checks=model_checks,cuda_checks=cuda_checks,memory_checks=memory_checks)
        binding=input_binding(output,candidate,require_preflight=False)
        # Manifest geometry and sampling contract are model-independent. Actual native
        # DiGiT workers independently rehash/validate the full g2 payload before use.
        from ae.pa_sage.trace_cache import PreparedEvaluationTrace
        p=cfg();traces={}
        for split,key in (('valid','validation_trace'),('test','test_trace')):
            trace=PreparedEvaluationTrace(ROOT/p[key],sha(ROOT/p[key]/'manifest.json'))
            trace.validate_contract(p['graph']['nodes'],p['graph']['edges'],p['fanouts'],p['batch_size'],splits(split))
            require(trace.budget['required_bytes']<=8*2**30,'Individual trace exceeds host reserve')
            traces[split]=dict(manifest_sha256=trace.manifest_sha256,budget=trace.budget)
        require(sum(v['budget']['required_bytes'] for v in traces.values())<=16*2**30,'Trace budget exceeded')
        write(output/'trace_contracts.json',traces)
        require(verify()==candidate and verify_entry()==entry,'Code changed during preflight')
        save(passed=True,complete=True,stage='complete',compatibility=compatibility(),
             evidence_sha256={n:sha(output/n) for n in ('inputs.json','parent_data_inputs.json','model_checks.json','cuda_checks.json','memory_checks.json','memory_checks.log','admission.json','trace_contracts.json','tests.log','cuda_checks.log')},
             seconds=time.time()-start,finished_unix=time.time(),
             scope='GAT math/CUDA tests and maximum-batch attention allocation, selected PA graph/source input hashes, trace payload hashes/contracts and live resource admission. Existing g2 SSD payload reused; full g2 artifact verification also runs inside native DiGiT workers. No raw SSD access or new performance evidence.')
    except BaseException as exc:save(stage='failed',error=type(exc).__name__+': '+str(exc));raise
    return state

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args()
    require(a.gpu>=0,'Invalid GPU');os.environ.update(environment(a.gpu));print(json.dumps(check(a.output,a.gpu),indent=2))
if __name__=='__main__':main()
