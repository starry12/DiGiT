"""Both UKL arms: preserve old raw sampling, blocks and training with common GPU execution."""
import argparse,copy,faulthandler,json,os,time,unittest
faulthandler.enable();faulthandler.dump_traceback_later(30,repeat=True)
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
print('common GPU micro: imports',flush=True)
from candidates.ukl_native_sampling_v10 import sampling as frozen
from candidates.ukl_postprocess_v22 import device
from candidates.ukl_training_adapter_v12.fixture import OwnedCPU
from candidates.ukl_real_multi_v9.memory import Registration
from candidates.ukl_training_adapter_v12.adapter import request_from_blocks as old_request
from candidates.ukl_postprocess_v22.request import request_from_blocks as new_request
from candidates.ukl_runtime_prepare_v11r1.data import features
from .block_checks import equal_blocks
from .checks import fixtures
from .training import prepare_gpu_model
from .gpu import verify_cuda_uuid
from . import protocol as P,sampling

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args()
    if os.environ.get('UKL_COMMON_MICRO')!='1':raise RuntimeError('Use bounded GPU micro service')
    verify_model,_=prepare_gpu_model(11);verify_cuda_uuid(os.environ['CUDA_VISIBLE_DEVICES'])
    if sampling.select('gids')!=(device.Native,device.Sampler):raise RuntimeError('Wrong production GIDS route')
    test=unittest.TestCase();cases=block_cases=training_steps=0;peak=0
    for arm in ('gids','digit'):
        grouped=arm=='digit'
        print('arm',arm,flush=True)
        for fixture,(arrays,meta) in enumerate(fixtures()):
            print('fixture',fixture,'begin',flush=True)
            models=[copy.deepcopy(verify_model) for _ in range(2)]
            opts=[torch.optim.Adam(m.parameters(),lr=.001) for m in models]
            with OwnedCPU(arrays,meta) as owned:
                peak=max(peak,owned.arena.size)
                reg=Registration(owned.arena);natives=[]
                try:
                    old=frozen.Native(owned.graph,reg);natives.append(old)
                    NewNative,NewSampler=sampling.select(arm)
                    new=NewNative(owned.graph,reg);natives.append(new)
                    if arm=='gids' and new.lib._name!=old.lib._name:raise RuntimeError('Native GIDS sampler changed')
                    if arm=='digit' and new.lib._name==old.lib._name:raise RuntimeError('Optimized DiGiT sampler missing')
                    pointers=None
                    for f in (1,2,3,5,10,31,32):
                        print('raw parity',fixture,f,flush=True)
                        for seed in (0,1,19,2**63+5,2**64-1):
                            roots=np.array([0,1,2,4,6],np.int32)
                            want=old.sample(roots,f,grouped,seed,packed=grouped)
                            got=new.sample(roots,f,grouped,seed,packed=grouped)
                            for k in want:np.testing.assert_array_equal(got[k],want[k],err_msg=k)
                            current={k:v.data_ptr() for k,v in new.buffers.items()}
                            if pointers is None:pointers=current
                            if current!=pointers:raise RuntimeError('Sampling buffers were reallocated')
                            cases+=1
                    left=frozen.Sampler(old,grouped,seed=17);right=NewSampler(new,grouped,seed=17)
                    for step in range(3):
                        print('block/model parity',fixture,step,flush=True)
                        roots=np.array([0,1,2],np.int64)
                        before=left.sample_blocks(roots,step);after=right.sample_blocks(roots,step)
                        for x,y in zip(before[:2],after[:2]):test.assertTrue(torch.equal(x,y))
                        equal_blocks(test,before[2],after[2]);block_cases+=1
                        br=old_request(arm,owned.graph,*before,roots)
                        ar=new_request(arm,owned.graph,*after,roots)
                        np.testing.assert_array_equal(br.logical_ids,ar.logical_ids)
                        np.testing.assert_array_equal(br.storage_rows,ar.storage_rows)
                        for model,opt,result,request in zip(models,opts,(before,after),(br,ar)):
                            x=torch.from_numpy(features(request.logical_ids)).cuda()
                            blocks=[b.to('cuda:0') for b in result[2]]
                            torch.cuda.manual_seed(501+step)
                            loss=torch.nn.functional.cross_entropy(model(blocks,x),torch.tensor([0,1,2],device='cuda:0'))
                            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
                        for p,q in zip(models[0].parameters(),models[1].parameters()):
                            torch.testing.assert_close(p,q,rtol=1e-4,atol=1e-5)
                            torch.testing.assert_close(p.grad,q.grad,rtol=1e-4,atol=1e-5)
                        training_steps+=1
                        del before,after,blocks,result,request,br,ar,x,loss
                    torch.cuda.synchronize()
                finally:
                    for native in natives:native.close()
                    reg.close()
            del models,opts
            print('fixture',fixture,'released',flush=True)
    r=dict(passed=True,time=time.time(),cases=cases,block_cases=block_cases,training_steps=training_steps,
           buffers_reused=True,gids_native_unchanged=True,digit_native_optimized=True,anonymous_graph_bytes=peak,
           graph_released=True,raw_ssd_access=False,full_graph_test=False,
           selected_uuid=os.environ['CUDA_VISIBLE_DEVICES'],manifest_sha256=P.verify_manifest())
    Path(a.out).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
    faulthandler.cancel_dump_traceback_later()
if __name__=='__main__':main()
