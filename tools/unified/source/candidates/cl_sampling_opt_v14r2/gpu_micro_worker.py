"""Bounded CUDA parity: tiny owned anonymous arenas; no raw SSD access."""
import argparse,copy,faulthandler,json,os,time,unittest
faulthandler.enable()
faulthandler.dump_traceback_later(30, repeat=True)
print("micro: importing numpy/torch", flush=True)
from pathlib import Path
import numpy as np
import torch
print("micro: configuring threads before DGL imports", flush=True)
torch.set_num_threads(1)
print("micro: importing sampling modules", flush=True)
from candidates.ukl_native_sampling_v10 import sampling as S
from candidates.ukl_training_adapter_v12.fixture import OwnedCPU
from candidates.ukl_real_multi_v9.memory import Registration
from candidates.ukl_runtime_prepare_v11r1.data import features
from candidates.ukl_postprocess_v22 import device as old
from .block_checks import equal_blocks
from . import device as new,protocol as P
from .checks import fixtures
from .training import prepare_gpu_model
from .gpu import verify_cuda_uuid


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args()
    if os.environ.get('CL_OPT_MICRO')!='1':raise RuntimeError('Use bounded micro service')
    print("micro: preparing GPU model", flush=True)
    model,_=prepare_gpu_model(11)
    print("micro: model ready", flush=True)
    verify_cuda_uuid(os.environ['CUDA_VISIBLE_DEVICES'])
    test=unittest.TestCase();cases=blocks_cases=training_steps=0;peak_arena=0
    for index,(arrays,meta) in enumerate(fixtures()):
        print("micro: fixture",index,"start",flush=True)
        models=[copy.deepcopy(model) for _ in range(2)]
        opts=[torch.optim.Adam(m.parameters(),lr=.001) for m in models]
        with OwnedCPU(arrays,meta) as owned:
            if owned.arena.size>2*2**20:raise RuntimeError('Tiny graph bound exceeded')
            peak_arena=max(peak_arena,owned.arena.size)
            reg=Registration(owned.arena);natives=[]
            try:
                natives.append(old.Native(owned.graph,reg));natives.append(new.Native(owned.graph,reg))
                seeds=np.array([0,1,2,4,6],np.int32)
                for grouped in (False,True):
                    for fanout in (1,2,3,5,10,31,32):
                        print("micro: sampling", index, grouped, fanout, flush=True)
                        for seed in (0,1,19,2**63+5,2**64-1):
                            reference=owned.native.sample(seeds,fanout,grouped,seed,packed=True)
                            for native in natives:
                                actual=native.sample(seeds,fanout,grouped,seed,packed=True)
                                for k in reference:np.testing.assert_array_equal(actual[k],reference[k],err_msg=k)
                            cases+=1
                for grouped in (False,True):
                    for step in range(3):
                        print("micro: blocks/model", index, grouped, step, flush=True)
                        roots=np.array([0,2,4],np.int64)
                        left=old.Sampler(natives[0],grouped,seed=17).sample_blocks(roots,step)
                        right=new.Sampler(natives[1],grouped,seed=17).sample_blocks(roots,step)
                        test.assertTrue(torch.equal(left[0],right[0]));equal_blocks(test,left[2],right[2]);blocks_cases+=1
                        if grouped:
                            for m,opt,result in zip(models,opts,(left,right)):
                                ids,_,blocks=result;x=torch.from_numpy(features(ids.numpy())).cuda()
                                torch.cuda.manual_seed(501+step)
                                loss=torch.nn.functional.cross_entropy(m(blocks,x),torch.tensor([0,1,2],device='cuda:0'))
                                opt.zero_grad(set_to_none=True);loss.backward();opt.step()
                            for p,q in zip(models[0].parameters(),models[1].parameters()):
                                torch.testing.assert_close(p,q,rtol=1e-5,atol=1e-6)
                                torch.testing.assert_close(p.grad,q.grad,rtol=1e-5,atol=1e-6)
                            training_steps+=1
                        del left,right
                torch.cuda.synchronize()
            finally:
                for native in natives:native.close()
                reg.close()
        del models,opts
        print("micro: fixture released", index, flush=True)
    r=dict(passed=True,time=time.time(),cases=cases,block_cases=blocks_cases,training_steps=training_steps,
           anonymous_graph_bytes=peak_arena,graph_released=True,raw_ssd_access=False,
           full_graph_test=False,selected_uuid=os.environ['CUDA_VISIBLE_DEVICES'],manifest_sha256=P.verify_manifest())
    Path(a.out).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
    faulthandler.cancel_dump_traceback_later()
if __name__=='__main__':main()
