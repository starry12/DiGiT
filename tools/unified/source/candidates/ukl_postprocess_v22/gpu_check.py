"""Tiny real CUDA sampling/blocks test; no raw SSD and <2 MiB anonymous graph."""
import copy,json,os,time,unittest
from pathlib import Path
import numpy as np
import torch
from candidates.ukl_native_sampling_v10 import sampling as S
from candidates.ukl_native_sampling_v10.test_sampling import fixture
from candidates.ukl_training_adapter_v12.fixture import OwnedCPU
from candidates.ukl_real_multi_v9.memory import Registration
from candidates.ukl_mixed512_v21.training import prepare_gpu_model
from candidates.ukl_runtime_prepare_v11r1.data import features
from . import host,buffer,device
from .request import request_from_blocks
from .test_host import equal_blocks


def main():
    from candidates.ukl_mixed512_v21.gpu import check_idle
    expected='GPU-48169b5b-94da-de49-94f9-64cf8e4fd8f2'
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=expected:raise RuntimeError('Tiny qualification is restricted to the separately checked idle GPU0')
    check_idle((0,expected))
    torch.set_num_threads(1)
    model,_=prepare_gpu_model(11)
    models=[copy.deepcopy(model) for _ in range(3)]
    opts=[torch.optim.Adam(m.parameters(),lr=.001) for m in models]
    test=unittest.TestCase();cases=0;peak_arena=0
    arrays,meta=fixture(high=True)
    with OwnedCPU(arrays,meta) as owned:
        if owned.arena.size>2*2**20:raise RuntimeError('Tiny GPU test bound')
        peak_arena=owned.arena.size
        registration=Registration(owned.arena)
        natives=[]
        try:
            natives=[S.Native(owned.graph,registration),buffer.Native(owned.graph,registration),device.Native(owned.graph,registration)]
            for grouped in (False,True):
                for batch,roots in enumerate(([1],[0,2,4],[4,0,1,6])):
                    roots=np.array(roots,np.int64)
                    reference=S.Sampler(owned.native,grouped,seed=17).sample_blocks(roots,batch)
                    for native,cls in zip(natives,(host.Sampler,host.Sampler,device.Sampler)):
                        actual=cls(native,grouped,seed=17).sample_blocks(roots,batch)
                        test.assertTrue(torch.equal(reference[0],actual[0]));equal_blocks(test,reference[2],actual[2])
                        req=request_from_blocks('digit' if grouped else 'gids',owned.graph,*actual,roots)
                        np.testing.assert_array_equal(req.logical_ids,reference[0].numpy())
                        cases+=1
            for step in range(3):
                for m,opt,native,cls in zip(models,opts,natives,(host.Sampler,host.Sampler,device.Sampler)):
                    ids,_,blocks=cls(native,True,seed=17).sample_blocks(np.array([0,2,4],np.int64),step)
                    x=torch.from_numpy(features(ids.numpy())).cuda();blocks=[b.to('cuda:0') for b in blocks]
                    torch.cuda.manual_seed(501+step)
                    loss=torch.nn.functional.cross_entropy(m(blocks,x),torch.tensor([0,1,2],device='cuda:0'))
                    opt.zero_grad(set_to_none=True);loss.backward();opt.step()
                for m in models[1:]:
                    for a,b in zip(models[0].parameters(),m.parameters()):
                        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
                        torch.testing.assert_close(a.grad,b.grad,rtol=1e-5,atol=1e-6)
            del reference,actual,req,ids,blocks
        finally:
            # A failed close raises before registration/graph ownership release.
            for native in natives:native.close()
            registration.close()
    r=dict(passed=True,time=time.time(),cases=cases,training_steps=3,
           gradient_parameter_rtol=1e-5,gradient_parameter_atol=1e-6,
           anonymous_graph_bytes=peak_arena,graph_released=True,raw_ssd_access=False,
           full_graph_test=False,gpu=os.environ.get('CUDA_VISIBLE_DEVICES'))
    Path('results/ukl_postprocess_20261004_v22/gpu_check.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r))

if __name__=='__main__':main()
