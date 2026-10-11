"""Bounded accumulation through CUDA grouped sampling/model; feature reads simulated."""
import tempfile
import subprocess
from .common import *


def main():
    require(not (OUT/'gpu_checks.json').exists(),'Preserve evidence')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='2','Use physical GPU2')
    fields=subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=uuid,memory.used',
                                   '--format=csv,noheader,nounits'],text=True).strip().split(',')
    require(fields[0].strip()==GPU_UUID and int(fields[1])<1024,'GPU busy or changed')
    setup()
    import numpy as np
    import torch
    import dgl
    from .loader import BoundedGIDS
    from candidates.pa_sage_512b_pair_v1.tests import fixture
    from candidates.pa_sage_512b_pair_v1.runtime_helpers import verify_digit_blocks
    from digit.sampler import DiGiTNeighborSampler
    from models import SAGE
    torch.manual_seed(0);dgl.seed(0)
    with tempfile.TemporaryDirectory(prefix='digit-bounded-window-') as folder:
        graph,arrays,bundle,features=fixture(folder,pin=True)
        sampler=DiGiTNeighborSampler([4,3,2],bundle,cuda_mode='required',metadata_mode='gpu_i32_uva_eid64')
        sampler._ensure_cuda_metadata(graph,torch.device('cuda:0'))
        loader=BoundedGIDS.__new__(BoundedGIDS)
        loader.accumulator_flag=loader.window_buffering_flag=True;loader.heterograph=False
        loader.wb_size=2;loader.required_accesses=6832.4;loader.cache_dim=128;loader.gids_device='cuda:0'
        loader.feature_index_mode='explicit';loader.graph_reorganize=False;loader.feature_map=None
        loader.strict_feature_row_validation=True
        loader.feature_index_batches={k:0 for k in ('logical','explicit','legacy_map')}
        loader.feature_index_rows=dict(loader.feature_index_batches);loader.begin_epoch()
        hints=[];read_indices=[];read_sizes=[]
        loader.window_buffering=lambda b:hints.append(loader.resolve_batch_feature_rows(b).clone())
        def read(bs):
            result=[];read_sizes.append(len(bs))
            for batch in bs:
                rows=loader.resolve_batch_feature_rows(batch);read_indices.append(rows.clone())
                result.append(torch.tensor(np.asarray(bundle.arrays['reordered_features'][rows.cpu().numpy()]),device='cuda'))
            return result,sum(len(b[0]) for b in bs)
        loader._read_many=read
        roots=[torch.tensor([0,2,4,6],device='cuda') for _ in range(32)]+[torch.tensor([8],device='cuda')]
        source=iter(sampler.sample_blocks(graph,root) for root in roots)
        model=SAGE(128,128,2,3,.2).cuda();opt=torch.optim.Adam(model.parameters(),lr=.001)
        seen=[]
        while True:
            try:inp,out,blocks,x=loader.fetch_feature(128,source,'cuda:0')
            except StopIteration:break
            verify_digit_blocks(blocks,arrays,[4,3,2])
            np.testing.assert_array_equal(x.cpu().numpy(),features[inp.cpu().numpy()])
            loss=torch.nn.functional.cross_entropy(model([b.int() for b in blocks],x),out%2)
            opt.zero_grad();loss.backward();opt.step()
            require(torch.isfinite(loss).item(),'Nonfinite loss');seen.extend(out.cpu().tolist())
        pipe=loader.pipeline.summary()
        require(seen==[0,2,4,6]*32+[8],'Root tail or ordering mismatch')
        require(pipe['drained'] and pipe['delivered_batches']==sampler._cuda_call_counter==33,'Batch loss')
        require(pipe['max_merged_batches']<=4 and pipe['max_pending_batches']<=5,'Unbounded GPU pipeline')
        require(pipe['batch_cap_flushes']>0 and pipe['max_read_rows']<=262144,'Cap not exercised')
        require(len(hints)==len(read_indices)==33,'Lost hints/reads')
        for h,r in zip(hints,read_indices):require(torch.equal(h,r),'Wrong physical hint/read address')
        graph._graph.unpin_memory_()
    paths=[HERE/name for name in ('gpu_tests.py','loader.py','window.py','common.py','protocol.json')]
    write(OUT/'gpu_checks.json',dict(passed=True,model_updates=33,grouped_cuda_batches=33,roots=129,
        tail_roots=1,pipeline=pipe,read_sizes=read_sizes,features_bit_exact=True,raw_ssd_access=False,
        feature_reads_simulated=True,binary_sha256=sha(BINARY),
        inherited_native_gpu_checks_sha256=sha(parent.OUT/'gpu_checks.json'),
        tested_sha256={str(f.relative_to(ROOT)):sha(f) for f in paths}))
    print('Bounded CUDA sampler/model check passed (33 updates, no SSD)',flush=True)


if __name__=='__main__':main()
