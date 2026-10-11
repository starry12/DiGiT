"""Explicit tiny CUDA check for shared groups; no graph-scale load or SSD I/O."""
import argparse
import tempfile
import subprocess
import numpy as np
from candidates.uks_group_reuse_v2.common import *

def _run(output):
    source=verify()
    # Refuse even a tiny check on a device already used by a training worker.
    import subprocess
    from .resources import target_gpu_busy
    require(not target_gpu_busy(),'Target GPU2 is occupied')
    from candidates.uks_mixed_1k2k_v1.runtime import setup_sampling_imports
    setup_sampling_imports()
    import torch,dgl,digit.sampler as module
    from digit.reorganization import reorganize_to_bundle
    from candidates.uks_group_reuse_v2.tests import fixture,invoke
    from candidates.uks_group_reuse_v2.overlay import apply
    from candidates.uks_group_incremental_v1.common import raw_extension
    from digit.sampler import DIGIT_STORAGE_ROW
    cuda=raw_extension();records=[]
    with tempfile.TemporaryDirectory() as tmp:
        n=16;ptr=np.arange(n+1,dtype=np.int64)*n;idx=np.tile(np.arange(n,dtype=np.int64),n)
        features=np.arange(n*256,dtype=np.float32).reshape(n,256)
        base=reorganize_to_bundle(ptr,idx,features,Path(tmp)/'bundle',dataset_name='UKS',dataset_size='fixture',group_size=2,replication_ratio=.2,page_size=4096,seed=0)
        oldp=base.arrays['reordered_indptr'];oldi=base.arrays['reordered_indices']
        data=Path(tmp)/'data';folder=Path(tmp)/'reuse'
        fixture(data,n,base.arrays['group_members'],[oldi[oldp[i]:oldp[i+1]].tolist() for i in range(n)])
        invoke('build',data,folder,n);proof=invoke('validate',data,folder,n)
        require(proof['added_group_references']>0,'Fixture did not reuse groups')
        bundle=apply(base,np.load(folder/'reordered_indptr.npy'),np.load(folder/'reordered_indices.npy'),proof)
        tensors=(torch.from_numpy(ptr.copy()),torch.from_numpy(idx.copy()),torch.arange(len(idx),dtype=torch.int64))
        graph=dgl.graph(('csc',tensors),num_nodes=n).formats('csc');graph.pin_memory_()
        try:
            for fanout in (2,5,10):
                for seed in (0,23,59):
                    reference=None
                    for mode in ('legacy','incremental'):
                        from types import SimpleNamespace
                        extension=SimpleNamespace(**{k:getattr(cuda,k) for k in dir(cuda) if not k.startswith('__')})
                        if mode=='incremental':extension.sample_group_aware_i32_uva64=cuda.sample_group_aware_i32_uva64_incremental
                        module._cuda_extension=extension
                        torch.manual_seed(seed);dgl.seed(seed);dgl.random.seed(seed)
                        sampler=module.DiGiTNeighborSampler([fanout,3,2],bundle,cuda_mode='required',metadata_mode='gpu_i32_uva_eid64',random_seed=seed)
                        inp,roots,blocks=sampler.sample_blocks(graph,torch.tensor([0,2,5,9],device='cuda'))
                        rows=blocks[0].srcdata[DIGIT_STORAGE_ROW].cpu().numpy()
                        np.testing.assert_array_equal(bundle.arrays['storage_to_node'][rows],inp.cpu().numpy())
                        np.testing.assert_array_equal(bundle.arrays['reordered_features'][rows],features[inp.cpu().numpy()])
                        values=[inp.cpu().numpy(),roots.cpu().numpy(),rows]
                        for b in blocks:
                            u,v=b.edges(order='eid');eid=b.edata[dgl.EID].cpu().numpy()
                            np.testing.assert_array_equal(b.srcdata[dgl.NID][u].cpu().numpy(),idx[eid])
                            np.testing.assert_array_equal(b.dstdata[dgl.NID][v].cpu().numpy(),np.searchsorted(ptr,eid,side='right')-1)
                            values.append(eid)
                        if reference is not None:
                            for a,b in zip(reference,values):np.testing.assert_array_equal(a,b)
                        reference=values;records.append(dict(fanout=fanout,seed=seed,mode=mode,passed=True))
            torch.cuda.synchronize()
        finally:graph._graph.unpin_memory_()
    write(output,dict(passed=True,source_sha256=source,cases=records,raw_ssd_access=False,native_training_accepted=False))

def run(output):
    with exclusive():_run(output)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=OUT/'gpu_checks.json')
    run(parser.parse_args().output)
