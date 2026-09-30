"""Small native CUDA sampler/model parity; simulated features, no SSD access."""
import random
import subprocess
import tempfile
from .common import *
from .profile import HostProfile, install, off_summary, validate_profile


def exercise(graph,arrays,bundle,features,arm,mode):
    import numpy as np
    import torch
    import dgl
    from digit.sampler import DiGiTNeighborSampler
    from .sampler import ProfiledDiGiTNeighborSampler
    from candidates.pa_sage_512b_pair_v4.loader import BoundedGIDS
    from candidates.pa_sage_512b_pair_v1.runtime_helpers import verify_digit_blocks
    from candidates.pa_sage_legacy_align_v1.graph import verify_blocks
    from models import SAGE
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0);dgl.seed(0);dgl.random.seed(0)
    p=HostProfile() if mode=='host' else None
    klass=ProfiledDiGiTNeighborSampler if p else DiGiTNeighborSampler
    sampler=(klass([4,3,2],bundle,cuda_mode='required',metadata_mode='gpu_i32_uva_eid64') if arm=='digit'
             else dgl.dataloading.NeighborSampler([4,3,2],replace=False))
    if arm=='digit':
        if p:sampler.profile=p
        sampler._ensure_cuda_metadata(graph,torch.device('cuda:0'))
    loader=BoundedGIDS.__new__(BoundedGIDS)
    loader.accumulator_flag=loader.window_buffering_flag=True;loader.heterograph=False
    loader.wb_size=2;loader.required_accesses=6832.4;loader.cache_dim=128;loader.gids_device='cuda:0'
    loader.feature_index_mode='explicit' if arm=='digit' else 'logical'
    loader.graph_reorganize=False;loader.feature_map=None;loader.strict_feature_row_validation=True
    loader.feature_index_batches={k:0 for k in ('logical','explicit','legacy_map')}
    loader.feature_index_rows=dict(loader.feature_index_batches);loader.begin_epoch()
    store=torch.tensor(np.asarray(bundle.arrays['reordered_features']) if arm=='digit' else features,device='cuda')
    hints=[];reads=[]
    loader.window_buffering=lambda b:hints.append(loader.resolve_batch_feature_rows(b).clone())
    def read(bs):
        rows=[loader.resolve_batch_feature_rows(b) for b in bs]
        reads.extend(r.clone() for r in rows)
        return [store[r] for r in rows],sum(len(b[0]) for b in bs)
    loader._read_many=read
    if p:install(p,sampler,graph,loader,arm)
    roots=[torch.tensor([0,2,4,6],device='cuda') for _ in range(32)]+[torch.tensor([8],device='cuda')]
    sample_ns=[0]
    def batches():
        for root in roots:
            start=time.perf_counter_ns();batch=sampler.sample(graph,root)
            sample_ns[0]+=time.perf_counter_ns()-start
            yield batch
    source=iter(batches());model=SAGE(128,128,2,3,.2).cuda();optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    signatures=[];seen=[];losses=[]
    try:
        while True:
            try:inp,out,blocks,x=loader.fetch_feature(128,source,'cuda:0')
            except StopIteration:break
            (verify_digit_blocks if arm=='digit' else verify_blocks)(blocks,arrays,[4,3,2])
            np.testing.assert_array_equal(x.cpu().numpy(),features[inp.cpu().numpy()])
            sig=hashlib.sha256()
            for t in [inp,out]+[t for b in blocks for t in (b.srcdata[dgl.NID],b.dstdata[dgl.NID],b.edata[dgl.EID],*b.edges(order='eid'))]:
                sig.update(t.cpu().numpy().tobytes())
            # Inspect without extra resolver calls: the timed path has exactly 2*N.
            row=blocks[0].srcdata['digit_storage_row'] if arm=='digit' else inp
            sig.update(row.cpu().numpy().tobytes())
            signatures.append(sig.hexdigest());seen.extend(out.cpu().tolist())
            loss=torch.nn.functional.cross_entropy(model([b.int() for b in blocks],x),out%2)
            optimizer.zero_grad();loss.backward();optimizer.step()
            require(torch.isfinite(loss).item(),'Nonfinite small-graph loss');losses.append(loss.item())
        torch.cuda.synchronize()
        require(seen==[0,2,4,6]*32+[8],'Lost tail or roots')
        pipe=loader.pipeline.summary()
        require(pipe['drained'] and pipe['delivered_batches']==33 and pipe['max_merged_batches']<=4,'Pipeline failure')
        require(len(hints)==len(reads)==33,'Incomplete address hints/reads')
        for h,r in zip(hints,reads):require(torch.equal(h,r),'Hint/read address mismatch')
        profile=p.summary() if p else off_summary()
        validate_profile(dict(sampling_profile=profile,updates=33,arm=arm,timing=dict(sample_host_seconds=sample_ns[0]/1e9)))
        return dict(arm=arm,mode=mode,model_updates=33,roots=129,tail=1,signatures=signatures,
                    losses=losses,profile=profile,pipeline=pipe)
    finally:
        if p:p.close()


def main():
    require(not (OUT/'gpu_checks.json').exists(),'Preserve GPU evidence')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='2','Use physical GPU2')
    fields=subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=uuid,memory.used',
        '--format=csv,noheader,nounits'],text=True).strip().split(',')
    require(fields[0].strip()==GPU_UUID and int(fields[1])<1024,'GPU busy or changed')
    setup();baseline.verify()
    import torch
    from candidates.pa_sage_512b_pair_v1.tests import fixture
    results=[]
    with tempfile.TemporaryDirectory(prefix='digit-host-profile-') as folder:
        graph,arrays,bundle,features=fixture(folder,pin=True)
        try:
            for arm in ('gids','digit'):
                off=exercise(graph,arrays,bundle,features,arm,'off');torch.cuda.empty_cache()
                host=exercise(graph,arrays,bundle,features,arm,'host');torch.cuda.empty_cache()
                require(off['signatures']==host['signatures'],'Instrumentation changed sampled blocks/physical rows: '+arm)
                results.extend([off,host])
        finally:graph._graph.unpin_memory_()
    write(OUT/'gpu_checks.json',dict(passed=True,results=results,model_updates=132,
        exact_block_and_row_parity=True,feature_reads_simulated=True,raw_ssd_access=False,
        performance_evidence=False,binary_sha256=sha(BINARY),
        tested_sha256={str(f.relative_to(ROOT)):sha(f) for f in source_files()}))
    print('GPU off/host parity passed: 132 updates, exact sampled blocks/rows, no SSD',flush=True)


if __name__=='__main__':main()
