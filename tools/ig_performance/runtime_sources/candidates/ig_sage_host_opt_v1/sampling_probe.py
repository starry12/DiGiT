"""Separate real-IG CUDA-event sampling diagnostic, outside formal training.

No model update, feature fetch or raw SSD access. The unchanged native sampler
already exposes events around group selection and first-occurrence EID lookup.
"""
import argparse
import fcntl
from .common import *


def execute(output,binding_path):
    execution=verify();binding=read(binding_path);check_inputs(binding)
    require(binding['candidate_sha256']==execution,'Probe code binding differs')
    setup()
    import numpy as np
    import torch,dgl
    from sampler_config import configure
    from bounded_io import load_csc
    from uva_sampler import UVANeighborSampler,native
    from digit.artifacts import ArtifactBundle
    from .admission import check_live
    from .affinity import apply_affinity,snapshot
    from .worker import reset
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    plan=check_live();require(plan['passed'] and plan['free_bytes']>plan['total_bytes']-2**30,'Probe GPU/host unavailable')
    affinity=apply_affinity('affinity');output.mkdir(parents=True,exist_ok=False)
    configure('full');p=cfg();cm=read(DATA/'csc/manifest.json');names=['original_indptr.npy','original_indices.npy','original_eids.npy']
    progress(output,'loading_original_csc')
    graph,csc=load_csc(*[DATA/'csc'/name for name in names],p['nodes'],host_cap=70*2**30,expected_sha256=[cm['files'][name]['payload_sha256'] for name in names])
    require(graph.num_edges()==p['edges'],'Wrong graph')
    manifest=read(DATA/'full/manifest.json');arrays={k:np.load(DATA/'full'/v['path'],mmap_mode='r') for k,v in manifest['files'].items() if k!='reordered_features'}
    bundle=ArtifactBundle(DATA/'full',manifest,arrays)
    sampler=UVANeighborSampler(p['fanouts'],bundle,random_seed=0,host_cap=100*2**30)
    metadata=sampler._ensure_cuda_metadata(graph,torch.device('cuda:0'));progress(output,'metadata_ready')
    order=np.load(ROOT/p['train_order'],mmap_mode='r');count=8
    native.profile_prepare(count,torch.cuda.current_stream().cuda_stream);reset(0)
    host_rows=[]
    for i in range(20+count):
        roots=torch.from_numpy(np.array(order[i*1024:(i+1)*1024],copy=True)).cuda()
        torch.cuda.synchronize()
        if i>=20:native.profile_arm(i+1)
        start=time.perf_counter();inputs,outputs,blocks=sampler.sample_blocks(graph,roots);torch.cuda.synchronize()
        wall=time.perf_counter()-start
        if i>=20:
            require(not native.profile_pending(),'Kernel profile not completed')
            host_rows.append(dict(batch=i+1,host_synchronized_seconds=wall,roots_sha256=digest(roots),
                shape=[[b.num_src_nodes(),b.num_dst_nodes(),b.num_edges()] for b in blocks]))
    rows=[dict(r) for r in native.profile_collect()];native.profile_release()
    require(len(rows)==count and all(r['fanout']==10 and r['batch']==21+i and r['group_sample_ms']>=0 and r['resolve_eids_ms']>=0 for i,r in enumerate(rows)),'Invalid native kernel records')
    for row in rows:require(abs(row['total_ms']-row['group_sample_ms']-row['resolve_eids_ms'])<.01,'Kernel event split does not reconcile')
    report=dict(passed=True,candidate_sha256=execution,input_binding_sha256=sha(binding_path),
        warmup_samples=20,measured_samples=count,training_updates=0,feature_fetches=0,raw_ssd_access=False,
        native_module_sha256=sha(Path(native.__file__)),native_events=rows,host_records=host_rows,
        group_selection_ms=sum(r['group_sample_ms'] for r in rows),eid_resolution_ms=sum(r['resolve_eids_ms'] for r in rows),
        native_total_ms=sum(r['total_ms'] for r in rows),affinity=dict(initial=affinity,final=snapshot()),
        scope='Independent sampler-only probe on first 8 measured root batches; unchanged full graph/UVA/native kernels; no model or feature reads. Not full training timing.')
    check_inputs(binding);require(verify()==execution,'Probe source changed');write(output/'report.json',report)
    progress(output,'complete',passed=True,group_ms=report['group_selection_ms'],eid_ms=report['eid_resolution_ms'])
    sampler.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--binding',type=Path,required=True);a=p.parse_args()
    require(os.geteuid()==0 and __debug__,'Use inherited root service with unlimited memlock')
    with open('/tmp/digit-pa-sage-libnvm0.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);execute(a.output,a.binding)


if __name__=='__main__':main()
