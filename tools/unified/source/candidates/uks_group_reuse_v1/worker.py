"""Explicit isolated old/reuse DiGiT workers; same BFS window, hot set and I/O.

Preview by default. No AE deployment changes. A validated full overlay and root
permission are required for actual native smoke/audit/performance execution.
"""
import argparse
from .common import *

def graph_sampler(arm,seed,folder,reuse):
    require(arm=='digit','This candidate isolates DiGiT adjacency reuse only')
    from candidates.uks_mixed_1k2k_v1.runtime import setup_sampling_imports
    setup_sampling_imports()
    import numpy as np,torch,dgl,digit.sampler as module
    from candidates.uks_native_v1.binding import protocol,bundle,check
    from candidates.uks_group_incremental_v1.common import extension
    from .overlay import load
    p=protocol();binding=check();n=p['nodes'];e=binding['receipts']['csc']['normalized_edges']
    # Use anonymous copies for BOTH control and overlay before CUDA pinning.
    # Do not inherit the old file-backed CSC registration route.
    arrays=[]
    for name,count in (('indptr',n+1),('indices',e),('eids',e)):
        src=np.load(DATA/'csc'/(name+'.npy'),mmap_mode='r')
        require(src.dtype==np.int64 and src.shape==(count,),'Wrong original CSC')
        dst=np.empty(count,dtype=np.int64)
        for lo in range(0,count,1048576):dst[lo:lo+1048576]=src[lo:lo+1048576]
        require(dst.flags.owndata and not isinstance(dst,np.memmap),'CSC is not anonymous');arrays.append(dst)
    tensors=tuple(torch.from_numpy(a) for a in arrays)
    graph=dgl.graph(('csc',tensors),num_nodes=n,idtype=torch.int64).formats('csc')
    graph.create_formats_()
    require(all(a.data_ptr()==b.data_ptr() for a,b in zip(tensors,graph.adj_tensors('csc'))),'CSC unexpectedly copied')
    graph.pin_memory_();require(graph.is_pinned(),'Anonymous CSC not pinned')
    artifact=bundle()
    if reuse:artifact=load(artifact,folder)
    os.environ['UKS_GROUP_KERNEL']='incremental';module._cuda_extension=extension()
    sampler=module.DiGiTNeighborSampler(p['fanouts'],artifact,cuda_mode='required',metadata_mode='gpu_i32_uva_eid64',random_seed=seed)
    sampler._ensure_cuda_metadata(graph,torch.device('cuda:0'))
    return graph,arrays,artifact,sampler

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overlay',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--variant',choices=('old','reuse'),required=True)
    parser.add_argument('--mode',choices=('smoke','audit','performance'),default='smoke')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--execute',action='store_true')
    args=parser.parse_args();source=verify();output=args.output.resolve()
    require(output.parent==(OUT/'runs').resolve(),'Worker outputs must be direct children of candidate results/runs')
    print(json.dumps(dict(variant=args.variant,mode=args.mode,overlay=str(args.overlay),output=str(output),
        source_sha256=source,execute=args.execute,same_bfs_roots=True,same_frequency_hotset=True,raw_ssd_writes=False),indent=2),flush=True)
    if not args.execute:return
    require(os.geteuid()==0 and os.environ.get('CUDA_VISIBLE_DEVICES')=='2','Use local sudo and CUDA_VISIBLE_DEVICES=2')
    with exclusive():
        receipt=check_prepared(args.overlay);require(not output.exists(),'Preserve previous worker results')
        gpu=read(OUT/'gpu_checks.json')
        require(gpu['passed'] and gpu['source_sha256']==source,'Run the current tiny CUDA check before native workers')
        if args.mode!='smoke':
            smokes=[read(p) for p in (OUT/'runs').glob('*/report.json')]
            require(any(r.get('passed') and r.get('mode')=='smoke' and r.get('reuse_variant')==args.variant and
                r.get('candidate_sha256')==source and r.get('prepared_sha256')==sha(args.overlay/'prepared.json')
                for r in smokes),'A matching successful smoke is required before audit/performance')
        from candidates.uks_freq_bfs_retry_v1 import worker as base
        base.admission()
        from ae.common import check_device
        check_device();output.mkdir(parents=True)
        audits=[]
        def select(arm,seed):
            result=graph_sampler(arm,seed,args.overlay,args.variant=='reuse')
            if args.mode=='audit':
                import numpy as np
                from candidates.uks_merge_audit_v1.audit import Audit
                from candidates.uks_freq_bfs_v1.common import hot_path
                audit=Audit(result[3]);audit.hot[:]=False
                audit.hot[np.load(hot_path('digit'),mmap_mode='r')]=True;audits.append(audit)
            return result
        if args.mode=='smoke':
            from candidates.uks_freq_bfs_retry_v1 import smoke as target
            entry=lambda:target.smoke('digit',output)
        else:
            target=base;entry=lambda:base.run('digit_cpu2',output,'off')
        old=target.graph_sampler;target.graph_sampler=select
        try:
            result=entry()
            detail=audits[0].finish() if audits else None
            if detail:require(detail['totals']['mapped_pairs_both_cpu_cold']==result['training']['request_sizes']['pair_requests'],'Pair eligibility/native count mismatch')
            require(check_prepared(args.overlay)==receipt and verify()==source,'Inputs/source changed')
            write(output/'report.json',dict(result,reuse_variant=args.variant,mode=args.mode,candidate_sha256=source,
                prepared_sha256=sha(args.overlay/'prepared.json'),diagnostic=detail,
                performance_result=args.mode=='performance',feature_layout_changed=False,anonymous_host_csc=True,raw_ssd_writes=False))
        except BaseException as error:
            write(output/'failure.json',dict(passed=False,error=repr(error),candidate_sha256=source));raise
        finally:target.graph_sampler=old

if __name__=='__main__':main()
