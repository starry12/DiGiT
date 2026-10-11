"""Fresh-process UKS native worker, same BFS roots and CPU2 for both arms."""
import argparse
import io
import json
import os
import time
from pathlib import Path
import numpy as np
from .common import OUT, verify, require, write, sha
from .core import profile, audit, select_hot, validate_order, batches, digest_array
from .io import save_array, selected_scores


def profile_on_graph(graph, sampler, output, nodes):
    import torch, dgl
    from candidates.uks_freq_bfs_v1.common import LARGE, OUT as BFS_OUT, DATA, load_roots, hot_path
    from candidates.uks_native_v1.binding import check as check_binding
    from .io import identity
    began = time.perf_counter(); binding = check_binding()
    receipt = json.loads((BFS_OUT/'bfs_receipt.json').read_text())
    require(sha(LARGE/'canonical.npy') == receipt['canonical_sha256'], 'BFS source changed')
    canonical = np.load(LARGE/'canonical.npy')
    train = np.load(DATA/'synthetic/train.npy')
    validate_order(canonical, train, nodes); del train
    from itertools import islice
    window = np.concatenate(list(islice(batches(canonical),320)))
    require(np.array_equal(window,load_roots('digit')), 'BFS window/profile mismatch'); del window
    def event(**kw): write(output/'progress.json', dict(kw, updated_unix=time.time()))
    def check():
        require(not (output/'STOP').exists(), 'Cooperative stop requested')
    def factory(seed):
        sampler.random_seed = seed; sampler._cuda_call_counter = 0
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); dgl.seed(seed)
        def sample(roots,index):
            inp, _, blocks = sampler.sample_blocks(graph,torch.from_numpy(roots.copy()).cuda())
            result = inp.detach().cpu().numpy().astype(np.int64,copy=True)
            del blocks
            return result
        return sample
    counts, prof = profile(canonical,nodes,factory,event,check)
    write(output/'full_epoch_profile.json',prof)
    count_file = save_array(output/'frequency_counts.u32',counts,check)
    old = np.load(hot_path('digit')); baseline = np.load(hot_path('gids'))
    scores = score_binding = None
    if prof['observed_nodes'] < len(baseline):
        path = DATA/'rank/revpr.npy'
        expected = binding['files'][str(path)]['sha256']
        with path.open('rb') as stream:
            version=np.lib.format.read_magic(stream)
            shape,fortran,dtype=np.lib.format._read_array_header(stream,version)
            header_bytes=stream.tell()
        require(shape==(nodes,) and not fortran and dtype==np.float64,'RevPR NPY schema changed')
        scores, score_binding = selected_scores(path,nodes,baseline,check,expected,header_bytes)
    hot, selection = select_hot(counts,len(baseline),baseline,scores,check)
    del counts,scores
    hot_file = save_array(output/'full_epoch_hot.i64',hot,check)
    # NPY compatibility with the retained native worker, streamed via O_DIRECT.
    from candidates.ukl_runtime_prepare_v11r1.data import direct_file
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header,dict(descr='<i8',fortran_order=False,shape=hot.shape))
    def chunks():
        yield header.getvalue()
        for lo in range(0,len(hot),1048576): yield hot[lo:lo+1048576]
    npy = direct_file(output/'full_epoch_hot.npy',chunks(),check,256*2**20)
    eligibility = audit(canonical,nodes,factory,{'old_freq100':old,'full_epoch_freq':hot,'revpr':baseline},event,check)
    require(prof['roots_sha256'] == eligibility['epochs'][0]['roots_sha256'], 'Misaligned epoch roots')
    write(output/'cache_eligibility.json',eligibility)
    sampler.random_seed = 0; sampler._cuda_call_counter = 0
    torch.manual_seed(0); torch.cuda.manual_seed_all(0); dgl.seed(0)
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    result = dict(profile=prof,selection=selection,audit=eligibility,hot_file=hot_file,
                  hot_npy=npy,counts_file=count_file,revpr_scores=score_binding,
                  preparation_seconds=time.perf_counter()-began,
                  sampler_seed_reset=0,native_gpu_feature_cache_not_yet_created=True)
    write(output/'profile_bundle.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=('gids','digit'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); source=verify(); output=args.output.resolve()
    require(os.geteuid()==0 and os.environ.get('CUDA_VISIBLE_DEVICES')=='2','Use bounded controller')
    require(output.parent.parent == OUT/'runs' and not output.exists(),'Unexpected/existing output')
    from candidates.uks_group_first_cpu2_v1.common import exclusive,check_prepared,DEFAULT_OUTPUT,prepared_hashes
    from candidates.uks_group_first_cpu2_v1.sampling import graph_sampler
    from candidates.uks_group_first_cpu2_v1.affinity import install
    from candidates.uks_group_first_v1.common import OUT as FIRST_OUT, verify as first_source
    from candidates.uks_group_compare_v2.common import matched_roots
    with exclusive():
        gpu=json.loads((FIRST_OUT/'gpu_checks.json').read_text())
        require(gpu['passed'] and gpu['source_sha256']==first_source(),'Group-first kernel not accepted')
        prepared=check_prepared(DEFAULT_OUTPUT); before=prepared_hashes()
        # UUID compatibility must be installed before importing legacy modules
        # that bind identity() by value. Match the accepted author worker order.
        from candidates.uks_freq_bfs_retry_v1 import worker as base
        base.admission()
        from ae.common import check_device
        check_device();output.mkdir()
        old_sampler,old_roots,old_hot=base.graph_sampler,base.load_roots,base.hot_path
        cache_epoch={}
        def sampling(arm,seed):
            result=graph_sampler(arm,seed,'gids' if arm=='gids' else 'first')
            if arm=='digit':
                from candidates.uks_native_v1.binding import protocol
                cache_epoch.update(profile_on_graph(result[0],result[3],output,protocol()['nodes']))
            return result
        base.graph_sampler=sampling; base.load_roots=matched_roots
        base.hot_path=lambda arm: output/'full_epoch_hot.npy' if arm=='digit' else old_hot(arm)
        try:
            with install():
                result=base.run('gids_default' if args.arm=='gids' else 'digit_cpu2',output,'off')
            require(result['roots_sha256']==digest_array(matched_roots(args.arm)),'Formal roots changed')
            require(result['complete_region']['raw_allocator_cache']['resident_pages_start']==0,'GPU cache was warmed')
            require(check_prepared(DEFAULT_OUTPUT)==prepared and prepared_hashes()==before,'Prepared inputs changed')
            require(verify()==source,'Source changed')
            result.update(cache_epoch=cache_epoch,source_sha256=source,matched_cpu_affinity=True,
                          bfs_enabled=True,hot_policy='revpr' if args.arm=='gids' else 'full_epoch_freq_revpr_fallback',
                          order_policy='matched_full_epoch_BFS_blocks',raw_ssd_writes=False,
                          accuracy_evaluated=False,full_epoch_training=False)
            write(output/'report.json',result)
        finally:
            base.graph_sampler,base.load_roots,base.hot_path=old_sampler,old_roots,old_hot


if __name__=='__main__':main()
