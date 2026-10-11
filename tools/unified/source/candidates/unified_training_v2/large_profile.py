import time
from pathlib import Path
import numpy as np
from candidates.group_cache_migration_v1.common import require,write
from candidates.group_cache_migration_v1.large import graph_files
from candidates.group_cache_migration_v1.core import validate_order,digest_array
from candidates.group_cache_migration_v1.io import read_array,save_array
from .profiling import window_profile,short_hot,validate_profile

def prepare_cache(arm, native, sampling, accepted, previous_hot, output, check, event, config):
    files = graph_files(accepted)
    matched = files['digit_roots.i64']
    metadata = dict(matched_bfs_roots_sha256=matched['sha256'],
                    capacity_rows=accepted['cpu_cache_rows'], gpu_cache_bytes=4*2**30,
                    gpu_reset='fresh provider after profiling; preload must leave residency zero',
                    profile=None, audit=None, hot_policy='RevPR' if arm == 'gids' else 'short_window_Freq_RevPR_set_fallback')
    if arm == 'gids': return None, metadata
    started = time.perf_counter()
    # Maximum live extra: counts 4N + flags N + train/order ~2.4N plus selection.
    # This stage runs BEFORE the >=40 GiB CPU feature cache and N*4 slot lookup.
    canonical = read_array(files['bfs.i64'], check=check)
    train = read_array(files['train.i64'], check=check)
    validate_order(canonical, train, native.graph.nodes); del train
    _, Sampler = sampling.select('digit')
    def factory(seed):
        sampler = Sampler(native, grouped=True, seed=seed)
        def sample(roots, index):
            inp, _, blocks = sampler.sample_blocks(roots, index)
            ids = inp.detach().cpu().numpy().astype(np.int64, copy=True) if hasattr(inp,'detach') else np.asarray(inp,dtype=np.int64).copy()
            del blocks
            return ids
        return sample
    counts, receipt = window_profile(canonical, native.graph.nodes, factory, config, event, check)
    write(output/'short_window_profile.json', receipt)
    count_file = save_array(output/'frequency_counts.u32', counts, check)
    baseline = read_array(files['gids_hot.i64'], check=check)
    score_receipt=None
    hot, selection = short_hot(counts,baseline,check)
    del counts
    hot_file = save_array(output/'short_window_hot.i64', hot, check)
    write(output/'hot_selection.json', dict(selection, file=hot_file, counts=count_file, revpr_scores=score_receipt))
    eligibility = dict(complete=False, reason='This study measures native cache serving after a short profiling window; no extra audit epochs')
    del baseline, canonical
    import torch
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    # Sampling may consume RNG state, but the formal model/optimizer must start
    # from the same seed as the GIDS arm. Formal sampler is constructed later.
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    metadata.update(profile=receipt, selection=selection, audit=eligibility,
                    hot_file=hot_file, counts_file=count_file,
                    preparation_seconds=time.perf_counter()-started,
                    sampler_seed_reset=0, native_gpu_feature_cache_not_yet_created=True)
    return hot, metadata


def validate_receipt(report, arm, config):
    try:
        value = report['cache_epoch']
        require(value['capacity_rows'] == report['cache']['cpu_rows'], 'Capacity mismatch')
        require(value['gpu_cache_bytes'] == report['cache']['gpu_feature_bytes'], 'GPU capacity mismatch')
        require(report['cpu_affinity']['cpus'] == [2] and report['cpu_affinity_initial']['cpus'] == [2], 'Affinity mismatch')
        require(report['counters']['whole']['raw_allocator_cache']['resident_pages_start'] == 0, 'GPU cache not cold')
        for name in ('whole','measured'):
            group=report['group_sampling'][name]
            require(0<=group['grouped_sampled_edges']<=group['outer_sampled_edges'],'Invalid sampled grouping counts')
            require(group['grouped_sampled_edges']%2==0,'Partial group emitted')
            if arm=='gids':require(group['grouped_sampled_edges']==0,'GIDS used grouped sampling')
        if arm == 'digit':
            validate_profile(value['profile'],config,report['common_training']['train_root_count'])
            require(value['profile']['seed'] == 23 and value['profile']['feature_reads'] == 0, 'Invalid profiling')
            require(value['hot_file']['sha256'] == value['selection']['hot_sha256'], 'Hot-set provenance')
            require(value['native_gpu_feature_cache_not_yet_created'], 'Profiling warmed GPU feature cache')
            require(report['numa_admission']['adapter_sha256'] == report['manifest_sha256'], 'NUMA evidence changed')
        return True
    except (KeyError,TypeError,ValueError,RuntimeError,OSError): return False
