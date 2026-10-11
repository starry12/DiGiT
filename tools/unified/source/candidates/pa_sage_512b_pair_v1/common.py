"""Shared 512-B native comparison, with independently frozen provenance."""
import math
import os
from pathlib import Path
import sys
from candidates.pa_sage_legacy_align_v1 import common as legacy
from candidates.pa_sage_legacy_align_v1.common import (
    ROOT, PY, GPU_UUID, read, write, sha, require, identity, check_inputs,
    object_sha, progress, hashlib, json, time,
)

HERE = Path(__file__).resolve().parent
OUT = ROOT / 'results/pa_sage_512b_pair_20260926_v1'
LARGE = Path('/mnt/n0/digit/pa_sage_512b_pair_20260926_v1')
MODULE = 'candidates.pa_sage_512b_pair_v1'
PARENT = ROOT / 'candidates/io_accounting_v1'
SAMPLER = ROOT / 'candidates/pa_sage_bidir_native_v2'
BINARY = HERE / 'runtime/BAM_Feature_Store/BAM_Feature_Store.so'


def setup():
    legacy.setup()
    sys.path[:0] = [str(HERE/'runtime'), str(legacy.HERE/'runtime'), str(SAMPLER/'runtime')]


def parents():
    from candidates.pa_sage_cpu_affinity_v1.common import verify as affinity
    from candidates.pa_sage_ablation_cache_v3.common import verify as row_cache
    return dict(affinity=affinity(), row_cache=row_cache())


def native_sources():
    sources = {str(p.relative_to(ROOT)):sha(p) for p in (HERE/'native').rglob('*') if p.is_file()}
    for rel, digest in read(PARENT/'manifest.json')['files'].items():
        if rel.startswith('native/bam/include/'):
            path=PARENT/rel
            require(sha(path)==digest,'Frozen include changed: '+rel)
            sources[str(path.relative_to(ROOT))]=digest
    sources['bam/build/lib/libnvm.so']=sha(ROOT/'bam/build/lib/libnvm.so')
    return sources


def verify():
    manifest=read(HERE/'manifest.json')
    require(parents()==manifest['parents'],'Frozen parent changed')
    for rel,digest in manifest['files'].items():
        require(sha(ROOT/rel)==digest,'Candidate changed: '+rel)
    require(sha(BINARY)==manifest['binary_sha256'],'Native binary changed')
    require(sha(OUT/'inputs.json')==manifest['inputs_sha256'],'Input binding changed')
    require(sha(OUT/'cpu_checks.json')==manifest['cpu_checks_sha256'],'CPU evidence changed')
    require(sha(OUT/'gpu_checks.json')==manifest['gpu_checks_sha256'],'GPU evidence changed')
    return sha(HERE/'manifest.json')


def validate_completion(report):
    legacy.validate_completion(report)
    p=read(HERE/'protocol.json');arm=report['arm']
    require(arm in p['arms'] and report['protocol']==p,'Wrong protocol or arm')
    require(report['block_index_bits']==32 and report['page_bytes']==512,'Wrong block/cache width')
    require(report['window_buffer'] and report['accumulator'] and not report['mixed_io'],'Wrong read path')
    require(report['cpu_cache_rows']==p['cpu_rows'],'Unequal CPU feature capacity')
    require(report['gpu_cache']['capacity_pages']*512==p['cache_mib']*2**20,'Unequal GPU cache capacity')
    require(report['gpu_cache']['policy']==p['gpu_policies'][arm],'Wrong GPU replacement policy')
    for phase in ('at_training_start','at_training_end'):
        masks=report['affinity'][phase]
        require(bool(masks) and all(v==[p['bound_logical_cpu']] for v in masks.values()),'CPU affinity escaped')
    require(report['continuous_cpu_frequency_observation'] is False,'Frequency observer was enabled')
    if arm=='digit':
        require(report['native_grouped_batches']==report['updates'],'Not every batch used grouped CUDA sampling')
        require(report['cpu_cache']['replica_aliases'] and report['cpu_cache']['padding_uncached'],'Invalid logical CPU mapping')
        if report['smoke']:
            require(report['smoke_checks']['storage_inverse_exact'],'Unchecked physical feature rows')
    return True


def compare(gids,digit):
    for r in (gids,digit):
        validate_completion(r);require(not r['smoke'],'Short run cannot be full performance')
    require(gids['arm']=='gids' and digit['arm']=='digit','Wrong paired arms')
    for key in ('root_order_sha256','initial_model_sha256','candidate_sha256','input_binding_sha256','binary_sha256'):
        require(gids[key]==digit[key],'Pair differs: '+key)
    return dict(passed=True,complete=True,epochs_per_arm=1,updates_per_arm=1179,
        speedup=gids['timing']['e2e_seconds']/digit['timing']['e2e_seconds'],
        reduction_percent=100*(1-digit['timing']['e2e_seconds']/gids['timing']['e2e_seconds']),
        rows=[dict(arm=r['arm'],**r['timing'],logical_feature_requests=r['logical_feature_requests'],
                   gpu_cache_hit_rate=r['gpu_cache']['hit_rate'],ssd_bytes=r['device_io']['completed_bytes']) for r in (gids,digit)],
        limits=['Single first epoch per arm, seed 0, no accuracy claim',
                'Grouped sampling intentionally differs; roots, model and training budget are equal',
                'DiGiT is a 512-byte line/WB/accumulator adaptation, not the original 4-KiB mixed-I/O path',
                'Training host spans are nested or asynchronous; their sum is not E2E'])
