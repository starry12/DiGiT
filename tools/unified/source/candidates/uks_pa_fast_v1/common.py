import json
from pathlib import Path
from candidates.uks_window_diag_v1.common import ROOT,PY,LOCKS,require,sha,write,inputs,load_roots,describe,window_ranges,validate_inherited_locks
from candidates.uks_mixed_1k2k_v1.common import SCRATCH_BYTES,heavy_gate,PARENT
HERE=ROOT/'candidates/uks_pa_fast_v1'
OUT=ROOT/'results/uks_pa_fast_20261011_v1'
BINARY=HERE/'runtime/BAM_Feature_Store/BAM_Feature_Store.so'

def verify():
    from candidates.uks_window_diag_v1.common import verify as previous
    require(previous()=='96af92ba4b0ff88e2a5185a622c9b1a2363fbbeee8d3ae6e55a0c4edc05ce97e','Previous test changed')
    m=json.loads((HERE/'manifest.json').read_text())
    for name,digest in m['files'].items():require(sha(ROOT/name)==digest,'Source changed: '+name)
    return sha(HERE/'manifest.json')

def native_sources():
    return {str(p.relative_to(HERE)):sha(p) for p in sorted((HERE/'native').rglob('*')) if p.is_file()}

def binary_receipt():
    r=json.loads((HERE/'runtime/build_receipt.json').read_text())
    require(r['native_files']==native_sources() and r['binary_sha256']==sha(BINARY),'Async native build changed')
    return r

def aggregate(reports):
    import numpy as np
    require({'smoke_old','smoke_fast','gids','digit_old','digit'}<=set(reports),'Incomplete candidate acceptance/pair')
    a,b=reports['smoke_old'],reports['smoke_fast']
    for k in ('roots_sha256','initial_model_sha256','shapes','sample_hashes'):
        require(a[k]==b[k],'Smoke sampling or initial state changed: '+k)
    np.testing.assert_allclose(a['losses'],b['losses'],rtol=1e-5,atol=1e-6)
    require(reports['gpu_checks']['passed'],'Missing GPU model acceptance')
    require(a['feature_bit_exact'] and b['feature_bit_exact'],'Missing exact native feature check')
    ref=reports['gids']
    for name in ('gids','digit_old','digit'):
        r=reports[name]
        require(r['passed'] and r['updates']==320 and r['measured_batches']==300,'Incomplete short comparison')
        require(r['training']['reconciled'] and r['complete_region']['raw_allocator_cache']['resident_pages_start']==0,'Counters/cold gate failed')
        for k in ('roots_sha256','initial_model_sha256','source_sha256'):require(r[k]==ref[k],'Unmatched '+k)
        for k in ('cpu_rows','cpu_feature_bytes','gpu_feature_cache_bytes'):require(r['cache'][k]==ref['cache'][k],'Unequal cache '+k)
    old,new=reports['digit_old'],reports['digit']
    require(new['optimized'] and not old['optimized'] and not ref['optimized'],'Wrong optimization arms')
    require(new['pipeline']['outstanding']==0 and new['pipeline']['submitted']==new['pipeline']['collected']==320,'Async pipeline incomplete')
    require(new['capture_calls']==320,'Missing dense graph replays')
    require(old['shapes']==new['shapes'] and old['hot_file_sha256']==new['hot_file_sha256'],'Optimized workload changed')
    np.testing.assert_allclose(old['losses'],new['losses'],rtol=1e-4,atol=1e-5)
    return dict(passed=True,complete=True,speedup=ref['measured_seconds']/new['measured_seconds'],
        old_speedup=ref['measured_seconds']/old['measured_seconds'],digit_optimization_speedup=old['measured_seconds']/new['measured_seconds'],
        arms={a:describe(reports[a]['training'],reports[a]['measured_seconds']) for a in ('gids','digit_old','digit')},
        smoke_feature_bit_exact=True,smoke_sampling_bit_exact=True,warmup_batches=20,measured_batches=300,
        aggregation='one fresh-process triplet; no max selection',accuracy_evaluated=False,
        diagnostics_required=False,diagnostics_complete=False,
        limits=['Synthetic UKS features/labels; not real accuracy evidence.','Async prefetch drains at each measurement boundary.','One triplet does not establish stable speedup.'])
