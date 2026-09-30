"""Pure component estimates; live checks use nvidia-smi only after the grid gate."""
import copy
import math
import subprocess
from .common import GPU, GPU_UUID, ROOT, require, read
from .backend import allocation


def estimate(p, m, arm='digit', profile=False):
    n, e = p['graph']['nodes'], p['graph']['edges']
    files = copy.deepcopy(m['files'])
    files['reordered_indices']['shape'][0] += e - m['dataset']['num_edges']
    names = ('reordered_indptr', 'reordered_indices', 'group_members', 'group_storage_base',
             'supernode_to_group', 'node_to_primary_row')
    parts = {key: math.prod(files[key]['shape']) * (8 if key == 'reordered_indptr' else 4) for key in names}
    storage = m['feature']['num_storage_rows']
    require(max(n + m['grouping']['num_groups'] - 1, storage - 1) <= 2**31-1, 'Compact IDs overflow')
    frontier = p['batch_size']; nodes = edges = 0
    for f in reversed(p['fanouts']):
        edges += frontier * f; frontier = min(n, frontier * (f+1)); nodes += frontier
    dma = 0 if profile else allocation(p['arms'][arm])['gpu_dma_allocation_bytes']
    parameters = 2*(128*p['hidden'] + max(0,p['layers']-2)*p['hidden']**2 + p['hidden']*p['classes'])
    parts.update(cache_data=dma, cache_metadata_allowance=dma//4,
        bam_storage_range_allowance=0 if profile else storage*512//4096*32,
        exact_cpu_row_map=0 if profile else storage*4,
        useful_io_bitmap_and_counters=0 if profile else dma//4096*8+48,
        model_optimizer_grad=0 if profile else parameters*32,
        batch_features_activations=0 if profile else nodes*max(128,p['hidden'],p['classes'])*4*8,
        sampled_edges_and_sort_scratch=edges*8*32, labels_masks_seed_indices=64*2**20,
        trace_scratch=128*2**20, runtime_reserve=4*2**30)
    meta = sum(parts[key] for key in names)
    host = dict(pinned_original_csc=8*(n+1+2*e), host_metadata_allowance=2*meta,
        cpu_feature_cache=0 if profile else p['arms'][arm]['cpu_feature_bytes'],
        frequency_vector=n*8 if profile else 0, logical_slot_builder=n*4,
        validation_and_runtime_reserve=64*2**30)
    return dict(schema='digit-cache-admission-v3', formal_bound=False, arm=arm, profile=profile,
        components_bytes=parts, required_bytes=math.ceil(sum(parts.values())*1.2),
        host_components_bytes=host, host_required_bytes=max(192*2**30, math.ceil(sum(host.values())*1.25)),
        gpu_safety_multiplier=1.2, host_safety_multiplier=1.25)


def live(plan):
    value = subprocess.check_output(['nvidia-smi', '-i', GPU,
        '--query-gpu=uuid,memory.free,memory.total', '--format=csv,noheader,nounits'], text=True, timeout=30)
    fields = value.strip().split(',')
    require(len(fields) == 3 and fields[0].strip() == GPU_UUID, 'Unexpected GPU')
    free, total = [int(x.strip())*2**20 for x in fields[1:]]
    from ae.common import host
    ram = host()
    return dict(plan, gpu_uuid=GPU_UUID, gpu_free_bytes=free, gpu_total_bytes=total,
        host_available_bytes=ram, passed=(plan['required_bytes'] <= total and
            free >= max(total-2**30, plan['required_bytes']) and ram >= plan['host_required_bytes']))
