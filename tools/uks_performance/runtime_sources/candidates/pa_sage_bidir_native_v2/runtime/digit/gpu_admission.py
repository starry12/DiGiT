"""Conservative admission for the current eager, int64 DiGiT CUDA adapter.

This is an engineering reservation, not an allocator/OOM proof. No device
arrays or DGL graphs are constructed here.
"""

# Locate DiGiT independently of the checkout directory and working directory.
from pathlib import Path as _DigitPath
import sys as _digit_sys
_digit_root = next((p for p in _DigitPath(__file__).resolve().parents
                    if (p / ".digit-root").is_file()), None)
if _digit_root is None:
    raise RuntimeError("Cannot locate the DiGiT project root")
if str(_digit_root) not in _digit_sys.path:
    _digit_sys.path.insert(0, str(_digit_root))
import digit_paths as _digit_paths
import math

GIB = 1024**3
METADATA = ('reordered_indptr', 'reordered_indices', 'group_members',
            'group_storage_base', 'supernode_to_group', 'node_to_primary_row')


def estimate(manifest, batch_size=1024, fanouts=(16,5,5), hidden=128,
             classes=172, cache_mib=512, reserve_gib=4, metadata_mode='gpu'):
    n = int(manifest['dataset']['num_nodes'])
    e = int(manifest['dataset']['num_edges'])
    dim = int(manifest['feature']['dim'])
    group = int(manifest['grouping']['group_size'])
    if (min(n,dim,batch_size,hidden,classes,cache_mib) <= 0 or e < 0
            or not fanouts or any(f <= 0 for f in fanouts)
            or fanouts[0] < group or not math.isfinite(reserve_gib) or reserve_gib < 2):
        raise ValueError('invalid geometry/fanout or safety reserve below 2 GiB')
    metadata = {name:8*math.prod(manifest['files'][name]['shape']) for name in METADATA}
    full_host_metadata = sum(metadata.values()) + 8*(n+1+2*e)
    # The adapter retains int64 original EIDs as well as original indices.
    original_bytes = 8*(n+1+2*e)
    if metadata_mode not in ('gpu','cpu_eid','gpu_i32'):raise ValueError('invalid metadata mode')
    metadata['original_csc'] = original_bytes if metadata_mode!='cpu_eid' else 0
    if metadata_mode=='gpu_i32':
        if max(n+int(manifest['grouping'].get('num_groups',0))-1,e-1,
               int(manifest['feature'].get('num_storage_rows',math.ceil(n*1.2)+group))-1)>2**31-1:
            raise ValueError('gpu_i32 ID/EID range exceeds signed int32; use gpu')
        for name in METADATA:
            if name!='reordered_indptr':metadata[name]//=2
        metadata['original_csc']=8*(n+1)+8*e
    frontier = batch_size
    edges = nodes = 0
    for f in reversed(fanouts):
        edges += frontier*f
        frontier = min(n, frontier*(f+1))
        nodes += frontier
    # Conservative model+Adam+grad state and sampled blocks/compaction workspace.
    parameters = 2*(dim*hidden + max(0,len(fanouts)-2)*hidden*hidden + hidden*classes)
    components = dict(metadata)
    storage_rows = int(manifest['feature'].get('num_storage_rows',math.ceil(n*1.2)+group))
    components.update(cache_data=cache_mib*1024**2,
        cache_metadata_allowance=cache_mib*1024**2//4,
        # BaM allocates data_page_t for the entire storage range, not only
        # resident cache entries. Use a conservative 64 B per 4-KiB page.
        bam_storage_range_allowance=math.ceil(storage_rows*dim*4/4096)*64,
        model_optimizer_grad=parameters*32,
        batch_features_activations=nodes*max(dim,hidden,classes)*4*8,
        sampled_edges_and_sort_scratch=edges*8*32,
        labels_masks_seed_indices=n*32,
        runtime_reserve=math.ceil(reserve_gib*GIB))
    subtotal = sum(components.values())
    return dict(schema='digit-gpu-admission-v1', components_bytes=components,
        metadata_bytes=sum(metadata.values()), required_bytes=math.ceil(subtotal*1.2),
        safety_multiplier=1.2, formal_bound=False, backend=metadata_mode,
        original_csc_gpu_bytes_avoided=original_bytes if metadata_mode=='cpu_eid' else 0,
        fanouts=list(fanouts), batch_size=batch_size,
        # Host DGL construction can transiently retain COO plus both CSCs and
        # normalization copies. Features remain mmap but no RSS guarantee here.
        # UVA can pin graph feature data even when GIDS supplies fetched rows.
        # Include the complete feature backing, not just its mmap object.
        host_required_bytes=math.ceil((64*e+64*n+n*dim*4+full_host_metadata)*1.2))


def check_live(plan, device=0):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('GPU admission unavailable: CUDA query failed; refusing training')
    free,total = torch.cuda.mem_get_info(device)
    result = dict(plan, gpu_name=torch.cuda.get_device_name(device), device=device,
                  free_bytes=free, total_bytes=total, passed=free>=plan['required_bytes'])
    from pathlib import Path
    info = dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(info['MemAvailable'].split()[0])*1024
    result.update(host_available_bytes=available)
    result['passed'] = result['passed'] and available>=plan['host_required_bytes']
    return result


def enforce_training(args):
    """Called before OGB dataset construction, also when bypassing the runner."""
    import json
    from pathlib import Path
    if args.data != 'OGB':
        return None
    artifact = args.digit_artifact
    if not artifact and getattr(args, '_smoke', None):
        # Baseline smoke uses the conservative grouped envelope as an upper
        # planning allowance before constructing the same full OGB/UVA graph.
        artifact = args._smoke['config']['artifact']
    if args.data != 'OGB' or not artifact:
        return None
    path = Path(artifact)/'manifest.json'
    if path.stat().st_size > 1024**2:
        raise ValueError('oversized artifact manifest')
    manifest = _digit_paths.json_loads(path.read_text())
    if args.model_type != 'sage' or not args.uva_graph:
        raise ValueError('OGB admission currently supports SAGE with UVA graph only')
    plan = estimate(manifest,args.batch_size,tuple(map(int,args.fan_out.split(','))),
                    args.hidden_channels,args.num_classes,args.cache_size,
                    metadata_mode=args.digit_metadata_mode)
    result = check_live(plan,args.device)
    print('GPU/host admission: '+json.dumps(result,sort_keys=True),flush=True)
    if not result['passed']:
        raise RuntimeError('GPU/host memory admission failed before graph construction')
    return result
