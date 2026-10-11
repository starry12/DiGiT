"""Keep the 4 KiB GPU path; replace only GIDS CPU membership and row addressing."""
from .common import read, sha, require


def node_map(hot, logical_nodes, storage_rows, np):
    require(storage_rows >= logical_nodes and storage_rows % 8 == 0, 'Bad payload extent')
    hot = np.asarray(hot)
    require(hot.dtype == np.int64 and hot.ndim == 1 and 0 < len(hot) < 2**31,
            'Bad node selection')
    require(hot[0] >= 0 and hot[-1] < logical_nodes and np.all(np.diff(hot) > 0),
            'Node IDs must be sorted, unique and in range')
    slots = np.zeros(storage_rows, dtype=np.int32)
    slots[hot] = np.arange(1, len(hot) + 1, dtype=np.int32)
    return slots


def install_cache(loader, arm, base, bundle, p, torch, np, binding, storage_rows):
    if arm == 'gids_node':
        path = binding['node_cache']['path']
        require(sha(path) == binding['node_cache']['sha256'], 'Node hot set changed')
        hot = np.load(path, mmap_mode='r')
        slots = node_map(hot, p['graph']['nodes'], storage_rows, np)
        rowmap = torch.from_numpy(slots).to(loader.gids_device)
        del slots
        rows = torch.from_numpy(hot.copy()).to(loader.gids_device)
        loader.cpu_backing_buffer(128, len(hot))
        loader.BAM_FS.set_cpu_buffer_rows(rows.data_ptr(), len(hot), rowmap.data_ptr(), storage_rows)
        refs = dict(primary_rows=hot, row_map=rowmap, device_rows=rows)
        rowmap_bytes = rowmap.numel() * rowmap.element_size()
    else:
        path = base / ('full_cpu_rows.npy' if arm == 'digit_full' else 'gids_cpu_rows.npy')
        hot = np.load(path, mmap_mode='r')
        loader.cpu_backing_buffer(128, len(hot))
        loader.set_cpu_buffer(torch.from_numpy(hot.copy()), len(hot))
        refs = dict(primary_rows=hot)
        rowmap_bytes = 0
    require(len(hot) == p['cpu_cache_rows'], 'CPU feature budget differs')
    result = dict(rows=len(hot), cpu_feature_bytes=len(hot) * 512,
                  hot_sha256=sha(path), mapping=p['arms'][arm]['cpu_mapping'],
                  row_map_gpu_bytes=rowmap_bytes, gpu_feature_bytes=p['gpu_cache_bytes'],
                  padding_uncached=True)
    validate_binding(result, arm, p, binding)
    return result, refs


def validate_binding(value, arm, p, binding):
    require(value['rows'] == p['cpu_cache_rows'] and
            value['cpu_feature_bytes'] == p['cpu_cache_rows'] * 512 and
            value['gpu_feature_bytes'] == p['gpu_cache_bytes'], 'Unequal feature-cache budget')
    require(value['mapping'] == p['arms'][arm]['cpu_mapping'] and value['padding_uncached'],
            'Wrong cache membership semantics')
    if arm == 'gids_node':
        require(value['hot_sha256'] == binding['node_cache']['sha256'] and
                value['row_map_gpu_bytes'] == binding['node_cache']['storage_rows'] * 4,
                'Wrong node set or metadata budget')
    else:
        rel = p['base_layout'] + ('/full_cpu_rows.npy' if arm == 'digit_full' else '/gids_cpu_rows.npy')
        require(value['hot_sha256'] == binding['files'][rel]['sha256'] and
                value['row_map_gpu_bytes'] == 0, 'Historical cache changed')
