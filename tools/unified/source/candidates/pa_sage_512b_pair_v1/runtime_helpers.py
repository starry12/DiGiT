"""Directed CSC, 512-byte capacity admission and logical CPU cache aliases."""
from .common import *


def load_graph(binding, nodes, edges, pin=True):
    import numpy as np
    import torch
    import dgl
    arrays = [np.load(binding['csc'][k], mmap_mode='c') for k in ('indptr', 'indices')]
    # Both arms use CSC-position EIDs. DiGiT's UVA resolver needs explicit IDs.
    eids = torch.arange(edges, dtype=torch.int64)
    graph = dgl.graph(('csc', (torch.from_numpy(arrays[0]), torch.from_numpy(arrays[1]), eids)),
                      num_nodes=nodes, idtype=torch.int64).formats('csc')
    require(graph.num_nodes()==nodes and graph.num_edges()==edges, 'Graph extent changed')
    if pin:
        graph._graph.pin_memory_()
        require(graph.is_pinned() and all(t.is_pinned() for t in graph.adj_tensors('csc')),
                'Explicit int64 CSC registration failed')
    return graph, arrays


def load_bundle(binding):
    import numpy as np
    from digit.artifacts import ArtifactBundle
    root=Path(binding['bundle']);manifest=read(root/'manifest.json')
    require(sha(root/'manifest.json')==binding['bundle_manifest_sha256'], 'Bundle changed')
    arrays={k:np.load(root/v['path'],mmap_mode='r') for k,v in manifest['files'].items()
            if k!='reordered_features'}
    return ArtifactBundle(root,manifest,arrays)


def admission(p, manifest, arm):
    """Conservative runtime budget, including every 512-B physical range entry."""
    setup()
    from digit.gpu_admission import estimate
    base=estimate(manifest,batch_size=p['batch_size'],fanouts=p['fanouts'],hidden=p['hidden'],
                  classes=p['classes'],cache_mib=p['cache_mib'],metadata_mode='cpu_eid')
    parts=base['components_bytes']
    compact=('reordered_indices','group_members','group_storage_base','supernode_to_group','node_to_primary_row')
    for k in compact:parts[k]//=2
    rows=manifest['feature']['num_storage_rows'] if arm=='digit' else p['nodes']
    if arm=='gids':
        for k in compact+('reordered_indptr',):parts[k]=0
    parts['bam_storage_range_allowance']=rows*32  # one 32-byte descriptor per 512-B line
    parts['labels_masks_seed_indices']=p['nodes']*8+p['train_count']*16
    parts['exact_cpu_row_map']=rows*4 if arm=='digit' else 0
    parts['cpu_preload_indices']=p['cpu_rows']*8
    # Window hints and merged reads retain several blocks/features at once.
    parts['window_retained_batches_allowance']=2*2**30
    return dict(arm=arm,components_bytes=parts,required_bytes=math.ceil(sum(parts.values())*1.15),
                host_required_bytes=p['host_admission_gib']*2**30,
                original_csc_host_bytes=8*(p['nodes']+1+2*p['edges']),
                storage_rows=rows,page_bytes=512,original_csc_gpu_bytes=0)


def install_cache(loader, arm, binding, bundle, p):
    import numpy as np
    import torch
    from candidates.pa_sage_ablation_cache_v3.cache import logical_slots, map_storage
    count=p['cpu_rows']
    if arm=='gids':
        hot=torch.load(p['pin_file'],map_location='cpu')[:count].clone()
        require(hashlib.sha256(hot.numpy().tobytes()).hexdigest()==binding['hot_rows_sha256'], 'RevPR changed')
        loader.cpu_backing_buffer(128,count);loader.set_cpu_buffer(hot,count)
        return dict(rows=count,hot_set='RevPR',logical_hot_sha256=binding['hot_rows_sha256'],
                    replica_aliases=False,padding_uncached=True,row_map_bytes=0), {}
    hot=np.load(binding['digit_hot_path'],mmap_mode='r')
    require(len(hot)==count and sha(binding['digit_hot_path'])==binding['digit_hot_sha256'],'Frequency hot set changed')
    slots=logical_slots(hot,p['nodes'],np);storage=bundle.arrays['storage_to_node']
    primary=np.array(bundle.arrays['node_to_primary_row'][hot],dtype=np.int64)
    require(np.array_equal(storage[primary],hot),'Primary inverse mismatch')
    rowmap=torch.empty(len(storage),dtype=torch.int32,device=loader.gids_device)
    digest=hashlib.sha256();cached_rows=0
    for lo in range(0,len(storage),1048576):
        mapped=map_storage(storage[lo:lo+1048576],slots,np)
        digest.update(mapped.tobytes());cached_rows+=int(np.count_nonzero(mapped))
        rowmap[lo:lo+len(mapped)].copy_(torch.from_numpy(mapped))
    rows=torch.from_numpy(primary).to(loader.gids_device)
    loader.cpu_backing_buffer(128,count)
    loader.BAM_FS.set_cpu_buffer_rows(rows.data_ptr(),count,rowmap.data_ptr(),len(storage))
    return dict(rows=count,hot_set='Freq',logical_hot_sha256=binding['digit_hot_sha256'],
                replica_aliases=True,padding_uncached=True,row_map_bytes=rowmap.numel()*4,
                row_map_sha256=digest.hexdigest(),cached_physical_rows=cached_rows), dict(rowmap=rowmap,rows=rows)


def verify_digit_blocks(blocks, arrays, fanouts):
    import numpy as np
    import dgl
    ptr,idx=arrays
    for b,fanout in zip(blocks,fanouts):
        u,v=b.edges(order='eid')
        src=b.srcdata[dgl.NID][u].cpu().numpy();dst=b.dstdata[dgl.NID][v].cpu().numpy()
        eids=b.edata[dgl.EID].cpu().numpy()
        require(np.all((eids>=0)&(eids<len(idx))),'EID outside original CSC')
        require(np.array_equal(src,idx[eids]) and np.array_equal(dst,np.searchsorted(ptr,eids,side='right')-1),
                'Grouped sample has nonexistent edge')
        owners=b.dstdata[dgl.NID].cpu().numpy();degrees=b.in_degrees().cpu().numpy()
        expected=np.minimum(ptr[owners+1]-ptr[owners],fanout)
        require(np.array_equal(degrees,expected),'Grouped sample cardinality mismatch')
    # Parallel graph edges can resolve to the same original EID; endpoints/cardinality
    # remain checked. Do not impose the standard DGL unique-EID oracle on that resolver.
    return True
