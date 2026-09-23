"""Candidate-specific budget: original int64 CSC stays in pinned host memory."""
import math,copy
from training.sage.common import *

def estimate():
    setup()
    from digit.gpu_admission import estimate as base_estimate
    p=cfg();m=copy.deepcopy(read(ROOT/p['base_layout']/'final/bundle/manifest.json'))
    source=source_config();n=p['graph']['nodes'];e=p['graph']['edges'];old_e=m['dataset']['num_edges'];reverse=e-old_e
    m['dataset']['num_edges']=e;m['files']['reordered_indices']['shape'][0]+=reverse
    plan=base_estimate(m,batch_size=p['batch_size'],fanouts=p['fanouts'],hidden=p['hidden'],classes=p['classes'],cache_mib=4096,metadata_mode='cpu_eid')
    parts=plan['components_bytes']
    for key in ('reordered_indices','group_members','group_storage_base','supernode_to_group','node_to_primary_row'):parts[key]//=2
    parts['bam_storage_range_allowance']=m['feature']['num_storage_rows']*512//4096*32
    parts['labels_masks_seed_indices']=64*2**20
    parts['common_degree_counter']=n*4
    parts['useful_io_bitmap_and_counters']=p['gpu_cache_bytes']//4096*8+48
    plan.update(schema='digit-bidir-uva64-admission-v1',backend=p['metadata_mode'],required_bytes=math.ceil(sum(parts.values())*1.2),
                host_required_bytes=320*2**30,original_csc_host_bytes=8*(n+1+2*e),original_csc_gpu_bytes=0,
                metadata_bytes=sum(parts[k] for k in ('reordered_indptr','reordered_indices','group_members','group_storage_base','supernode_to_group','node_to_primary_row')),
                host_budget_note='320 GiB includes pinned int64 CSC, mmap validation, graph metadata, CPU preload, source audit and <=16 GiB evaluation traces',
                source_profile='64-bit offsets/EIDs; compact checked int32 node/storage IDs only; L40 CUDA sm89',protocol_sha256=sha(P))
    require(max(n+m['grouping']['num_groups']-1,m['feature']['num_storage_rows']-1)<=2**31-1,'Node/storage IDs need wider type')
    return plan
