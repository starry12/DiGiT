"""Conservative separate host/pinned/GPU budget; not a measured training peak."""
from candidates.ukl_runtime_prepare_v11r1 import protocol as U
GIB=2**30;MIB=2**20

def budget(arm,page_state_bytes=32):
    if arm not in ('gids','digit') or page_state_bytes!=32:raise ValueError('Unsupported arm/native page ABI')
    n,rows=U.N,U.N if arm=='gids' else U.ROWS;extent=(rows+7)//8*8
    host=dict(graph_arena=U.BASE.EXTENT,
      cpu_features=(n//10)*512,hot_ids=(n//10)*8,hot_physical_rows=(n//10)*8,
      storage_inverse=0 if arm=='gids' else U.ROWS*4,
      backend_page_initialization=16*MIB,
      logical_slot_lookup=n*4,mapping_chunk=64*MIB,framework_and_sampling=16*GIB)
    # Every SAGE frontier is bounded by 1024 * 6 * 6 * 11 = 405504.
    gpu=dict(feature_cache=4*GIB,cpu_row_slot_map=extent*4,page_state=extent*page_state_bytes,
      cache_and_queue_allowance=2*GIB,sampling_scratch=256*MIB,model_blocks_activations=2*GIB,framework_reserve=2*GIB,pair_staging=16*MIB,pair_hash=4*MIB)
    return dict(arm=arm,nodes=n,logical_storage_rows=rows,padded_storage_rows=extent,
      host_components=host,host_accounted=sum(host.values()),host_cgroup_max=288*GIB,host_cgroup_high=287*GIB,
      host_min=352*GIB+128*MIB,host_reserve=64*GIB,own_file_cache_limit=128*MIB,
      source_read_rate=GIB,source_write_rate=GIB,application_read_rate=512*MIB,
      pinned_graph_and_features=host['graph_arena']+host['cpu_features'],memlock=256*GIB,
      gpu_components=gpu,gpu_accounted=sum(gpu.values()),gpu_free_min=44*GIB,
      max_input_rows=405504,feature_dim=128,row_bytes=512,page_bytes=512,
      cpu_rows=n//10,gpu_feature_cache_bytes=4*GIB,gpu_policy='legacy' if arm=='gids' else 'fifo',
      graph_file_registration=False,full_training_peak_measured=False)
