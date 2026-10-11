"""New protocol; preserves the older full-epoch candidate as historical evidence."""
N=787801471
E=47614527250

def plan():
 return dict(schema='ukl-window-native-v2',dataset='UKL',nodes=N,source_edges=E,
  model='sage',feature_dim=128,row_bytes=512,classes=19,hidden=128,layers=3,fanouts=[10,5,5],batch_size=1024,
  seed=0,optimizer=dict(name='Adam',lr=.001,weight_decay=.001),dropout=.2,
  train_fraction=.1,train_nodes=N//10,warmup_batches=20,measured_batches=300,rounds=5,
  selection='maximum_same_round_speedup_out_of_five',accuracy=False,validation=False,full_epoch=False,
  gpu_cache_bytes=4*2**30,cpu_cache_rows=N//10,group_size=2,replication_ratio=.2,
  proposed_io=dict(cache_line_bytes=512,read_bytes=[512,1024],status='geometry proposed; SSD backend not integrated'),
  features=dict(mode='logical-consistent synthetic',replicas_identical=True,feature_seed=20260922,label_seed=20260923),
  gids=dict(cpu_hot='RevPR',gpu_cache='baseline legacy',order='fixed random window',cpu_affinity='default'),
  digit=dict(cpu_hot='independent grouped Freq,100 batches,seed23',gpu_cache='FIFO',order='BFS of full training-induced graph,then fixed window',cpu_affinity='CPU2'),
  same_node_set=False,window_identity='Bind per-arm root hashes; repeat identical windows across all five rounds',
  gpu_selection='one idle GPU among 0..3 for all ten formal workers',
  timing='synchronized window wall time, exclude preparation/preload/model init/warmup; no added per-batch synchronization',
  checks=['short native acceptance before formal rounds','normal exit','exact batch counts','finite values','I/O reconciliation','same GPU/source/input identities across rounds'],
  native_training_ready=False,full_graph_build_enabled=False,large_output='/mnt/n0/digit/ukl_window_native_v2',
  limits=['Performance only; no convergence claim.','Maximum ratio is not average or stable performance.','BFS and random windows need not contain identical nodes.','Native prototype changes graph representation for both arms; no upstream GIDS compatibility claim.'])

def budget(available_bytes=None):
 en=E+N;g_primary=(N-N//10)//2;g_replica=(N//5)//2;g=g_primary+g_replica
 original=8*(N+1)+4*en
 # A selected g2 group replaces two edge occurrences by one; future builder must prove this identity.
 grouped=8*(N+1)+4*(en-g)
 groups=8*g+8*g+8*N  # two i32 members, i64 base, i64 primary map; direct group IDs
 cpu_cache=(N//10)*512
 slots=4*N
 parts=dict(original_csc=original,grouped_csc=grouped,group_and_storage_maps=groups,
            cpu_features=cpu_cache,logical_cpu_slot_map=slots,root_window=320*1024*8,
            feature_generation_buffer=64*2**20)
 resident=sum(parts.values());headroom=64*2**30
 return dict(schema='ukl-memory-budget-v2',formal_admission=False,nodes=N,normalized_edges_upper=en,
  group_scenario=dict(primary=g_primary,replica=g_replica,total=g,assumption='maximum groups; not measured'),
  components_bytes=parts,host_resident_scenario_bytes=resident,headroom_bytes=headroom,
  host_required_scenario_bytes=resident+headroom,current_available_bytes=available_bytes,
  scenario_fits_current_available=(resident+headroom<=available_bytes) if available_bytes else None,
  original_plus_grouped_graph_bytes=original+grouped,
  legacy_csc_plus_eids_upper_bytes=8*(N+1)+16*en,
  full_eid_array_bytes=0,full_graph_dgl=False,full_graph_gpu_copy=False,
  original_feature_disk_bytes=N*512,
  proposed_packed_feature_disk_bytes=(N+2*g_replica)*512,
  proposed_packed_layout_implemented=False,
  gpu_known_bytes=dict(feature_cache=4*2**30,prototype_graph_arrays=0),gpu_total_required_bytes=None,
  preprocessing=dict(revpr_two_float64_vectors_bytes=16*N,frequency_uint64_counts_bytes=8*N,
                     uks_builder_per_edge_byte_marks_upper_bytes=en,
                     note='Do not add all preparation phases to runtime; release temporary arrays, budget each phase separately.'),
  unresolved=['actual normalized edge/group/storage counts','pinning and registration overhead','model/optimizer and sampled frontier peaks','SSD page-cache metadata','external-transpose/sort disk and RAM peaks','full BFS workspace'],
  recommendation='Keep original/grouped CSC and maps in host/UVA; consider partitioned graph access if measured peak plus headroom does not fit.')
