from candidates.ukl_window_native_v2.protocol import plan as old_plan,budget as old_budget,N,E

def plan():
 p=old_plan();p.update(schema='ukl-sparse-overlay-v3',graph_representation='one original CSC + sparse covered EIDs + per-owner group IDs',full_graph_build_enabled=True,large_output='/mnt/n0/digit/ukl_sparse_overlay_v3',native_training_ready=False)
 return p

def budget(available_bytes=None):
 old=old_budget(available_bytes);g=old['group_scenario']['total'];parts=dict(old['components_bytes']);del parts['grouped_csc']
 parts.update(overlay_owner_offsets=8*(N+1),overlay_covered_eids=16*g,overlay_group_ids=4*g)
 resident=sum(parts.values());required=resident+old['headroom_bytes']
 return dict(schema='ukl-sparse-overlay-budget-v3',components_bytes=parts,host_resident_scenario_bytes=resident,host_required_scenario_bytes=required,headroom_bytes=old['headroom_bytes'],saved_bytes=old['host_resident_scenario_bytes']-resident,current_available_bytes=available_bytes,scenario_fits_current_available=required<=available_bytes if available_bytes else None,formal_admission=False,g2_build_explicit_array_scenario_bytes=max(parts['original_csc']+30*N+24*g+8*2**30,parts['original_csc']+32*N+24*g),group_scenario=old['group_scenario'],gpu_total_required_bytes=None,transpose=dict(bins=4096,java_heap_bytes=4*2**30,gather_explicit_array_cap_bytes=8*2**30,pair_disk_upper_bytes=E*8,csc_disk_upper_bytes=8*(N+1)+4*(E+N)),limitations=['Planning upper bound, not measured runtime RSS','Full mmap/UVA registration and native training pending','Production group selection requires a bound hot-set input'])
