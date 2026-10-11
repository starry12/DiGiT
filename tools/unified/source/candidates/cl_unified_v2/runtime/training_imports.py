import importlib


def preload():
    names=(
        'candidates.cl_unified_v2.profile',
        'candidates.cl_unified_v2.integration',
        'candidates.cl_unified_v2.runtime.device',
        'candidates.cl_unified_v2.runtime.external',
        'candidates.cl_unified_v2.runtime.counters',
        'candidates.cl_unified_v2.runtime.timing',
        'candidates.cl_unified_v2.runtime.numa_guard',
        'candidates.ukl_training_native_v15r11.external',
        'candidates.group_cache_migration_v1.large',
        'candidates.group_cache_migration_v1.prepare',
        'candidates.group_cache_migration_v1.core',
        'candidates.group_cache_migration_v1.io',
        'candidates.unified_training_v2.large_profile',
        'candidates.ukl_postprocess_v22.request',
        'candidates.uks_pa_fast_v1.model',
        'candidates.large_graph_fast_v1.pipeline',
    )
    for name in names:importlib.import_module(name)
    return names
