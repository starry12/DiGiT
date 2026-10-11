"""Reuse only a previous accepted same-series full profiling receipt."""
import os
from pathlib import Path
from .common import read,write,verify,require
from candidates.unified_training_v2.config import digest

def prepare_cache(arm,native,sampling,accepted,previous_hot,output,check,event):
    from candidates.group_cache_migration_v1.large import prepare_cache as original
    from candidates.group_cache_migration_v1.io import read_array
    from candidates.group_cache_migration_v1.core import digest_array
    if arm=='gids':return original(arm,native,sampling,accepted,previous_hot,output,check,event)
    from .common import config
    from .runtime.protocol import SCOPE
    c=config(SCOPE)
    if c['cache_profile']['scope']=='short_window':
        from candidates.unified_training_v2.large_profile import prepare_cache as bounded
        prepare=lambda *args:bounded(*args,c)
    else:prepare=original
    binding=dict(source_sha256=verify(),graph_sha256=digest(accepted['stages']['graph']),capacity=accepted['cpu_cache_rows'],profile_policy=c['cache_profile'])
    reuse=os.environ.get('UKL_UNIFIED_PROFILE')
    if reuse:
        path=Path(reuse);record=read(path)
        require(record['binding']==binding and record['metadata']['profile']['complete'],'Wrong reused profile binding')
        metadata=record['metadata']
        if c['cache_profile']['scope']=='short_window':
            from candidates.unified_training_v2.profiling import validate_profile
            validate_profile(metadata['profile'],c,metadata['profile']['canonical_root_count'])
        hot=read_array(metadata['hot_file'],check=check)
        require(len(hot)==accepted['cpu_cache_rows'] and digest_array(hot)==metadata['selection']['hot_sha256'],'Reused hot set changed')
        metadata=dict(metadata,reused_from=str(path))
        write(output/'profile_binding.json',dict(binding=binding,metadata=metadata));return hot,metadata
    hot,metadata=prepare(arm,native,sampling,accepted,previous_hot,output,check,event)
    write(output/'profile_binding.json',dict(binding=binding,metadata=metadata));return hot,metadata
