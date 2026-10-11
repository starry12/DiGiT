"""Reuse accepted layout; require a separately validated adjacency overlay."""
from types import SimpleNamespace
from candidates.cache_epoch_v1.ukl import dataset as original
from candidates.group_cache_migration_v1.large_overlay import metadata, binding as overlay_binding
ROOT=original.ROOT;OUT=original.OUT;N=original.N;ROWS=original.ROWS;LOCK=original.LOCK
source_device=original.source_device;verify_manifest=original.verify_manifest
METADATA=metadata('UKL')
def bound(): return overlay_binding('UKL')
BASE=SimpleNamespace(binding=bound, source_device=source_device,
                     EXTENT=sum((v['length']+4095)//4096*4096 for v in bound().values()))
binding=bound
