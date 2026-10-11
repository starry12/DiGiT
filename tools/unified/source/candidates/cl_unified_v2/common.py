import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/cl_unified_20261011_v2'
from candidates.group_cache_migration_v1.common import read,write,sha,require

def config(mode):
    from candidates.unified_training_v2.config import load_config
    c=load_config('CL',mode)
    # Private async ABI; runtime acceptance requires fresh CUDA and native checks.
    c['execution_capabilities']=dict(digit_async_prefetch=True,digit_model_optimization=True,
        native_sampler='v2_group_reuse_strict_group_first_existing_owned_CL',gpu_capture=True)
    return c

_SOURCE_GUARD=None

def verify_full():
    from candidates.group_cache_migration_v1.common import verify as parent
    parent();m=read(HERE/'manifest.json')
    for name,h in m['files'].items():require(sha(ROOT/name)==h,'CL unified source changed: '+name)
    return sha(HERE/'manifest.json')

def verify():
    return _SOURCE_GUARD.check() if _SOURCE_GUARD is not None else verify_full()

def begin_source_guard(expected):
    global _SOURCE_GUARD
    if _SOURCE_GUARD is not None:raise RuntimeError('Source guard already initialized')
    from .source_guard import SourceGuard,manifests
    _SOURCE_GUARD=SourceGuard(ROOT,manifests(ROOT),verify_full,expected)
    return _SOURCE_GUARD
