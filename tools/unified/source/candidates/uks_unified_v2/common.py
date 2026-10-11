import json
from pathlib import Path
from candidates.group_cache_migration_v1.common import read,write,sha,require
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/uks_unified_20261011_v2'
def config(mode):
    from candidates.unified_training_v2.config import load_config
    c=load_config('UKS',mode)
    c['execution_capabilities']=dict(digit_async_prefetch=True,digit_model_optimization=True,gpu_capture=True)
    return c
def verify():
    from candidates.uks_pa_fast_v1.common import verify as parent
    parent()
    for name,h in read(HERE/'manifest.json')['files'].items():
        require(sha(ROOT/name)==h,'UKS unified source changed: '+name)
    return sha(HERE/'manifest.json')
