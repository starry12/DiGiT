"""Isolated strict group-first dispatch; original shared CSC/EID ABI retained."""
import importlib.util
import sys
from pathlib import Path
from candidates.group_cache_migration_v1.common import OUT, require, read, sha, ROOT
def load():
    name='IGGroupFirstMigration';p=OUT/'build'/ (name+'.so');r=read(str(p)+'.json')
    require(sha(p)==r['sha256'],'IG sampler binary changed')
    for rel,h in r['sources'].items():require(sha(ROOT/rel)==h,'IG sampler source changed')
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);sys.modules[name]=mod
    return sys.modules[name]
def expected_configuration(arm):
    from sampler_config import expected_configuration as parent
    from ae.common import binding
    value=parent(arm)
    if arm=='full':value.update(module=binding(OUT/'build/IGGroupFirstMigration.so'),selection_policy='strict_group_first')
    return value
def configure(arm):
    from . import fast_uva as uva_sampler
    from sampler_config import configure as parent
    parent(arm)
    if arm=='full':
        module=load();require(module.GROUP_FIRST_API==1,'Missing group priority ABI')
        module.set_optimized(True);module.set_cooperative(True);module.set_bucket(False,-1,2**63-1)
        uva_sampler.native=module
    return expected_configuration(arm)
