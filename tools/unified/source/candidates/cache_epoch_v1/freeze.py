"""Freeze new sources and unchanged parent dependencies after CPU tests."""
import ast
import json
import unittest
from pathlib import Path
from .common import ROOT,HERE,OUT,sha,write,verify,require


def main():
    require(not (HERE/'manifest.json').exists(),'Candidate already frozen')
    from .tests import Core
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Core))
    require(result.wasSuccessful(),'CPU tests failed')
    files={}
    for p in HERE.rglob('*.py'):
        ast.parse(p.read_text(),str(p));files[str(p.relative_to(ROOT))]=sha(p)
    files[str((HERE/'README.md').relative_to(ROOT))]=sha(HERE/'README.md')
    # Bind the original large-graph dependencies and their native libraries.
    for name in ['ukl_common_gpu_five_20261006_v27','cl_common_gpu_five_20261006_v17']:
        path=ROOT/'results'/name/'manifest.json'
        for rel,digest in json.loads(path.read_text()).items():
            require(sha(ROOT/rel)==digest,'Frozen parent dependency changed: '+rel)
            files[rel]=digest
    from candidates.uks_group_first_cpu2_v1.common import verify as verify_uks
    parent=verify_uks()
    # Capture all source files used by the UKS adapter dependency family, leaving
    # its own manifest verification enabled in native graph/preparation calls.
    prefixes=('uks_','ig_sage_affinity_pair_v1','ig_sage_host_opt_v1','pa_sage_cache_policy_v1','pa_sage_mixed_group_v1')
    for directory in (ROOT/'candidates').iterdir():
        if directory.is_dir() and directory.name.startswith(prefixes):
            for p in directory.rglob('*'):
                if p.is_file() and p.suffix in ('.py','.json','.cu','.h','.cpp') and '__pycache__' not in p.parts:
                    files[str(p.relative_to(ROOT))]=sha(p)
    write(HERE/'manifest.json',dict(files=files,parent_uks=parent,policy='complete epoch, same BFS batches, independent sampling seed'))
    source=verify()
    write(OUT/'cpu_checks.json',dict(passed=True,tests=result.testsRun,source_sha256=source,
          gpu_used=False,raw_ssd_access=False,native_training_accepted=False))
    print('Frozen '+source)


if __name__=='__main__':main()
