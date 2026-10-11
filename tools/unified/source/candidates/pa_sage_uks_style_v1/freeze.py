"""Freeze sources after CPU/native build checks; GPU/SSD acceptance is separate."""
import ast
import os
import subprocess
import sys
import time
from .common import *

def main():
    setup()
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    subprocess.run([PY,'-B','-m','unittest',MODULE+'.tests','-v'],env=env,check=True)
    for f in HERE.glob('*.py'):ast.parse(f.read_text(),filename=str(f))
    files={str(f.relative_to(HERE)):sha(f) for f in sorted(HERE.rglob('*')) if f.is_file() and '__pycache__' not in f.parts and f.name not in ('manifest.json','build.log') and not f.name.endswith('.partial.so')}
    # Bind imported implementation files and the accepted strict sampler binary.
    dependencies={}
    dirs=['ae','candidates/pa_sage_bidir_native_v2','candidates/pa_sage_ae_512b_v1','candidates/uks_group_first_v1',
          'candidates/uks_pa_fast_v1','candidates/cache_epoch_v1','candidates/group_cache_migration_v1',
          'candidates/pa_sage_cache_policy_v1','candidates/pa_sage_cache_policy_v6','candidates/pa_sage_512b_latency_v1',
          'candidates/uks_window_diag_v1','candidates/uks_group_first_cpu2_v1','candidates/pa_sage_bfs_main_v1',
          'candidates/uks_native_v1','candidates/uks_group_reuse_v2','candidates/pa_sage_mixed_bfs_v5',
          'candidates/uks_group_compare_v2','candidates/io_accounting_v1','candidates/ig_after_cl_v1']
    for d in dirs:
        for f in (ROOT/d).rglob('*'):
            if f.is_file() and f.suffix in ('.py','.json','.h','.cu','.so') and '__pycache__' not in f.parts and f.name!='build_receipt.json':dependencies[str(f.relative_to(ROOT))]=sha(f)
    dependencies['bam/build/lib/libnvm.so']=sha(ROOT/'bam/build/lib/libnvm.so')
    from .provenance import check_transition,PREVIOUS
    require(sha(HERE/'prior_manifest.json')==PREVIOUS,'Wrong pre-repair manifest')
    manifest=dict(files=files,dependencies=dependencies,created_unix=time.time(),frozen_parents_unchanged=True)
    check_transition(read(HERE/'prior_manifest.json'),manifest)
    write(HERE/'manifest.json',manifest)
    source=verify();binary_receipt();OUT.mkdir(parents=True,exist_ok=True)
    write(OUT/'cpu_checks.json',dict(passed=True,source_sha256=source,unit_tests=15,native_compiled=True,gpu_accepted=False,ssd_accepted=False))
    print(source)

if __name__=='__main__':main()
