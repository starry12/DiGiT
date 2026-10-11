"""Seal the isolated implementation once, after CPU/CUDA fixtures pass."""
from pathlib import Path
from .common import HERE,ROOT,write,sha,require

def main():
    from candidates.group_cache_migration_v1.common import verify as parent
    from .cache import HISTORY
    require(not (HERE/'manifest.json').exists(),'Never re-sign a frozen candidate')
    own=[p for p in HERE.rglob('*') if p.is_file() and p.suffix in ('.py','.json','.cu','.cuh','.h','.sh','.so') and '__pycache__' not in p.parts]
    dependencies=[ROOT/p for p in ('candidates/pa_sage_cache_policy_v1/mapping.py',
        'candidates/pa_sage_cache_policy_v1/common.py','candidates/cache_epoch_v1/io.py',
        'candidates/cache_epoch_v1/common.py','candidates/ukl_runtime_prepare_v11r1/data.py',
        'bam/build/lib/libnvm.so')]
    dependencies.extend(p for p in (ROOT/'candidates/io_accounting_v1/native').rglob('*')
                        if p.is_file() and p.suffix in ('.h','.hpp','.cuh'))
    dependencies.extend(HISTORY/p for p in ('smoke_digit_full/digit_full/report.json',
        'smoke_digit_full/digit_full/full_epoch_profile.json','smoke_digit_full_digit_full_receipt.json'))
    value=dict(schema='digit-ig-demand-pair-v1',optimized_parent_sha256='35b42b5e69b6caf0e4e22cc2589a4ad1fddc472d83378d265bded1694074178e',parent_sha256=parent(),
               files={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(own+dependencies))},
               scope='same IG inputs; exact physical batch demand and CPU eligibility before pairing; locked GPU state prevents unsolicited sibling fill on requested-row hits')
    write(HERE/'manifest.json',value);print(dict(manifest_sha256=sha(HERE/'manifest.json'),files=len(value['files'])))

if __name__=='__main__':main()
