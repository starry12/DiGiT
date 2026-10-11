"""Freeze a new candidate after CPU checks; never overwrite a frozen version."""
from .common import *

def main():
    require(not (HERE/'manifest.json').exists(),'Preserve frozen candidate')
    checks=read(OUT/'cpu_checks.json');require(checks['passed'] and checks['builder_sha256']==sha(BINARY),'Missing current CPU checks')
    dependencies=[ROOT/'candidates/uks_freq_bfs_retry_v1/manifest.json',ROOT/'candidates/uks_merge_audit_v1/audit.py',
        ROOT/'candidates/uks_mixed_1k2k_v1/manifest.json',ROOT/'candidates/uks_group_incremental_v1/manifest.json',
        ROOT/'results/uks_native_20260929_v1/binding.json',ROOT/'results/uks_freq_bfs_20260930_v1/bfs_receipt.json',
        Path('/srv/digit-ae/admin/identity_v1/stable_identity.py'),Path('/srv/digit-ae/admin/identity_v1/registry.json')]
    files=list(HERE.glob('*.py'))+[HERE/'reuse.cpp',HERE/'README.md',HERE/'protocol.json']
    write(HERE/'manifest.json',dict(files={f.name:sha(f) for f in files},dependencies={str(p):sha(p) for p in dependencies},builder_sha256=sha(BINARY)))
    source=verify();write(OUT/'ready.json',dict(source_sha256=source,cpu_checks=checks,full_graph_prepared=False,native_training_accepted=False))
    print('Frozen '+source)

if __name__=='__main__':main()
