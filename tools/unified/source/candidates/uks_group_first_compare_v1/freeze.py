from .common import *

def main():
    require(not (HERE/'manifest.json').exists(),'Preserve frozen candidate')
    checks=read(OUT/'cpu_checks.json');require(checks['passed'],'CPU checks missing')
    from candidates.uks_group_first_v1.common import HERE as parent
    build=read(parent/'runtime/build_receipt.json')
    require(build['source_sha256']==verify_first() and build['binary_sha256']==sha(BINARY),'Priority binary mismatch')
    deps=['candidates/pa_sage_mixed_group_v1/queue.py',
          str(BINARY.relative_to(ROOT)),str((parent/'runtime/build_receipt.json').relative_to(ROOT))]
    files=list(HERE.glob('*.py'))+[HERE/'README.md']
    write(HERE/'manifest.json',dict(weighted_source=verify_weighted(),first_source=verify_first(),
          files={p.name:sha(p) for p in files},dependencies={p:sha(ROOT/p) for p in deps}))
    source=verify();print('Frozen '+source)
    write(OUT/'ready.json',dict(source_sha256=source,cpu_checks=checks,native_training_accepted=False))

if __name__=='__main__':main()
