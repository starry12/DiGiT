"""Bind new orchestration to existing immutable v1/v2 preparation candidates."""
from .common import *

def main():
    require(not (HERE/'manifest.json').exists(),'Preserve frozen experiment')
    checks=read(OUT/'cpu_checks.json');require(checks['passed'],'Missing CPU checks')
    deps=['candidates/pa_sage_mixed_group_v1/queue.py',
          'candidates/ig_sage_affinity_pair_v1/affinity.py']
    files=list(HERE.glob('*.py'))+[HERE/'README.md']
    write(HERE/'manifest.json',dict(v1_source=v1.verify(),v2_source=v2.verify(),
          files={p.name:sha(p) for p in files},dependencies={p:sha(ROOT/p) for p in deps}))
    source=verify()
    write(OUT/'ready.json',dict(source_sha256=source,cpu_checks=checks,
          preparation_complete=False,native_training_accepted=False))
    print('Frozen '+source)

if __name__=='__main__':main()
