from .common import *


def main():
    require(not (HERE/'manifest.json').exists(),'Preserve frozen candidate')
    checks=read(OUT/'cpu_checks.json');require(checks['passed'],'CPU checks missing')
    deps=['candidates/pa_sage_mixed_group_v1/queue.py',
        'candidates/ig_sage_affinity_pair_v1/affinity.py',
        'candidates/ig_sage_host_opt_v1/affinity.py',
        'results/uks_group_first_compare_20261010_v1/summary.json']
    write(HERE/'manifest.json',dict(parent_source=verify_parent(),
        files={p.name:sha(p) for p in HERE.iterdir() if p.is_file() and p.suffix in ('.py','.md')},
        dependencies={rel:sha(ROOT/rel) for rel in deps}))
    source=verify();write(OUT/'cpu_checks.json',dict(checks,source_sha256=source))
    write(OUT/'ready.json',dict(code_ready=True,source_sha256=source,cpu_checks_passed=True,
        native_training_accepted=False,policy='all arms CPU2',rounds=5))
    print('Frozen '+source)


if __name__=='__main__':main()
