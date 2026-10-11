"""Reuse the frozen tiny sampler check; full native smoke is required per arm."""
from .common import *


def main():
    from candidates.uks_group_reuse_v1.gpu_check import run
    source = verify()
    run(OUT/'parent_gpu_checks.json')
    result = read(OUT/'parent_gpu_checks.json')
    require(result['passed'] and result['source_sha256'] == verify_parent(), 'GPU check failed')
    write(OUT/'gpu_checks.json', dict(passed=True, source_sha256=source,
          parent_receipt_sha256=sha(OUT/'parent_gpu_checks.json'),
          native_training_accepted=False, raw_ssd_access=False))


if __name__ == '__main__':
    main()
