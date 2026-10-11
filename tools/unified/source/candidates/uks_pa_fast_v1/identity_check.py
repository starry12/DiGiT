"""Read-only preflight through the exact worker import path; no CUDA or SSD I/O."""
import json
from pathlib import Path
from unittest.mock import patch


def main():
    from . import worker
    from .identity import install
    from .common import OUT, verify, write
    from candidates.uks_group_first_cpu2_v1.common import check_prepared, DEFAULT_OUTPUT
    from candidates.uks_native_v1 import binding, common
    import stable_identity
    registry = install()
    prepared = check_prepared(DEFAULT_OUTPUT)
    bound = binding.check()
    assert getattr(binding.identity, '_ae_stable_identity', False)
    assert getattr(common.identity, '_ae_stable_identity', False)
    target = '/mnt/n0/digit/uks_prepare_20260929_v3/csc/indices.npy'
    expected = bound['files'][target]['identity']
    actual = stable_identity.stat_identity(target)
    registry.verify(target, expected)
    rejected = []
    for key in ('inode', 'bytes', 'mtime_ns', 'ctime_ns'):
        wrong = dict(expected)
        wrong[key] += 1
        assert binding.identity(target) != wrong, 'Accepted changed ' + key
        rejected.append(key)
    with patch.object(stable_identity, 'filesystem_uuid', return_value='unregistered-filesystem'):
        try:
            registry.verify(target, expected)
        except RuntimeError:
            rejected.append('filesystem_uuid')
        else:
            raise AssertionError('Accepted wrong filesystem UUID')
    result = dict(passed=True, source_sha256=verify(), worker_import=True,
                  prepared_overlays=sorted(prepared), binding_file_count=len(bound['files']),
                  registry_entries_checked=registry.require_all(),
                  device_renumbered=actual['device'] != expected['device'],
                  unchanged_identity_fields=all(actual[k] == expected[k] for k in rejected if k != 'filesystem_uuid'),
                  rejected_changes=rejected, gpu_access=False, raw_ssd_access=False)
    OUT.mkdir(exist_ok=True)
    write(OUT/'identity_checks.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
