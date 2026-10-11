"""Wait for the controller's init cgroup before importing training libraries."""
import os
from pathlib import Path
from .phase import await_phase


if __name__=='__main__':
    if os.environ.get('UKL_V27_BOUNDED_WORKER')!='1':
        raise RuntimeError('Bounded worker required')
    try:
        await_phase(Path(os.environ['UKL_V27_OUTPUT']), 'init')
        from candidates.group_cache_migration_v1.common import stable
        stable('UKL')
        from .worker import main
        main()
    except BaseException as error:
        from .worker import persist_failure
        persist_failure(error)
        raise
