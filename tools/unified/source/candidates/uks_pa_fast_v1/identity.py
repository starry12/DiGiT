"""Install authenticated UUID compatibility before any by-value UKS imports."""
from pathlib import Path
import sys

ADMIN = Path('/srv/digit-ae/admin/identity_v1')


def install():
    sys.path.insert(0, str(ADMIN))
    import stable_identity
    if Path(stable_identity.__file__).resolve() != ADMIN / 'stable_identity.py':
        raise RuntimeError('Unexpected identity implementation')
    registry = stable_identity.install('UKS')
    # Installing the provider hook cannot replace references already copied by
    # "from ... import identity". Refuse late initialization instead of ignoring
    # provenance checks or rewriting the authenticated historical identities.
    for name in ('candidates.uks_native_v1.common', 'candidates.uks_native_v1.binding'):
        module = sys.modules.get(name)
        if module is not None and not getattr(module.identity, '_ae_stable_identity', False):
            raise RuntimeError('UUID compatibility installed too late: ' + name)
    return registry
