"""Apply the existing DiGiT CPU2 policy to both arms, only inside this worker."""
import contextlib
import os
from candidates.ig_sage_host_opt_v1.affinity import snapshot
from .common import require


def apply_affinity(arm):
    require(arm in ('gids','digit_full'),'Unknown arm')
    before=snapshot()
    require(2 in os.sched_getaffinity(0),'CPU2 outside worker CPU set')
    for tid in before['thread_masks']:
        try:os.sched_setaffinity(int(tid),{2})
        except ProcessLookupError:pass
    value=dict(arm=arm,policy='one_core',cpu=2,before=before,after=snapshot())
    validate_affinity(dict(initial=value,final=value['after']),arm)
    return value


def validate_affinity(value,arm):
    require(arm in ('gids','digit_full'),'Unknown arm')
    initial=value['initial']
    require(initial['arm']==arm and initial['policy']=='one_core' and initial['cpu']==2,'Wrong affinity policy')
    require(bool(initial['before']['thread_masks']),'Missing original CPU masks')
    for snap in (initial['after'],value['final']):
        masks=snap['thread_masks']
        require(bool(masks) and all(mask==[2] for mask in masks.values()),'Worker thread escaped CPU2')
    return True


@contextlib.contextmanager
def install():
    # The retained worker imports these functions inside run()/smoke(). Replace
    # that process-local dispatch for one call, without editing any frozen file.
    from candidates.ig_sage_affinity_pair_v1 import affinity as original
    old_apply,old_validate=original.apply_affinity,original.validate_affinity
    applied=[]
    def tracked(arm):
        result=apply_affinity(arm);applied.append(result);return result
    original.apply_affinity,original.validate_affinity=tracked,validate_affinity
    try:yield applied
    finally:original.apply_affinity,original.validate_affinity=old_apply,old_validate
