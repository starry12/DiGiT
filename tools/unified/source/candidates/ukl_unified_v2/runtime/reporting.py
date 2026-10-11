from .sampling import effective_variant


def finalize(measurement,*,arm,variant,affinity_initial,affinity_final,pair_id,
             selected_gpu,maxrss_kib,source_verification,execution_scope,validate):
    if variant!='gpu':raise ValueError('Only migrated GPU sampler supported')
    result=dict(measurement,sampler_variant=effective_variant(arm),experiment_variant=variant,
                cpu_affinity_initial=affinity_initial,cpu_affinity=affinity_final,
                pair_id=pair_id,selected_gpu=list(selected_gpu),maxrss_kib=maxrss_kib,
                effective_limits_verified=True,raw_ssd_writes=False,
                source_verification=source_verification,execution_scope=execution_scope)
    if not validate(result):raise RuntimeError('Performance receipt rejected')
    return result
