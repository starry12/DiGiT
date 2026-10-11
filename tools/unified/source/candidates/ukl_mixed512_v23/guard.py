"""Aggregate safeguards plus independent phase cache limits; no baseline credit."""
from .guard_base import *
from .guard_base import _io_stat, Guard as BaseGuard
from .kernel_scope import KernelCursor
from .phase import INIT_LIMIT, DATA_LIMIT


class Guard(BaseGuard):
    def __init__(self, *args, phases=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.phases=phases

    def _check(self,sample):
        if self.phases is None or not self.require_cgroup:
            return super().check(sample)
        phase=self.phases.phase
        self.file_cache_limit=INIT_LIMIT if phase in ('boot','init') else INIT_LIMIT+DATA_LIMIT
        result=super().check(sample)
        reasons=list(result['reasons'])
        try:
            if sample.get('accounting_phase')!=phase:
                raise ValueError('phase observation mismatch')
            if phase not in ('boot','init','data'):
                raise ValueError('unknown phase')
            if phase!='boot':
                children=sample['phase_cgroups']
                if set(children)!={'init','data'}:raise ValueError('missing phase metrics')
                for name,limit in (('init',INIT_LIMIT),('data',DATA_LIMIT)):
                    m=children[name]['memory']
                    if any(type(m[k]) is not int or m[k]<0 for k in ('file','file_dirty','file_writeback')):
                        raise ValueError('invalid phase memory counters')
                    if m['file']>limit:reasons.append(name+'_file_cache_limit')
                    if m['file_dirty']>64*MIB:reasons.append(name+'_dirty_above_64mib')
                    if m['file_writeback']>64*MIB:reasons.append(name+'_writeback_above_64mib')
                    for key in ('max','oom','oom_kill'):
                        if children[name]['events'][key]!=0:reasons.append(name+'_cgroup_event_'+key)
                if sample['cgroup']['file']-children['data']['memory']['file']>INIT_LIMIT:
                    reasons.append('initialization_aggregate_file_limit')
                if phase=='init' and children['data']['memory']['file']!=0:
                    reasons.append('data_cache_before_handoff')
        except (KeyError,TypeError,ValueError) as exc:
            reasons.append('phase_accounting_unavailable: '+str(exc))
        self.failed=list(dict.fromkeys(reasons))
        result.update(ok=not self.failed,reasons=self.failed,
                      phase_cache_limits=dict(initialization=INIT_LIMIT,data=DATA_LIMIT,phase=phase))
        return result

    def check(self,sample):
        # Classify current kernel counters; never grant historical cache credit.
        import copy
        from candidates.ukl_training_native_v15r11.accounting import classify
        value=copy.deepcopy(sample);shared=[]
        try:
            if self.require_cgroup and 'cgroup' in value:
                m=value['cgroup'];c=classify(m);m['file']=c['non_shmem_file']
                if c['shared']>640*MIB:shared.append('aggregate_shared_above_640mib')
                for name,v in value.get('phase_cgroups',{}).items():
                    c=classify(v['memory']);v['memory']['file']=c['non_shmem_file']
                    if c['shared']>320*MIB:shared.append(name+'_shared_above_320mib')
        except (ValueError,KeyError,TypeError) as e:shared.append('memory_classification: '+str(e))
        result=self._check(value)
        result['reasons']+=shared;result['ok']=not result['reasons']
        self.failed=result['reasons'];return result
