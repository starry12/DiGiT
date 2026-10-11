"""Accept only the lossless compact receipt derived from the current overlay."""
import json,os
from pathlib import Path
from .convert import identity
DEFAULT='/mnt/n0/digit/ukl_compact_cover_20261011_v1/covered.u32.json'
def binding(dataset='UKL'):
    if dataset!='UKL':raise ValueError('UKL only')
    from candidates.group_cache_migration_v1.large_overlay import binding as parent
    b=parent(dataset);r=json.loads(Path(os.environ.get('DIGIT_UKL_COMPACT_RECEIPT',DEFAULT)).read_text())
    expected=dict(ptr=b['indptr'],gptr=b['group_ptr'],covered=b['covered'])
    if (not r['passed'] or r['encoding']!='owner_relative_u32' or not r['reconstructed_all_values']
        or not r['all_source_payloads_verified'] or r['sources']!=expected
        or r['output']['length']*2!=b['covered']['length']):raise RuntimeError('Invalid compact provenance')
    if identity(r['output']['path'])!=r['output']['identity']:raise RuntimeError('Compact output changed')
    b['covered']=r['output'];return b

def projection():
    from candidates.group_cache_migration_v1.large_overlay import binding as parent
    from candidates.cache_epoch_v1.ukl import dataset as original
    b=parent('UKL');old=sum((x['length']+4095)//4096*4096 for x in b.values())
    n=b['covered']['length'];new=old-(n+4095)//4096*4096+(n//2+4095)//4096*4096
    extra=max(0,new-original.BASE.EXTENT)
    return dict(original_arena_bytes=old,compact_arena_bytes=new,saved_bytes=old-new,
                host_min_bytes=352*2**30+128*2**20+extra,cgroup_max_bytes=288*2**30+extra,
                memlock_bytes=256*2**30+extra,allocator_hard_cap_bytes=336*2**30,
                numa_balance_unchanged=True,whole_graph_gpu_acceptance=False)
if __name__=='__main__':print(json.dumps(projection(),indent=2))
