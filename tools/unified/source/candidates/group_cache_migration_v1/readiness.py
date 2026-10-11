"""Import/namespace checks using original small metadata, no graph arrays/GPU."""
import importlib
from unittest.mock import patch
from .common import HERE,ROOT,require
from .prepare import source
def check():
    records={}
    for ds in ('UKL','CL'):
        base=source(ds);b=base['base_specs'];v=dict(output_units=base['base_metadata']['units'])
        receipt=dict(base=base,validation=v,files={'ptr.i64':b['group_ptr'],'units.i32':b['group_ids'],'covered.i64':b['covered']})
        with patch('candidates.group_cache_migration_v1.large_overlay.check_prepared',return_value=receipt):
            names=['protocol','session','worker','bootstrap','cli','start','native_loader','performance_engine','phase','cgroup_limits']
            modules={name:importlib.import_module('candidates.group_cache_migration_v1.'+ds.lower()+'.'+name) for name in names}
            p=modules['protocol'];require(p.ROUNDS==1 and (p.WARMUP,p.MEASURED)==(20,300),'Wrong run extent')
            budget=p.budget('digit');require(budget['arena_bytes']==sum((x['length']+4095)//4096*4096 for x in b.values()),'Wrong sparse graph budget')
            prefix='/system.slice/digit-group-cache-'+ds.lower()+'-'
            require(modules['phase'].PREFIX==prefix,'Wrong phase namespace')
            require(modules['cgroup_limits'].service_group(prefix+'fixture.service/data')==prefix+'fixture.service','Wrong admission namespace')
            from .build import LARGE_OUT
            sb=importlib.import_module('candidates.group_cache_migration_v1.'+ds.lower()+'.sampler_build')
            require(sb.OUT==LARGE_OUT,'Wrong native sampler build')
            inputs=importlib.import_module('candidates.group_cache_migration_v1.'+ds.lower()+'.inputs')
            accepted=inputs.accepted_inputs();arms={arm:inputs.arm_inputs(accepted,arm) for arm in ('gids','digit')}
            require(arms['gids']['roots']==arms['digit']['roots'] and arms['gids']['labels']==arms['digit']['labels'],'Root/label inputs not matched')
            for arm in arms:
                roots,labels=inputs.load_window(arms[arm]);require(roots.shape==labels.shape==(320,1024),'Wrong prepared window shape')
            records[ds]=dict(imports=len(modules),cpu_only=True,original_extent_bytes=budget['arena_bytes'],matched_roots=True,matched_labels=True)
    from .ig.common import setup,cfg,P
    setup()
    for name in ('worker','features','cache','sampler','controller','review','validation','summary'):
        importlib.import_module('candidates.group_cache_migration_v1.ig.'+name)
    from .ig.controller import plan
    jobs=plan();require(len(jobs)==4 and sum(not j['smoke'] for j in jobs)==2,'Wrong IG smoke/formal schedule')
    require((cfg()['warmup_batches'],cfg()['measured_batches'])==(20,300),'Wrong IG extent')
    records['IG']=dict(imports=8,cpu_only=True,jobs=jobs)
    return records
if __name__=='__main__':
    import json
    print(json.dumps(check(),indent=2))
