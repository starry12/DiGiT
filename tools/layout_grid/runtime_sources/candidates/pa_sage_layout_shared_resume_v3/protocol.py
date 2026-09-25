"""Compile and validate the full grid without opening large arrays or CUDA."""
import argparse,copy
from .common import ROOT,HERE,Path,read,write,sha,require,POLICY,verification_policy
from candidates.pa_sage_layout_sweep_v1.common import point,grid

REFERENCE=HERE/'reference_protocol.json'
FIXED=('seed','fanouts','batch_size','optimizer','hidden','classes','layers','dropout',
       'graph','gpu_cache_bytes','cpu_cache_rows','metadata_mode','bfs_enabled','data',
       'smoke_epochs','smoke_batches')

def validate(p):
    ref=read(REFERENCE)
    require(p['verification_policy']==verification_policy(p['point']),'Wrong declared verification policy')
    require(p['epochs']==1 and p['evaluation']=='disabled' and p['warmup_batches']==0,'Expected first-epoch performance protocol')
    require('validation_trace' not in p and 'test_trace' not in p,'Evaluation traces must be absent')
    require(p['schema']=='digit-pa-sage-layout-performance-v2','Wrong training protocol schema')
    require(p['feature_mode'] in ('real','shared'),'Unknown feature mode')
    require(p['feature_mode']=='shared' or p['point']['id']=='g2_r20','Real pilot is only g2/r20')
    require(p['pool_spec_sha256']==sha(HERE/'pools.json'),'Pool specification changed')
    spec=p['point'];require(spec==point(spec['group_size'],spec['replica_percent']),'Invalid grid point')
    require(p['group_size']==spec['group_size'],'Sampling group size differs from point')
    require(p['reference_sha256']==sha(REFERENCE),'Wrong training reference')
    for key in FIXED:require(p[key]==ref[key],'Changed controlled training field: '+key)
    require(p['arms']=={'digit_full':ref['arms']['digit_full']},'Wrong cache policy')
    base=Path(p['base_layout'])
    require(base.is_absolute() and base.name==spec['id'],'Point needs its own absolute layout path')
    require(Path(p['overlay'])==base/'overlay' and Path(p['ssd_ready'])==base/'ssd_ready.json','Wrong per-point inputs')
    require(p['orders_file']==str(ROOT/ref['base_layout']/'orders.json'),'Unpaired root orders')
    require(p['hot_nodes']==str(ROOT/ref['base_layout']/'hot_nodes.npy'),'Changed logical hot-set path')
    require(p['hot_sha256']==read(ROOT/ref['base_layout']/'prepared.json')['bindings']['hot_nodes.npy'],'Changed hot-set identity')
    return p

def compile_protocols(plan,output):
    output=Path(output);require(not output.exists(),'Keep old protocols; select a new output directory')
    require(plan['schema']=='digit-pa-sage-layout-sweep-plan-v1','Wrong layout plan')
    ref=read(REFERENCE);require(plan['training_reference']==ref,'Layout plan uses a different training reference')
    require(len(plan['points'])==15 and {p['id'] for p in plan['points']}=={p['id'] for p in grid()},'Incomplete or duplicate grid')
    require(plan['hot_rows']==ref['cpu_cache_rows'],'Changed CPU cache budget')
    expected_hot=read(ROOT/ref['base_layout']/'prepared.json')['bindings']['hot_nodes.npy']
    require(plan['inputs']['hot_nodes']['sha256']==expected_hot,'Changed hot profile')
    protocols=[]
    for cell in plan['points']:
        spec=point(cell['group_size'],cell['replica_percent']);require(cell['id']==spec['id'],'Point label mismatch')
        p=copy.deepcopy(ref);p.pop('ablation_order');p.pop('validation_trace');p.pop('test_trace');p.pop('baseline_policy');p.update(epochs=1,evaluation='disabled',warmup_batches=0);p['schema']='digit-pa-sage-layout-performance-v2'
        p.update(point=spec,group_size=spec['group_size'],reference_sha256=sha(REFERENCE),
                 base_layout=str(Path(cell['filesystem_destination']).resolve()),arms={'digit_full':ref['arms']['digit_full']},
                 orders_file=str(ROOT/ref['base_layout']/'orders.json'),hot_nodes=str(ROOT/ref['base_layout']/'hot_nodes.npy'),
                 hot_sha256=expected_hot,layout_policy='Per-point directed grouping plus validated reverse-edge overlay; fixed logical hot set',
                 model_policy='Fresh seed0 model and Adam per point; training correctness only; no evaluation, accuracy or convergence claim',
                 measurement='One fresh smoke then one complete first training epoch per point; no validation/test; setup and order time separate; one seed/repeat')
        p['overlay']=str(Path(p['base_layout'])/'overlay');p['ssd_ready']=str(Path(p['base_layout'])/'ssd_ready.json')
        p.update(feature_mode='shared',pool_spec_sha256=sha(HERE/'pools.json'),verification_policy=verification_policy(p['point']))
        validate(p);protocols.append(p)
    require(len({p['base_layout'] for p in protocols})==15,'Layouts alias each other')
    output.mkdir(parents=True,exist_ok=False)
    entries=[]
    for p in protocols:
        path=(output/(p['point']['id']+'.json')).resolve();write(path,p)
        entries.append(dict(point=p['point'],protocol=str(path),protocol_sha256=sha(path),
            preview_command=['/home/embed/miniconda3/envs/gids/bin/python','-B','-m','candidates.pa_sage_layout_shared_resume_v3.cli','--protocol',str(path)],
            native_ready=False))
    write(output/'index.json',dict(schema='digit-layout-performance-grid-v2',points=entries,auto_launch=False))
    return entries

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    entries=compile_protocols(read(a.plan),a.output);print('Prepared %d point protocols; no native execution'%len(entries))
if __name__=='__main__':main()
