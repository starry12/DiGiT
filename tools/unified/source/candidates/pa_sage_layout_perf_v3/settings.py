"""Shared parameter checks used by the actual native worker and CPU fixtures."""
from .common import require

def check_manifest(p,m):
    g=m['grouping'];spec=p['point']
    require(type(g['group_size']) is int and g['group_size']==p['group_size']==spec['group_size'],'Wrong layout group size')
    require(g['replication_ratio']==spec['replication_ratio'],'Wrong layout replication ratio')
    require(m['dataset']['num_nodes']==p['graph']['nodes'],'Wrong layout node count')
    require(m['feature']['dim']==128 and m['feature']['row_bytes']==512,'Wrong feature width')
    require(g['num_replica_groups']*g['group_size']<=m['dataset']['num_nodes']*spec['replica_percent']//100,'Replica budget exceeded')
    require(m['io']['page_size']==4096 and m['io']['alignment_rows']==8,'Wrong slot geometry')

def sampler_kwargs(p,m):
    check_manifest(p,m)
    return dict(cuda_mode='required',metadata_mode=p['metadata_mode'],random_seed=p['seed'])

def loader_kwargs(p,m,offset,geometry):
    check_manifest(p,m)
    require(type(offset) is int and offset>=0 and offset%4096==0,'Unaligned SSD offset')
    return dict(page_size=4096,off=offset,cache_dim=128,num_ele=m['feature']['num_storage_rows']*128,
        num_ssd=1,ssd_list=[0],cache_size=p['gpu_cache_bytes']//2**20,ctrl_idx=0,
        window_buffer=False,accumulator_flag=False,feature_index_mode='explicit',gpu_cache_policy='fifo',
        cpu_feature_path='mapped',mixed_io=True,device_io_stats=True,
        mixed_io_geometry=geometry.with_payload_offset(offset).to_native_mapping())

def create_model(p,value,device,seed_fn,model_class,torch_module,hash_fn):
    seed_fn(value)
    model=model_class(128,p['hidden'],p['classes'],num_layers=p['layers'],dropout=p['dropout']).to(device)
    kwargs=dict(p['optimizer']['kwargs']);kwargs['betas']=tuple(kwargs['betas'])
    optimizer=torch_module.optim.Adam(model.parameters(),**kwargs)
    initial=hash_fn(model);seed_fn(value);model.train()
    return model,optimizer,initial
