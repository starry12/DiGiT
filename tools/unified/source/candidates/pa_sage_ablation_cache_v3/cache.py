"""Exact logical hot set across reordered and replicated physical rows."""
import hashlib
from .common import sha,require

def logical_slots(hot,n,np):
    hot=np.asarray(hot,dtype=np.int64)
    require(hot.ndim==1 and 0<len(hot)<2**31,'Invalid hot-set extent')
    require(hot[0]>=0 and hot[-1]<n and np.all(np.diff(hot)>0),'Hot IDs must be sorted, unique and in range')
    slots=np.zeros(n,dtype=np.int32);slots[hot]=np.arange(1,len(hot)+1,dtype=np.int32)
    return slots

def map_storage(storage,slots,np):
    storage=np.asarray(storage);require(np.all((storage>=-1)&(storage<len(slots))),'Bad storage-to-node map')
    result=np.zeros(len(storage),dtype=np.int32);valid=storage>=0;result[valid]=slots[storage[valid]]
    return result

def validate_binding(v,arm,p):
    policy=p['arms'][arm]
    require(v['hot_set']==policy['cpu_hot_set'] and v['mapping']==policy['cpu_mapping'],'Wrong CPU cache policy')
    require(v['rows']==p['cpu_cache_rows'] and v['logical_hot_sha256']==p['cache_inputs'][policy['cpu_hot_set']]['sha256'],'Wrong CPU cache set/budget')
    require(v['all_replicas_mapped'] and v['padding_uncached'],'Unverified CPU row semantics')
    if arm in ('gr','ns'):require(v['row_map_bytes']>0 and len(v['row_map_sha256'])==64,'Missing exact row map')

def install_cache(loader,arm,base,bundle,p,torch,np):
    policy=p['arms'][arm];desc=p['cache_inputs'][policy['cpu_hot_set']]
    hot=np.load(base/desc['file'],mmap_mode='r');require(sha(base/desc['file'])==desc['sha256'],'Hot-set identity changed')
    require(len(hot)==p['cpu_cache_rows'],'Wrong logical cache extent')
    binding=dict(hot_set=policy['cpu_hot_set'],mapping=policy['cpu_mapping'],rows=len(hot),logical_hot_sha256=desc['sha256'],all_replicas_mapped=True,padding_uncached=True,row_map_bytes=0)
    refs={}
    if arm in ('gr','ns'):
        slots=logical_slots(hot,p['graph']['nodes'],np)
        primary=np.array(bundle.arrays['node_to_primary_row'][hot],dtype=np.int64)
        storage=bundle.arrays['storage_to_node']
        require(np.array_equal(storage[primary],hot),'Primary-row inverse differs')
        rowmap=torch.empty(len(storage),dtype=torch.int32,device=loader.gids_device)
        digest=hashlib.sha256();cached_physical_rows=0
        for lo in range(0,len(storage),1048576):
            values=map_storage(storage[lo:lo+1048576],slots,np);digest.update(values.tobytes());cached_physical_rows+=int(np.count_nonzero(values))
            rowmap[lo:lo+len(values)].copy_(torch.from_numpy(values))
        rows=torch.from_numpy(primary.copy()).to(loader.gids_device)
        loader.cpu_backing_buffer(128,len(hot))
        loader.BAM_FS.set_cpu_buffer_rows(rows.data_ptr(),len(hot),rowmap.data_ptr(),len(storage))
        refs.update(row_map=rowmap,device_rows=rows,primary_rows=primary)
        binding.update(row_map_sha256=digest.hexdigest(),row_map_bytes=rowmap.numel()*4,cached_physical_rows=cached_physical_rows)
    else:
        filename='gids_cpu_rows.npy' if arm=='gids' else 'full_cpu_rows.npy'
        primary=np.load(base/filename,mmap_mode='r')
        if arm=='digit_full':require(np.array_equal(bundle.arrays['storage_to_node'][primary],hot),'Frequency physical rows differ')
        else:require(np.array_equal(primary,hot),'GIDS logical rows differ')
        loader.cpu_backing_buffer(128,len(hot));loader.set_cpu_buffer(torch.from_numpy(primary.copy()),len(hot))
        refs['primary_rows']=primary
    validate_binding(binding,arm,p)
    return binding,refs
