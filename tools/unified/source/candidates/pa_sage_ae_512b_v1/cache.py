"""Preserve the exact AE CPU row selections; only make their lookup explicit."""
from .common import *
def validate_rows(hot,storage_rows,np):
    require(hot.dtype==np.int64 and hot.ndim==1 and 0<len(hot)<2**31,'Bad CPU rows')
    require(hot[0]>=0 and hot[-1]<storage_rows and np.all(hot[1:]>hot[:-1]),'CPU rows not sorted/unique/in bounds')
    # The deployed page cache represents complete aligned eight-row runs.
    require(len(hot)%8==0 and np.all(hot[::8]%8==0),'AE CPU pages not aligned')
    require(np.all(hot.reshape(-1,8)==hot[::8,None]+np.arange(8)),'AE page membership is incomplete')
def install_cache(loader,arm,base,bundle,p,torch,np,binding,storage_rows):
    path=base/('full_cpu_rows.npy' if arm=='digit_full' else 'gids_cpu_rows.npy')
    hot=np.load(path,mmap_mode='r');validate_rows(hot,storage_rows,np)
    require(len(hot)==p['cpu_cache_rows'],'CPU budget changed')
    # No new RevPR selection, frequency profile or replica aliasing.
    rowmap=torch.zeros(storage_rows,dtype=torch.int32,device=loader.gids_device)
    rows=torch.from_numpy(hot.copy()).to(loader.gids_device)
    for lo in range(0,len(hot),1048576):
        hi=min(len(hot),lo+1048576)
        rowmap[rows[lo:hi]]=torch.arange(lo+1,hi+1,dtype=torch.int32,device=loader.gids_device)
    loader.cpu_backing_buffer(128,len(hot))
    loader.BAM_FS.set_cpu_buffer_rows(rows.data_ptr(),len(hot),rowmap.data_ptr(),storage_rows)
    value=dict(rows=len(hot),cpu_feature_bytes=len(hot)*512,hot_sha256=sha(path),
        mapping=p['arms'][arm]['cpu_mapping'],row_map_gpu_bytes=storage_rows*4,
        gpu_feature_bytes=p['gpu_cache_bytes'],cpu_selection_unchanged=True,replica_aliases_added=False)
    validate_binding(value,arm,p,binding)
    return value,dict(primary_rows=hot,row_map=rowmap,device_rows=rows)
def validate_binding(value,arm,p,binding):
    rel=p['base_layout']+('/full_cpu_rows.npy' if arm=='digit_full' else '/gids_cpu_rows.npy')
    require(value['rows']==p['cpu_cache_rows'] and value['cpu_feature_bytes']==p['cpu_cache_rows']*512
            and value['gpu_feature_bytes']==p['gpu_cache_bytes'],'Unequal feature budgets')
    require(value['mapping']=='unchanged_AE_physical_rows' and value['cpu_selection_unchanged']
            and not value['replica_aliases_added'] and value['hot_sha256']==binding['files'][rel]['sha256'], 'CPU selection changed')
