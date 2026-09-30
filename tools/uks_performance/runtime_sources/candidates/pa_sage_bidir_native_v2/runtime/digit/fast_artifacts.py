"""Opt-in vectorized invariant checks with a capped working set and phase progress."""
import mmap
import numpy as np
from .artifacts import ArtifactValidationError

CHUNK = 1 << 20
MAX_WORKING_BYTES = 8 * 1024**3

def require(ok, message):
    if not ok:
        raise ArtifactValidationError(message)

def validate_arrays(manifest, arrays):
    n = int(manifest['dataset']['num_nodes'])
    rows = int(manifest['feature']['num_storage_rows'])
    grouping = manifest['grouping']
    groups, primary, size = (int(grouping[k]) for k in ('num_groups','num_primary_groups','group_size'))
    alignment = int(manifest['io']['alignment_rows'])
    width = alignment if manifest['schema_version'] == 2 else size
    # Dense bool scratch plus all potentially resident integer mappings and temporary batches.
    # Features are never read here. Hash workers use their own small streaming buffers.
    reservation = rows+n+groups+sum(a.nbytes for k,a in arrays.items() if k!='reordered_features')+128*CHUNK
    # Large sequential CSC arrays are released after each batch and never needed together.
    reservation -= arrays['reordered_indices'].nbytes+arrays['reordered_indptr'].nbytes
    require(reservation <= MAX_WORKING_BYTES, 'fast validation reservation exceeds 8 GiB; use bounded')
    with open('/proc/meminfo') as stream:
        available = next(int(line.split()[1])*1024 for line in stream if line.startswith('MemAvailable:'))
    require(available >= reservation+1024**3, 'insufficient host memory for fast validation')
    for a in arrays.values():
        require(isinstance(a,np.memmap) and a.mode=='r' and a.flags.c_contiguous,
                'fast validation requires read-only C-contiguous memmaps')
    occupied=np.zeros(rows,dtype=np.bool_)
    seen_primary=np.zeros(n,dtype=np.bool_)
    seen_groups=np.zeros(groups,dtype=np.bool_)
    def take(name, indices):
        a=arrays[name].reshape(-1)
        indices=np.asarray(indices,dtype=np.int64)
        require(not indices.size or (indices.min()>=0 and indices.max()<a.size),'array index outside range')
        result=np.array(a[indices],copy=True)
        return result
    def release():
        for name,a in arrays.items():
            if name!='reordered_features':a._mmap.madvise(mmap.MADV_DONTNEED)
    def mark(bitmap, indices, message):
        flat=np.asarray(indices,dtype=np.int64).reshape(-1)
        unique=np.unique(flat)
        require(unique.size==flat.size and not np.any(bitmap[unique]),message)
        bitmap[unique]=True
    def phase(name,done,total):
        print('Fast validation {}: {}/{}'.format(name,done,total),flush=True)
    try:
        batch=max(1,CHUNK//max(width,size))
        phase('groups',0,groups)
        next_progress=0
        for start in range(0,groups,batch):
            end=min(groups,start+batch)
            ids=np.arange(start,end,dtype=np.int64)
            bases=take('group_storage_base',ids).astype(np.int64,copy=False)
            owners=take('group_owner',ids)
            require(np.all((owners>=0)&(owners<n)),'group owner outside range')
            require(np.all((bases>=0)&(bases%alignment==0)&(bases<=rows-max(width,size))),
                    'group storage range/alignment invalid')
            slots=bases[:,None]+np.arange(width,dtype=np.int64)
            mark(occupied,slots,'overlapping group storage')
            members=take('group_members',ids[:,None]*size+np.arange(size,dtype=np.int64))
            require(np.all((members>=0)&(members<n)),'invalid group member')
            locations=bases[:,None]+np.arange(size,dtype=np.int64)
            require(np.array_equal(take('storage_to_node',locations),members),'group storage mapping mismatch')
            if manifest['schema_version']==2 and alignment>size:
                padding=bases[:,None]+np.arange(size,alignment,dtype=np.int64)
                require(np.all(take('storage_to_node',padding)==-1),'group padding used')
            count=max(0,min(end,primary)-start)
            if count:
                mark(seen_primary,members[:count],'primary groups overlap')
                require(np.array_equal(take('node_to_primary_row',members[:count]),locations[:count]),'primary row mismatch')
            mapped=take('supernode_to_group',ids)
            require(np.all((mapped>=0)&(mapped<groups)),'invalid supernode permutation')
            mark(seen_groups,mapped,'invalid supernode permutation')
            release()
            if end>=next_progress or end==groups:
                phase('groups',end,groups);next_progress=end+max(batch,groups//10)
        del occupied,seen_primary,seen_groups
        for name,total in [('storage_to_node',rows),('node_to_primary_row',n),
                           ('reordered_indptr',n+groups+1),('reordered_indices',arrays['reordered_indices'].size)]:
            phase(name,0,total);previous=0
            for start in range(0,total,CHUNK):
                end=min(total,start+CHUNK);ids=np.arange(start,end,dtype=np.int64)
                values=take(name,ids)
                if name=='storage_to_node':
                    require(np.all((values>=-1)&(values<n)),'invalid storage node')
                elif name=='node_to_primary_row':
                    require(np.all((values>=0)&(values<rows)),'invalid primary row')
                    require(np.array_equal(take('storage_to_node',values),ids),'primary inverse mismatch')
                elif name=='reordered_indptr':
                    require((start!=0 or values[0]==0) and values[0]>=previous and np.all(values[1:]>=values[:-1]),
                            'CSC pointer is not monotonic')
                    previous=int(values[-1])
                else:
                    require(np.all((values>=0)&(values<n+groups)),'invalid CSC node')
                release()
            if name=='reordered_indptr':
                require(previous==arrays['reordered_indices'].size,'CSC terminal pointer mismatch')
            phase(name,total,total)
    finally:
        release()
