"""Real 128D BaM exact-row/FIFO adapter; no device operation at import."""
import hashlib,importlib.util,json,os,sys
from pathlib import Path
import numpy as np
from .budget import budget
from .build import ROOT,OUT,BINARY
from candidates.ukl_native_sampling_v10r4 import fork_guard as H
from candidates.cl_ssd_writer_v10 import protocol as W
MAX_ROWS=405504;RETAINED=[]

def verify_binary(arm=None):
    from .protocol import STAGE
    from .build import binary
    arm=arm or STAGE
    r=json.loads((OUT/('build_'+arm+'_receipt.json')).read_text())
    if hashlib.sha256(binary(arm).read_bytes()).hexdigest()!=r['binary_sha256']:
        raise RuntimeError('CL arm backend binary changed')
    return r


def prepare_dependencies():
    if H._ACTIVE:raise RuntimeError('Load backend/framework before graph ownership')
    verify_binary()
    from .protocol import verify_manifest
    verify_manifest()
    import torch,dgl
    H.prewarm_cpu_block()
    name='BAM_Feature_Store'
    if name in sys.modules:raise RuntimeError('Use a fresh worker, conflicting native import')
    from .protocol import STAGE
    from .build import binary as arm_binary
    binary=arm_binary(STAGE)
    spec=importlib.util.spec_from_file_location(name,str(binary));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);sys.modules[name]=module
    if module.cache_policy_abi()!=5 or list(module.ukl_layout_abi())!=([2,32,512,512] if STAGE=='gids' else [3,24,512,512]):raise RuntimeError('Backend layout ABI mismatch')
    expected=978408104 if STAGE=='gids' else 1174089720
    if list(module.cl_capacity_abi())!=[1,97840809,expected,128]:raise RuntimeError('CL capacity ABI mismatch')
    return module


def hot_rows(arm,hot,primary,inverse,nodes,rows,check=lambda:None):
    if hot.dtype!=np.int64 or hot.ndim!=1 or len(hot)!=nodes//10 or not hot.flags.c_contiguous:raise ValueError('Exact logical hot set')
    for lo in range(0,len(hot),1048576):
        check();v=hot[max(0,lo-1):lo+1048576]
        if np.any(v<0) or np.any(v>=nodes) or np.any(v[1:]<=v[:-1]):raise ValueError('Hot range/order')
    if arm=='gids':return hot
    if arm!='digit' or primary is None or inverse is None or len(primary)!=nodes or len(inverse)!=rows:raise ValueError('DiGiT mapping extent')
    result=np.empty(len(hot),np.int64)
    for lo in range(0,len(hot),1048576):
        check();ids=hot[lo:lo+1048576];p=primary[ids]
        if np.any(p<0) or np.any(p>=rows) or not np.array_equal(inverse[p],ids):raise ValueError('Primary/inverse mismatch')
        result[lo:lo+len(ids)]=p
    return result


def map_chunks(arm,hot,inverse,nodes,rows,chunk=1048576,lookup=None):
    if type(chunk)is not int or not 0<chunk<=1048576:raise ValueError('Mapping chunk cap')
    padded=(rows+7)//8*8
    if arm not in ('gids','digit') or (arm=='gids' and rows!=nodes) or (arm=='digit' and (inverse is None or len(inverse)!=rows)):raise ValueError('Arm mapping shape')
    for lo in range(0,padded,chunk):
        hi=min(lo+chunk,padded);end=min(hi,rows);slots=np.zeros(hi-lo,np.uint32)
        ids=np.arange(lo,end,dtype=np.int64) if arm=='gids' else np.asarray(inverse[lo:end])
        if np.any(ids<0) or np.any(ids>=nodes):raise ValueError('Invalid logical alias')
        if lookup is not None:
            if lookup.dtype!=np.uint32 or len(lookup)!=nodes:raise ValueError('Logical slot lookup extent')
            slots[:len(ids)]=lookup[ids]
        else:
            pos=np.searchsorted(hot,ids);match=pos<len(hot)
            if len(hot):match &= hot[np.minimum(pos,len(hot)-1)]==ids
            slots[:len(ids)][match]=pos[match].astype(np.uint32)+1
        yield lo,slots


def install(fs,arm,hot,primary,inverse,nodes,rows,check=lambda:None,event=lambda **kw:None,owner=None):
    selected=hot_rows(arm,hot,primary,inverse,nodes,rows,check);check()
    from .external import ExternalCache
    from candidates.ukl_real_multi_v9.memory import CudaAPI
    options=dict(interleave=True) if arm=='digit' else {}
    owner.external=ExternalCache(fs,selected,(rows+7)//8*8,CudaAPI(),check,event,full=True,**options);del selected
    lookup=np.zeros(nodes,np.uint32) # budgeted CPU-only anonymous logical lookup
    for lo in range(0,len(hot),1048576):
        check();ids=hot[lo:lo+1048576];lookup[ids]=np.arange(lo+1,lo+1+len(ids),dtype=np.uint32)
    for lo,slots in map_chunks(arm,hot,inverse,nodes,rows,lookup=lookup):
        check();fs.policy_write_cpu_map(lo,slots);event(stage='cache_map_chunk',rows_done=lo+len(slots),rows_total=(rows+7)//8*8)
    del lookup
    fs.policy_finish_cpu_cache();fs.policy_configure(3 if arm=='gids' else 2,4*2**30)
    return dict(cpu_rows=len(hot),cpu_feature_bytes=len(hot)*512,gpu_feature_bytes=4*2**30,
                gpu_row_map_bytes=((rows+7)//8*8)*4,gpu_policy='legacy' if arm=='gids' else 'fifo',page_bytes=512,feature_row_bytes=512,
                all_replica_aliases_mapped=True,padding_uncached=True)


class Provider:
    """Process-scoped native cache, for an admitted fresh GPU worker only.

    No global-ID-sized torch tensor and no full storage map copied to GPU.
    Only uint32 slot map persists on GPU; aliases are populated in bounded chunks.
    """
    def __init__(self,module,storage,arm,graph,hot,inverse,check,event=lambda **kw:None):
        from .protocol import require_admitted_worker
        require_admitted_worker(arm)
        from .storage import accepted_storage
        if storage!=accepted_storage():raise RuntimeError('SSD acceptance changed')
        if not H._ACTIVE:raise RuntimeError('Owned graph lifecycle required')
        H.assert_dontfork(graph.arena)
        import torch
        if not torch.cuda.is_initialized() or torch.cuda.current_device()!=0:raise RuntimeError('Preinitialize selected GPU before graph ownership')
        self.arm,self.graph,self.inverse,self.check=arm,graph,inverse,check
        self.rows=graph.nodes if arm=='gids' else graph.storage_rows
        self.region=storage['arms'][arm]['region'];self.device=torch.device('cuda:0')
        if self.region['storage_rows']!=self.rows or self.region['row_bytes']!=512 or self.region['payload_bytes']!=((self.rows+7)//8*8)*512:raise RuntimeError('SSD region extent')
        self.controllers=module.GIDS_Controllers();self.fs=module.BAM_Feature_Store_float();self.closed=False
        RETAINED.append((self.fs,self.controllers))
        self.external=None
        try:
            self.controllers.init_GIDS_controllers(1,1024,128,[0])
            self.fs.init_controllers(self.controllers,512,self.region['device_offset_bytes']//512,4096,((self.rows+7)//8*8)*128,1,0 if arm=='gids' else 1,0)
            self.fs.set_mixed_io_geometry(True,512,512,1,1,1,hashlib.sha256(b'UKL-v21-512B-row-page-explicit-g2-512B-1KiB').hexdigest())
            self.fs.set_device_io_stats(True);self.fs.set_cpu_feature_path(0,131072,1)
            self.cache=install(self.fs,arm,hot,graph.arrays['primary'],inverse,graph.nodes,self.rows,check,event,self)
            from .counters import snapshot
            self.cache['preload_io']=snapshot(self.fs)['device']
            self.cache['segmented_anonymous']=True
            self.cache['allocation_chunk_bytes']=16*2**20
            event(stage='cache_initialization_complete',cpu_rows=len(hot))
            if self.fs.get_gpu_cache_stats()[6]!=0 or sum(self.fs.get_feature_access_stats())!=0:raise RuntimeError('Preload polluted cache or training counters')
            self.fs.reset_device_io_stats();self.fs.begin_useful_io_region()
        except BaseException:
            if self.external is not None:self.external.close()
            raise
    def __call__(self,request):
        import torch
        if self.closed or request.arm!=self.arm or request.row_count!=self.rows or len(request.logical_ids)>MAX_ROWS:raise ValueError('Feature request scope')
        request.byte_offsets(self.region['device_offset_bytes']);ids=request.logical_ids;rows=request.storage_rows
        if len(ids)!=len(rows) or np.any(ids<0) or np.any(ids>=self.graph.nodes):raise ValueError('Logical request bounds')
        expected=rows if self.arm=='gids' else self.inverse[rows]
        if not np.array_equal(expected,ids):raise ValueError('Logical/physical feature alias mismatch')
        self.check();index=torch.from_numpy(rows.copy()).to(self.device)
        from .pairing import group_bases
        bases=group_bases(self.graph,rows) if self.arm=='digit' else np.full(len(rows),-1,np.int64)
        flags=torch.from_numpy(bases).to(self.device)
        result=torch.empty((len(rows),128),device=self.device,dtype=torch.float32)
        self.fs.read_feature(result.data_ptr(),index.data_ptr(),len(rows),128,128,0,flags.data_ptr())
        return result
    def snapshot(self):
        from .counters import snapshot
        return snapshot(self.fs)
    def close(self):
        if not self.closed:
            import torch
            torch.cuda.synchronize()
            if self.snapshot()['device']['outstanding']:raise RuntimeError('Feature I/O still outstanding')
            # Existing BaM cache allocations have process lifetime. Retain all
            # handles through graph unregister; worker must exit before acceptance.
            self.external.close()
            self.inverse=None;self.graph=None;self.check=None
            RETAINED.append((self.fs,self.controllers));self.closed=True
