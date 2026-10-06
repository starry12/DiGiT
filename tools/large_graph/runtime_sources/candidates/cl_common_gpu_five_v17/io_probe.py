"""Tiny readonly real-SSD row/pair checks before loading the full graph."""
import mmap
import os
from pathlib import Path
import numpy as np


def prefix_ids(arm, inputs):
    if arm == 'gids':
        return np.arange(16,dtype=np.int64)
    entry=inputs['stages']['graph']['files']['storage_to_node.i32']
    def identity(s):return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
    fd=os.open(entry['path'],os.O_RDONLY|os.O_DIRECT|os.O_NOFOLLOW)
    buf=mmap.mmap(-1,4096)
    try:
        if identity(os.fstat(fd))!=entry['identity']:
            raise RuntimeError('Probe inverse source changed')
        if os.preadv(fd,[buf],0)!=4096:
            raise RuntimeError('Short direct inverse prefix read')
        result=np.frombuffer(buf,np.int32,count=16).astype(np.int64)
        if identity(os.fstat(fd))!=entry['identity']:
            raise RuntimeError('Probe inverse source changed during read')
        return result
    finally:
        buf.close();os.close(fd)


def run(module,storage,arm,inputs,check,event):
    import torch
    from .external import ExternalCache
    from candidates.ukl_real_multi_v9.memory import CudaAPI
    from candidates.ukl_runtime_prepare_v11r1.data import features
    from .backend import RETAINED
    from .counters import snapshot,interval
    check();event('mixed_io_probe_begin')
    logical=prefix_ids(arm,inputs)
    fs=module.BAM_Feature_Store_float();ctrl=module.GIDS_Controllers()
    RETAINED.append((fs,ctrl));cache=None
    try:
        ctrl.init_GIDS_controllers(1,1024,128,[0])
        fs.init_controllers(ctrl,512,storage['arms'][arm]['region']['device_offset_bytes']//512,16,16*128,1,0 if arm=='gids' else 1,0)
        fs.set_mixed_io_geometry(True,512,512,1,1,1,'UKL-v21-readonly-small-pair-probe')
        fs.set_device_io_stats(True);fs.set_cpu_feature_path(0,131072,1)
        cache=ExternalCache(fs,np.array([5],np.int64),16,CudaAPI(),check)
        slots=np.zeros(16,np.uint32);slots[5]=1
        fs.policy_write_cpu_map(0,slots);fs.policy_finish_cpu_cache()
        fs.policy_configure(3 if arm=='gids' else 2,16*2**20)
        fs.reset_device_io_stats();fs.begin_useful_io_region()
        cases=[]
        # Explicit odd heads stress replica parity and 4-KiB page crossing.
        specs=[('cold_pair',[1,2],[1,1],(0,1)),
               ('both_hit',[1,2],[1,1],(0,0)),
               ('single',[3],[3],(1,0)),
               ('one_hit',[3,4],[3,3],(1,0)),
               ('cpu_sibling',[5,6],[5,5],(1,0)),
               ('cross_4k_pair',[7,8],[7,7],(0,1))]
        for name,rows,heads,expected in specs:
            check()
            ids=torch.tensor(rows,dtype=torch.int64,device='cuda:0')
            bases=torch.tensor(heads,dtype=torch.int64,device='cuda:0')
            x=torch.empty((len(rows),128),device='cuda:0')
            before=snapshot(fs)
            fs.read_feature(x.data_ptr(),ids.data_ptr(),len(rows),128,128,0,bases.data_ptr())
            torch.cuda.synchronize();after=snapshot(fs)
            if not np.array_equal(x.cpu().numpy(),features(logical[rows])):
                raise RuntimeError('512B/1KiB probe feature mismatch: '+name)
            result=interval(before,after,len(rows))
            counts=result['request_sizes']
            wanted=expected if arm=='digit' else (expected[0]+2*expected[1],0)
            if (counts['primary_512b'],counts['primary_1k'])!=wanted:
                raise RuntimeError('Probe physical request sizes differ: '+name)
            if name=='cpu_sibling' and result['serving']['cpu_served_rows']!=1:
                raise RuntimeError('CPU sibling was incorrectly merged')
            cases.append(dict(name=name,passed=True,rows=rows,request_sizes=counts))
        cache.close();check()
        result=dict(passed=True,arm=arm,raw_ssd_writes=False,cases=cases,cache_released=cache.closed)
        event('mixed_io_probe_complete',receipt=result)
        return result
    finally:
        if cache is not None:cache.close()
