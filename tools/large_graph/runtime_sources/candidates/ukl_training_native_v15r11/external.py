"""External anonymous CPU cache shared by real SSD precheck and full training."""
import numpy as np
from .segmented import SegmentedArena,CHUNK
RETAINED=[]
class ExternalCache:
    def __init__(self,fs,rows,storage_rows,api,check,event=lambda **kw:None,full=False):
        if rows.dtype!=np.int64 or rows.ndim!=1 or not rows.flags.c_contiguous or len(rows)==0:raise ValueError('Contiguous int64 hot rows required')
        if full and len(rows)!=78780147:raise ValueError('Exact full hot set required')
        self.fs,self.closed,self.event=fs,False,event
        self.pool=SegmentedArena((len(rows)*512+4095)//4096*4096,api,full=full)
        try:
            self.pool.prepare(check,event);check()
            fs.policy_begin_external_cpu_cache(self.pool.address,self.pool.device_address,len(rows),storage_rows)
            for lo in range(0,len(rows),CHUNK//512):
                check();part=rows[lo:lo+CHUNK//512]
                fs.policy_preload_chunk(lo,part)
                event(stage='native_preload_chunk',rows_done=lo+len(part),rows_total=len(rows));check()
        except BaseException:
            self.close();raise
    def release_event(self,**kw):
        try:self.event(**kw)
        except Exception:pass # A failed progress write must not interrupt unregister.
    def close(self):
        if self.closed:return
        try:
            self.release_event(stage="cache_release_begin")
            self.fs.policy_release_external_cpu_cache()
            self.pool.close(event=self.release_event);self.closed=True
            self.release_event(stage="cache_release_complete")
        except BaseException:RETAINED.append(self);raise
