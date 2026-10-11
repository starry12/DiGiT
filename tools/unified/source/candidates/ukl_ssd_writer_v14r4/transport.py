"""Small synchronous CPU NVMe transport; instantiation is privileged device access."""
import ctypes as C
import os
from pathlib import Path

class NVMe:
    def __init__(self, binary, device, regions, resume_offsets):
        if os.geteuid()!=0:raise RuntimeError('Native transport requires root')
        self.write_ranges=[(r['device_offset_bytes']+n,r['end_bytes_exclusive']) for r,n in zip(regions,resume_offsets)]
        if len(self.write_ranges)!=2:raise ValueError('Two resume ranges required')
        self.lib=C.CDLL(str(binary));self.handle=None
        self.lib.ukl_transport_error.restype=C.c_char_p
        self.lib.ukl_transport_open.argtypes=[C.c_char_p,C.c_uint64,C.POINTER(C.c_uint64)];self.lib.ukl_transport_open.restype=C.c_void_p
        self.lib.ukl_transport_io.argtypes=[C.c_void_p,C.c_int,C.c_uint64,C.c_void_p,C.c_size_t]
        for name in ('flush','close'):getattr(self.lib,'ukl_transport_'+name).argtypes=[C.c_void_p]
        ranges=(C.c_uint64*4)(*[v for r in regions for v in (r['device_offset_bytes'],r['end_bytes_exclusive'])])
        self.handle=self.lib.ukl_transport_open(device['serial'].encode(),device['capacity_bytes'],ranges)
        if not self.handle:raise RuntimeError(self.error())
        self.identity=dict(device);self.closed=False
    def error(self):return self.lib.ukl_transport_error().decode(errors='replace')
    def read(self,offset,size):
        b=C.create_string_buffer(size)
        if self.lib.ukl_transport_io(self.handle,0,offset,b,size):raise RuntimeError(self.error())
        return b.raw
    def write(self,offset,data):
        if not data or not any(lo<=offset and offset+len(data)<=hi for lo,hi in self.write_ranges):
            raise RuntimeError('Write overlaps retained content or lies outside resume suffix')
        b=C.create_string_buffer(data,len(data))
        if self.lib.ukl_transport_io(self.handle,1,offset,b,len(data)):raise RuntimeError(self.error())
    def flush(self):
        if self.lib.ukl_transport_flush(self.handle):raise RuntimeError(self.error())
    def close(self):
        if self.handle:
            if self.lib.ukl_transport_close(self.handle):raise RuntimeError('Native DMA cleanup failed: '+self.error())
            self.handle=None;self.closed=True
