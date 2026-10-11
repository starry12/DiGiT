"""CL accepted CPU preparation facade; no arrays, CUDA or device opens."""
from types import SimpleNamespace
from candidates.cl_night_prepare_v9 import protocol as C
ROOT=C.ROOT;OUT=C.OUT;N=C.N;ROWS=C.ROWS;LOCK=C.BASE.LOCK
METADATA=C.metadata()['metadata']
EXTENT=sum((v['length']+4095)//4096*4096 for v in C.metadata()['specs'].values())
binding=C.binding;source_device=C.source_device;verify_manifest=C.verify_manifest
BASE=SimpleNamespace(EXTENT=EXTENT,binding=binding,source_device=source_device)
