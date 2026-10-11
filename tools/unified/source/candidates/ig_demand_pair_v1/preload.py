"""Reusable, all-byte-verified CPU feature pack; no raw SSD payload writes."""
import hashlib
import os
from pathlib import Path
import numpy as np
from candidates.cache_epoch_v1.io import direct_chunks, identity
from candidates.ukl_runtime_prepare_v11r1.data import direct_file
from ae.common import read, write, require


def selected_chunks(path, expected, physical_hot, header, progress):
    pending=b'';row=0;loaded=0
    for chunk in direct_chunks(path,expected):
        blob=pending+bytes(chunk);chunk.release()
        if header:
            require(header<len(blob),'Feature header exceeds chunk')
            blob=blob[header:];header=0
        length=len(blob)//4096*4096;pending=blob[length:]
        values=np.frombuffer(blob,dtype='<f4',count=length//4).reshape(-1,1024)
        lo,hi=np.searchsorted(physical_hot,[row,row+len(values)])
        if hi>lo:
            selected=np.ascontiguousarray(values[physical_hot[lo:hi]-row])
            require(np.isfinite(selected).all(),'Nonfinite cached features')
            loaded+=len(selected);yield selected
        row+=len(values);progress(loaded,row)
    require(not pending and loaded==len(physical_hot),'Incomplete feature pack')


def populate(features, expected_cache_sha):
    # Use the already-bound source path, not the relocatable ${PROJECT_ROOT}
    # spelling in the payload plan.
    path=Path(features.source_path).resolve()
    source=dict(identity=identity(path),bytes=path.stat().st_size,sha256=features.plan['file_sha256'])
    key=hashlib.sha256((features.physical_hot_digest+expected_cache_sha).encode()).hexdigest()
    folder=Path('/mnt/n0/digit/ig_uks_path_v1/cache')/key
    folder.mkdir(parents=True,exist_ok=True)
    pack=folder/'features.bin';receipt=folder/'receipt.json'
    expected=dict(source=source,physical_hot_sha256=features.physical_hot_digest,
                  cpu_cache_sha256=expected_cache_sha,rows=len(features.physical_hot))
    import time
    last=[0.]
    def progress(done,scanned=0):
        if time.monotonic()-last[0]>30:
            features.mark('cpu_cache_pack',rows_done=done,total_rows=len(features.physical_hot),storage_rows_scanned=scanned)
            last[0]=time.monotonic()
    created=not receipt.exists()
    if created:
        require(not pack.exists(),'Uncommitted pack; retain evidence and use a new pack directory')
        free=os.statvfs(folder);need=len(features.physical_hot)*4096+8*2**30
        require(free.f_bavail*free.f_frsize>=need,'Insufficient disk for verified CPU pack')
        chunks=selected_chunks(path,source,features.physical_hot,features.plan['header_bytes'],progress)
        result=direct_file(pack,chunks,rate=1024**3)
        require(result['bytes']==len(features.physical_hot)*4096 and result['sha256']==expected_cache_sha,
                'Pack differs from accepted CPU cache bytes')
        write(receipt,dict(expected=expected,file=result))
    r=read(receipt);require(r['expected']==expected,'Pack source/hot-set identity changed')
    loaded=0
    import IGPerfNative as native
    for chunk in direct_chunks(pack,r['file']):
        values=np.frombuffer(chunk,dtype='<f4').reshape(-1,1024)
        rows=np.ascontiguousarray(features.physical_hot[loaded:loaded+len(values)])
        require(np.isfinite(values).all(),'Nonfinite packed features')
        native.short_fill(features.store,rows,values);loaded+=len(values)
        del values;chunk.release();progress(loaded)
    require(loaded==len(features.physical_hot) and identity(path)==source['identity'],'Changed source or incomplete pack preload')
    features.cache_sha=expected_cache_sha
    return dict(created=created,receipt=str(receipt),sha256=r['file']['sha256'],bytes=r['file']['bytes'],
                full_source_scan_this_worker=created,all_pack_bytes_verified=True)
