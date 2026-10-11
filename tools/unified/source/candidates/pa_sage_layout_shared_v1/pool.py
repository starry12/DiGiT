"""Read-only reuse of identified, previously written PA and IG test extents."""
import argparse
import hashlib
import os
import time
import numpy as np
from .common import ROOT,HERE,Path,read,write,sha,require,setup,verify

OUT=ROOT/'results/pa_sage_layout_shared_20260925_v1'

def identity(path):
    s=Path(path).stat()
    return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)

def spec(mode):
    p=read(HERE/'pools.json')[mode]
    require(sha(p['state'])==p['state_sha256'],'Registered payload changed')
    state=read(p['state'])
    require(state['status']=='verified' and state['device']=='/dev/libnvm0' and
            state['device_offset_bytes']==p['offset'] and state['payload_bytes']==p['capacity_bytes'] and
            state['feature_file_sha256']==p['source_full_sha256'],'Wrong pool state')
    require(p['offset']%4096==0 and p['verify_bytes']%4096==0 and 0<p['verify_bytes']<=p['capacity_bytes'],'Invalid pool extent')
    require(identity(p['source'])==p['source_identity'],'Pool source file changed')
    return p

def source_rows(mode):
    p=spec(mode)
    # The IG source is raw float32 despite its .npy filename. The PA source has
    # a real NPY header. Never infer one format from a suffix.
    return np.memmap(p['source'],mode='r',dtype='<f4',offset=p['header_bytes'],shape=(p['capacity_bytes']//512,128))

def check_receipt(mode):
    p=spec(mode);path=OUT/'pools'/(mode+'_receipt.json');r=read(path)
    require(r['passed'] and r['mode']==mode and r['raw_ssd_writes'] is False,'Wrong pool receipt')
    require(r['pool_spec_sha256']==sha(HERE/'pools.json') and r['state_sha256']==p['state_sha256'],'Pool binding changed')
    require(r['verified_bytes']==p['verify_bytes'] and r['offset']==p['offset'] and r['all_finite'],'Incomplete pool readback')
    require(r['source_identity']==identity(p['source']),'Source changed since pool readback')
    require(r['actual_sha256']==r['expected_sha256'] and len(r['actual_sha256'])==64,'Readback mismatch')
    return dict(r,receipt_path=str(path))

def native_verify(mode):
    from candidates.pa_sage_layout_native_v2.common import device_locks
    from ae.common import check_device
    require(os.geteuid()==0 and os.environ.get('CUDA_VISIBLE_DEVICES')=='2','Root / physical GPU 2 required')
    with device_locks():
        execution=verify();p=spec(mode);path=OUT/'pools'/(mode+'_receipt.json')
        require(not path.exists(),'Preserve prior readback');check_device();setup()
        import torch
        from GIDS import GIDS
        require(torch.cuda.mem_get_info()[0]>torch.cuda.mem_get_info()[1]-2**30,'GPU is busy')
        source=source_rows(mode);rows=p['verify_bytes']//512
        loader=GIDS(page_size=4096,off=p['offset'],num_ele=rows*128,num_ssd=1,cache_size=64,
                    cache_dim=128,ssd_list=[0],ctrl_idx=0,feature_index_mode='explicit',mixed_io=False)
        actual_hash=hashlib.sha256();expected_hash=hashlib.sha256();started=time.time();low=float('inf');high=-float('inf')
        for lo in range(0,rows,65536):
            hi=min(rows,lo+65536);n=hi-lo
            ids=torch.arange(lo,hi,device='cuda',dtype=torch.int64)
            values=torch.empty((n,128),device='cuda',dtype=torch.float32)
            loader.BAM_FS.read_feature(values.data_ptr(),ids.data_ptr(),n,128,128,0,0)
            torch.cuda.synchronize();actual=values.cpu().numpy();expected=np.array(source[lo:hi],copy=True)
            require(np.array_equal(actual.view(np.uint32),expected.view(np.uint32)),'Pool readback differs at row '+str(lo))
            require(np.isfinite(actual).all(),'Nonfinite pool values at row '+str(lo))
            actual_hash.update(memoryview(actual).cast('B'));expected_hash.update(memoryview(expected).cast('B'))
            low=min(low,float(actual.min()));high=max(high,float(actual.max()))
            if lo%(65536*32)==0 or hi==rows:
                write(OUT/'pools'/(mode+'_progress.json'),dict(stage='full_readonly_verification',mode=mode,
                     verified_bytes=hi*512,total_bytes=p['verify_bytes'],updated_unix=time.time()))
                print(mode,hi*512,'/',p['verify_bytes'],flush=True)
        require(spec(mode)==p and verify()==execution,'Source/code changed during verification')
        write(path,dict(passed=True,mode=mode,offset=p['offset'],verified_bytes=rows*512,all_finite=True,
            min_value=low,max_value=high,raw_ssd_writes=False,source_identity=identity(p['source']),
            state_sha256=p['state_sha256'],pool_spec_sha256=sha(HERE/'pools.json'),candidate_sha256=execution,
            actual_sha256=actual_hash.hexdigest(),expected_sha256=expected_hash.hexdigest(),elapsed_seconds=time.time()-started))

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('real','shared'),required=True);a=p.parse_args();native_verify(a.mode)
if __name__=='__main__':main()
