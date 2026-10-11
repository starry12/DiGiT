"""Standalone CPU preparation; large scans are explicitly requested, not imports."""
import argparse,ctypes as C,hashlib,json,os
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def identity(p):
    s=Path(p).stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
def exact_permutation(out,train):
    if out.dtype!=np.int64 or out.shape!=(train,):raise ValueError('Canonical roots width/extent')
    seen=np.zeros(train,np.bool_)
    for at in range(0,train,1048576):
        ids=out[at:at+1048576]
        if ids.min()<0 or ids.max()>=train or len(np.unique(ids))!=len(ids) or seen[ids].any():raise ValueError('Not an exact training permutation')
        seen[ids]=True
    if not seen.all():raise ValueError('Incomplete BFS')
def generate(ptr,idx,train,out):
    if ptr.dtype!=np.int64 or idx.dtype not in (np.int32,np.int64) or out.dtype!=np.int64 or out.shape!=(train,):raise ValueError('BFS widths')
    if not all(a.flags.c_contiguous for a in (ptr,idx,out)):raise ValueError('Contiguous BFS arrays required')
    lib=C.CDLL(str(HERE/'runtime/libbfs.so'));fn=lib.bfs64 if idx.dtype==np.int64 else lib.bfs
    fn.argtypes=[C.c_void_p,C.c_void_p,C.c_int64,C.c_int64,C.c_int64,C.c_void_p];fn.restype=C.c_int
    rc=fn(ptr.ctypes.data,idx.ctypes.data,len(ptr)-1,len(idx),train,out.ctypes.data)
    if rc:raise ValueError('Invalid BFS graph: '+str(rc))
    exact_permutation(out,train)
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    root=HERE.parents[1];csc=root/'data/igb/csc';config=json.loads((root/'candidates/ig_demand_pair_v1/protocol.json').read_text())
    files=[csc/'original_indptr.npy',csc/'original_indices.npy'];plan=dict(train_nodes=config['train_nodes'],orientation='incoming_CSC_train_induced',source=[str(x) for x in files],output=str(a.output))
    if not a.execute:print(json.dumps(plan));return
    before=[identity(x) for x in files]
    if a.output.exists() or a.output.with_suffix('.partial.npy').exists():raise ValueError('Preserve old BFS output')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    ptr,idx=(np.load(x,mmap_mode='r',allow_pickle=False) for x in files)
    if len(ptr)!=config['nodes']+1 or len(idx)!=config['edges']:raise ValueError('Wrong IG graph')
    partial=a.output.with_suffix('.partial.npy');out=np.lib.format.open_memmap(partial,mode='w+',dtype='<i8',shape=(config['train_nodes'],))
    generate(ptr,idx,config['train_nodes'],out);out.flush();del out
    hashes=[sha(x) for x in files]
    if before!=[identity(x) for x in files]:raise ValueError('BFS sources changed')
    # Source payloads are verified again by the admitted graph loader before use.
    with partial.open('rb') as f:os.fsync(f.fileno())
    os.rename(partial,a.output)
    receipt=dict(passed=True,exact_permutation=True,**plan,source_identities=before,source_file_sha256=hashes,output_sha256=sha(a.output))
    with a.output.with_suffix('.json').open('x') as f:json.dump(receipt,f,indent=2)
if __name__=='__main__':main()
