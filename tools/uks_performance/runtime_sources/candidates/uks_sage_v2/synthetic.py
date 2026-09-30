"""Deterministic block-based synthetic data; called only behind the idle guard."""
import hashlib,os,time
from pathlib import Path
from .common import write,identity

def generate(output,protocol,notify=None):
    import numpy as np
    p=protocol;spec=p['synthetic'];out=Path(output)
    if np.__version__!=spec['numpy_version']:raise RuntimeError('Use frozen NumPy '+spec['numpy_version'])
    n=p['nodes'];d=p['feature_dim'];block=spec['block_rows'];needed=p['measurement']['total_batches']*p['batch_size']
    if min(n,d,block,p['classes'])<=0 or not needed<=p['training']['train_nodes']<=n:raise ValueError('Invalid source dimensions or insufficient training roots')
    out.mkdir(parents=True,exist_ok=False);started=time.time();files={}
    for name,shape,dtype,seed in [('node_feat.npy',(n,d),'<f4',spec['feature_seed']),('node_label.npy',(n,),'<i8',spec['label_seed'])]:
        path=out/name
        with path.open('xb') as f:
            np.lib.format.write_array_header_1_0(f,dict(descr=dtype,fortran_order=False,shape=shape));header_bytes=f.tell()
        digest=hashlib.sha256(path.read_bytes())
        with path.open('ab') as f:
            for index,lo in enumerate(range(0,n,block)):
                count=min(block,n-lo);rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed,index])))
                if name=='node_feat.npy':values=rng.random((count,d),dtype=np.float32)*np.float32(2)-np.float32(1)
                else:values=rng.integers(0,p['classes'],size=count,dtype=np.int64)
                blob=np.asarray(values,dtype=dtype).tobytes(order='C');f.write(blob);digest.update(blob)
                if notify:notify(stage='synthetic_'+name,rows_done=lo+count,total_rows=n)
            f.flush();os.fsync(f.fileno())
        files[name]=dict(sha256=digest.hexdigest(),header_bytes=header_bytes,identity=identity(path),shape=list(shape),dtype=dtype)
    count=p['training']['train_nodes'];rng=np.random.Generator(np.random.PCG64(p['training']['split_seed']))
    selected=np.sort(rng.choice(n,size=count,replace=False)).astype('<i8');order=selected[np.random.Generator(np.random.PCG64(p['training']['order_seed'])).permutation(count)]
    needed=p['measurement']['total_batches']*p['batch_size']
    if count<needed:raise ValueError('Training subset smaller than benchmark roots')
    for name,values in [('train_nodes.npy',selected),('benchmark_roots.npy',order[:needed])]:
        path=out/name
        with path.open('xb') as f:np.save(f,values);f.flush();os.fsync(f.fileno())
        files[name]=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),identity=identity(path),shape=list(values.shape),dtype=values.dtype.str)
    result=dict(passed=True,kind='synthetic_sources_only',files=files,seconds=time.time()-started,accuracy_claim=False,native_ready=False,numpy=np.__version__,protocol=protocol)
    write(out/'ready.json',result);return result
