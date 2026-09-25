"""Keep NPY container hashes separate from logical CSC array hashes."""
import numpy as np
from .common import Path,read,require

def descriptors(data,graph):
    data=Path(data);receipt=read(data/'prepared.json');prepared=receipt['graph']
    require(receipt['passed'] and receipt['validation']['passed'],'CSC preparation did not pass')
    require(receipt['validation']['nodes']==graph['nodes'] and receipt['validation']['edges']==graph['edges'],'Wrong prepared graph extent')
    require(set(prepared)=={'indptr','indices','eids'},'Unexpected CSC preparation schema')
    require(set(graph['csc_sha256'])==set(prepared),'Incomplete logical CSC binding')
    result={}
    for name,logical_digest in graph['csc_sha256'].items():
        d=prepared[name];filename='original_'+name+'.npy'
        require(d['file']==filename and d['array_sha256']==logical_digest,'Wrong logical CSC array: '+name)
        require(receipt['bindings'][filename]==d['sha256'],'Conflicting CSC file hash records')
        expected_shape=[graph['nodes']+1 if name=='indptr' else graph['edges']]
        require(d['shape']==expected_shape and len(d['sha256'])==64,'Wrong CSC descriptor: '+name)
        a=np.load(data/filename,mmap_mode='r',allow_pickle=False)
        require(a.dtype==np.dtype('<i8') and list(a.shape)==expected_shape,'Wrong CSC header: '+name)
        result[name]=dict(path=data/filename,file_sha256=d['sha256'],array_sha256=logical_digest,shape=expected_shape)
    return result

def bind(data,graph,add):
    values=descriptors(data,graph)
    for d in values.values():add(d['path'],d['file_sha256'])
    return {name:{k:v for k,v in d.items() if k!='path'} for name,d in values.items()}
