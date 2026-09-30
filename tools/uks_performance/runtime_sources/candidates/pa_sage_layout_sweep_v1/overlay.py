"""New per-point reverse-edge overlay; never rewrite the source graph or bundle."""
import numpy as np
from .common import *
from candidates.pa_sage_bidir_native_v2.overlay import rewrite,exact_validate

def build_overlay(layout,directed_indptr,bidirectional,edge_count,output,fixture=False):
    from .build import exclusive
    if fixture:return _build_overlay(layout,directed_indptr,bidirectional,edge_count,output,True)
    with exclusive():return _build_overlay(layout,directed_indptr,bidirectional,edge_count,output,False)

def _build_overlay(layout,directed_indptr,bidirectional,edge_count,output,fixture):
    layout=Path(layout);output=Path(output)
    require(not output.exists(),'Overlay output already exists')
    receipt=read(layout/'build_receipt.json');bundle=layout/'final/bundle'
    require(receipt['passed'] and sha(bundle/'manifest.json')==receipt['manifest_sha256'],'Unverified layout')
    m=read(bundle/'manifest.json');n=m['dataset']['num_nodes']
    if fixture:require(n<=4096 and m['dataset']['num_edges']<=131072,'Overlay fixture too large')
    else:require(host()>100*2**30,'Insufficient host reserve for overlay')
    require(type(edge_count) is int and edge_count>=0,'Invalid original nonself edge count')
    oldptr=array(bundle/'reordered_indptr.npy');oldidx=array(bundle/'reordered_indices.npy')
    original=array(directed_indptr)
    bp,bi,be=(array(Path(bidirectional)/('original_'+name+'.npy')) for name in ('indptr','indices','eids'))
    require(len(original)==len(bp)==n+1 and len(be)==len(bi)==int(bp[-1]),'Wrong graph shape')
    if fixture:require(len(bi)<=2*131072+4096,'Overlay fixture exceeds edge limit')
    require(int(bp[-1])-int(original[-1])==edge_count,'Wrong reverse-edge extent')
    paths=[bundle/'manifest.json',Path(directed_indptr)]+[bundle/m['files'][name]['path'] for name in ('reordered_indptr','reordered_indices','group_members')]
    paths += [Path(bidirectional)/('original_'+name+'.npy') for name in ('indptr','indices','eids')]
    bindings={}
    for path in paths:
        before=identity(path);digest=sha(path)
        require(identity(path)==before,'Overlay input changed while hashing')
        bindings[str(path)]=dict(sha256=digest,identity=before)
    code=code_bindings()
    for name in ('reordered_indptr','reordered_indices','group_members'):
        require(bindings[str(bundle/m['files'][name]['path'])]['sha256']==m['files'][name]['sha256'],'Layout arrays changed')
    require(bindings[str(Path(directed_indptr))]['sha256']==m['source']['indptr']['sha256'],'Directed graph differs from layout source')
    if not fixture:
        prepared=read(Path(bidirectional)/'prepared.json')
        require(prepared['passed'],'Bidirectional graph is not prepared')
        for name in ('original_indptr.npy','original_indices.npy','original_eids.npy'):
            require(bindings[str(Path(bidirectional)/name)]['sha256']==prepared['bindings'][name],'Bidirectional graph receipt differs')
    output.mkdir(parents=True,exist_ok=False)
    rp=np.lib.format.open_memmap(output/'reordered_indptr.npy',mode='w+',dtype='<i8',shape=oldptr.shape)
    ri=np.lib.format.open_memmap(output/'reordered_indices.npy',mode='w+',dtype='<i8',shape=(len(oldidx)+edge_count,))
    rewrite(oldptr,oldidx,original,bp,bi,be,edge_count,rp,ri,notify=lambda **kw:progress(output,**kw))
    rp.flush();ri.flush()
    # A graph with no groups is legal. The inherited validator supports it.
    result=exact_validate(rp,ri,bp,bi,array(bundle/'group_members.npy'),notify=lambda **kw:progress(output,**kw))
    for path,desc in bindings.items():require(identity(path)==desc['identity'],'Overlay input changed')
    require(code_bindings()==code,'Overlay code changed')
    result.update(point=receipt['point'],layout_manifest_sha256=receipt['manifest_sha256'],inputs=bindings,code_sha256=code,
                  files={name:sha(output/name) for name in ('reordered_indptr.npy','reordered_indices.npy')},
                  raw_ssd_access=False,native_ready=False,fixture=fixture)
    write(output/'overlay_receipt.json',result);return result
