"""Bind accepted filesystem products; canonical imports avoid duplicate extensions."""
from types import SimpleNamespace
import numpy as np
from .common import *

def protocol():
    p=read(PREP/'plan/protocol.json')
    p['ssd_offsets']={'gids':4*2**40,'digit':int(4.25*2**40)}
    p['execution'].update(profile_batches=100,native_short_updates=4,native_ready=False)
    return p

def bind(output):
    state=read(PREP/'status.json');require(state['complete'] and state['passed'],'Filesystem preparation incomplete')
    from candidates.uks_prepare_v3.common import verify as original
    require(original()==state['source_sha256'],'Preparation sources changed')
    files={};receipts={}
    for stage,digest in state['completed'].items():
        receipt=DATA/stage/'receipt.json';require(sha(receipt)==digest,'Preparation receipt changed')
        value=read(receipt);require(value['passed'],'Unaccepted preparation');receipts[stage]=value
        for name,expected in value['files'].items():
            path=DATA/stage/name;before=identity(path)
            write(output/'progress.json',dict(stage='binding',file=str(path)))
            require(sha(path)==expected and identity(path)==before,'Preparation data changed: '+str(path))
            files[str(path)]=dict(identity=before,sha256=expected)
    require(receipts['g2']['expanded_adjacency_multiset_exact'] and receipts['payload']['logical_aliases_bit_exact'],'Missing graph/feature acceptance')
    p=protocol();write(output/'protocol.json',p)
    result=dict(passed=True,source_sha256=verify(),protocol_sha256=sha(output/'protocol.json'),files=files,receipts=receipts,raw_ssd_writes=False)
    write(output/'binding.json',result);return result

def check():
    b=read(OUT/'binding.json');require(b['source_sha256']==verify() and b['protocol_sha256']==sha(OUT/'protocol.json'),'Binding context changed')
    for path,d in b['files'].items():require(identity(path)==d['identity'],'Input changed: '+path)
    return b

def bundle():
    from .runtime import setup_sampling_imports
    setup_sampling_imports()
    from digit.artifacts import ArtifactBundle,_build_manifest
    p=protocol();b=check();g=b['receipts']['g2'];c=b['receipts']['csc']
    arrays={name[:-4]:np.load(DATA/'g2'/name,mmap_mode='r') for name in g['files']}
    arrays['reordered_features']=np.load(DATA/'payload/features.npy',mmap_mode='r')
    m=_build_manifest(arrays,'UKS','full',p['nodes'],c['normalized_edges'],256,2,.2,g['num_primary_groups'],4096,{},dict(source_sha256=b['source_sha256']),minimum_transfer_bytes=4096,target_request_bytes=4096)
    return ArtifactBundle(DATA/'g2',m,arrays)

def graph_sampler(arm,seed):
    from .runtime import setup_sampling_imports
    setup_sampling_imports()
    import torch,dgl
    from digit.sampler import DiGiTNeighborSampler
    p=protocol();arrays=[np.load(DATA/'csc'/(name+'.npy'),mmap_mode='c') for name in ('indptr','indices','eids')]
    tensors=tuple(torch.from_numpy(v) for v in arrays)
    graph=dgl.graph(('csc',tensors),num_nodes=p['nodes'],idtype=torch.int64).formats('csc');graph.create_formats_();graph.pin_memory_()
    require(graph.is_pinned(),'Graph not pinned')
    for x,y in zip(tensors,graph.adj_tensors('csc')):require(x.data_ptr()==y.data_ptr(),'CSC unexpectedly copied')
    artifact=bundle() if arm=='digit' else None
    sampler=DiGiTNeighborSampler(p['fanouts'],artifact,cuda_mode='required',metadata_mode='gpu_i32_uva_eid64',random_seed=seed) if arm=='digit' else dgl.dataloading.NeighborSampler(p['fanouts'])
    if arm=='digit':sampler._ensure_cuda_metadata(graph,torch.device('cuda:0'))
    return graph,arrays,artifact,sampler
