"""Checkpointed filesystem-only UKS preparation; no GPU or raw SSD operations."""
import argparse,json,os,shutil,subprocess,time,fcntl,contextlib
from pathlib import Path
import numpy as np
from .common import ROOT,HERE,OUT,LARGE,read,sha,require,verify,heavy_gate
from .protocol import compile_plan
from .data import row_block,training_ids,epoch_order
from candidates.pa_sage_cache_policy_v6.common import write,identity
CPP=HERE/'bin/prepare_graph'

def convert(raw,path,shape,dtype='<i8'):
    raw=Path(raw);path=Path(path);dt=np.dtype(dtype)
    require(raw.stat().st_size==int(np.prod(shape))*dt.itemsize,'Raw output extent mismatch')
    dst=np.lib.format.open_memmap(path,mode='w+',dtype=dt,shape=shape)
    if dst.size:
        src=np.memmap(raw,mode='r',dtype=dt,shape=(dst.size,))
        flat=dst.reshape(-1)
        for lo in range(0,len(src),1048576):flat[lo:lo+1048576]=src[lo:lo+1048576]
    dst.flush();return path

def native(args,folder):
    with (folder/'build.log').open('a') as f:
        f.write(json.dumps([str(CPP),*map(str,args)])+'\n');f.flush()
        subprocess.run([str(CPP),*map(str,args)],stdout=f,stderr=subprocess.STDOUT,check=True)

def build_csc(source,n,folder,bucket_nodes=1048576):
    edges=np.load(source,mmap_mode='r');require(edges.dtype==np.int64 and edges.ndim==2 and 2 in edges.shape and edges.flags.c_contiguous,'Expected C-order int64 edges')
    rows=edges.shape[0]==2;e=edges.shape[1] if rows else edges.shape[0]
    native(['csc',source,n,e,int(rows),folder,bucket_nodes],folder)
    total,removed=map(int,(folder/'counts.txt').read_text().split())
    require(total==e-removed+n,'Normalized edge total mismatch')
    convert(folder/'indptr.bin',folder/'indptr.npy',(n+1,));convert(folder/'indices.bin',folder/'indices.npy',(total,))
    ids=np.lib.format.open_memmap(folder/'eids.npy',mode='w+',dtype='<i8',shape=(total,))
    for lo in range(0,total,1048576):ids[lo:lo+1048576]=np.arange(lo,min(total,lo+1048576),dtype=np.int64)
    ids.flush()
    return dict(nodes=n,source_edges=e,normalized_edges=total,removed_self_edges=removed,eid_dtype='int64',stable_input_order_with_self_loop_last=True)

def build_rank(csc,n,folder):
    native(['rank',csc/'indptr.npy',csc/'indices.npy',n,folder],folder)
    convert(folder/'revpr.bin',folder/'revpr.npy',(n,),'<f8');convert(folder/'hot_nodes.bin',folder/'hot_nodes.npy',(n//10,))
    return dict(iterations=20,damping=.85,graph='reverse of normalized directed graph',hot_rows=n//10)

def build_g2(csc,rank,n,folder):
    native(['g2',csc/'indptr.npy',csc/'indices.npy',rank/'hot_nodes.npy',n,folder],folder)
    primary,replica,rows,units=map(int,(folder/'counts.txt').read_text().split());g=primary+replica
    require(replica*2<=n//5 and rows<2**31 and n+g<2**31,'Layout limits exceeded')
    shapes=dict(group_members=(g,2),group_owner=(g,),group_storage_base=(g,),supernode_to_group=(g,),node_to_primary_row=(n,),storage_to_node=(rows,),reordered_indptr=(n+g+1,),reordered_indices=(units,))
    for name,shape in shapes.items():convert(folder/(name+'.bin'),folder/(name+'.npy'),shape)
    native(['validate',csc/'indptr.npy',csc/'indices.npy',n,folder],folder)
    return dict(expanded_adjacency_multiset_exact=True,primary_inverse_exact=True,num_primary_groups=primary,num_replica_groups=replica,num_storage_rows=rows,reordered_units=units,group_size=2,replication_ratio=.2,alignment_rows=4,
                group_rng='mt19937_64 seed0, rejection-sampled Fisher-Yates; independent reconstruction, not byte-identical PA layout')

def synthetic(p,folder):
    n=p['nodes'];block=p['synthetic']['block_rows']
    features=np.lib.format.open_memmap(folder/'features.npy',mode='w+',dtype='<f4',shape=(n,256))
    labels=np.lib.format.open_memmap(folder/'labels.npy',mode='w+',dtype='<i8',shape=(n,))
    for lo in range(0,n,block):
        take=min(block,n-lo);features[lo:lo+take]=row_block(p,lo,take);labels[lo:lo+take]=row_block(p,lo,take,labels=True)
        if lo%(block*64)==0:write(folder/'progress.json',dict(stage='logical_features',nodes_done=lo,nodes=n,updated_unix=time.time()))
    features.flush();labels.flush();train=training_ids(p);roots,receipt=epoch_order(p,train)
    np.save(folder/'train.npy',train);np.save(folder/'roots.npy',roots);return receipt

def payload(synthetic_dir,g2,folder):
    features=np.load(synthetic_dir/'features.npy',mmap_mode='r');mapping=np.load(g2/'storage_to_node.npy',mmap_mode='r')
    dst=np.lib.format.open_memmap(folder/'features.npy',mode='w+',dtype='<f4',shape=(len(mapping),256))
    for lo in range(0,len(mapping),65536):
        ids=mapping[lo:lo+65536];mask=ids>=0;require(not mask.any() or ids[mask].max()<len(features),'Invalid feature mapping')
        part=np.zeros((len(ids),256),np.float32);part[mask]=features[ids[mask]];dst[lo:lo+len(ids)]=part
        if lo%(65536*64)==0:write(folder/'progress.json',dict(stage='grouped_features',rows_done=lo,rows=len(mapping),updated_unix=time.time()))
    dst.flush()
    # Complete logical-value readback, including zero padding.
    for lo in range(0,len(mapping),65536):
        ids=mapping[lo:lo+65536];mask=ids>=0;got=np.asarray(dst[lo:lo+len(ids)])
        require(np.array_equal(got[mask].view(np.uint32),np.asarray(features[ids[mask]]).view(np.uint32)),'Replica feature mismatch')
        require(np.all(got[~mask]==0),'Padding must be zero')
    return dict(rows=len(mapping),row_bytes=1024,logical_aliases_bit_exact=True,padding_zero=True,raw_ssd_writes=False)

def run(resume=False):
    heavy_gate();source=verify();p=compile_plan();src=Path(p['paths']['source'])/'edge_index.npy'
    require(os.geteuid()==0,'Preparation uses administrator-owned experiment locks')
    edge_header=np.load(src,mmap_mode='r')
    require(edge_header.dtype==np.int64 and edge_header.shape==(2,p['source_edges']),'Unexpected UKS source header')
    del edge_header
    require(CPP.is_file(),'Compile filesystem builder first')
    require(np.__version__==p['synthetic']['numpy_version'],'Synthetic generator version changed')
    with contextlib.ExitStack() as stack:
        for name in ('/run/digit-ae-selfservice/exclusive.lock','/tmp/digit-pa-bidir-controller.lock','/tmp/digit-pa-sage-libnvm0.lock'):
            f=stack.enter_context(open(name,'r'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from ae.common import host
        require(host()>=128*2**30,'Filesystem preparation needs 128 GiB available RAM, not a training admission')
        require(shutil.disk_usage('/mnt/n0').free>=1536*2**30,'Reserve 1.5 TiB free for retained raw/NPY, payload and scratch')
        OUT.mkdir(parents=True,exist_ok=True);status=OUT/'status.json'
        if status.exists():require(resume,'Use --resume for this preparation')
        else:require(not resume and not LARGE.exists(),'Preserve existing data; use a new run')
        state=read(status) if resume else dict(source_sha256=source,input_identity=identity(src),completed={},complete=False,started_unix=time.time())
        require(state['source_sha256']==source and state['input_identity']==identity(src),'Source/inputs changed')
        LARGE.mkdir(parents=True,exist_ok=True);write(OUT/'protocol.json',p)
        stages=[('csc',lambda d:build_csc(src,p['nodes'],d)),('rank',lambda d:build_rank(LARGE/'csc',p['nodes'],d)),
                ('g2',lambda d:build_g2(LARGE/'csc',LARGE/'rank',p['nodes'],d)),('synthetic',lambda d:synthetic(p,d)),
                ('payload',lambda d:payload(LARGE/'synthetic',LARGE/'g2',d))]
        try:
            if 'source_sha256_file' not in state:
                state.update(stage='hash_source');write(status,state);state['source_sha256_file']=sha(src);write(status,state)
            for stage,fn in stages:
                if stage in state['completed']:
                    r=read(LARGE/stage/'receipt.json')
                    require(sha(LARGE/stage/'receipt.json')==state['completed'][stage],'Stage receipt changed')
                    for name,digest in r['files'].items():require(sha(LARGE/stage/name)==digest,'Output changed: '+name)
                    continue
                folder=LARGE/stage
                # Failed folders are retained. Restart only this stage into a new attempt.
                if folder.exists():folder.rename(LARGE/(stage+'_failed_'+str(time.time_ns())))
                folder.mkdir();state.update(stage=stage,pid=os.getpid(),updated_unix=time.time(),error=None);write(status,state)
                before=time.time();value=fn(folder)
                require(identity(src)==state['input_identity'] and verify()==source,'Sources changed during preparation')
                value.update(passed=True,native_ready=False,seconds=time.time()-before,files={f.name:sha(f) for f in folder.glob('*.npy')})
                write(folder/'receipt.json',value);state['completed'][stage]=sha(folder/'receipt.json');write(status,state)
            state.update(stage='filesystem_complete',complete=True,passed=True,native_ready=False,raw_ssd_writes=False,finished_unix=time.time());write(status,state)
        except BaseException as e:
            state.update(passed=False,stage='failed',error=repr(e),updated_unix=time.time());write(status,state);raise

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--execute',action='store_true');a.add_argument('--resume',action='store_true');v=a.parse_args()
    require(v.execute,'Explicit filesystem preparation required');run(v.resume)
