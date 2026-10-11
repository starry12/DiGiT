"""Parameter-driven filesystem layout; derived from the accepted PA preparation.
No CUDA imports, model training, device mapping, or raw SSD writes.
"""
import gc, math, os, time
from types import SimpleNamespace
import numpy as np
from .common import *
from .cpu_runtime import artifacts

def group(ptr,idx,order,hot,priority,g=2,replica_percent=20,sink=None,notify=None):
    n=len(ptr)-1;used=np.zeros(n,dtype=bool);covered=np.zeros(len(idx),dtype=bool)
    counts=np.zeros(n,dtype=np.int32);rng=np.random.default_rng(0);primary=replica=0
    cap=(n*replica_percent//100)//g;last=time.time()
    stages=[('primary',order),('replica_priority',priority),('replica_fallback',order)]
    seen=np.zeros(n,dtype=bool)
    for phase,owners in stages:
        for pos,value in enumerate(owners):
            owner=int(value);a=int(ptr[owner]);b=int(ptr[owner+1])
            if phase!='primary':
                if replica>=cap:break
                if seen[owner]:continue
                seen[owner]=True
            if b-a<g:
                if phase in ('primary','replica_fallback'):break
                continue
            neighbors=idx[a:b]
            mask=(~used[neighbors] & ~hot[neighbors]) if phase=='primary' else (used[neighbors] & ~hot[neighbors] & ~covered[a:b])
            positions=np.flatnonzero(mask)
            if len(positions)>=g:
                _,first=np.unique(neighbors[positions],return_index=True)
                candidates=positions[first]+a;take=len(candidates)//g
                if phase!='primary':take=min(take,cap-replica)
                if take:
                    rng.shuffle(candidates);selected=candidates[:take*g];members=idx[selected].reshape(-1,g)
                    assert not covered[selected].any() and not hot[members].any()
                    if sink:sink(phase,owner,members,primary+replica)
                    if phase=='primary':used[members]=True;primary+=take
                    else:replica+=take
                    covered[selected]=True;counts[owner]+=take
            if notify and time.time()-last>=10:
                notify(phase=phase,owners_done=pos+1,primary=primary,replica=replica);last=time.time()
    assert int(used.sum())==primary*g and int(covered.sum())==(primary+replica)*g
    return dict(primary=primary,replica=replica,counts=counts,used=used,covered=covered,rng=rng.bit_generator.state)

def make_metadata(ctx, data, report, bootstrap=False):
    _build_manifest = artifacts()._build_manifest
    folder=data/('bootstrap' if bootstrap else 'final')
    require(not folder.exists(),'Existing incomplete metadata: preserve it and use a new data root')
    folder.mkdir();bundle=folder/'bundle';bundle.mkdir();start=time.time()
    p=ctx.cfg();src=ctx.source_config();N=src['num_nodes'];g=p['paper_specified']['group_size'];align=p['reconstruction_choices']['io_page_bytes']//512
    ctx.check_resources()
    ptr,idx=map(array,ctx.csc_paths());E=len(idx)
    hot_nodes=np.array([],dtype=np.int64) if bootstrap else array(data/'hot_nodes.npy')
    hot=np.zeros(N,dtype=bool);hot[hot_nodes]=True
    degree=np.diff(ptr);order=np.argsort(-degree,kind='stable');del degree
    # Separate complete passes; each phase writes only its own new files.
    with (folder/'members.bin').open('xb') as mf,(folder/'owners.bin').open('xb') as of:
        def sink(phase,owner,members,offset):
            mf.write(members.astype('<i8',copy=False).tobytes());of.write(np.full(len(members),owner,dtype='<i8').tobytes())
        result=group(ptr,idx,order,hot,np.array([],dtype=np.int64),g=g,replica_percent=ctx.point['replica_percent'],sink=sink,
              notify=lambda **kw:progress(report,'bootstrap' if bootstrap else 'final',**kw))
        mf.flush();of.flush();os.fsync(mf.fileno());os.fsync(of.fileno())
    del order
    P=result['primary'];B=result['replica'];G=P+B;counts=result['counts'];covered=result['covered'];used=result['used']
    cold=np.flatnonzero(~used & ~hot);hot_start=G*align;cold_start=((hot_start+len(hot_nodes)+align-1)//align)*align
    ROWS=((cold_start+len(cold)+align-1)//align)*align
    def mm(name,shape):return np.lib.format.open_memmap(bundle/(name+'.npy'),mode='w+',dtype='<i8',shape=shape)
    def flush(a):
        a.flush()
        with open(a.filename,'rb') as f:os.fsync(f.fileno())
    m=mm('group_members',(G,g));
    if G:m[:]=np.memmap(folder/'members.bin',mode='r',dtype='<i8',shape=(G,g))
    flush(m)
    owners=mm('group_owner',(G,));
    if G:owners[:]=np.memmap(folder/'owners.bin',mode='r',dtype='<i8',shape=(G,))
    flush(owners)
    save(folder/'covered.npy',covered);save(folder/'owner_group_counts.npy',counts)
    save(bundle/'group_storage_base.npy',np.arange(G,dtype=np.int64)*align)
    save(bundle/'supernode_to_group.npy',np.arange(G,dtype=np.int64))
    storage=mm('storage_to_node',(ROWS,));storage[:]=-1
    for lo in range(0,G,262144):
        hi=min(G,lo+262144);storage[lo*align:hi*align].reshape(-1,align)[:,:g]=m[lo:hi]
    storage[hot_start:hot_start+len(hot_nodes)]=hot_nodes;storage[cold_start:cold_start+len(cold)]=cold;flush(storage)
    node=mm('node_to_primary_row',(N,))
    for lo in range(0,P,262144):
        hi=min(P,lo+262144);node[m[lo:hi]]=(np.arange(lo,hi)[:,None]*align+np.arange(g))
    node[hot_nodes]=np.arange(hot_start,hot_start+len(hot_nodes));node[cold]=np.arange(cold_start,cold_start+len(cold));flush(node)
    if not bootstrap:save(data/'full_cpu_rows.npy',np.arange(hot_start,hot_start+len(hot_nodes),dtype=np.int64))
    rp=mm('reordered_indptr',(N+G+1,));rp[0]=0;np.cumsum(np.diff(ptr)-(g-1)*counts,out=rp[1:N+1]);rp[N+1:]=rp[N];flush(rp)
    ri=mm('reordered_indices',(int(rp[-1]),));go=np.argsort(owners,kind='stable');co=np.asarray(owners[go])
    unique,first,number=np.unique(co,return_index=True,return_counts=True)
    local=np.arange(G,dtype=np.int64)-np.repeat(first,number);dest=rp[co+1]-counts[co]+local;ri[dest]=N+go
    del co,go,local,dest,unique,first,number,cold,hot,used,result
    for lo in range(0,N,100000):
        hi=min(N,lo+100000);a=int(ptr[lo]);b=int(ptr[hi]);keep=~covered[a:b]
        rc=np.diff(ptr[lo:hi+1])-g*counts[lo:hi];require(np.all(rc>=0),'Negative raw degree')
        prefix=np.r_[0,np.cumsum(rc)];offset=np.repeat(rp[lo:hi]-prefix[:-1],rc)
        require(len(offset)==int(keep.sum()),'Raw accounting mismatch');ri[offset+np.arange(len(offset))]=idx[a:b][keep]
        if lo%5000000==0:progress(report,'bootstrap' if bootstrap else 'final',phase='rewrite',owners_done=hi,groups=G,seconds=time.time()-start)
    flush(ri)
    arrays={q.stem:array(q) for q in bundle.glob('*.npy')}
    arrays['reordered_features']=SimpleNamespace(dtype=np.dtype('<f4'),shape=(ROWS,128))
    manifest=_build_manifest(arrays,'synthetic' if ctx.fixture else 'OGB','fixture' if ctx.fixture else 'papers100M',N,E,128,g,p['paper_specified']['replication_ratio'],P,4096,
        dict(features=src['source_features'],indptr=read(data/'sources.json')['files']['original_indptr'],indices=read(data/'sources.json')['files']['original_indices']),
        dict(graph_source=src['source_contract'],pa_sage_layout_sweep_v1=dict(point=ctx.point,protocol_sha256=sha(ctx.contract_path),bootstrap=bootstrap,hot_count=len(hot_nodes),hot_start=hot_start)),
        minimum_transfer_bytes=4096,target_request_bytes=4096)
    for name,desc in manifest['files'].items():
        if name!='reordered_features':desc['sha256']=sha(bundle/desc['path'])
    write(folder/'descriptor.json',manifest)
    write(folder/'metadata_ready.json',dict(passed=True,primary=P,replica=B,groups=G,storage_rows=ROWS,hot_count=len(hot_nodes),hot_start=hot_start,
          protocol_sha256=sha(ctx.contract_path),descriptor_sha256=sha(folder/'descriptor.json'),seconds=time.time()-start))
    # These are temporary duplicates of the now checksummed NPY arrays owned by this build.
    (folder/'members.bin').unlink();(folder/'owners.bin').unlink()

def validate_metadata(ctx,data,report,bootstrap=False):
    folder=data/('bootstrap' if bootstrap else 'final');bundle=folder/'bundle';m=read(folder/'descriptor.json');ready=read(folder/'metadata_ready.json')
    N=m['dataset']['num_nodes'];g=m['grouping']['group_size'];G=ready['groups'];P=ready['primary'];align=m['io']['alignment_rows'];R=ready['storage_rows']
    arrays={name:array(bundle/d['path']) for name,d in m['files'].items() if name!='reordered_features'}
    for name,d in m['files'].items():
        if name!='reordered_features':require(sha(bundle/d['path'])==d['sha256'],'Metadata hash mismatch '+name)
    ptr,idx=map(array,ctx.csc_paths());covered=array(folder/'covered.npy');members=arrays['group_members'];owners=arrays['group_owner']
    rp=arrays['reordered_indptr'];ri=arrays['reordered_indices'];storage=arrays['storage_to_node'];primary=arrays['node_to_primary_row']
    counts=np.bincount(owners,minlength=N);require(np.array_equal(counts,array(folder/'owner_group_counts.npy')),'Group owner counts')
    require(np.array_equal(np.diff(rp[:N+1]),np.diff(ptr)-(g-1)*counts),'Rewritten degrees');require(np.all(rp[N:]==len(ri)),'Supernode adjacency')
    require(int(covered.sum())==G*g and ready['replica']*g<=(N*ctx.point['replica_percent']//100),'Replica budget')
    hot=np.array([],dtype=np.int64) if bootstrap else array(data/'hot_nodes.npy');hotmask=np.zeros(N,dtype=bool);hotmask[hot]=True
    for lo in range(0,G,262144):
        hi=min(G,lo+262144);slots=storage[lo*align:hi*align].reshape(-1,align)
        require(np.array_equal(slots[:,:g],members[lo:hi]) and np.all(slots[:,g:]==-1),'Storage/padding mismatch')
        require(not hotmask[members[lo:hi]].any(),'Hot node grouped');require(np.all(np.diff(np.sort(members[lo:hi],axis=1),axis=1)>0),'Duplicate group member')
    for lo in range(0,N,1048576):
        hi=min(N,lo+1048576);rows=primary[lo:hi]
        require(np.all((rows>=0)&(rows<R)) and np.array_equal(storage[rows],np.arange(lo,hi)),'Primary inverse mismatch')
    require(np.array_equal(storage[ready['hot_start']:ready['hot_start']+len(hot)],hot),'Hot layout')
    # Independent exact covered-edge multiset comparison, preserving parallel edges.
    keys=(owners[:,None]*N+members).reshape(-1).copy();keys.sort();positions=np.flatnonzero(covered);original=np.empty(len(positions),dtype=np.int64)
    for lo in range(0,len(positions),1048576):
        take=positions[lo:lo+1048576];original[lo:lo+len(take)]=(np.searchsorted(ptr,take,side='right')-1)*N+idx[take]
    original.sort();require(np.array_equal(keys,original),'Expanded group edges differ from original');del keys,original,positions
    seen=np.zeros(G,dtype=bool);raw_total=group_total=0
    for lo in range(0,N,100000):
        hi=min(N,lo+100000);a=int(ptr[lo]);b=int(ptr[hi]);start=int(rp[lo]);units=ri[start:int(rp[hi])];mask=units>=N
        require(np.array_equal(units[~mask],idx[a:b][~covered[a:b]]),'Raw adjacency differs')
        positions=np.flatnonzero(mask)+start;ids=units[mask]-N
        require(len(np.unique(ids))==len(ids) and not seen[ids].any(),'Duplicate group edge');seen[ids]=True
        actual=np.searchsorted(rp[:N+1],positions,side='right')-1;require(np.array_equal(owners[ids],actual),'Group owner differs')
        raw_total+=int((~mask).sum());group_total+=len(ids)
    require(seen.all() and raw_total+g*group_total==len(idx),'Full expanded edge count')
    write(folder/'validation.json',dict(passed=True,expanded_adjacency_multiset_exact=True,storage_inverse_exact=True,group_padding_checked=True,
          hot_excluded=True,expanded_edges=len(idx),descriptor_sha256=sha(folder/'descriptor.json')))

def payload(ctx,data,report):
    load_artifact_bundle = artifacts().load_artifact_bundle
    folder=data/'final';require(read(folder/'validation.json')['passed'],'Final metadata not validated');m=read(folder/'descriptor.json')
    mapping=array(folder/'bundle/storage_to_node.npy');features=array(ctx.source_config()['source_features']['path']);rows=len(mapping)
    target=folder/'bundle/reordered_features.npy';checkpoint=folder/'payload_checkpoint.json';chunk=131072
    binding=dict(protocol_sha256=sha(ctx.contract_path),mapping_sha256=sha(folder/'bundle/storage_to_node.npy'),source_sha256=read(data/'sources.json')['files']['features']['sha256'])
    if checkpoint.exists():
        state=read(checkpoint);require(state['binding']==binding,'Changed payload binding');out=np.lib.format.open_memmap(target,mode='r+')
        for part in state['chunks']:require(hashlib.sha256(out[part['lo']:part['hi']].tobytes()).hexdigest()==part['sha256'],'Existing chunk changed')
    else:
        require(not target.exists(),'Unowned existing payload');out=np.lib.format.open_memmap(target,mode='w+',dtype='<f4',shape=(rows,128));state=dict(binding=binding,chunks=[])
    start=state['chunks'][-1]['hi'] if state['chunks'] else 0
    for lo in range(start,rows,chunk):
        hi=min(rows,lo+chunk);nodes=mapping[lo:hi];valid=nodes>=0;block=np.zeros((hi-lo,128),dtype='<f4');block[valid]=features[nodes[valid]];out[lo:hi]=block
        out.flush()
        with target.open('rb') as f:os.fsync(f.fileno())
        state['chunks'].append(dict(lo=lo,hi=hi,sha256=hashlib.sha256(block.tobytes()).hexdigest()));write(checkpoint,state)
        if len(state['chunks'])%16==0:progress(report,'payload',rows_done=hi,total_rows=rows)
    del out;payload_array=array(target)
    for lo in range(0,rows,chunk):
        hi=min(rows,lo+chunk);nodes=mapping[lo:hi];valid=nodes>=0
        require(np.array_equal(payload_array[lo:hi][valid].view(np.uint32),features[nodes[valid]].view(np.uint32)),'Feature readback differs')
        require(not np.any(payload_array[lo:hi][~valid].view(np.uint32)),'Nonzero padding')
        if lo%(chunk*64)==0:progress(report,'payload',phase='filesystem_readback',rows_checked=hi,total_rows=rows)
    m['files']['reordered_features']['sha256']=sha(target);write(folder/'bundle/manifest.json',m)
    # Runtime GPU admission and raw SSD acceptance are deliberately separate.
    if ctx.fixture:load_artifact_bundle(folder/'bundle',validation_mode='legacy')
    write(folder/'payload_ready.json',dict(passed=True,all_feature_rows_bit_exact=True,raw_ssd_io=False,payload_bytes=rows*512,
          manifest_sha256=sha(folder/'bundle/manifest.json'),file_sha256=m['files']['reordered_features']['sha256']))
