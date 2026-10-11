"""Rebuild g2 metadata, same-random-order hot profile, traces and filesystem payload."""
import argparse, copy, gc, hashlib, json, math, os, shutil, subprocess, sys, time
from pathlib import Path
from types import SimpleNamespace
from ae.pa_sage.common import *

setup_imports()
import numpy as np

def group(ptr,idx,order,hot,priority,g=2,ratio=.2,sink=None,notify=None):
    n=len(ptr)-1;used=np.zeros(n,dtype=bool);covered=np.zeros(len(idx),dtype=bool)
    counts=np.zeros(n,dtype=np.int32);rng=np.random.default_rng(0);primary=replica=0
    cap=int(n*ratio)//g;last=time.time()
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

def sources(data, report):
    data.mkdir(parents=True, exist_ok=True)
    p=cfg(); src=source_config(); bindings={}
    # Actual complete hashes, not inherited acceptance flags.
    files=[('features',src['source_features']['path'],src['source_features']['sha256']),
           ('edges',src['source_contract']['original_edges']['path'],src['source_contract']['original_edges']['sha256']),
           ('labels',src['label_identity']['path'],src['label_identity']['sha256'])]
    for path in csc_paths():files.append((path.stem,str(path),src['source_contract']['csc'][path.name]['sha256']))
    for name,path,expected in files:
        progress(report,'sources',checking=name)
        actual=sha(path);require(actual==expected,'Source hash mismatch: '+name)
        st=Path(path).stat();bindings[name]=dict(path=path,sha256=actual,bytes=st.st_size,mtime_ns=st.st_mtime_ns)
    all_splits={name:splits(name) for name in ('train','valid','test')}
    joined=np.concatenate(list(all_splits.values()));require(len(np.unique(joined))==len(joined),'Overlapping official splits')
    orders={}
    for seed in p['reconstruction_choices']['seeds']:
        for epoch in range(p['paper_specified']['epochs']):
            roots=np.random.default_rng(np.random.SeedSequence([seed,epoch])).permutation(all_splits['train'])
            orders['s%d_e%d'%(seed,epoch)]=hashlib.sha256(roots.tobytes()).hexdigest()
    write(data/'orders.json',dict(bfs_enabled=False,policy='paired_random_permutation',hashes=orders))
    base_rows=ROOT/'data/papers/gids_cpu_rows.npy';rows=array(base_rows)
    require(len(rows)==p['reconstruction_choices']['cpu_cache_rows'] and np.all(np.diff(rows)>0),'Invalid baseline cache rows')
    require(rows.min()>=0 and rows.max()<src['num_nodes'],'Baseline cache row out of range')
    require(np.all(rows.reshape(-1,8)==rows[::8,None]+np.arange(8)) and np.all(rows[::8]%8==0),'Cache pages not complete')
    if not (data/'gids_cpu_rows.npy').exists():shutil.copyfile(base_rows,data/'gids_cpu_rows.npy')
    require(sha(base_rows)==sha(data/'gids_cpu_rows.npy'),'Baseline row copy differs')
    write(data/'sources.json',dict(passed=True,protocol_sha256=sha(PROTOCOL),files=bindings,
         gids_cache_rows_sha256=sha(base_rows),gids_cache_selection='verified current RevPR page-adapted rows; graph-dependent, not fanout-dependent',
         split_counts={k:len(v) for k,v in all_splits.items()},raw_ssd_io=False))


def make_metadata(data, report, bootstrap=False):
    from digit.artifacts import _build_manifest
    folder=data/('bootstrap' if bootstrap else 'final')
    require(not folder.exists(),'Existing incomplete metadata: preserve it and use a new data root')
    folder.mkdir();bundle=folder/'bundle';bundle.mkdir();start=time.time()
    p=cfg();src=source_config();N=src['num_nodes'];g=p['paper_specified']['group_size'];align=p['reconstruction_choices']['io_page_bytes']//512
    require(host()>100*2**30,'Need at least100 GiB available for metadata build')
    ptr,idx=map(array,csc_paths());E=len(idx)
    hot_nodes=np.array([],dtype=np.int64) if bootstrap else array(data/'hot_nodes.npy')
    hot=np.zeros(N,dtype=bool);hot[hot_nodes]=True
    degree=np.diff(ptr);order=np.argsort(-degree,kind='stable');del degree
    # Separate complete passes; each phase writes only its own new files.
    with (folder/'members.bin').open('xb') as mf,(folder/'owners.bin').open('xb') as of:
        def sink(phase,owner,members,offset):
            mf.write(members.astype('<i8',copy=False).tobytes());of.write(np.full(len(members),owner,dtype='<i8').tobytes())
        result=group(ptr,idx,order,hot,np.array([],dtype=np.int64),g=g,ratio=p['paper_specified']['replication_ratio'],sink=sink,
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
    m=mm('group_members',(G,g));m[:]=np.memmap(folder/'members.bin',mode='r',dtype='<i8',shape=(G,g));flush(m)
    owners=mm('group_owner',(G,));owners[:]=np.memmap(folder/'owners.bin',mode='r',dtype='<i8',shape=(G,));flush(owners)
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
    manifest=_build_manifest(arrays,'OGB','papers100M',N,E,128,g,.2,P,4096,
        dict(features=src['source_features'],indptr=read(data/'sources.json')['files']['original_indptr'],indices=read(data/'sources.json')['files']['original_indices']),
        dict(graph_source=src['source_contract'],pa_sage_random_v2=dict(protocol_sha256=sha(PROTOCOL),bootstrap=bootstrap,hot_count=len(hot_nodes),hot_start=hot_start)),
        minimum_transfer_bytes=4096,target_request_bytes=4096)
    for name,desc in manifest['files'].items():
        if name!='reordered_features':desc['sha256']=sha(bundle/desc['path'])
    write(folder/'descriptor.json',manifest)
    write(folder/'metadata_ready.json',dict(passed=True,primary=P,replica=B,groups=G,storage_rows=ROWS,hot_count=len(hot_nodes),hot_start=hot_start,
          protocol_sha256=sha(PROTOCOL),descriptor_sha256=sha(folder/'descriptor.json'),seconds=time.time()-start))
    # These are temporary duplicates of the now checksummed NPY arrays owned by this build.
    (folder/'members.bin').unlink();(folder/'owners.bin').unlink()


def validate_metadata(data,report,bootstrap=False):
    folder=data/('bootstrap' if bootstrap else 'final');bundle=folder/'bundle';m=read(folder/'descriptor.json');ready=read(folder/'metadata_ready.json')
    N=m['dataset']['num_nodes'];g=m['grouping']['group_size'];G=ready['groups'];P=ready['primary'];align=m['io']['alignment_rows'];R=ready['storage_rows']
    arrays={name:array(bundle/d['path']) for name,d in m['files'].items() if name!='reordered_features'}
    for name,d in m['files'].items():
        if name!='reordered_features':require(sha(bundle/d['path'])==d['sha256'],'Metadata hash mismatch '+name)
    ptr,idx=map(array,csc_paths());covered=array(folder/'covered.npy');members=arrays['group_members'];owners=arrays['group_owner']
    rp=arrays['reordered_indptr'];ri=arrays['reordered_indices'];storage=arrays['storage_to_node'];primary=arrays['node_to_primary_row']
    counts=np.bincount(owners,minlength=N);require(np.array_equal(counts,array(folder/'owner_group_counts.npy')),'Group owner counts')
    require(np.array_equal(np.diff(rp[:N+1]),np.diff(ptr)-(g-1)*counts),'Rewritten degrees');require(np.all(rp[N:]==len(ri)),'Supernode adjacency')
    require(int(covered.sum())==G*g and ready['replica']*g<=int(N*.2),'Replica budget')
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


def profile(data,report):
    import torch,dgl
    from digit.artifacts import ArtifactBundle
    from digit.sampler import DiGiTNeighborSampler,DIGIT_STORAGE_ROW
    from digit.gpu_admission import estimate,check_live
    folder=data/'bootstrap';require(read(folder/'validation.json')['passed'],'Bootstrap not validated')
    m=read(folder/'descriptor.json');p=cfg();f=p['paper_specified']['fanouts']
    plan=estimate(m,fanouts=f,cache_mib=1,metadata_mode='gpu_i32')
    # This stage is sampling-only: no GIDS cache/payload/model is created.
    plan['stage']='bootstrap_sampling_only';live=check_live(plan);write(data/'profile_admission.json',live);require(live['passed'],'Sampling-only admission failed')
    import runner
    runner.startup();progress(report,'profile',phase='graph_load');graph_obj=graph()
    arrays={name:array(folder/'bundle'/d['path']) for name,d in m['files'].items() if name!='reordered_features'}
    sampler=DiGiTNeighborSampler(f,ArtifactBundle(folder/'bundle',m,arrays),cuda_mode='required',metadata_mode='gpu_i32',random_seed=0)
    sampler._ensure_cuda_metadata(graph_obj,torch.device('cuda:0'));runner.seed(0)
    order=np.random.default_rng(np.random.SeedSequence([0,0])).permutation(splits('train'));frequency=np.zeros(m['dataset']['num_nodes'],dtype=np.int64)
    for batch,lo in enumerate(range(0,len(order),1024)):
        roots=torch.from_numpy(order[lo:lo+1024].copy()).cuda();inp,out,blocks=sampler.sample_blocks(graph_obj,roots)
        logical=inp.cpu().numpy();rows=blocks[0].srcdata[DIGIT_STORAGE_ROW].cpu().numpy()
        require(np.array_equal(out.cpu().numpy(),order[lo:lo+1024]),'Root order mismatch')
        require(np.array_equal(arrays['storage_to_node'][rows],logical),'Profile storage mapping')
        require(len(np.unique(logical))==len(logical),'Duplicate logical feature input');frequency[logical]+=1
        if batch%25==0:progress(report,'profile',completed_batches=batch+1,total_batches=math.ceil(len(order)/1024))
        del roots,inp,out,blocks
    N=len(frequency);K=p['reconstruction_choices']['cpu_cache_rows'];threshold=int(np.partition(frequency,N-K)[N-K])
    above=np.flatnonzero(frequency>threshold);ties=np.flatnonzero(frequency==threshold)[:K-len(above)];hot=np.sort(np.r_[above,ties]).astype(np.int64)
    require(len(hot)==K,'Hot count');save(data/'logical_frequency.npy',frequency);save(data/'hot_nodes.npy',hot)
    write(data/'hot_ready.json',dict(passed=True,protocol_sha256=sha(PROTOCOL),bootstrap_descriptor_sha256=sha(folder/'descriptor.json'),
          fanouts=f,bfs_enabled=False,batches=batch+1,hot_nodes=K,threshold=threshold,zero_frequency_selected=int((frequency[hot]==0).sum()),
          hot_sha256=sha(data/'hot_nodes.npy'),frequency_sha256=sha(data/'logical_frequency.npy'),raw_ssd_io=False,optimizer_updates=0))


def traces(data,report):
    import runner
    from digit.eval_trace import build_trace
    runner.startup();progress(report,'traces',phase='graph_load');g=graph();g.unpin_memory_();p=cfg();results={}
    for split in ('valid','test'):
        progress(report,'traces',split=split)
        trace=build_trace(g,splits(split),p['paper_specified']['fanouts'],1024,0,data/(split+'_trace'),
                         source=dict(split='official_'+split,protocol_sha256=sha(PROTOCOL),graph_source=source_config()['source_contract']))
        results[split]={'manifest_sha256':trace.manifest_sha256,'examples':trace.manifest['sampling']['num_test_nodes'],'batches':trace.manifest['sampling']['num_batches']}
    write(data/'traces_ready.json',dict(passed=True,protocol_sha256=sha(PROTOCOL),traces=results))


def payload(data,report):
    from digit.artifacts import load_artifact_bundle
    folder=data/'final';require(read(folder/'validation.json')['passed'],'Final metadata not validated');m=read(folder/'descriptor.json')
    mapping=array(folder/'bundle/storage_to_node.npy');features=array(source_config()['source_features']['path']);rows=len(mapping)
    target=folder/'bundle/reordered_features.npy';checkpoint=folder/'payload_checkpoint.json';chunk=131072
    binding=dict(protocol_sha256=sha(PROTOCOL),mapping_sha256=sha(folder/'bundle/storage_to_node.npy'),source_sha256=read(data/'sources.json')['files']['features']['sha256'])
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
    load_artifact_bundle(folder/'bundle',validation_mode='fast')
    write(folder/'payload_ready.json',dict(passed=True,all_feature_rows_bit_exact=True,raw_ssd_io=False,payload_bytes=rows*512,
          manifest_sha256=sha(folder/'bundle/manifest.json'),file_sha256=m['files']['reordered_features']['sha256']))


def complete(data,report):
    from digit.gpu_admission import estimate,check_live
    p=cfg();m=read(data/'final/bundle/manifest.json');plan=estimate(m,fanouts=p['paper_specified']['fanouts'],cache_mib=p['reconstruction_choices']['gpu_cache_bytes']//2**20,metadata_mode='gpu_i32')
    plan['host_required_bytes']+=2*p['reconstruction_choices']['cpu_cache_rows']*512+2*m['dataset']['num_nodes']*8
    plan['components_bytes']['common_degree_counter']=m['dataset']['num_nodes']*4
    plan['required_bytes']=math.ceil(sum(plan['components_bytes'].values())*plan['safety_multiplier'])
    live=check_live(plan);write(data/'admission.json',live)
    names=['sources.json','orders.json','hot_ready.json','hot_nodes.npy','gids_cpu_rows.npy','full_cpu_rows.npy','traces_ready.json','valid_trace/manifest.json','test_trace/manifest.json',
           'final/descriptor.json','final/validation.json','final/bundle/manifest.json','final/payload_ready.json','admission.json']
    write(data/'prepared.json',dict(passed=True,protocol_sha256=sha(PROTOCOL),filesystem_payload_verified=True,bfs_enabled=False,
          bindings={name:sha(data/name) for name in names},GPU_training_admission_passed=live['passed'],raw_ssd_payload_ready=False,
          full_training_ready=False,bulk_source_hashes_recomputed=True))
    progress(report,'complete',filesystem_data_ready=True,GPU_training_admission_passed=live['passed'],raw_ssd_payload_ready=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['sources','bootstrap','profile','final','traces','payload','complete']);parser.add_argument('--data',type=Path,default=DEFAULT_DATA);parser.add_argument('--report',type=Path,default=DEFAULT_REPORT);a=parser.parse_args()
    a.report.mkdir(parents=True,exist_ok=True);a.data.mkdir(parents=True,exist_ok=True)
    if a.stage in ('bootstrap','final'):
        make_metadata(a.data,a.report,a.stage=='bootstrap');validate_metadata(a.data,a.report,a.stage=='bootstrap')
    else:globals()[a.stage](a.data,a.report)

if __name__=='__main__':main()
