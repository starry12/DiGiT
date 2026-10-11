"""Bounded CPU preparation, never CUDA or libnvm."""
import argparse,gc,json,mmap,os,resource,signal,shutil,time
from pathlib import Path
import numpy as np
from . import protocol as P,stream
from candidates.ukl_runtime_prepare_v11r1 import data as D
from candidates.ukl_native_sampling_v10r4 import fork_guard as H
from candidates.ukl_native_sampling_v10 import sampling as S
from candidates.ukl_native_sampling_v10r4.worker import progress_callback
from candidates.ukl_real_multi_v9.memory import Arena
from candidates.ukl_training_sequence_v19r3.lifecycle import record_failure,raise_failures

def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(str(tmp),str(path))

def limits():
    cg=next(l[3:] for l in Path('/proc/self/cgroup').read_text().splitlines() if l.startswith('0::'))
    if not cg.startswith('/system.slice/digit-cl-night-v9-'):raise RuntimeError('Scoped night service required')
    path=Path('/sys/fs/cgroup')/cg.lstrip('/')
    values={k:int((path/k).read_text()) for k in ('memory.max','memory.high','memory.swap.max','pids.max')}
    q,period=map(int,(path/'cpu.max').read_text().split());soft,hard=resource.getrlimit(resource.RLIMIT_MEMLOCK)
    io={l.split()[0]:dict(t.split('=') for t in l.split()[1:]) for l in (path/'io.max').read_text().splitlines()}
    values.update(cgroup=cg,cpu_quota=q,cpu_period=period,memlock_soft=soft,memlock_hard=hard,io_max=io)
    if not P._limits_valid(values,P.budget(),P.source_device()[1]):raise RuntimeError('Effective limits differ')
    return values

def read_at(path,offset,size):
    page=offset//4096*4096;length=((offset-page+size+4095)//4096)*4096
    fd=os.open(str(path),os.O_RDONLY|os.O_DIRECT|os.O_NOFOLLOW);buf=mmap.mmap(-1,length)
    try:
        view=memoryview(buf)
        try:got=os.preadv(fd,[view],page)
        finally:view.release()
        if got<offset-page+size:raise IOError('Short verification read')
        return buf[offset-page:offset-page+size]
    finally:buf.close();os.close(fd)

def feature_check(path,rows,mapping=None):
    selected=np.unique(np.r_[np.array([0,rows-1],np.int64),np.random.default_rng(23).integers(0,rows,30)])
    for row in selected:
        logical=int(row) if mapping is None else int(np.frombuffer(read_at(mapping,int(row)*4,4),np.int32)[0])
        actual=np.frombuffer(read_at(path,int(row)*512,512),np.float32)
        if not np.array_equal(actual,D.features([logical])[0]):raise RuntimeError('Feature readback mismatch')
    return dict(passed=True,rows_checked=len(selected),logical_node_consistent=True)

def prepare(stage,source,meta,predecessor,target,check,event,ownership,*,batch_size=1024,window_batches=320,freq_batches=100):
    """Real stage functions also run on small ordinary files in CPU regression."""
    n,rows=meta['nodes'],meta['storage_rows'];files={};checks={}
    arena=graph=native=sampler=None;failure=cleanup_failure=None
    def save(name,array):
        chunks=D.array_chunks(array)
        try:return D.direct_file(target/name,chunks,check,P.APPLICATION_RATE)
        finally:chunks.close()
    def tick(done=0,total=0,stage=0):check();event('native_progress',done=int(done),total=int(total),native_stage=stage)
    try:
        if stage=='mapping':
            event('mapping_begin');check(rows*4+P.GIB)
            mapping=stream.inverse_mapping(source,n,rows,check,lambda part,done,total:event('mapping',part=part,done=done,total=total))
            files['storage_to_node.i32']=save('storage_to_node.i32',mapping)
            del mapping;checks=dict(primary_roundtrip=True,all_rows_mapped=True)
        elif stage=='split':
            event('split_begin');check(12*P.GIB)
            train,gids,freq=D.split(n,batch_size,window_batches,freq_batches)
            for name,array in [('train.i64',train),('gids_roots.i64',gids),('freq_roots.i64',freq),('gids_labels.i64',D.labels(gids))]:
                files[name]=save(name,array)
            checks=dict(train_nodes=len(train),sorted_unique=True,split_seed=0,freq_seed=23)
            del train,gids,freq,array
        elif stage.startswith('features-'):
            mapped=stage=='features-digit';count=rows if mapped else n
            mapping=predecessor['stages']['mapping']['files']['storage_to_node.i32'] if mapped else None
            chunks=stream.mapped_feature_chunks(mapping,n,check) if mapped else stream.natural_feature_chunks(n,check)
            name=stage.replace('features-','')+'_features.f32'
            event('features',rows=0,total_rows=count)
            try:files[name]=D.direct_file(target/name,chunks,check,P.APPLICATION_RATE,lambda size:event('features',rows=size//512,total_rows=count))
            finally:chunks.close()
            checks=feature_check(target/name,count,None if mapping is None else mapping['path'])
        elif stage=='graph':
            algorithms=D.Algorithms(P.BUILD);S._binary(False)
            extent=sum((v['length']+4095)//4096*4096 for v in source.values())
            extra=P.budget('graph')['memory_max']-extent
            arena=Arena(source,admission_check=lambda rem:check(rem+extra),max_bytes=extent)
            ownership.track(arena);H.dontfork_arena(arena)
            arena.load(progress_callback(event,lambda:check(arena.remaining_bytes+extra),P.APPLICATION_RATE))
            H.assert_dontfork(arena);graph=S.Graph(arena,meta);native=S.Native(graph)
            prior=predecessor['stages']['split']['files']
            event('load_train');check(80*P.GIB)
            train=np.empty(n//10,np.int64)
            for lo,chunk in stream.read_chunks(prior['train.i64'],np.int64,check):train[lo:lo+len(chunk)]=chunk
            event('bfs');order,stats=algorithms.bfs(graph.arrays['ptr'],graph.arrays['idx'],train,tick)
            sorted_order=np.sort(order)
            if not np.array_equal(sorted_order,train):raise RuntimeError('BFS changed training nodes')
            del sorted_order
            roots=D.bfs_window(order,batch_size,window_batches)
            for name,array in [('bfs.i64',order),('digit_roots.i64',roots),('digit_labels.i64',D.labels(roots))]:files[name]=save(name,array)
            checks['bfs']=dict(stats,same_training_set=True,train_nodes=len(train),seed=0)
            del array,train,order,roots
            freq=np.concatenate([x for _,x in stream.read_chunks(prior['freq_roots.i64'],np.int64,check)])
            check(10*P.GIB);counts=np.zeros(n,np.uint64);sampler=S.Sampler(native,grouped=True,seed=23);requests=0
            for batch in range(freq_batches):
                check();nodes,layers=sampler.layers(freq[batch*batch_size:(batch+1)*batch_size],batch)
                counts[nodes]+=1;requests+=len(nodes);del nodes,layers
                event('freq_presampling',batch=batch+1,total_batches=freq_batches,force=True)
            files['freq_counts.u64']=save('freq_counts.u64',counts)
            event('freq_topk');hot=algorithms.topk(counts,tick);files['freq_hot.i64']=save('freq_hot.i64',hot)
            checks['freq']=dict(batches=freq_batches,hot_nodes=len(hot),independent=True,seed=23,logical_unique_requests=requests,backend='native CPU')
            del freq,counts,hot;sampler=None
            info=P.metadata()['revpr'];chunks=(v for _,v in stream.read_chunks(info,np.int64,check))
            try:files['gids_hot.i64']=D.direct_file(target/'gids_hot.i64',chunks,check,P.APPLICATION_RATE)
            finally:chunks.close()
        else:raise ValueError('Unknown stage')
    except BaseException as error:failure=record_failure(error)
    finally:
        try:
            sampler=None
            if native is not None:native.close()
            if graph is not None:graph.close()
            native=graph=None;gc.collect()
            if arena is not None:arena.close();ownership.confirm_released(arena)
        except BaseException as error:cleanup_failure=record_failure(error)
    raise_failures(failure,cleanup_failure)
    return files,checks

def execute(out,target):
    digest=P.verify_manifest();source=P.binding();prior=P.require_predecessor();effective=limits()
    before=json.loads((out/'before.json').read_text());lease=before['controller_lease']
    if before['manifest_sha256']!=digest or before['predecessor']!=prior:raise RuntimeError('Admission changed')
    if target.parent!=P.DATA_ROOT or target.resolve()!=target or target.stat().st_dev!=source['indices']['identity'][0] or any(target.iterdir()):raise RuntimeError('Unsafe output directory')
    write(out/'effective_limits.json',effective)
    stopped=[False];signal.signal(signal.SIGTERM,lambda *a:stopped.__setitem__(0,True));signal.signal(signal.SIGINT,lambda *a:stopped.__setitem__(0,True))
    def check(remaining=0):
        if stopped[0] or (out/'STOP').exists():raise RuntimeError('Cooperative stop requested')
        if Path('/proc/%d/stat'%lease['pid']).read_text().rsplit(')',1)[1].split()[19]!=lease['start_ticks']:raise RuntimeError('Controller lease expired')
        available=next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
        if available<remaining+P.RESERVE:raise RuntimeError('Memory reserve: available=%d required=%d'%(available,remaining+P.RESERVE))
        if shutil.disk_usage(target).free<64*P.GIB:raise RuntimeError('Filesystem free space below 64 GiB')
    last=[0.,None]
    def event(stage,force=False,**kw):
        check();now=time.monotonic()
        if not force and stage==last[1] and now-last[0]<5:return
        last[:]=[now,stage];write(out/'status.json',dict(stage=stage,time=time.time(),complete=False,**kw))
    check();ownership=H.OwnershipGuard().enter()
    try:files,checks=prepare(P.STAGE,source,P.metadata()['metadata'],prior,target,check,event,ownership)
    finally:ownership.release()
    if P.binding()!=source or P.verify_manifest()!=digest:raise RuntimeError('Sources changed')
    report=dict(passed=True,stage=P.STAGE,manifest_sha256=digest,files=files,checks=checks,predecessor=prior,
        effective_limits=effective,source_revalidated=True,normal_release=True,ownership=ownership.receipt(),
        maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,gpu_called=False,raw_ssd_access=False)
    if not P.validate_worker_report(report):raise RuntimeError('Preparation receipt rejected')
    write(out/'worker.json',report);write(out/'status.json',dict(complete=True,stage='complete',time=time.time()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=P.STAGES,required=True);a=p.parse_args();P.configure_stage(a.stage)
    if os.environ.get('CL_NIGHT_V9_WORKER')!='1':raise RuntimeError('Use bounded controller')
    out=Path(os.environ['CL_NIGHT_V9_OUTPUT']).resolve(strict=True);target=Path(os.environ['CL_NIGHT_V9_DATA']).resolve(strict=True)
    try:execute(out,target)
    except BaseException as error:
        write(out/'failure.json',dict(passed=False,error=record_failure(error),stage=P.STAGE,time=time.time()))
        raise
if __name__=='__main__':main()
