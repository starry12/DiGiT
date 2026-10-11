"""CPU preparation worker. Never registers graph mappings with CUDA."""
import argparse
import gc
import json
import mmap
import os
from pathlib import Path
import resource
import signal
import shutil
import time
import numpy as np
from . import protocol as P
from . import data as D
from candidates.ukl_native_sampling_v10r4 import fork_guard as H
from candidates.ukl_native_sampling_v10 import sampling as S
from candidates.ukl_real_multi_v9.memory import Arena
from candidates.ukl_native_sampling_v10r4.worker import progress_callback

def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(str(tmp),str(path))

def limits():
    cg=next(l[3:] for l in Path('/proc/self/cgroup').read_text().splitlines() if l.startswith('0::'))
    path=Path('/sys/fs/cgroup')/cg.lstrip('/')
    values={k:int((path/k).read_text()) for k in ('memory.max','memory.high','memory.swap.max','pids.max')}
    quota,period=map(int,(path/'cpu.max').read_text().split());soft,hard=resource.getrlimit(resource.RLIMIT_MEMLOCK)
    io={l.split()[0]:dict(t.split('=') for t in l.split()[1:]) for l in (path/'io.max').read_text().splitlines()}
    values.update(cgroup=cg,cpu_quota=quota,cpu_period=period,memlock_soft=soft,memlock_hard=hard,io_max=io)
    if not P._limits_valid(values,P.budget(),P.source_device()[1]):raise RuntimeError('effective limits differ')
    return values

def feature_check(path,rows,mapping=None):
    chosen=np.unique(np.concatenate((np.array([0,rows-1],np.int64),np.random.default_rng(23).integers(0,rows,size=30))))
    fd=os.open(str(path),os.O_RDONLY|os.O_DIRECT|os.O_NOFOLLOW);buf=mmap.mmap(-1,4096)
    try:
        for row in chosen:
            offset=int(row)*512;page=offset//4096*4096
            v=memoryview(buf)
            try:got=os.preadv(fd,[v],page)
            finally:v.release()
            end=offset-page+512
            if got<end:raise IOError('feature check short read')
            actual=np.frombuffer(buf[offset-page:end],np.float32)
            logical=int(row) if mapping is None else int(mapping[row])
            if not np.array_equal(actual,D.features([logical])[0]):raise RuntimeError('feature sample mismatch')
    finally:buf.close();os.close(fd)
    return dict(passed=True,rows_checked=len(chosen),logical_node_consistent=True)

def execute(out,target):
    digest=P.verify_manifest();predecessor=P.require_predecessor();source=P.binding();effective=limits()
    write(out/'effective_limits.json',effective)
    if target.parent!=P.DATA_ROOT or target.is_symlink():raise RuntimeError('output directory scope')
    source_device=P.binding()['indices']['identity'][0]
    if target.stat().st_dev!=source_device:raise RuntimeError('output moved off source filesystem')
    if any(target.iterdir()):raise RuntimeError('output directory not empty')
    def stopped():
        if (out/'STOP').exists():raise RuntimeError('controller requested cooperative stop')
    def check(remaining=0):
        stopped()
        available=next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
        if available<remaining+P.RESERVE:raise RuntimeError('remaining allocation/host reserve')
        if shutil.disk_usage(target).free<32*P.GIB:raise RuntimeError('output free space below 32 GiB')
    signal.signal(signal.SIGTERM,lambda *args:(_ for _ in ()).throw(RuntimeError('termination requested')))
    last=[0.0]
    def event(stage,**kw):
        write(out/'status.json',dict(stage=stage,time=time.time(),complete=False,**kw))
    def tick(done=0,total=0,stage=0):
        check()
        if time.monotonic()-last[0]>2:
            event('bfs_or_topk',native_stage=stage,done=int(done),total=int(total));last[0]=time.monotonic()
    def save(name,array):
        return D.direct_file(target/name,D.array_chunks(array),check,P.APPLICATION_WRITE_RATE)
    algorithms=D.Algorithms(P.BUILD)  # Load dependencies before ownership guard.
    S._binary(False)                 # Hash-check accepted CPU native sampler.
    ownership=H.OwnershipGuard().enter();arena=None;graph=None;native=None;mapping=None
    files={};extra={}
    try:
        if P.STAGE=='graph':
            event('load_graph',gib=P.EXTENT/P.GIB)
            arena=Arena(source,admission_check=lambda remaining:check(remaining+P.HEADROOM),max_bytes=P.EXTENT)
            ownership.track(arena)
            H.dontfork_arena(arena)
            arena.load(progress_callback(lambda stage,**v:event(stage,**v),lambda:check(arena.remaining_bytes+P.HEADROOM),P.APPLICATION_READ_RATE))
            H.assert_dontfork(arena)
            graph=S.Graph(arena,P.METADATA);native=S.Native(graph)
            event('split_and_windows');check(8*P.GIB)
            train,gids_roots,freq_roots=D.split(P.N,P.BATCH,P.WINDOW_BATCHES,P.FREQ_BATCHES)
            for name,array in [('train.i64',train),('gids_roots.i64',gids_roots),('freq_roots.i64',freq_roots)]:files[name]=save(name,array)
            del array
            event('bfs');check(80*P.GIB)
            order,stats=algorithms.bfs(graph.arrays['ptr'],graph.arrays['idx'],train,tick)
            event('bfs_training_set_check')
            sorted_order=np.sort(order)
            if not np.array_equal(sorted_order,train):raise RuntimeError('BFS changed training set')
            del sorted_order
            digit_roots=D.bfs_window(order,P.BATCH,P.WINDOW_BATCHES);files['bfs.i64']=save('bfs.i64',order)
            files['digit_roots.i64']=save('digit_roots.i64',digit_roots)
            extra['bfs']=dict(stats,train_nodes=len(train),same_training_set=True,
                              direction='training-induced undirected, multiedge degree',seed=0)
            del train,order
            event('freq_presampling',batch=0,total_batches=P.FREQ_BATCHES);check(10*P.GIB)
            counts=np.zeros(P.N,np.uint64);sampler=S.Sampler(native,grouped=True,seed=23);requests=0
            for batch in range(P.FREQ_BATCHES):
                check();nodes,layers=sampler.layers(freq_roots[batch*P.BATCH:(batch+1)*P.BATCH],batch)
                counts[nodes]+=1;requests+=len(nodes);del nodes,layers
                event('freq_presampling',batch=batch+1,total_batches=P.FREQ_BATCHES)
            files['freq_counts.u64']=save('freq_counts.u64',counts)
            event('freq_topk');hot=algorithms.topk(counts,tick)
            files['freq_hot.i64']=save('freq_hot.i64',hot)
            extra['freq']=dict(batches=P.FREQ_BATCHES,seed=23,hot_nodes=len(hot),logical_unique_requests=requests,
                               independent=True,backend='accepted native CPU shared sampling core',ties='node ID')
            del counts,hot,sampler
            event('inverse_storage_map');check(5*P.GIB)
            mapping=D.inverse_map(graph.arrays,P.N,P.ROWS,check)
            files['storage_to_node.i32']=save('storage_to_node.i32',mapping);del mapping;mapping=None
            files['gids_labels.i64']=save('gids_labels.i64',D.labels(gids_roots))
            files['digit_labels.i64']=save('digit_labels.i64',D.labels(digit_roots))
            del gids_roots,digit_roots,freq_roots
            # Materialize the existing logical RevPR set as the baseline CPU
            # cache list. Hash-check with direct reads, never import v5 runtime.
            event('gids_revpr_hot');hot_arena=None;hot_view=None
            try:
                hot_identity=D.ident(P.REVPR_PATH)
                if P.REVPR_PATH.is_symlink() or hot_identity[2]!=P.N//10*8:raise RuntimeError('RevPR extent')
                spec=dict(path=str(P.REVPR_PATH),offset=0,length=hot_identity[2],identity=hot_identity,sha256=P.REVPR_SHA)
                hot_arena=Arena({'hot':spec},admission_check=lambda remaining:check(remaining+P.GIB),max_bytes=P.GIB)
                ownership.track(hot_arena)
                H.dontfork_arena(hot_arena)
                hot_arena.load(progress_callback(lambda stage,**v:event(stage,**v),check,P.APPLICATION_READ_RATE))
                hot_view=np.frombuffer(hot_arena.mm,np.int64,count=P.N//10)
                for lo in range(0,len(hot_view),1048576):
                    check();part=hot_view[max(0,lo-1):lo+1048576]
                    if np.any(part<0) or np.any(part>=P.N) or np.any(part[1:]<=part[:-1]):raise ValueError('RevPR hot range/order')
                del part
                files['gids_hot.i64']=save('gids_hot.i64',hot_view)
            finally:
                del hot_view
                if hot_arena is not None:
                    hot_arena.close();ownership.confirm_released(hot_arena)
        else:
            rows=P.N if P.STAGE=='features-gids' else P.ROWS
            if P.STAGE=='features-digit':
                info=predecessor['graph_files']['storage_to_node.i32']
                spec=dict(path=info['path'],offset=0,length=info['bytes'],identity=info['identity'],sha256=info['sha256'])
                arena=Arena({'mapping':spec},admission_check=lambda remaining:check(remaining+P.GIB),max_bytes=P.EXTENT)
                ownership.track(arena)
                H.dontfork_arena(arena)
                arena.load(progress_callback(lambda stage,**v:event(stage,**v),lambda:check(arena.remaining_bytes+P.GIB),P.APPLICATION_READ_RATE))
                mapping=np.frombuffer(arena.mm,dtype=np.int32,count=rows)
                mapping.flags.writeable=False
            event(P.STAGE,rows=0,total_rows=rows)
            def chunks():
                for lo in range(0,rows,16384):
                    check();hi=min(rows,lo+16384)
                    ids=np.arange(lo,hi,dtype=np.int64) if mapping is None else mapping[lo:hi]
                    if np.any(ids<0) or np.any(ids>=P.N):raise ValueError('feature logical ID out of range')
                    yield D.features(ids)
            def progress(written):
                if time.monotonic()-last[0]>5:event(P.STAGE,rows=written//512,total_rows=rows);last[0]=time.monotonic()
            name=next(iter(P.expected_files()))
            files[name]=D.direct_file(target/name,chunks(),check,P.APPLICATION_WRITE_RATE,progress)
            extra['feature_checks']=feature_check(target/name,rows,mapping)
            del mapping;mapping=None
    finally:
        if native is not None:native.close()
        if graph is not None:graph.close()
        if mapping is not None:del mapping
        gc.collect()
        if arena is not None:
            arena.close();ownership.confirm_released(arena)
        ownership.release()
    if P.binding()!=source:raise RuntimeError('source identity changed')
    if P.verify_manifest()!=digest:raise RuntimeError('manifest changed')
    report=dict(passed=True,stage=P.STAGE,files=files,effective_limits=effective,
                manifest_sha256=digest,predecessor=predecessor,normal_release=True,
                source_revalidated=True,gpu_called=False,raw_ssd_access=False,
                application_read_rate=P.APPLICATION_READ_RATE,application_write_rate=P.APPLICATION_WRITE_RATE,
                maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,ownership=ownership.receipt(),**extra)
    if not P.validate_worker_report(report):raise RuntimeError('incomplete preparation receipt')
    write(out/'worker.json',report);event('complete');write(out/'status.json',dict(complete=True,stage='complete',time=time.time()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=P.STAGES,required=True);a=p.parse_args()
    if os.environ.get('UKL_V11R1_BOUNDED_WORKER')!='1':raise RuntimeError('bounded controller required')
    P.configure_stage(a.stage)
    execute(Path(os.environ['UKL_V11R1_OUTPUT']).resolve(strict=True),Path(os.environ['UKL_V11R1_DATA']).resolve(strict=True))

if __name__=='__main__':main()
