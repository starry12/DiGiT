"""CPU-only sampled-layer microbenchmark; never a training speedup claim."""
import gc,json,os,time,statistics
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .host import number_layer,make_block,storage_rows
from candidates.ukl_native_sampling_v10.sampling import stable_unique
from candidates.ukl_sage_compact_v1.sampler import block_from_edges,STORAGE_ROW

def baseline(targets,r,graph):
    import torch
    nodes=stable_unique(np.concatenate((targets,r['src'])))
    block=block_from_edges(targets,r['src'],r['dst'],r['eid'])
    addresses=nodes.astype(np.int64,copy=True)
    _,first=np.unique(r['src'],return_index=True)
    lookup={int(n):i for i,n in enumerate(nodes)}
    for i in first:addresses[lookup[int(r['src'][i])]]=r['rows'][i]
    block.srcdata[STORAGE_ROW]=torch.from_numpy(addresses)
    return block

def candidate(targets,r,graph):
    import torch
    layer=number_layer(targets,r);block=make_block(layer)
    block.srcdata[STORAGE_ROW]=torch.from_numpy(storage_rows(layer,graph,False))
    return block

def main():
    import torch
    torch.set_num_threads(1);os.nice(19)
    cpus=os.sched_getaffinity(0)-{2}
    if cpus:os.sched_setaffinity(0,{min(cpus)})
    rng=np.random.RandomState(9);records=[]
    for count in (1024,8192,32768):
        targets=np.arange(count,dtype=np.int32);edges=count*10
        src=rng.randint(0,count*8,edges).astype(np.int32)
        r=dict(src=src,dst=np.repeat(targets,10),eid=np.arange(edges,dtype=np.int64)+2**34,rows=src.astype(np.int64))
        g=SimpleNamespace(nodes=count*8)
        for fn in (baseline,candidate):fn(targets,r,g)
        times={'baseline':[],'candidate':[]}
        for repeat in range(5):
            for fn in ((baseline,candidate) if repeat%2==0 else (candidate,baseline)):
                gc.collect();start=time.perf_counter();block=fn(targets,r,g);seconds=time.perf_counter()-start
                times[fn.__name__].append(seconds);del block
        med={k:statistics.median(v) for k,v in times.items()}
        records.append(dict(targets=count,edges=edges,seconds=times,median_seconds=med,speedup=med['baseline']/med['candidate']))
    result=dict(scope='CPU sampled-layer postprocessing only',gpu_used=False,ssd_access=False,training_speedup_measured=False,records=records)
    p=Path('results/ukl_postprocess_20261004_v22/host_benchmark.json');p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
