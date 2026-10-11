"""Independent integer CPU sampling oracle plus bounded native GPU comparisons."""
import importlib.util
import statistics
import subprocess
from .common import *
from .build import SOURCE, PARENT_SOURCE, preserved_kernels

MASK = (1 << 64) - 1


def splitmix(value):
    value = (value + 0x9e3779b97f4a7c15) & MASK
    value = ((value ^ (value >> 30)) * 0xbf58476d1ce4e5b9) & MASK
    value = ((value ^ (value >> 27)) * 0x94d049bb133111eb) & MASK
    return value ^ (value >> 31)


def fixture(degree, group_size, pattern='mixed'):
    n, groups = 8, 3
    members = [(g*3+j) % n for g in range(groups) for j in range(group_size)]
    bases, mapping, primary = [100, 500, 900], [2,0,1], [30+3*i for i in range(n)]
    row = []
    for i in range(degree):
        is_group = pattern == 'groups' or (pattern == 'mixed' and i % 3 != 0)
        row.append(n + (i*7 % groups) if is_group else (i*5 % n))
    rows = [[], row, list(reversed(row)), row[:1], row[:31], row[:32], row[:33], row]
    ptr, idx, optr, oidx = [0], [], [0], []
    for units in rows:
        idx.extend(units); ptr.append(len(idx))
        expanded = []
        for item in units:
            if item < n: expanded.append(item)
            else:
                g = mapping[item-n]; expanded.extend(members[g*group_size:(g+1)*group_size])
        oidx.extend(reversed(expanded));optr.append(len(oidx))
    eids = [2**40 + (len(oidx)-i)*11 for i in range(len(oidx))]
    return dict(n=n,groups=groups,group_size=group_size,ptr=ptr,idx=idx,members=members,
                bases=bases,mapping=mapping,primary=primary,optr=optr,oidx=oidx,eids=eids,
                seeds=[0,1,2,3,4,5,6,7,1,0,7],degree=degree,pattern=pattern)


def oracle(f, fanout, random_seed):
    sources, physical, flags, eids, groups, counts = [], [], [], [], [], []
    for position, owner in enumerate(f['seeds']):
        units = f['idx'][f['ptr'][owner]:f['ptr'][owner+1]]
        selected = set(); remaining = fanout; src=[]; rows=[]; group_flags=[]; ngroups=0
        while remaining > 0 and len(selected) < 128:
            # Explicit lane-major occurrence order: no node-ID deduplication.
            eligible = []
            for lane in range(32):
                for local in range(lane,len(units),32):
                    cost = 1 if units[local] < f['n'] else f['group_size']
                    if local not in selected and cost <= remaining: eligible.append((local,cost))
            weight = sum(cost for _,cost in eligible)
            if not weight: break
            counter = (random_seed ^ ((owner*0xd6e8feb86659fd93)&MASK)
                       ^ ((position*0xa0761d6478bd642f)&MASK) ^ len(selected))
            threshold = splitmix(counter) % weight
            for local,cost in eligible:
                if threshold < cost: break
                threshold -= cost
            selected.add(local); item=units[local]
            if item < f['n']:
                src.append(item);rows.append(f['primary'][item]);group_flags.append(0);remaining-=1
            else:
                g=f['mapping'][item-f['n']]
                src.extend(f['members'][g*f['group_size']:(g+1)*f['group_size']])
                rows.extend(range(f['bases'][g],f['bases'][g]+f['group_size']))
                group_flags.extend([1]*f['group_size']);remaining-=f['group_size'];ngroups+=1
        first={}
        for i in range(f['optr'][owner],f['optr'][owner+1]):first.setdefault(f['oidx'][i],f['eids'][i])
        count=len(src);src += [-1]*(fanout-count);rows += [-1]*(fanout-count);group_flags += [0]*(fanout-count)
        sources.extend(src);physical.extend(rows);flags.extend(group_flags)
        eids.extend([-1 if value<0 else first.get(value,-2) for value in src])
        groups.append(ngroups);counts.append(count)
    return sources,physical,flags,eids,groups,counts


def upload(f, original='uva'):
    import torch
    arrays=[]
    for name in ('ptr','idx','members','bases','mapping','primary'):
        dtype=torch.int64 if name=='ptr' else torch.int32
        arrays.append(torch.tensor(f[name] or [0],dtype=dtype,device='cuda:0'))
    for name in ('optr','oidx','eids'):
        t=torch.tensor(f[name] or [0],dtype=torch.int64)
        arrays.append(t.pin_memory() if original=='uva' else t.cuda())
    seeds=torch.tensor(f['seeds'],dtype=torch.int64,device='cuda:0')
    return arrays,seeds


def outputs(f, fanout):
    import torch
    slots=len(f['seeds'])*fanout
    return [torch.full((len(f['seeds']) if i>=4 else slots,),123 if i==2 else -777,
            dtype=torch.uint8 if i==2 else torch.int64,device='cuda:0') for i in range(6)]


def call(module, mode, f, arrays, seeds, out, fanout, random_seed, stream, selection_only=False):
    pointers=[t.data_ptr() for t in arrays]
    if selection_only:pointers[6:9]=[0,0,0]
    function = module.sample_group_aware_i32_uva64 if mode=='legacy' else module.sample_group_aware_i32_uva64_incremental
    function(*pointers,seeds.data_ptr(),len(f['seeds']),f['n'],f['groups'],f['group_size'],fanout,
             random_seed,*(t.data_ptr() for t in out),stream.cuda_stream)


def correctness(module):
    import torch
    cases=[]; dimensions=[]
    for g in (1,2,3,8):
        for fanout in sorted(set((g,max(g,10),max(g,31),max(g,32),max(g,33),64,127,128))):
            for pattern in ('normal','mixed','groups'):
                dimensions.append((65,g,fanout,pattern))
    for degree in (0,1,7,31,32,33,63,64,257,1024,65539):
        dimensions.append((degree,2,10,'mixed'))
    for index,(degree,g,fanout,pattern) in enumerate(dimensions):
        f=fixture(degree,g,pattern); mode='device' if index%2 else 'uva'
        arrays,seeds=upload(f,mode); stream=torch.cuda.Stream()
        for seed in (0,17,2**64-1):
            expected=oracle(f,fanout,seed)
            for implementation in ('legacy','incremental'):
                out=outputs(f,fanout);stream.wait_stream(torch.cuda.current_stream())
                call(module,implementation,f,arrays,seeds,out,fanout,seed,stream)
                stream.synchronize()
                for actual,truth in zip(out,expected):
                    require(actual.cpu().tolist()==truth,'CPU oracle mismatch: '+str((degree,g,fanout,pattern,seed,implementation)))
            cases.append(dict(degree=degree,group_size=g,fanout=fanout,pattern=pattern,seed=seed,
                              original_metadata=mode,all_six_outputs_exact=True))
    return dict(passed=True,cases=cases,duplicate_occurrences_preserved=True,
        remaining_below_group_cost=True,lane_and_member_order_preserved=True,
        nonmonotonic_eids_above_int32=True,source_rng_independent_cpu_oracle=True)


def boundaries(module):
    import torch
    f=fixture(65,2);arrays,seeds=upload(f);stream=torch.cuda.Stream();probe=[]
    for mode in VARIANTS:
        out=outputs(f,10);stream.wait_stream(torch.cuda.current_stream())
        call(module,mode,f,arrays,seeds,out,10,0,stream,selection_only=True);stream.synchronize()
        require(bool((out[3]==-777).all()),'Selection-only modified EIDs')
        producer=torch.cuda.Stream();consumer=torch.cuda.Stream()
        delayed=[torch.empty_like(t) if t.device.type=='cuda' else t for t in arrays]
        with torch.cuda.stream(producer):
            producer.wait_stream(torch.cuda.current_stream('cuda:0'))
            torch.cuda._sleep(100000)
            for a,b in zip(delayed,arrays):
                if a.device.type=='cuda':a.copy_(b)
        consumer.wait_stream(producer);consumer.wait_stream(stream)
        out=outputs(f,10);consumer.wait_stream(torch.cuda.current_stream())
        call(module,mode,f,delayed,seeds,out,10,17,consumer);consumer.synchronize()
        require(all(a.cpu().tolist()==b for a,b in zip(out,oracle(f,10,17))),'Nondefault producer dependency failed')
        function=module.sample_group_aware_i32_uva64 if mode=='legacy' else module.sample_group_aware_i32_uva64_incremental
        function(*([0]*10),0,8,3,2,10,0,*([0]*7))
        for fanout,g,num_seeds in ((0,2,0),(129,2,0),(10,0,0),(10,11,0),(10,2,-1)):
            try:function(*([0]*10),num_seeds,8,3,g,fanout,0,*([0]*7))
            except ValueError:pass
            else:raise RuntimeError('Invalid sampler dimensions accepted')
        probe.append(dict(mode=mode,selection_only=True,zero_seeds=True,rejected_dimensions=5,nondefault_stream=True))
    return probe


def microbenchmarks(module):
    import torch
    records=[]
    for degree in (1,8,32,128,1024,4096):
        for pattern in ('normal','mixed','groups'):
            f=fixture(degree,2,pattern);f['seeds']=[1,2,7,1]*32
            arrays,seeds=upload(f);stream=torch.cuda.Stream();out={m:outputs(f,10) for m in VARIANTS}
            stream.wait_stream(torch.cuda.current_stream())
            for _ in range(4):
                for mode in VARIANTS:call(module,mode,f,arrays,seeds,out[mode],10,17,stream,selection_only=True)
            stream.synchronize()
            require(all(torch.equal(a,b) for a,b in zip(out['legacy'],out['incremental'])),'Microbenchmark selection differs')
            timings={m:[] for m in VARIANTS}
            for turn in range(4):
                for mode in (VARIANTS if turn%2==0 else tuple(reversed(VARIANTS))):
                    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    begin.record(stream)
                    for _ in range(20):call(module,mode,f,arrays,seeds,out[mode],10,17,stream,selection_only=True)
                    end.record(stream);end.synchronize()
                    timings[mode].append(begin.elapsed_time(end)*1000/20)
            means={m:statistics.median(timings[m]) for m in VARIANTS}
            records.append(dict(degree=degree,pattern=pattern,fanout=10,seeds=len(f['seeds']),
                microseconds=means,samples=timings,incremental_over_legacy=means['incremental']/means['legacy']))
    return dict(cases=records,scope='Warmed synthetic selection-only CUDA event intervals including launch gaps; no EID/DGL/I/O/model',
        performance_evidence=False)


def main():
    require(not (OUT/'native_checks.json').exists(),'Preserve native evidence')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='2','Use physical GPU2')
    fields=subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True).strip().split(',')
    require(fields[0].strip()==GPU_UUID and int(fields[1])<1024,'GPU2 busy/changed')
    setup(); import DiGiTGroupIncrementalCUDA as module
    require(Path(module.__file__).resolve()==SAMPLER_BINARY and module.INCREMENTAL_GROUP_API==1,'Wrong group module')
    tested={str(p.relative_to(ROOT)):sha(p) for p in (SOURCE,PARENT_SOURCE,HERE/'build.py',Path(__file__),HERE/'common.py')}
    result=dict(passed=True,correctness=correctness(module),boundaries=boundaries(module),microbenchmark=microbenchmarks(module),
        preserved_kernels=preserved_kernels(),binary_sha256=sha(SAMPLER_BINARY),tested_sha256=tested,
        raw_ssd_access=False,performance_evidence=False)
    require(all(sha(ROOT/p)==v for p,v in tested.items()),'Test sources changed')
    write(OUT/'native_checks.json',result)
    print('Native group CPU oracle cases:',len(result['correctness']['cases']),flush=True)
    for r in result['microbenchmark']['cases']:print(r,flush=True)


if __name__=='__main__':main()
