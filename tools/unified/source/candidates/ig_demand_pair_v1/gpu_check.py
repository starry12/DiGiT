"""Tiny CUDA fixtures. No controller creation, NVMe access, or dataset loading."""
import argparse
import os
from pathlib import Path
import numpy as np

CHECKED_SOURCES=('gpu_check.py','adjacency_tests.py','adjacency_sampler.py','fast_uva.py',
                 'features.py','mapping.py','native/optimized_helpers.cuh',
                 'native/demand_helpers.cuh','native/demand_fixtures.cuh','native/include/page_cache.h')

def run():
    from .common import setup, HERE, sha
    setup()
    import torch, IGPerfNative as native
    torch.set_num_threads(1)
    assert Path(native.__file__).resolve()==HERE/'runtime/IGPerfNative.so'
    assert native.OPTIMIZED_IG_ABI==1
    assert native.DEMAND_PAIR_ABI==1
    demand_cases=0
    rng=np.random.default_rng(83)
    # Missing sibling, CPU sibling, raw flags, duplicates, cross-launch pairs,
    # different physical replicas, hot-region boundary and hash collisions.
    fixtures=[([0],[1],[0]*8,8),([0,1],[1,1],[0]*8,8),
              ([0,1],[1,1],[0,1]+[0]*6,8),([1,0],[1,1],[1]+[0]*7,8),
              ([0,1],[0,0],[0]*8,8),([0,3],[1,1],[0]*8,8),
              ([0,0,1],[1,1,0],[0]*8,8),([6,7],[1,1],[0]*8,7)]
    for n in (1,31,512,17000,33000):
        rows=rng.integers(0,65536,size=n).tolist()
        flags=rng.integers(0,2,size=n).tolist()
        cpu=rng.integers(0,3,size=65536)==0
        fixtures.append((rows,flags,cpu.astype(int).tolist(),60000))
    # Each physical pair straddles the 16,384-row launch boundary.
    rows=list(range(0,32768,2))+list(range(1,32768,2))
    fixtures.append((rows,[1]*len(rows),[0]*32768,32768))
    for rows,flags,cpu,hot in fixtures:
        cold={row for row in rows if row<hot and not cpu[row]}
        expected=[bool(flag and row<hot and (row^1)<hot and not cpu[row] and (row^1) in cold)
                  for row,flag in zip(rows,flags)]
        assert list(native.demand_fixture(rows,flags,cpu,hot))==expected
        demand_cases+=1
    for row in (0,1):
        for sibling_valid in (False,True):
            stats=native.pair_hit_fixture(row,sibling_valid)
            # Actual acquire_page_pair, with no controller attached: returning
            # without an SSD access is mandatory even if the sibling is absent.
            assert stats[2:7]==[0]*5 and stats[12]==7+row
            assert stats[13]==stats[16]==1 and stats[7]==1
            assert stats[14+row]!=0
            if not sibling_valid:assert stats[14+(row^1)]==0
    cache=np.arange(3072,dtype=np.float32).reshape(3,1024)/32
    expected=cache[[0,0,0,2,2,1]]
    for mixed in (False,True):
        got=np.array(native.alias_fixture(mixed),dtype=np.float32).reshape(6,1024)
        np.testing.assert_array_equal(got,expected)
    # Check fused annotation against the original scatter-reduce definition,
    # on a non-default stream and with a mapped anonymous host primary table.
    from .sampler import load
    sampler_native=load()
    rng=np.random.default_rng(19);stream=torch.cuda.Stream()
    cases=0
    for edges in (0,1,79,1024):
        with torch.cuda.stream(stream):
            primary=torch.empty(97,dtype=torch.int32,pin_memory=True)
            primary.copy_(torch.from_numpy(rng.permutation(97).astype(np.int32)))
            ptr=sampler_native.mapped_pointer(primary.data_ptr())['device_pointer']
            logical=torch.tensor(rng.integers(0,97,size=47),dtype=torch.int64,device='cuda')
            src=torch.tensor(rng.integers(0,47,size=edges),dtype=torch.int64,device='cuda')
            erows=torch.tensor(rng.integers(0,160,size=edges),dtype=torch.int64,device='cuda')
            eflags=torch.tensor(rng.integers(0,2,size=edges).astype(bool),device='cuda')
            rows=torch.empty_like(logical);flags=torch.empty_like(logical,dtype=torch.bool)
            errors=torch.zeros(1,dtype=torch.int64,device='cuda')
            native.annotate(logical.data_ptr(),ptr,src.data_ptr(),erows.data_ptr(),eflags.data_ptr(),
                            len(logical),edges,97,160,rows.data_ptr(),flags.data_ptr(),errors.data_ptr(),stream.cuda_stream)
            preferred=torch.full_like(logical,torch.iinfo(torch.int64).max)
            preferred.scatter_reduce_(0,src[eflags],erows[eflags],reduce='amin',include_self=True)
            grouped=preferred!=torch.iinfo(torch.int64).max
            reference=torch.where(grouped,preferred,primary.to('cuda',dtype=torch.int64)[logical])
        stream.synchronize()
        assert errors.item()==0
        assert torch.equal(rows,reference) and torch.equal(flags,grouped)
        cases+=1
    # Invalid logical and edge IDs must be reported without an out-of-bounds read.
    with torch.cuda.stream(stream):
        logical[0]=-1;src[0]=999
        eflags[0]=True;errors.zero_()
        native.annotate(logical.data_ptr(),ptr,src.data_ptr(),erows.data_ptr(),eflags.data_ptr(),
                        len(logical),len(src),97,160,rows.data_ptr(),flags.data_ptr(),errors.data_ptr(),stream.cuda_stream)
    stream.synchronize();assert errors.item()>=2
    from .adjacency_tests import gpu_fixture
    adjacency=gpu_fixture()
    return dict(passed=True,raw_ssd_access=False,large_graph_loaded=False,
                native_sha256=sha(HERE/'runtime/IGPerfNative.so'),
                sources={name:sha(HERE/name) for name in CHECKED_SOURCES},alias_reader_modes=2,
                annotation_parity_cases=cases,invalid_annotation_rejected=True,
                demand_filter_cases=demand_cases,locked_requested_hit_cases=4,
                raw_adjacency_fixture=adjacency,visible_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                torch_peak_allocated_bytes=torch.cuda.max_memory_allocated())


def require_accepted():
    from .common import HERE,OUT,read,require,sha
    r=read(OUT/'gpu_check.json')
    require(r['passed'] and not r['raw_ssd_access'] and r['native_sha256']==sha(HERE/'runtime/IGPerfNative.so'),
            'Missing matching optimized CUDA fixtures')
    require(r['sources']=={name:sha(HERE/name) for name in CHECKED_SOURCES},'CUDA fixture source changed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from ae.common import write
    result=run();write(a.output,result);print(result)
