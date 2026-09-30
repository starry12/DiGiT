"""GPU pair discovery and DMA scatter checks; no SSD access or cache acceptance."""
from .common import *
def main():
    check_ready()
    from .runtime import prepare_imports
    _,m=prepare_imports()
    import torch,numpy as np
    cases=0
    # Both flags, CPU fallback, missing partner, interleaved rows, and padding.
    for seed in range(12):
        rng=np.random.RandomState(seed);ids=rng.permutation(2048)[:1300].astype(np.int64)
        flags=rng.randint(0,2,len(ids)).astype(np.bool_);cpu=(rng.random_sample(2048)<.25).astype(np.int32)
        eligible={int(row):i for i,row in enumerate(ids) if flags[i] and row%4<2 and not cpu[row]}
        wanted=np.array([eligible.get(int(row)^1,-1) if int(row) in eligible else -1 for row in ids],np.int32)
        ts=[torch.from_numpy(x).cuda() for x in (ids,flags,cpu)]
        cap=1
        while cap<2*len(ids):cap*=2
        table=torch.empty(cap,dtype=torch.int32,device='cuda');partners=torch.empty(len(ids),dtype=torch.int32,device='cuda');errors=torch.zeros(1,dtype=torch.int64,device='cuda')
        m.probe_pair_hash(*[t.data_ptr() for t in ts],len(ids),len(cpu),table.data_ptr(),cap,partners.data_ptr(),errors.data_ptr())
        require(errors.item()==0 and np.array_equal(partners.cpu().numpy(),wanted),'GPU pairing differs from independent eligibility oracle')
        for i,j in enumerate(wanted):
            if j>=0:require(wanted[j]==i and abs(int(ids[j])-int(ids[i]))==1,'Pair not symmetric')
        cases+=1
    # Duplicate eligible rows must fail, rather than silently leaving output stale.
    ids=torch.tensor([0,1,0],dtype=torch.int64,device='cuda');flags=torch.ones(3,dtype=torch.bool,device='cuda');cpu=torch.zeros(4,dtype=torch.int32,device='cuda');errors.zero_()
    m.probe_pair_hash(ids.data_ptr(),flags.data_ptr(),cpu.data_ptr(),3,4,table.data_ptr(),8,partners.data_ptr(),errors.data_ptr())
    require(errors.item()>0,'Duplicate guard missing')
    src=torch.arange(512,dtype=torch.float32,device='cuda');dest=torch.full((4096,),-77.,device='cuda')
    a=dest[256:512];b=dest[2048:2304]
    m.probe_pair_scatter(src.data_ptr(),a.data_ptr(),b.data_ptr())
    expected=torch.full_like(dest,-77.);expected[256:512]=src[:256];expected[2048:2304]=src[256:]
    require(torch.equal(dest,expected),'Scatter corrupted features or neighboring slots')
    write(OUT/'gpu_check.json',dict(passed=True,source_sha256=verify(),random_pairing_cases=cases,duplicate_guard=True,scatter_noncontiguous_slots_bit_exact=True,ssd_access=False,native_cache_acceptance=False))
    print('GPU pairing, duplicate guard, noncontiguous scatter: PASS')
if __name__=='__main__':main()
