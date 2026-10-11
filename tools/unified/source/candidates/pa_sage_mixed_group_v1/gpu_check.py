"""Tiny no-SSD production hash, window and paired-resident fixtures."""
import argparse
import random
from .common import *


def run(output):
    source=verify();setup()
    import torch
    import BAM_Feature_Store as native
    require(Path(sys.modules['BAM_Feature_Store.BAM_Feature_Store'].__file__).resolve()==BINARY,'Wrong test backend')
    require(native.PA_PAIR_ASYNC_API==1,'Wrong pair API')
    cpu=[0]*32
    cases=[([0,1],[1,1],cpu,[1,0],0),([1,0],[1,1],cpu,[1,0],0),
           ([0,1],[1,0],cpu,[-1,-1],0),([0,9],[1,1],cpu,[-1,-1],0),
           ([0,1,8,9],[1]*4,cpu,[1,0,3,2],0),
           ([0,1],[1,1],[0,1]+[0]*30,[-1,-1],0),
           ([0,1],[1,1],[1,0]+[0]*30,[-1,-1],0),
           ([2],[1],cpu,[-1],1),([-1],[0],cpu,[-1],1),([32],[0],cpu,[-1],1)]
    for ids,flags,slots,want,errors in cases:
        require(native.pair_plan_probe(ids,flags,slots)==want+[errors],'GPU pairing differs from oracle')
    duplicate=native.pair_plan_probe([0,0],[1,1],cpu)
    require(duplicate[-1]==1,'Duplicate row accepted')
    rng=random.Random(41)
    for _ in range(40):
        ids=rng.sample(range(4096),rng.randrange(1,2048))
        flags=[int((row&7)<=1 and rng.random()<.8) for row in ids]
        slots=[int(rng.random()<.35) for _ in range(4096)]
        eligible={row:i for i,row in enumerate(ids) if flags[i] and not slots[row]}
        want=[eligible.get(row^1,-1) if row in eligible else -1 for row in ids]
        require(native.pair_plan_probe(ids,flags,slots)==want+[0],'Randomized pairing oracle differs')
    for hints in (1,2):
        result=native.pair_resident_probe(hints)
        require(result[7]==0 and result[4:6]==[0,0] and result[6]==2*hints,'Resident pair data or pin lifecycle differs')
        require(result[(hints-1)*2:(hints-1)*2+2]==[0,0],'Window reference not drained')
        if hints==2:require(result[:2]==[1,129],'Lookahead pin was not retained')
    # Existing production FIFO and async stream probes are bound in this binary.
    for mode in range(4):
        result=native.fifo_window_probe(mode)
        require(result[9]==3,'FIFO fixture lost resident capacity')
    torch.cuda.synchronize()
    write(output,dict(passed=True,candidate_sha256=source,binary_sha256=sha(BINARY),
        pairing_cases=len(cases)+41,resident_window_cases=2,fifo_cases=4,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),raw_ssd_access=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
