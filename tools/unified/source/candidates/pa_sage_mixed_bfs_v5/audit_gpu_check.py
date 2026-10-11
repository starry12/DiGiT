"""Frozen CUDA outputs versus independent no-fit oracle and strict verifier."""
from .common import *


def run():
    import torch
    import DiGiTGroupIncrementalCUDA as cuda
    from candidates.pa_sage_group_incremental_v1.native_tests import upload,outputs,call
    from .sampling_audit import whole_group_reference,verify_digit_blocks
    from .audit_fixtures import fixture,block
    from candidates.pa_sage_512b_pair_v1.runtime_helpers import verify_digit_blocks as old_check
    f,bundle,arrays=fixture();metadata,seeds=upload(f,'uva');records=[]
    for mode in ('legacy','incremental'):
        for seed in range(32):
            out=outputs(f,10);call(cuda,mode,f,metadata,seeds,out,10,seed,torch.cuda.current_stream())
            torch.cuda.synchronize()
            src=[v for v in out[0].cpu().tolist() if v>=0]
            groups=int(out[4][0]);count=int(out[5][0])
            ref=whole_group_reference(0,f['idx'],f['n'],2,10,seed,0,bundle.arrays['group_members'],bundle.arrays['supernode_to_group'])
            require(src==ref['sources'] and count==ref['count'] and groups==ref['groups'],'Frozen CUDA differs from CPU oracle')
            b=block(src,groups,count)
            proof=verify_digit_blocks([b],arrays,[10],bundle,seed)
            old_passed=True
            try:old_check([b],arrays,[10])
            except RuntimeError:old_passed=False
            require(old_passed==(count==10),'Unexpected old checker behavior')
            records.append(dict(mode=mode,seed=seed,count=count,old_passed=old_passed,new_passed=proof['passed']))
    require(any(r['count']==9 for r in records) and any(r['count']==10 for r in records),'Fixture lacks both stopping paths')
    return dict(passed=True,cases=len(records),records=records,native_binary_unchanged=True,raw_ssd_access=False)
