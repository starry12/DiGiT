from .common import *

def estimate(arm):
    import copy
    from candidates.pa_sage_ae_512b_v1.admission import estimate as base
    plan=copy.deepcopy(base('digit_full' if arm=='digit' else 'gids'));p=cfg()
    if arm=='digit':
        proof=prepared()['reuse'];c=plan['components_bytes']
        # Converted GPU indices are i32; EIDs and original CSC remain UVA i64.
        actual=(DATA/'adjacency/units.i64').stat().st_size//8*4
        delta=actual-c['reordered_indices'];c['reordered_indices']=actual
        plan['phase_allocations']['metadata']+=delta;plan['required_bytes']+=delta
    plan['phase_allocations']['pipeline']=2*2**30 if arm=='digit' else 0
    plan['required_bytes']+=plan['phase_allocations']['pipeline']
    plan['arm']=arm;plan['protocol_sha256']=sha(HERE/'protocol.json')
    return plan

def admit(arm):
    import torch
    from ae.common import host
    require(host()>=160*2**30,'PA requires 160 GiB host availability')
    p=estimate(arm)
    require(torch.cuda.mem_get_info()[0]>=p['required_bytes'],'Insufficient GPU memory for explicit PA allocation plan')
    return p

def phase(folder,plan,name,remaining):
    from candidates.pa_sage_ae_512b_v1.admission import stage_check
    stage_check(folder,plan,name,remaining)
