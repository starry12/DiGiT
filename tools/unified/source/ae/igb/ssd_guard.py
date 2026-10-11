from ig_common import *
from ae.common import device_idle

def require_payload(arm):
    require(arm in ('gids','full'),'Unknown arm')
    state=check_payload('igb_'+arm); plan=read(CFG/'ssd_plan.json')[arm]
    require(state['offset']==plan['offset'] and state['bytes']==plan['bytes'] and state['file_sha256']==plan['file_sha256'],'Payload geometry/hash mismatch')
    return dict(plan=plan,verified_state=state)
