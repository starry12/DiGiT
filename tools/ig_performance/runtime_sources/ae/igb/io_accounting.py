"""Physical, primary and replay I/O kept separate, including zero-read cache hits."""
NAMES=('enabled','outstanding','submitted_commands','completed_commands','completed_bytes','active_ns','total_latency_ns','max_latency_ns','max_outstanding','replay_commands','replay_bytes')
def reconcile(io,expected_commands=None,expected_bytes=None,block_bytes=512):
    if not isinstance(io,(list,tuple)) or len(io)!=11 or any(type(x) is not int or x<0 or x>=2**64 for x in io):raise ValueError('Invalid 11-field uint64 device I/O vector')
    if block_bytes!=512:raise ValueError('Frozen device requires 512-byte logical blocks')
    for n in (expected_commands,expected_bytes):
        if n is not None and (type(n) is not int or n<0):raise ValueError('Invalid expected primary I/O')
    v=dict(zip(NAMES,io));commands=v['completed_commands']-v['replay_commands'];nbytes=v['completed_bytes']-v['replay_bytes']
    checks=dict(enabled=v['enabled']==1,drained=v['outstanding']==0,submitted_completed=v['submitted_commands']==v['completed_commands'],
        primary_nonnegative=commands>=0 and nbytes>=0,replay_bytes=v['replay_bytes']==v['replay_commands']*block_bytes,replay_bound=0<=v['replay_commands']<=commands,
        primary_aligned=nbytes%4096==0,zero_consistency=(commands==0)==(nbytes==0))
    if expected_commands is not None:checks['expected_primary_commands']=commands==expected_commands
    if expected_bytes is not None:checks['expected_primary_bytes']=nbytes==expected_bytes
    return dict(passed=all(checks.values()),checks=checks,raw=list(io),named=v,primary_commands=commands,primary_bytes=nbytes,expected_commands=expected_commands,expected_bytes=expected_bytes,device_logical_block_bytes=block_bytes)
