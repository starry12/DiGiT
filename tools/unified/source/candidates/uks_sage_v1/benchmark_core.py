"""Window timing scheduler. A native backend must supply real training steps.

This helper neither creates a storage backend nor certifies native execution.
"""
import hashlib,math,statistics,struct,time

def measure(roots,step,synchronize,snapshot,begin_region,protocol,clock=time.perf_counter):
    p=protocol;m=p['measurement'];batch=p['batch_size'];warm=m['warmup_batches'];width=m['window_batches'];count=m['windows']
    if len(roots)!=(warm+width*count)*batch:raise ValueError('Incorrect exact root count')
    if m['validation_batches'] or m['test_calls'] or m['epoch_time_claim']:raise ValueError('Performance-only contract required')
    def execute(lo,hi):
        for i in range(lo,hi):step(roots[i*batch:(i+1)*batch])
    synchronize();start=clock();execute(0,warm);synchronize();warm_seconds=clock()-start
    # Retain naturally warmed cache; start a fresh accounting region only.
    begin_region();windows=[]
    for j in range(count):
        lo=warm+j*width;before=snapshot();synchronize();start=clock();execute(lo,lo+width);synchronize();seconds=clock()-start;after=snapshot()
        if seconds<=0 or not math.isfinite(seconds):raise ValueError('Invalid window duration')
        windows.append(dict(index=j,batches=width,root_nodes=width*batch,seconds=seconds,before=before,after=after))
    times=[w['seconds']/w['batches'] for w in windows];median=statistics.median(times);spread=(max(times)-min(times))/median
    digest=hashlib.sha256()
    for root in roots:digest.update(struct.pack('<q',int(root)))
    total=sum(w['seconds'] for w in windows);updates=count*width
    return dict(kind='bounded_training_windows',native_acceptance=False,native_acceptance_note='Timing helper alone; backend source/I/O/model/monitor acceptance required',root_order_sha256=digest.hexdigest(),warmup_batches=warm,warmup_seconds=warm_seconds,measured_batches=updates,measured_seconds=total,seconds_per_batch=total/updates,training_roots_per_second=updates*batch/total,windows=windows,window_relative_spread=spread,window_stability_heuristic_passed=spread<=m['stability_relative_spread'],steady_state_proven=False,epoch_time_seconds=None,accuracy=None)
