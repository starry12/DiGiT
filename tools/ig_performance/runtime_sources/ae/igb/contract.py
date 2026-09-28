"""CPU-only schedule, bounded batch iterators and statistics contract."""
import math
import numpy as np
TRAIN=136278514;VALID=45426171;TEST=45426173;BATCH=1024

def batch_slices(count,size=BATCH,boundary=False):
    if count<=0 or size<=0:raise ValueError('Non-positive batch geometry')
    if boundary:
        yield 0,min(size,count)
        last=((count-1)//size)*size
        if last:yield last,count
    else:
        for start in range(0,count,size):yield start,min(start+size,count)

def train_order(seed,epoch,count=TRAIN):
    return np.random.default_rng(np.random.SeedSequence([seed,epoch])).permutation(count)

def epoch_seed(seed,epoch):
    value=seed+100003*epoch
    if not 0<=value<2**31:raise ValueError('Epoch RNG seed out of range')
    return value

def schedule():
    return [dict(seed=s,repeat=r,arm=a) for s in (0,1,2) for r in (0,1)
            for a in (('gids','full') if (s+r)%2==0 else ('full','gids'))]

def summarize_pairs(pairs):
    expected={(s,r) for s in (0,1,2) for r in (0,1)}
    if len(pairs)!=6 or {(x['seed'],x['repeat']) for x in pairs}!=expected:raise ValueError('Six matched pairs required')
    out={}
    for key in ('online_reduction','training_reduction','quality_difference'):
        values=[sum(x[key] for x in pairs if x['seed']==s)/2 for s in (0,1,2)]
        mean=sum(values)/3;sd=math.sqrt(sum((x-mean)**2 for x in values)/2);margin=4.302652729911275*sd/math.sqrt(3)
        out[key]=dict(seed_means=values,mean=mean,ci95=[mean-margin,mean+margin],df=2)
    out['quality_passed']=out['quality_difference']['mean']>=-.005
    out['performance_passed']=all(x>0 for k in ('online_reduction','training_reduction') for x in out[k]['seed_means'])
    return out
