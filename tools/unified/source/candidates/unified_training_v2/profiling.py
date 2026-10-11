from itertools import islice
import math
import numpy as np
from .order import arm_batches
from candidates.group_cache_migration_v1.core import profile,digest_array,select_hot


def window_profile(canonical,nodes,factory,config,event=lambda **kw:None,check=lambda:None):
    policy=config['cache_profile']
    if policy['scope']!='short_window' or policy['batches']!=320:raise ValueError('Short profiling only')
    parts=list(islice(arm_batches(canonical,config,0,'digit'),policy['batches']))
    if len(parts)!=320 or any(len(p)!=config['batch_size'] for p in parts):raise ValueError('Incomplete profile window')
    roots=np.concatenate(parts)
    def notify(**row):row['stage']='short_window_profile';event(**row)
    counts,r=profile(roots,nodes,factory,notify,check,batch_iterator=iter(parts))
    r.update(epochs=0,scope='short_window',full_epoch=False,canonical_sha256=digest_array(canonical),
             policy=dict(policy),canonical_root_count=len(canonical),root_order='DiGiT_BFS256_epoch0_prefix',functional_only=True)
    validate_profile(r,config,len(canonical))
    return counts,r


def validate_profile(r,config,root_count):
    p=config['cache_profile'];short=p['scope']=='short_window'
    batches=p['batches'] if short else math.ceil(root_count/config['batch_size'])
    roots=batches*config['batch_size'] if short else root_count
    if not (r['complete'] and r['epochs']==(0 if short else 1) and r['seed']==p['seed'] and r['batches']==batches and r['roots']==roots and r['feature_reads']==0 and r['optimizer_updates']==0):
        raise ValueError('Incomplete or wrong profiling scope')
    if short and (r.get('policy')!=p or r.get('full_epoch') is not False or r.get('scope')!='short_window'):
        raise ValueError('Wrong short profile provenance')
    return True


def short_hot(counts,revpr,check=lambda:None):
    hot,selection=select_hot(counts,len(revpr),revpr,np.zeros(len(revpr),np.float32),check)
    selection.update(fallback='existing_RevPR_set_then_logical_ID',revpr_scores_used=False,profile_scope='short_window')
    return hot,selection
