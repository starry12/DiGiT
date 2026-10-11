import numpy as np
from .common import *
from candidates.pa_sage_random_bfs_v1.dataset import roots as parent_roots,graph_sampler
from candidates.cache_epoch_v1.core import digest_array

def recombine(canonical,epoch=0,seed=0,chunk_nodes=256):
    """Keep local BFS sequences, but change their minibatch companions each epoch."""
    require(canonical.ndim==1 and chunk_nodes>0 and epoch>=0,'Invalid BFS chunk request')
    count=len(canonical)//chunk_nodes
    permutation=np.random.default_rng(np.random.SeedSequence([seed,epoch])).permutation(count)
    result=np.empty_like(canonical)
    result[:count*chunk_nodes]=canonical[:count*chunk_nodes].reshape(count,chunk_nodes)[permutation].reshape(-1)
    result[count*chunk_nodes:]=canonical[count*chunk_nodes:]
    return result

def roots(epoch=0,arm='digit'):
    if arm=='gids':return parent_roots(epoch,arm)
    require(arm=='digit','Unknown root arm')
    p=cfg()
    return recombine(np.load(DATA/'canonical.npy'),epoch,p['seed'],p['bfs_chunk_nodes'])

def check_report_roots(report,arm):
    if report['smoke']:
        order=roots(arm=arm)
        require(report['roots_sha256']==digest_array(order[:24*cfg()['batch_size']]),'Wrong smoke roots')
        require(report['full_order_sha256']==digest_array(order),'Wrong full smoke order')
        require(report['root_order_policy']==cfg()['root_order_by_arm'][arm],'Wrong smoke order policy')
        return
    require(report['root_order_policy']==cfg()['root_order_by_arm'][arm],'Wrong order policy')
    for epoch,r in enumerate(report['epochs']):
        require(r['root_sha256']==digest_array(roots(epoch,arm)),'Wrong epoch root order')

def validate_epochs(report):
    p=cfg();n=p['train_count'];bs=p['batch_size'];count=(n+bs-1)//bs
    require(report['passed'] and not report['smoke'] and report['updates']==20*count,'Incomplete full training')
    require(len(report['epochs'])==20 and report['warmup_batches']==0,'Wrong epoch/warmup count')
    for i,e in enumerate(report['epochs'],1):
        require(e['epoch']==i and e['updates']==count and e['examples']==n,'Incomplete epoch')
        require(len(e['shapes'])==count and all(s[1]==bs for s in e['shapes'][:-1]) and e['shapes'][-1][1]==n%bs,'Tail dropped or truncated batch')
        require(e['training']['reconciled'] and e['persistent_state'],'Invalid epoch state/counters')
        require([(w['batch_start'],w['batch_end']) for w in e['windows']]==epoch_windows(count),'Gapped training windows')
    require(report['complete_region']['reconciled'] and report['complete_region']['raw_allocator_cache']['resident_pages_start']==0,'Invalid full I/O region/cold start')
