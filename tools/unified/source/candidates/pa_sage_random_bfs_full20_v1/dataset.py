import numpy as np
from .common import *
from candidates.pa_sage_random_bfs_v1.dataset import roots,graph_sampler
from candidates.cache_epoch_v1.core import digest_array

def check_report_roots(report,arm):
    if report['smoke']:
        from candidates.pa_sage_random_bfs_v1.dataset import check_report_roots as check
        return check(report,arm)
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
