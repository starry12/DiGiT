"""Complete frozen IG root sequence, independent profiling seed, fixed layout."""
import copy
import numpy as np
from candidates.group_cache_migration_v1.common import ROOT, read, write, require, sha, verify
from candidates.group_cache_migration_v1.prepare import check_prepared
from candidates.group_cache_migration_v1.core import profile, select_hot, digest_array, validate_order

def overlay(arrays,manifest):
    receipt=check_prepared('IG');arrays=dict(arrays);manifest=copy.deepcopy(manifest)
    for key,name in [('reordered_indptr','ptr.i64'),('reordered_indices','units.i64')]:
        v=receipt['files'][name];a=np.memmap(v['path'],mode='r',dtype='<i8')
        arrays[key]=a;manifest['files'][key]=dict(v,shape=list(a.shape),dtype='<i8',format='raw')
    manifest.setdefault('metadata',{})['group_reuse']=receipt['validation'];return arrays,manifest

HISTORY=ROOT/'results/group_cache_migration_v1/runs/20261010-184301-745254/IG_pair'

def accepted_hot():
    base=HISTORY/'smoke_digit_full/digit_full'
    receipt=read(HISTORY/'smoke_digit_full_digit_full_receipt.json')
    report=base/'report.json'
    require(receipt['passed'] and sha(report)==receipt['report_sha256'],'Historical smoke report changed')
    r=read(report);profile=read(base/'full_epoch_profile.json')
    require(r['passed'] and r['cache_epoch']==profile and profile['complete'] and
            profile['source_sha256']==verify(),'Unaccepted or changed full epoch profile')
    target=base/'full_epoch_hot.npy'
    require(sha(target)==profile['hot_file_sha256'],'Changed full epoch hot IDs')
    hot=np.load(target,allow_pickle=False)
    require(digest_array(hot)==profile['selection']['hot_sha256'],'Changed hot set payload')
    return hot,profile,r['native']['cache_sha256']

def prepare_hot(graph,sampler,output,mark,smoke):
    hot,profile,cache_sha=accepted_hot()
    sampler.random_seed=0;sampler._cuda_call_counter=0
    write(output/'reused_hot_profile.json',dict(profile=profile,source=str(HISTORY),cache_sha256=cache_sha))
    mark('accepted_hot_set_reused',rows=len(hot))
    return hot,profile
