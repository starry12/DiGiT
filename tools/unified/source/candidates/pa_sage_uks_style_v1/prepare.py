"""Filesystem-only preparation under controller locks; source/SSD immutable."""
import argparse
import shutil
import subprocess
import time
import numpy as np
from .common import *

def source():
    p=cfg();base=ROOT/p['base_layout']/'final/bundle';m=read(base/'manifest.json');graph=ROOT/p['data'];r=read(graph/'prepared.json')
    require(r['passed'] and r['base_manifest_sha256']==sha(base/'manifest.json'),'Unverified bidirectional base')
    for rel,h in r['preparation_code_sha256'].items():require(sha(ROOT/rel)==h,'Base preparation implementation changed')
    specs={};files={}
    from candidates.group_cache_migration_v1.prepare import npy
    for key,name in [('ptr','original_indptr'),('idx','original_indices'),('gp','reordered_indptr'),('gi','reordered_indices')]:
        specs[key]=npy(graph/(name+'.npy'),r['bindings'][name+'.npy'])
    for key,name in [('members','group_members'),('super','supernode_to_group')]:
        specs[key]=npy(base/m['files'][name]['path'],m['files'][name]['sha256'])
    return dict(format='dense',nodes=p['graph']['nodes'],edges=p['graph']['edges'],groups=m['grouping']['num_groups'],specs=specs)

def plan():
    s=source();return dict(source=s,output=str(DATA),maximum_new_bytes=8*(s['nodes']+s['groups']+1+s['specs']['gi']['count'])+s['nodes']*8+2*2**30,
                           minimum_available_host_bytes=160*2**30,threads=4,raw_ssd_writes=False,feature_layout_changed=False)

def execute():
    p=cfg();v=plan();s=v['source'];execution=verify()
    require(not DATA.exists(),'Preserve previous preparation directory')
    from ae.common import host
    require(host()>=v['minimum_available_host_bytes'],'Preparation host admission failed')
    require(shutil.disk_usage(DATA.parent).free>=v['maximum_new_bytes']+16*2**30,'Insufficient preparation disk')
    DATA.mkdir();OUT.mkdir(parents=True,exist_ok=True);files={}
    def state(stage,**kw):write(OUT/'preparation_status.json',dict(stage=stage,updated_unix=time.time(),**kw))
    def add(path,expected=None):
        state('hash_source',path=str(path));item=bind(path,expected);files[item['path']]=item
    # Complete hashes for every consumed array and feature/label/split source.
    for a in s['specs'].values():add(a['path'],a['sha256'])
    base=ROOT/p['base_layout']/'final/bundle';m=read(base/'manifest.json');g=ROOT/p['data'];ready=read(g/'prepared.json')
    add(g/'original_eids.npy',ready['bindings']['original_eids.npy'])
    for name in ('group_storage_base','node_to_primary_row','storage_to_node'):
        add(base/m['files'][name]['path'],m['files'][name]['sha256'])
    from ae.pa_sage.common import source_config,splits
    sc=source_config()
    for d in (sc['source_features'],sc['label_identity'],sc['splits']['train'],sc['source_contract']['original_edges']):add(d['path'],d['sha256'])
    for path in (base/'manifest.json',g/'prepared.json',ROOT/p['base_layout']/'ssd_ready.json'):add(path)
    from candidates.group_cache_migration_v1.prepare import descriptor
    spec=DATA/'source.spec';descriptor(s,spec);reports=[]
    for mode in ('build','validate'):
        state('reuse_'+mode)
        with (DATA/(mode+'.log')).open('x') as log:
            r=subprocess.run([str(HERE/'bin/reuse'),mode,str(spec),str(DATA/'adjacency'),'4'],stdout=subprocess.PIPE,stderr=log,text=True,check=True)
        reports.append(json.loads(r.stdout));write(DATA/(mode+'.json'),reports[-1])
    require(reports[0]==reports[1] and reports[1]['passed'],'Independent graph validation differs')
    proof=dict(reports[1],expanded_multigraph_exact=True)
    for name in ('ptr.i64','units.i64'):add(DATA/'adjacency'/name)
    state('revpr')
    with (DATA/'revpr.log').open('x') as log:
        r=subprocess.run([str(HERE/'bin/rank'),str(spec),str(DATA/'revpr.f64'),'20','4'],stdout=subprocess.PIPE,stderr=log,text=True,check=True)
    rank_receipt=json.loads(r.stdout);require(rank_receipt['passed'],'Rank failed')
    from candidates.pa_sage_cache_policy_v1.selection import topk
    scores=np.memmap(DATA/'revpr.f64',mode='r',dtype=np.float64)
    hot=topk(scores,p['cpu_cache_rows']);np.save(DATA/'revpr_hot.npy',hot,allow_pickle=False)
    state('bfs');setup()
    from candidates.pa_sage_bfs_main_v1.order import extract,validate_roots
    from digit.bfs_order import build_bfs_train_order
    train=splits('train');require(len(train)==p['train_count'],'Training split count changed')
    edge=np.load(sc['source_contract']['original_edges']['path'],mmap_mode='r')
    induced=extract(edge,train,p['graph']['nodes'])
    canonical,bfs=build_bfs_train_order(induced,train,p['graph']['nodes']);validate_roots(canonical,train)
    np.save(DATA/'canonical.npy',canonical,allow_pickle=False)
    for name in ('revpr.f64','revpr_hot.npy','canonical.npy'):add(DATA/name)
    for item in files.values():check_binding(item)
    require(verify()==execution,'Source changed during preparation')
    write(DATA/'prepared.json',dict(passed=True,source_sha256=execution,files=files,reuse=proof,revpr=rank_receipt,bfs=bfs,
          bfs_graph_relation='training-induced undirected source connectivity equals AE bidirectional graph, ignoring self loops',raw_ssd_writes=False,feature_layout_changed=False))
    state('complete',passed=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');parser.add_argument('--lock-fds');a=parser.parse_args()
    if a.execute:validate_inherited_locks(a.lock_fds);execute()
    else:print(json.dumps(plan(),indent=2))
