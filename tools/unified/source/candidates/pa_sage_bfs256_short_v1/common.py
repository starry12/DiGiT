from candidates.pa_sage_random_bfs_v1.common import *
from candidates.pa_sage_random_bfs_v1 import common as BASE
from candidates.pa_sage_bfs256_full20_v1 import common as FULL
HERE=ROOT/'candidates/pa_sage_bfs256_short_v1'
OUT=ROOT/'results/pa_sage_bfs256_short_20261011_v1'
MODULE='candidates.pa_sage_bfs256_short_v1'
FULL_SHA='f6b4083e803299d0fcf822bd319e466e492cc45041ba4acc8783c5c8c5d08627'

def cfg():
    p=BASE.cfg();full=FULL.cfg()
    for key in ('root_order_by_arm','bfs_chunk_nodes','bfs_tail_policy','hot_set_policy'):
        p[key]=full[key]
    p.update(schema='pa-bfs256-short-v1',scope='20 warmup plus 300 measured batches; GIDS random and DiGiT BFS256; fixed parent hot set')
    return p

def verify():
    require(FULL.verify()==FULL_SHA,'Accepted BFS256 parent changed')
    for name,h in read(HERE/'manifest.json')['files'].items():
        require(sha(HERE/name)==h,'BFS256 short source changed: '+name)
    return sha(HERE/'manifest.json')
