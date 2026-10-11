"""256-node BFS recombination; all other PA full20 settings inherited."""
from candidates.pa_sage_random_bfs_full20_v1.common import *
from candidates.pa_sage_random_bfs_full20_v1 import common as BASE
HERE=ROOT/'candidates/pa_sage_bfs256_full20_v1'
OUT=ROOT/'results/pa_sage_bfs256_full20_20261011_v1'
MODULE='candidates.pa_sage_bfs256_full20_v1'
PARENT_SHA='afab748ee912c8f75f7b8f501a053facf797ed4e9f5d7e4e95da70e9f4f641af'


def cfg():
    p=BASE.cfg()
    p.update(schema='pa-bfs256-full20-v1',
             root_order_by_arm=dict(p['root_order_by_arm'],digit='BFS_256_node_chunks_recombined_each_epoch_seed0'),
             bfs_chunk_nodes=256,
             bfs_tail_policy='Shuffle complete 256-node chunks; append incomplete canonical chunk; retain every root',
             hot_set_policy='Reuse fixed parent full-epoch profile and hot set; not reprofiled for 256-node batch composition',
             scope='20 complete epochs per arm; unchanged GIDS random, DiGiT BFS chunks of 256 recombined every epoch')
    return p


def verify():
    require(BASE.verify()==PARENT_SHA,'Accepted parent changed')
    m=read(HERE/'manifest.json');require(m['parent_sha256']==PARENT_SHA,'Wrong parent')
    for name,h in m['files'].items():require(sha(HERE/name)==h,'Candidate changed: '+name)
    for name,h in m['inputs'].items():require(sha(name)==h,'Random root input changed: '+name)
    for name,h in m['dependencies'].items():require(sha(ROOT/name)==h,'Evaluation dependency changed: '+name)
    return sha(HERE/'manifest.json')

def epoch_windows(count):
    return [(lo,min(lo+100,count)) for lo in range(0,count,100)]
