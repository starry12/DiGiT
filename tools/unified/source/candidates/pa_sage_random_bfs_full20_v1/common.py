"""Order-only derivative of the accepted PA matched-BFS candidate."""
from candidates.pa_sage_random_bfs_v1.common import *
from candidates.pa_sage_random_bfs_v1 import common as BASE
HERE=ROOT/'candidates/pa_sage_random_bfs_full20_v1'
OUT=ROOT/'results/pa_sage_random_bfs_full20_20261011_v1'
MODULE='candidates.pa_sage_random_bfs_full20_v1'
PARENT_SHA='c52784a91357a4aa45216dbc88786cec1365f1adca9ddd3adc2c8484460b5f84'


def cfg():
    p=BASE.cfg()
    p.update(schema='pa-random-bfs-full20-v1',epochs=20,warmup_batches=0,measured_batches=1179,
             scope='20 complete training epochs per arm; GIDS random and DiGiT BFS, same full training set',
             evaluation='Separate process after each arm training: validation on all 20 checkpoints, test on epoch20 only',
             checkpoint_selection='fixed epoch20; no validation-based selection')
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
