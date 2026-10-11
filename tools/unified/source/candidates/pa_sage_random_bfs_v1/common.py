"""Order-only derivative of the accepted PA matched-BFS candidate."""
from candidates.pa_sage_uks_style_v1.common import *
from candidates.pa_sage_uks_style_v1 import common as BASE
HERE=ROOT/'candidates/pa_sage_random_bfs_v1'
OUT=ROOT/'results/pa_sage_random_bfs_20261011_v1'
MODULE='candidates.pa_sage_random_bfs_v1'
PARENT_SHA='6fbbb9258f4d502dac2fe53225f6799b1ae5ed2b761a483a9fd26042f781f20d'


def cfg():
    p=BASE.cfg()
    p.update(schema='pa-random-gids-bfs-digit-v1',
             root_order_by_arm=dict(gids='original_AE_random_seed0',digit='matched_parent_BFS_blocks_seed0'),
             bfs_enabled_by_arm=dict(gids=False,digit=True),
             model_policy='Same PA model and optimizer; GIDS random roots versus DiGiT BFS roots',
             scope='Order-only derivative of accepted PA UKS-style short pair; different short-window examples')
    p.pop('bfs_enabled')
    return p


def verify():
    require(BASE.verify()==PARENT_SHA,'Accepted parent changed')
    m=read(HERE/'manifest.json');require(m['parent_sha256']==PARENT_SHA,'Wrong parent')
    for name,h in m['files'].items():require(sha(HERE/name)==h,'Candidate changed: '+name)
    for name,h in m['random_inputs'].items():require(sha(name)==h,'Random root input changed: '+name)
    return sha(HERE/'manifest.json')
