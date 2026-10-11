import hashlib,json
from pathlib import Path
from candidates.ig_demand_pair_v1.common import *
HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/ig_unified_20261011_v2'
def verify():
    from candidates.ig_demand_pair_v1.common import verify as parent
    parent()
    m=read(HERE/'manifest.json')
    for rel,h in m['files'].items():require(sha(ROOT/rel)==h,'IG unified source changed: '+rel)
    return sha(HERE/'manifest.json')
def setup():
    os.environ.update(environment())
    sys.path[:0]=[str(HERE/'runtime'),str(ROOT/'ae/igb'),str(ROOT/'ae/igb/runtime')]

# Configuration and actual optimizer agree in this isolated derivative.
from candidates.ig_demand_pair_v1.common import cfg as parent_cfg
def cfg():
    p=parent_cfg();p['optimizer']['kwargs']['weight_decay']=0.0
    return p
