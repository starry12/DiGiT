"""Same native backend, bounded accumulator and independent acceptance."""
from candidates.pa_sage_512b_pair_v3.common import *
from candidates.pa_sage_512b_pair_v3 import common as parent

HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/pa_sage_512b_pair_20260926_v4'
MODULE='candidates.pa_sage_512b_pair_v4'
BOUND_KEYS=('accumulator_max_batches','accumulator_max_rows','accumulator_bound_scope')


def verify():
    m=read(HERE/'manifest.json')
    require(parent.verify()==m['parent_sha256'],'Frozen parent changed')
    for rel,digest in m['files'].items():require(sha(ROOT/rel)==digest,'Bounded candidate changed: '+rel)
    for name in ('inputs.json','cpu_checks.json','gpu_checks.json'):
        require(sha(OUT/name)==m['evidence_sha256'][name],'Evidence changed: '+name)
    require(sha(OUT/'inputs.json')==sha(parent.OUT/'inputs.json'),'Inputs changed')
    protocol=read(HERE/'protocol.json')
    require({k:v for k,v in protocol.items() if k not in BOUND_KEYS}==read(parent.HERE/'protocol.json'),
            'Unrelated protocol change')
    return sha(HERE/'manifest.json')


def validate_completion(report):
    p=read(HERE/'protocol.json')
    require(report['protocol']==p,'Wrong bounded protocol')
    parent.validate_completion(dict(report,protocol=read(parent.HERE/'protocol.json')))
    pipe=report['pipeline'];rows=max(s['input_nodes'] for s in report['shapes'])
    require(pipe['limit_batches']==p['accumulator_max_batches']==4 and pipe['limit_rows']==p['accumulator_max_rows']==262144,
            'Accumulator limits differ')
    require(pipe['max_merged_batches']<=4 and pipe['max_pending_batches']<=5,'Too many retained batches')
    require(pipe['max_read_rows']<=max(rows,262144),'Feature accumulation exceeded row budget')
    return True


def compare(gids,digit):
    for r in (gids,digit):validate_completion(r)
    reports=[dict(r,protocol=read(parent.HERE/'protocol.json')) for r in (gids,digit)]
    result=parent.compare(*reports)
    result['limits'].append('Both arms cap merged reads at 4 batches / 262144 rows; oversized individual batches remain intact')
    return result
