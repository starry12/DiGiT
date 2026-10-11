"""FIFO/window support with a separately built and checked 512-B runtime."""
from candidates.pa_sage_512b_pair_v1.common import *
from candidates.pa_sage_512b_pair_v1 import common as base
from candidates.pa_sage_512b_pair_v2 import common as parent

HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/pa_sage_512b_pair_20260926_v3'
MODULE='candidates.pa_sage_512b_pair_v3'
BINARY=HERE/'runtime/BAM_Feature_Store/BAM_Feature_Store.so'


def setup():
    base.setup()
    sys.path.insert(0,str(HERE/'runtime'))


def native_sources():
    sources=base.native_sources()
    for rel in list(sources):
        if rel.startswith(str(base.HERE.relative_to(ROOT))+'/native/'):
            del sources[rel]
    sources.update({str(f.relative_to(ROOT)):sha(f) for f in (HERE/'native').rglob('*') if f.is_file()})
    return sources


def verify():
    m=read(HERE/'manifest.json')
    require(parent.verify()==m['parent_sha256'],'Parent changed')
    for rel,digest in m['files'].items():require(sha(ROOT/rel)==digest,'Candidate changed: '+rel)
    require(sha(BINARY)==m['binary_sha256'],'New backend changed')
    for name in ('inputs.json','cpu_checks.json','gpu_checks.json'):
        require(sha(OUT/name)==m['evidence_sha256'][name],'Evidence changed: '+name)
    require(sha(OUT/'inputs.json')==sha(parent.OUT/'inputs.json'),'Inputs differ')
    require(sha(HERE/'protocol.json')==sha(parent.HERE/'protocol.json'),'Protocol differs')
    return sha(HERE/'manifest.json')


def validate_completion(report):
    parent.validate_completion(report)
    require(report['fifo_window_512b_api']==1,'Unverified FIFO/window backend')
    require(report['fifo_window_policy']=='ring ticket with protected/busy victim skipping at 512 B',
            'Missing FIFO window policy declaration')
    return True


def compare(gids,digit):
    for r in (gids,digit):validate_completion(r)
    result=parent.compare(gids,digit)
    result['limits'].append('512-B FIFO uses its insertion ticket then skips protected/busy victims; not strict fixed-victim FIFO')
    return result
