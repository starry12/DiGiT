"""Same training protocol, one I/O owner and bounded single-batch lookahead."""
from candidates.pa_sage_512b_latency_v1.common import *
from candidates.pa_sage_512b_latency_v1 import common as reference

HERE=ROOT/'candidates/pa_sage_512b_overlap_v1'
OUT=ROOT/'results/pa_sage_512b_overlap_20260926_v1'
MODULE='candidates.pa_sage_512b_overlap_v1'
BINARY=HERE/'runtime/BAM_Feature_Store/BAM_Feature_Store.so'
VARIANTS=('legacy','serial','overlap')


def setup():
    reference.setup()
    sys.path.insert(0,str(HERE/'runtime'))


def source_files():
    files=[p for p in HERE.iterdir() if p.is_file() and p.name!='manifest.json']
    files.append(BINARY.parent/'__init__.py')
    return sorted(p for p in files if p.exists())


def verify():
    from .native_sources import native_sources
    m=read(HERE/'manifest.json')
    require(reference.verify()==m['reference_sha256'],'Frozen combined reference changed')
    for rel,digest in m['files'].items():require(sha(ROOT/rel)==digest,'Overlap source changed: '+rel)
    require(native_sources()==m['native_sources'],'Native sources changed')
    require(sha(BINARY)==m['binary_sha256'],'Overlap backend changed')
    for name,digest in m['evidence_sha256'].items():require(sha(OUT/name)==digest,'Evidence changed: '+name)
    require(read(HERE/'protocol.json')==read(reference.HERE/'protocol.json'),'Training protocol changed')
    require(sha(OUT/'inputs.json')==sha(reference.OUT/'inputs.json'),'Input binding changed')
    return sha(HERE/'manifest.json')


def validate_completion(r):
    require(r['variant'] in VARIANTS,'Invalid overlap mode')
    reference.validate_completion(dict(r,variant='combined'))
    require(r['changes']==dict(frontier=True,functional=True),'Combined optimizations changed')
    require(r['io_mode']==r['variant'],'Mode metadata differs')
    pipe=r['pipeline']
    if r['variant']!='legacy':
        require(pipe['max_outstanding_reads']<=1,'Multiple requests escaped the I/O owner')
        require(pipe['submitted_reads']==pipe['collected_reads']==pipe['read_calls'],'Read ticket was lost')
        require(r['async_backend']['outstanding']==0,'Native request not collected')
    return True


def full_schedule():
    return ['legacy','serial','overlap','overlap','serial','legacy']
