"""Only group weight maintenance changes on the accepted dense-graph pipeline."""
from candidates.pa_sage_dense_graph_v1.common import *
from candidates.pa_sage_dense_graph_v1 import common as reference

HERE = ROOT / 'candidates/pa_sage_group_incremental_v1'
OUT = ROOT / 'results/pa_sage_group_incremental_20260926_v1'
MODULE = 'candidates.pa_sage_group_incremental_v1'
BINARY = reference.BINARY
SAMPLER_BINARY = HERE / 'runtime/DiGiTGroupIncrementalCUDA.so'
VARIANTS = ('legacy', 'incremental')


def setup():
    reference.setup()
    sys.path.insert(0, str(HERE/'runtime'))


def source_files():
    files = [p for p in HERE.iterdir() if p.is_file() and p.name != 'manifest.json']
    files += list((HERE/'native').glob('*.cu'))
    files += [HERE/'runtime/__init__.py', HERE/'runtime/build_receipt.json']
    return sorted(p for p in files if p.exists())


def verify():
    m = read(HERE/'manifest.json')
    require(reference.verify() == m['reference_sha256'], 'Frozen dense-graph reference changed')
    for rel, digest in m['files'].items():
        require(sha(ROOT/rel) == digest, 'Group candidate changed: '+rel)
    require(sha(BINARY) == m['binary_sha256'], 'I/O binary changed')
    require(sha(SAMPLER_BINARY) == m['sampler_binary_sha256'], 'Group binary changed')
    for name, digest in m['evidence_sha256'].items():
        require(sha(OUT/name) == digest, 'Evidence changed: '+name)
    require(read(HERE/'protocol.json') == read(reference.HERE/'protocol.json'), 'Protocol changed')
    require(sha(OUT/'inputs.json') == sha(reference.OUT/'inputs.json'), 'Inputs changed')
    return sha(HERE/'manifest.json')


def validate_completion(r):
    require(r['variant'] in VARIANTS and r['group_mode'] == r['variant'], 'Invalid group mode')
    require(r['incremental_group_api'] == 1 and r['sampler_binary_sha256'] == sha(SAMPLER_BINARY), 'Wrong native group API/binary')
    require(r['dense_mode'] == 'graph' and r['eid_mode'] == 'original' and r['io_mode'] == 'overlap', 'Unrelated optimization changed')
    reference.validate_completion(dict(r,variant='graph'))
    return True


def full_schedule():
    return ['legacy','incremental','incremental','legacy']
