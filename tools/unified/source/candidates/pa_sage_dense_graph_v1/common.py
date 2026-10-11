"""Partial final-layer capture on the accepted overlap sampler/I/O baseline."""
from candidates.pa_sage_512b_overlap_v1.common import *
from candidates.pa_sage_512b_overlap_v1 import common as reference

HERE = ROOT / 'candidates/pa_sage_dense_graph_v1'
OUT = ROOT / 'results/pa_sage_dense_graph_20260926_v1'
MODULE = 'candidates.pa_sage_dense_graph_v1'
BINARY = reference.BINARY
VARIANTS = ('eager', 'graph')


def source_files():
    return sorted(p for p in HERE.iterdir() if p.is_file() and p.name != 'manifest.json')


def environment_files():
    import torch.cuda.graphs
    return {str(Path(torch.cuda.graphs.__file__).resolve()): sha(torch.cuda.graphs.__file__)}


def verify():
    m = read(HERE / 'manifest.json')
    require(reference.verify() == m['reference_sha256'], 'Frozen overlap reference changed')
    for rel, digest in m['files'].items():
        require(sha(ROOT / rel) == digest, 'Dense capture source changed: ' + rel)
    require(sha(BINARY) == m['binary_sha256'], 'Frozen async I/O binary changed')
    require(environment_files() == m['environment_files'], 'Torch CUDA Graph implementation changed')
    for name, digest in m['evidence_sha256'].items():
        require(sha(OUT / name) == digest, 'Evidence changed: ' + name)
    require(read(HERE / 'protocol.json') == read(reference.HERE / 'protocol.json'), 'Training protocol changed')
    require(sha(OUT / 'inputs.json') == sha(reference.OUT / 'inputs.json'), 'Input binding changed')
    return sha(HERE / 'manifest.json')


def validate_completion(r):
    require(r['variant'] in VARIANTS and r['dense_mode'] == r['variant'], 'Invalid dense mode')
    require(r['io_mode'] == 'overlap' and r['eid_mode'] == 'original', 'Sampler or I/O mode changed')
    dense = r['dense_graph']
    full_batches = sum(s['output_nodes'] == 1024 for s in r['shapes'])
    require(dense['capture_batch_size'] == 1024 and dense['mode'] == r['variant'], 'Dense shape/mode changed')
    require(dense['graph_forwards'] == (full_batches if r['variant'] == 'graph' else 0), 'Missing graph replay')
    require(dense['eager_forwards'] == r['updates'] - dense['graph_forwards'], 'Lost eager/tail call')
    require(dense['prepared'] and dense['parameter_identity_preserved'] and dense['rng_preserved'], 'Capture changed state')
    require(dense['capture_count'] == (1 if r['variant'] == 'graph' else 0), 'Unexpected recapture')
    require(dense['optimizer_identity_preserved'] and dense['initial_parameters_unchanged'], 'Capture mutated parameters/optimizer')
    require(dense['retained_allocation_bytes'] <= 256 * 2**20, 'Graph pool exceeded planned allowance')
    reference.validate_completion(dict(r, variant='overlap', model_implementation='functional_mean_sage'))
    require(r['model_implementation'] == ('last_dense_cuda_graph' if r['variant'] == 'graph' else 'functional_mean_sage'), 'Wrong model')
    return True


def full_schedule():
    return ['eager', 'graph', 'graph', 'eager']
