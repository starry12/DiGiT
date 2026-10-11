"""Frozen incremental sampling, dense graph and overlap; vary mean normalization only."""
from candidates.pa_sage_group_incremental_v1.common import *
from candidates.pa_sage_group_incremental_v1 import common as reference

HERE = ROOT/'candidates/pa_sage_mean_fused_v1'
OUT = ROOT/'results/pa_sage_mean_fused_20260926_v1'
MODULE = 'candidates.pa_sage_mean_fused_v1'
BINARY = reference.BINARY
SAMPLER_BINARY = reference.SAMPLER_BINARY
NORMALIZATION_BINARY = HERE/'runtime/DiGiTMeanNormalizeCUDA.so'
VARIANTS = ('legacy','fused')


def setup():
    reference.setup()
    sys.path.insert(0,str(HERE/'runtime'))


def source_files():
    files=[p for p in HERE.iterdir() if p.is_file() and p.name!='manifest.json']
    files+=list((HERE/'native').glob('*.cu'))
    files += [HERE/'runtime/__init__.py',HERE/'runtime/build_receipt.json']
    return sorted(p for p in files if p.exists())


def environment_files():
    import dgl.ops.spmm
    files=reference.environment_files()
    path=Path(dgl.ops.spmm.__file__).resolve()
    files[str(path)]=sha(path)
    return files


def verify():
    m=read(HERE/'manifest.json')
    require(reference.verify()==m['reference_sha256'],'Incremental reference changed')
    for rel,digest in m['files'].items():require(sha(ROOT/rel)==digest,'Mean candidate changed: '+rel)
    for path,key in ((BINARY,'binary_sha256'),(SAMPLER_BINARY,'sampler_binary_sha256'),(NORMALIZATION_BINARY,'normalization_binary_sha256')):
        require(sha(path)==m[key],'Bound binary changed: '+key)
    for name,digest in m['evidence_sha256'].items():require(sha(OUT/name)==digest,'Evidence changed: '+name)
    require(environment_files()==m['environment_files'],'Bound DGL/Torch code changed')
    require(read(HERE/'protocol.json')==read(reference.HERE/'protocol.json'),'Protocol changed')
    require(sha(OUT/'inputs.json')==sha(reference.OUT/'inputs.json'),'Input binding changed')
    return sha(HERE/'manifest.json')


def as_reference(r):
    return dict(r,variant='incremental',model_implementation='last_dense_cuda_graph')


def validate_completion(r):
    require(r['variant'] in VARIANTS and r['mean_mode']==r['variant'],'Wrong mean mode')
    require(r['group_mode']=='incremental','Incremental sampler changed')
    require(r['mean_normalize_api']==1 and r['normalization_binary_sha256']==sha(NORMALIZATION_BINARY),'Wrong normalization API/binary')
    require(r['model_implementation']==('last_dense_cuda_graph' if r['variant']=='legacy' else 'last_dense_cuda_graph_fused_mean'),'Wrong model implementation')
    reference.validate_completion(as_reference(r))
    return True


def full_schedule():
    return ['legacy','fused','fused','legacy']
