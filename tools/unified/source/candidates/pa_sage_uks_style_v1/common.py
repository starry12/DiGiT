"""Independent PA candidate; original AE graph, data and optimizer preserved."""
import json
import os
from pathlib import Path
from ae.common import require,sha,write
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/pa_sage_uks_style_20261011_v1'
DATA=Path('/mnt/n0/digit/pa_sage_uks_style_20261011_v1')
PARENT=ROOT/'candidates/io_accounting_v1'
BINARY=HERE/'runtime/BAM_Feature_Store/BAM_Feature_Store.so'
PY='/home/embed/miniconda3/envs/gids/bin/python'
SCRATCH_BYTES=16*2**20
MODULE='candidates.pa_sage_uks_style_v1'
from candidates.uks_window_diag_v1.common import LOCKS,validate_inherited_locks

def read(path):return json.loads(Path(path).read_text())
def cfg():
    p=read(HERE/'protocol.json');old=read(ROOT/'candidates/pa_sage_ae_512b_v1/protocol.json')
    for key in ('seed','fanouts','batch_size','optimizer','hidden','classes','layers','dropout','graph','gpu_cache_bytes','cpu_cache_rows','base_layout','data'):
        require(p[key]==old[key],'PA parameter changed: '+key)
    require(p['page_bytes']==512 and p['group_size']==2 and p['warmup_batches']==20 and p['measured_batches']==300,'Wrong measurement/geometry')
    require(p['bfs_enabled'] and not p['window_buffer'] and not p['accumulator'],'Wrong scheduling policy')
    return p

def native_sources():
    return {str(p.relative_to(HERE)):sha(p) for p in sorted((HERE/'native').rglob('*')) if p.is_file() and not (p.parent==HERE/'native' and p.name in ('reuse.cpp','rank.cpp'))}

def binary_receipt():
    r=read(HERE/'runtime/build_receipt.json')
    require(r['native_files']==native_sources() and sha(BINARY)==r['binary_sha256'],'Native source/build changed')
    return r

def verify():
    m=read(HERE/'manifest.json');cfg()
    for name,h in m['files'].items():require(sha(HERE/name)==h,'Candidate changed: '+name)
    for name,h in m['dependencies'].items():require(sha(ROOT/name)==h,'Dependency changed: '+name)
    return sha(HERE/'manifest.json')

def heavy_gate():
    require(os.geteuid()==0 and os.environ.get('CUDA_VISIBLE_DEVICES')=='2' and __debug__,'Root and isolated GPU2 required')

def setup():
    from candidates.uks_pa_fast_v1.runtime import setup_sampling_imports
    setup_sampling_imports()

def identity(path):
    s=Path(path).stat()
    return dict(inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)

def uuid(path):
    import sys
    sys.path.insert(0,'/srv/digit-ae/admin/identity_v1')
    import stable_identity
    return stable_identity.filesystem_uuid(str(path))

def bind(path,expected=None):
    path=Path(path).resolve();before=identity(path);value=sha(path)
    require(identity(path)==before and (expected is None or value==expected),'File hash/identity changed: '+str(path))
    fs=uuid(path);require(bool(fs),'Unidentified filesystem')
    return dict(path=str(path),identity=before,sha256=value,filesystem_uuid=fs)

def check_binding(item):
    require(identity(item['path'])==item['identity'] and uuid(item['path'])==item['filesystem_uuid'],'Bound input changed: '+item['path'])

def prepared():
    r=read(DATA/'prepared.json')
    from .provenance import check_prepared_source
    require(r['passed'],'Preparation incomplete')
    check_prepared_source(r['source_sha256'])
    for item in r['files'].values():check_binding(item)
    return r

def window_ranges(count):return [(0,20)]+[(lo,min(lo+100,count)) for lo in range(20,count,100)]
