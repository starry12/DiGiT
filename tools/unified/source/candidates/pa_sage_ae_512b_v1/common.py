"""AE workload with an isolated 512-byte cache/read comparison."""
import json, os, sys, time
from pathlib import Path
from ae.common import sha, write, require
from ae.pa_sage.common import source_config
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PARENT=ROOT/'candidates/pa_sage_bidir_native_v2'
MIXED=ROOT/'candidates/pa_sage_mixed_group_v1'
DEPLOYED=Path('/srv/digit-ae/releases/digit_ae_20260923_v3')
P=HERE/'protocol.json'
OUT=ROOT/'results/pa_sage_ae_512b_20261010_v1'
PY='/home/embed/miniconda3/envs/gids/bin/python'
MODULE='candidates.pa_sage_ae_512b_v1'
GPU='2'
GPU_UUID='GPU-927ce617-743a-4bfe-6a60-8a8311cfc703'
def read(path):return json.loads(Path(path).read_text())
def cfg():
    p=read(P);ae=read(PARENT/'protocol.json')
    require(p['epochs']==1 and p['evaluation']=='disabled','One performance epoch only')
    for k in ('seed','fanouts','batch_size','optimizer','hidden','classes','layers','dropout',
              'graph','gpu_cache_bytes','cpu_cache_rows','base_layout','data','metadata_mode','bfs_enabled'):
        require(p[k]==ae[k],'AE workload setting changed: '+k)
    require(p['page_bytes']==512 and not p['window_buffer'] and not p['accumulator'],'Wrong read geometry/scheduling')
    require(p['ablation_order']==['gids','digit_full'],'Wrong arms')
    return p
def backend_binary(arm=None):return MIXED/'runtime/BAM_Feature_Store/BAM_Feature_Store.so'
def setup(arm=None):
    from candidates.pa_sage_bidir_native_v2.common import setup as parent
    parent()
    sys.path[:0]=[str(MIXED/'runtime'),str(ROOT/'candidates/pa_sage_512b_pair_v3/runtime')]
def progress(output,stage,**kw):
    v=dict(stage=stage,pid=os.getpid(),time_unix=time.time(),**kw);write(Path(output)/'progress.json',v);print(json.dumps(v),flush=True)
def verify():
    m=read(HERE/'manifest.json')
    from candidates.pa_sage_bidir_native_v2.common import verify as parent
    from candidates.pa_sage_mixed_group_v1.common import verify as mixed
    require(parent()==m['ae_parent_sha256'] and mixed()==m['mixed_backend_candidate_sha256'],'Frozen dependency changed')
    require(sha(DEPLOYED/'candidates/pa_sage_bidir_native_v2/protocol.json')==m['deployed_protocol_sha256'], 'Installed AE protocol changed')
    for rel,h in m['files'].items():require(sha(ROOT/rel)==h,'Source changed: '+rel)
    cfg();return sha(HERE/'manifest.json')
