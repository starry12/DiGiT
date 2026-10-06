"""Small-file provenance and isolation. No device imports or probes here."""
import contextlib
import fcntl
import os
from pathlib import Path
from candidates.pa_sage_cache_policy_v1.common import ROOT, read, sha, require, write_new, digest
HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/uks_sage_epoch_20260925_v1'
LARGE=Path('/mnt/n0/digit/uks_sage_epoch_20260925_v1')
GRID=ROOT/'results/pa_sage_layout_continue_20260925_v4'
PARENT=ROOT/'candidates/io_accounting_v1'
BINARY=HERE/'runtime/BAM_Feature_Store/BAM_Feature_Store.so'
SCRATCH_BYTES=16*2**20
ARMS=('gids','digit')


def require_grid_complete(state):
    expected={'g%d_r%02d'%(g,r) for g in (1,2,4) for r in (0,10,20,40,80)}
    require(state.get('complete') is True and state.get('passed') is True and
            len(state.get('completed',[]))==15 and set(state['completed'])==expected,
            'UKS heavy work is deferred until all 15 original grid points pass')


def heavy_gate():
    require_grid_complete(read(GRID/'status.json'))


@contextlib.contextmanager
def heavy_lock():
    heavy_gate()
    with contextlib.ExitStack() as stack:
        for path in (GRID/'controller.lock',Path('/tmp/digit-pa-bidir-controller.lock'),Path('/tmp/digit-pa-sage-libnvm0.lock')):
            require(path.is_file() and not path.is_symlink(),'Missing experiment lock: '+str(path))
            f=stack.enter_context(path.open('rb'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        heavy_gate();yield


def cpu_gate():
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and
            all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')),
            'CPU checks need hidden CUDA and single-thread numerical libraries')
    state=read(GRID/'status.json');stage=state.get('stage','')
    require(stage.startswith(('layout_','overlay_')) or state.get('complete') is True,
            'Defer CPU checks outside graph preparation; current stage: '+stage)
    return state


def verify():
    m=read(HERE/'manifest.json')
    for name,value in m['files'].items():require(sha(HERE/name)==value,'UKS source changed: '+name)
    for name,value in m['dependencies'].items():require(sha(ROOT/name)==value,'Dependency changed: '+name)
    return sha(HERE/'manifest.json')


def binary_receipt():
    value=read(HERE/'runtime/build_receipt.json')
    require(value['source_sha256']==verify() and value['binary_sha256']==sha(BINARY),'Missing/stale UKS native binary')
    return value
