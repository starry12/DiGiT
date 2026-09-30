"""Build only the independent group selection module; no GPU or SSD initialization."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE / 'native/digit_group_incremental_cuda.cu'
PARENT_SOURCE = ROOT / 'candidates/pa_sage_bidir_native_v2/native/digit_sampler_cuda.cu'
BINARY = HERE / 'runtime/DiGiTGroupIncrementalCUDA.so'
PYTHON = '/home/embed/miniconda3/envs/gids/bin/python'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preserved_kernels():
    old, new = PARENT_SOURCE.read_text(), SOURCE.read_text()
    group_start = 'template <typename IdT>\n__global__ void group_sample_kernel'
    eid_start = 'template <typename IdT>\n__global__ void resolve_eids_kernel'
    old_group = old[old.index(group_start):old.index(eid_start)]
    new_group = new[new.index(group_start):new.index('template <typename IdT>\n__global__ void group_sample_incremental_kernel')]
    old_eid = old[old.index(eid_start):old.index('void check_cuda_launch')].strip()
    new_eid = new[new.index(eid_start):new.index('void check_cuda_launch')].strip()
    if old_group != new_group or old_eid != new_eid:
        raise RuntimeError('Original group sampling or legacy EID kernel changed')
    return dict(group_kernel_sha256=hashlib.sha256(old_group.encode()).hexdigest(),
                legacy_eid_kernel_sha256=hashlib.sha256(old_eid.encode()).hexdigest(),
                group_kernel_byte_identical=True, legacy_eid_kernel_byte_identical=True)


def main():
    if BINARY.exists():
        raise RuntimeError('Preserve existing binary; use a new candidate for rebuilding')
    BINARY.parent.mkdir(parents=True, exist_ok=True)
    partial = BINARY.with_suffix('.partial.so')
    if partial.exists():
        raise RuntimeError('Preserve partial build evidence before rebuilding')
    kernels = preserved_kernels()
    files = (SOURCE, PARENT_SOURCE, Path(__file__), BINARY.parent / '__init__.py')
    sources = {str(path.relative_to(ROOT)): sha(path) for path in files}
    includes = shlex.split(subprocess.check_output([PYTHON, '-B', '-m', 'pybind11', '--includes'], text=True))
    command = ['/usr/local/cuda-12.4/bin/nvcc', '-std=c++14', '-O3', '-arch=sm_89',
               '--default-stream', 'per-thread', '-shared', '-Xcompiler', '-fPIC',
               '-Xptxas=-v'] + includes + [str(SOURCE), '-o', str(partial)]
    started = time.time()
    print(json.dumps(command), flush=True)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with (BINARY.parent / 'build.log').open('x') as log:
        subprocess.run(['nice', '-n', '10'] + command, check=True, env=env,
                       stdout=log, stderr=subprocess.STDOUT)
    if sources != {str(path.relative_to(ROOT)): sha(path) for path in files}:
        raise RuntimeError('Source changed during EID compilation')
    partial.rename(BINARY)
    receipt = dict(passed=True, source_sha256=sources, binary_sha256=sha(BINARY),
                   preserved_kernels=kernels, command=command, started_unix=started,
                   finished_unix=time.time(), gpu_or_ssd_access=False,
                   incremental_fanout_range=[1, 128], no_graph_sized_allocation=True)
    (BINARY.parent / 'build_receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print('Isolated group selection build complete', flush=True)


if __name__ == '__main__':
    main()
