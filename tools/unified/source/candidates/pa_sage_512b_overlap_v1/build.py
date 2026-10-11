"""One isolated nvcc process; does not open a GPU or raw storage device."""
import json
import os
import shlex
import subprocess
import time
from .native_sources import BINARY, HERE, PARENT, PY, ROOT, native_sources, object_sha, sha


def main():
    sources = native_sources()
    if BINARY.exists():
        raise RuntimeError('Preserve the existing frozen binary; use a new candidate for rebuilding')
    BINARY.parent.mkdir(parents=True, exist_ok=True)
    partial = BINARY.with_suffix('.partial.so')
    if partial.exists():
        raise RuntimeError('Preserve the existing partial build')
    includes = shlex.split(subprocess.check_output([PY, '-B', '-m', 'pybind11', '--includes'], text=True))
    cmd = ['/usr/local/cuda-12.4/bin/nvcc', '-std=c++11', '-O3', '-arch=sm_89',
           '--default-stream', 'per-thread', '-shared', '-Xcompiler', '-fPIC'] + includes + [
           '-I' + str(HERE / 'native/gids_module/include'),
           '-I' + str(HERE / 'native/bam/include'),
           '-I' + str(PARENT / 'native/bam/include'),
           '-I' + str(PARENT / 'native/bam/include/freestanding/include'),
           str(HERE / 'native/gids_module/gids_nvme.cu'), str(ROOT / 'bam/build/lib/libnvm.so'),
           '-Xlinker', '-rpath', '-Xlinker', str(ROOT / 'bam/build/lib'),
           '-Xcompiler', '-pthread', '-o', str(partial)]
    started = time.time()
    print(json.dumps(cmd), flush=True)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with (HERE / 'runtime/build.log').open('x') as stream:
        subprocess.run(['nice', '-n', '10'] + cmd, check=True, stdout=stream, stderr=subprocess.STDOUT, env=env)
    if sources != native_sources():
        raise RuntimeError('Native sources changed during compilation')
    partial.rename(BINARY)
    (BINARY.parent / '__init__.py').write_text('from .BAM_Feature_Store import *\n')
    receipt = dict(native_source_sha256=object_sha(sources), native_sources=sources,
                   binary_sha256=sha(BINARY), command=cmd, started_unix=started,
                   finished_unix=time.time(), passed=True, gpu_or_ssd_access=False)
    (HERE / 'runtime/build_receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print('Isolated asynchronous backend build complete', flush=True)


if __name__ == '__main__':
    main()
