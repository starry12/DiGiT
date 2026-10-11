"""Compile the isolated 512-B backend without a GPU or raw SSD operation."""
import shlex
import subprocess
import sys
import time
from .common import *


def main():
    sources = native_sources()
    require(not BINARY.exists(), 'Preserve existing binary; do not overwrite a build')
    BINARY.parent.mkdir(parents=True, exist_ok=True)
    partial = BINARY.with_suffix('.partial.so')
    require(not partial.exists(), 'Preserve previous partial build')
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
    with (HERE / 'runtime/build.log').open('x') as f:
        subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)
    require(sources == native_sources(), 'Native inputs changed during compilation')
    partial.rename(BINARY)
    (BINARY.parent / '__init__.py').write_text('from .BAM_Feature_Store import *\n')
    write(HERE / 'runtime/build_receipt.json', dict(native_source_sha256=object_sha(sources),
        binary_sha256=sha(BINARY), command=cmd, started_unix=started, finished_unix=time.time()))
    print('Isolated native build complete', flush=True)


if __name__ == '__main__':
    main()
