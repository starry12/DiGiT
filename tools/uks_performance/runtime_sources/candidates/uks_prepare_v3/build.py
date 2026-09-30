"""Print a build plan by default. --execute requires the original grid to finish."""
import argparse
import fcntl
import json
import os
import shlex
import subprocess
import sys
import time
from .common import HERE, ROOT, PARENT, BINARY, GRID, read, write_new, sha, verify, require, require_grid_complete, heavy_lock


def plan():
    return dict(source=str(HERE / 'native/gids_module/gids_nvme.cu'),
        output=str(BINARY), parent_headers=str(PARENT / 'native/bam/include'),
        compiler='/usr/local/cuda-12.4/bin/nvcc', arch='sm_89', parallel_jobs=1,
        cuda_device_execution=False, raw_ssd_access=False, auto_launch=False,
        execution_gate='original grid complete and passed, controller lock released',
        compiled=BINARY.exists(), native_accepted=False)


def execute():
    # No CUDA probing, compiler invocation, directory mutation or imports first.
    require_grid_complete(read(GRID / 'status.json'))
    with heavy_lock():
        identity = verify()
        require(not BINARY.exists() and not (HERE / 'runtime/build_receipt.json').exists(),
                'Do not overwrite an existing build; preserve it for review')
        from candidates.io_accounting_v1.common import verify_release as parent_verify
        parent_verify()
        py_includes = shlex.split(subprocess.check_output(
            [sys.executable, '-B', '-m', 'pybind11', '--includes'], text=True))
        BINARY.parent.mkdir(parents=True, exist_ok=True)
        partial = BINARY.with_suffix('.partial.so')
        require(not partial.exists(), 'Preserve the previous partial build before retrying')
        command = ['/usr/local/cuda-12.4/bin/nvcc', '-std=c++11', '-O3', '-arch=sm_89',
            '--default-stream', 'per-thread', '-shared', '-Xcompiler', '-fPIC'] + py_includes + [
            '-I' + str(HERE / 'native/gids_module/include'),
            '-I' + str(PARENT / 'native/bam/include'),
            '-I' + str(PARENT / 'native/bam/include/freestanding/include'),
            str(HERE / 'native/gids_module/gids_nvme.cu'), str(ROOT / 'bam/build/lib/libnvm.so'),
            '-Xlinker', '-rpath', '-Xlinker', '$ORIGIN/../../../../bam/build/lib',
            '-Xcompiler', '-pthread', '-o', str(partial)]
        started = time.time()
        with (HERE / 'runtime/build.log').open('x') as log:
            log.write(json.dumps(command) + '\n'); log.flush()
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        require(verify() == identity, 'Sources changed during compilation')
        partial.rename(BINARY)
        (BINARY.parent / '__init__.py').write_text('from .BAM_Feature_Store import *\n')
        write_new(HERE / 'runtime/build_receipt.json', dict(source_sha256=identity,
            binary_sha256=sha(BINARY), command=command, started_unix=started,
            finished_unix=time.time(), native_accepted=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    print(json.dumps(plan(), indent=2), flush=True)
    if args.execute:
        execute()


if __name__ == '__main__':
    main()
