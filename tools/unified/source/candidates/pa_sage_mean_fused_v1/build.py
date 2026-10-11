"""Single nvcc compile without initializing GPU/SSD or changing installed libraries."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE/'native/mean_normalize.cu'
BINARY = HERE/'runtime/DiGiTMeanNormalizeCUDA.so'
PYTHON = '/home/embed/miniconda3/envs/gids/bin/python'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    if BINARY.exists() or BINARY.with_suffix('.partial.so').exists():
        raise RuntimeError('Preserve existing build evidence')
    files = [SOURCE, Path(__file__), HERE/'runtime/__init__.py']
    sources = {str(p.relative_to(ROOT)):sha(p) for p in files}
    includes = shlex.split(subprocess.check_output([PYTHON,'-B','-m','pybind11','--includes'],text=True))
    cmd = ['/usr/local/cuda-12.4/bin/nvcc','-std=c++14','-O3','-arch=sm_89',
        '--default-stream','per-thread','--ftz=false','--prec-div=true',
        '-shared','-Xcompiler','-fPIC','-Xptxas=-v'] + includes + [str(SOURCE),'-o',str(BINARY.with_suffix('.partial.so'))]
    start=time.time()
    with (HERE/'runtime/build.log').open('x') as log:
        subprocess.run(['nice','-n','10']+cmd,check=True,stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))
    assert sources == {str(p.relative_to(ROOT)):sha(p) for p in files}
    BINARY.with_suffix('.partial.so').rename(BINARY)
    receipt=dict(passed=True,source_sha256=sources,binary_sha256=sha(BINARY),command=cmd,
        started_unix=start,finished_unix=time.time(),gpu_or_ssd_access=False,
        fp32_division='__fdiv_rn, no fast math, no flush-to-zero',api=1)
    (HERE/'runtime/build_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('Isolated mean normalization compiled:',sha(BINARY))


if __name__=='__main__':main()
