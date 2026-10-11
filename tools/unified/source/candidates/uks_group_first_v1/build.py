"""Compile only; does not create a CUDA context or run a GPU workload."""
import shlex,subprocess
from .common import *

def main():
    source=verify();require(not BINARY.exists(),'Preserve existing native build')
    inc=shlex.split(subprocess.check_output([PY,'-B','-m','pybind11','--includes'],text=True))
    partial=BINARY.with_suffix('.partial.so')
    require(not partial.exists(),'Preserve partial build')
    command=['/usr/local/cuda-12.4/bin/nvcc','-std=c++14','-O3','-arch=sm_89','-shared','-Xcompiler','-fPIC']+inc+[str(HERE/'native/sampler.cu'),'-o',str(partial)]
    with (HERE/'runtime/build.log').open('x') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
    require(verify()==source,'Source changed during build');partial.rename(BINARY)
    write(HERE/'runtime/build_receipt.json',dict(passed=True,source_sha256=source,
          binary_sha256=sha(BINARY),command=command,native_training_accepted=False))

if __name__=='__main__':main()
