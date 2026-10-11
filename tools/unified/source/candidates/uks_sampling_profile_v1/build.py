import subprocess,shlex,time
from .common import *
def main():
    source=verify();require(not BINARY.exists(),'Preserve previous build')
    old=(ROOT/'candidates/pa_sage_bidir_native_v2/native/digit_sampler_cuda.cu').read_text()
    new=(HERE/'native/sampler.cu').read_text()
    for name in ('group_sample_kernel','resolve_eids_kernel'):
        def body(s):
            start=s.index('__global__ void '+name);brace=s.index('{',start);n=1;i=brace+1
            while n:
                n+=(s[i]=='{')-(s[i]=='}');i+=1
            return s[start:i]
        require(body(old)==body(new),'Changed native kernel: '+name)
    inc=shlex.split(subprocess.check_output([PYTHON,'-B','-m','pybind11','--includes'],text=True))
    cmd=['/usr/local/cuda-12.4/bin/nvcc','-std=c++14','-O3','-arch=sm_89','-shared','-Xcompiler','-fPIC']+inc+[str(HERE/'native/sampler.cu'),'-o',str(BINARY.with_suffix('.partial.so'))]
    with (HERE/'runtime/build.log').open('x') as log:subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
    require(verify()==source,'Source changed during build');BINARY.with_suffix('.partial.so').rename(BINARY)
    write(HERE/'runtime/build_receipt.json',dict(passed=True,source_sha256=source,binary_sha256=sha(BINARY),kernel_bodies_unchanged=True,command=cmd,finished_unix=time.time()))
if __name__=='__main__':main()
