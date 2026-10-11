"""Compile isolated async backend; no GPU/SSD work."""
import json,shlex,subprocess,sys,time
from .common import HERE,ROOT,BINARY,PARENT,require,sha,write,native_sources

def main():
    require(not BINARY.exists(),'Preserve existing build')
    native=native_sources()
    includes=shlex.split(subprocess.check_output([sys.executable,'-B','-m','pybind11','--includes'],text=True))
    command=['/usr/local/cuda-12.4/bin/nvcc','-std=c++11','-O3','-arch=sm_89','--default-stream','per-thread','-shared','-Xcompiler','-fPIC']+includes+[
       '-I'+str(HERE/'native/gids_module/include'),'-I'+str(HERE/'native/bam/include'),
       '-I'+str(PARENT/'native/bam/include'),'-I'+str(PARENT/'native/bam/include/freestanding/include'),
       str(HERE/'native/gids_module/gids_nvme.cu'),str(ROOT/'bam/build/lib/libnvm.so'),
       '-Xlinker','-rpath','-Xlinker',str(ROOT/'bam/build/lib'),'-Xcompiler','-pthread','-o',str(BINARY.with_suffix('.partial.so'))]
    started=time.time()
    with (HERE/'runtime/build.log').open('w') as f:
        f.write(json.dumps(command)+'\n');f.flush();subprocess.run(command,check=True,stdout=f,stderr=subprocess.STDOUT)
    require(native==native_sources(),'Sources changed during build')
    BINARY.with_suffix('.partial.so').rename(BINARY)
    write(HERE/'runtime/build_receipt.json',dict(native_files=native,binary_sha256=sha(BINARY),command=command,started_unix=started,finished_unix=time.time(),gpu_ssd_accepted=False))
    print('Build passed')
if __name__=='__main__':main()
