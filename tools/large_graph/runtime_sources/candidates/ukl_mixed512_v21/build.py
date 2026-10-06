"""CPU compiler only; importing or compiling this module never initializes CUDA."""
import argparse,hashlib,json,os,resource,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/ukl_mixed512_20261004_v21';BINARY=OUT/'build/BAM_Feature_Store.so'
def command():
    includes=subprocess.check_output(['/home/embed/miniconda3/envs/gids/bin/python','-B','-m','pybind11','--includes'],text=True).split()
    return ['/usr/local/cuda-12.4/bin/nvcc','-std=c++11','-O3','-arch=sm_89','--default-stream','per-thread','-shared','-Xcompiler','-fPIC',*includes,
      '-I'+str(HERE/'native/gids_module/include'),'-I'+str(HERE/'native/bam/include'),
      '-I'+str(ROOT/'candidates/io_accounting_v1/native/bam/include'),'-I'+str(ROOT/'candidates/io_accounting_v1/native/bam/include/freestanding/include'),
      str(HERE/'native/gids_module/gids_nvme.cu'),str(ROOT/'bam/build/lib/libnvm.so'),'-Xlinker','-rpath','-Xlinker',str(ROOT/'bam/build/lib'),'-Xcompiler','-pthread','-o',str(BINARY)]
def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args();cmd=command()
    if not a.execute:print(json.dumps(cmd,indent=2));return
    if BINARY.exists():raise FileExistsError('Preserve previous binary')
    BINARY.parent.mkdir(exist_ok=True);start=time.time()
    def limits():
        os.nice(19);resource.setrlimit(resource.RLIMIT_AS,(8*2**30,8*2**30));os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    with (OUT/'build.log').open('x') as log:subprocess.run(cmd,check=True,timeout=240,stdout=log,stderr=subprocess.STDOUT,preexec_fn=limits)
    (OUT/'build_receipt.json').write_text(json.dumps(dict(binary_sha256=hashlib.sha256(BINARY.read_bytes()).hexdigest(),command=cmd,seconds=time.time()-start,cuda_initialized=False,native_accepted=False),indent=2)+'\n')
if __name__=='__main__':main()
