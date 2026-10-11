"""Bounded CPU/CUDA compilation for an independent sampling-only library."""
import argparse,hashlib,json,os,resource,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/cl_sampling_opt_20261005_v14/build'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');args=p.parse_args()
    commands={
        'libcompact_cpu.so':['/usr/bin/g++','-O2','-std=c++14','-fPIC','-shared',str(HERE/'native/cpu.cpp')],
        'libvalidate.so':['/usr/bin/g++','-O2','-std=c++14','-fPIC','-shared',str(HERE/'native/validate.cpp')],
        'libcompact_cuda.so':['/usr/local/cuda-12.4/bin/nvcc','-O2','-std=c++14','-arch=sm_89','--compiler-options=-fPIC','-shared',str(HERE/'native/cuda.cu')],
    }
    def limits():
        os.nice(19);os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
        resource.setrlimit(resource.RLIMIT_AS,(4*2**30,4*2**30))
    for name,cmd in commands.items():
        cmd=cmd+['-o',str(OUT/name)]
        if not args.execute:print(json.dumps(cmd));continue
        if (OUT/name).exists():raise FileExistsError('Preserve prior native binary')
        start=time.time()
        with (OUT/(name+'.log')).open('x') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180,preexec_fn=limits)
        (OUT/(name+'.json')).write_text(json.dumps(dict(passed=True,command=cmd,seconds=time.time()-start,
          sha256=sha(OUT/name),sources={str(p.relative_to(ROOT)):sha(p) for p in (HERE/'native').iterdir()},cuda_initialized=False),indent=2)+'\n')
if __name__=='__main__':main()
