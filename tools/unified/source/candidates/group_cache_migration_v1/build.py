"""Compile separate CPU/CUDA libraries without initializing a GPU."""
import argparse
import shlex
import subprocess
import time
from .common import *
LARGE_OUT=OUT/'build'
def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args()
    n=HERE/'native';cxx=['/usr/bin/g++','-O2','-std=c++17']
    nvcc=['/usr/local/cuda-12.4/bin/nvcc','-O2','-std=c++14','-arch=sm_89','-shared','-Xcompiler','-fPIC']
    includes=shlex.split(subprocess.check_output([PY,'-B','-m','pybind11','--includes'],text=True))
    commands={'reuse':cxx+['-fopenmp',str(n/'reuse.cpp')],
        'libcompact_cpu.so':cxx+['-shared','-fPIC',str(n/'cpu.cpp')],
        'libvalidate.so':cxx+['-shared','-fPIC',str(n/'validate.cpp')],
        'libcompact_cuda.so':nvcc+[str(n/'cuda.cu')],
        'IGGroupFirstMigration.so':nvcc+includes+[str(n/'ig_group_first.cu')]}
    LARGE_OUT.mkdir(parents=True,exist_ok=True)
    sources={str(f.relative_to(ROOT)):sha(f) for f in n.iterdir() if f.is_file()}
    sources['candidates/uks_group_reuse_v2/matching.h']=sha(ROOT/'candidates/uks_group_reuse_v2/matching.h')
    for name,cmd in commands.items():
        cmd+=['-o',str(LARGE_OUT/name)];print(json.dumps(cmd),flush=True)
        if not a.execute:continue
        require(not (LARGE_OUT/name).exists(),'Preserve binary '+name);start=time.time()
        with (LARGE_OUT/(name+'.log')).open('x') as f:
            subprocess.run(['nice','-n','10']+cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=600)
        write(LARGE_OUT/(name+'.json'),dict(passed=True,sha256=sha(LARGE_OUT/name),sources=sources,command=cmd,seconds=time.time()-start,gpu_initialized=False))
if __name__=='__main__':main()
