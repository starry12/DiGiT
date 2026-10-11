"""Independent native ABI. Compilation never initializes CUDA or touches SSD."""
import argparse, hashlib, json, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/ukl_compact_cover_20261011_v1/build'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verified_binary(name):
    p=OUT/name;r=json.loads((OUT/(name+'.json')).read_text())
    if not r['passed'] or sha(p)!=r['sha256']:raise RuntimeError('Compact binary changed')
    for rel,h in r['sources'].items():
        if sha(ROOT/rel)!=h:raise RuntimeError('Compact native source changed: '+rel)
    return p
def main():
    p=argparse.ArgumentParser();p.add_argument('--cuda',action='store_true');a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    commands={'libcompact_cpu.so':['g++','-O2','-std=c++17','-shared','-fPIC',str(HERE/'native/cpu.cpp')],
              'libvalidate.so':['g++','-O2','-std=c++17','-shared','-fPIC',str(HERE/'native/validate.cpp')]}
    if a.cuda:commands['libcompact_cuda.so']=['/usr/local/cuda-12.4/bin/nvcc','-O2','-std=c++14','-arch=sm_89','-shared','-Xcompiler','-fPIC',str(HERE/'native/cuda.cu')]
    sources={str(p.relative_to(ROOT)):sha(p) for p in (HERE/'native').iterdir() if p.is_file()}
    for name,cmd in commands.items():
        if (OUT/name).exists():verified_binary(name);continue
        cmd+=['-o',str(OUT/name)]
        with (OUT/(name+'.log')).open('x') as f:subprocess.run(['nice','-n','10']+cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
        (OUT/(name+'.json')).write_text(json.dumps(dict(passed=True,sha256=sha(OUT/name),sources=sources,command=cmd,gpu_initialized=False),indent=2))
if __name__=='__main__':main()
