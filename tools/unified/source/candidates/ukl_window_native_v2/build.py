"""Explicit bounded build; never launches a workload or modifies old binaries."""
import argparse,hashlib,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=ROOT/'results/ukl_window_native_20260930_v2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(cuda=False):
 build=OUT/'build';build.mkdir(exist_ok=True)
 commands=[['/usr/bin/g++','-O2','-std=c++14','-fPIC','-shared',str(HERE/'native/cpu.cpp'),'-o',str(build/'libcompact_cpu.so')]]
 if cuda:commands.append(['/usr/local/cuda/bin/nvcc','-O2','-std=c++14','-arch=sm_89','--compiler-options=-fPIC','-shared',str(HERE/'native/cuda.cu'),'-o',str(build/'libcompact_cuda.so')])
 for command in commands:subprocess.run(['/usr/bin/nice','-n','19']+command,check=True,timeout=180)
 receipt=dict(passed=True,commands=commands,sources={str(p.relative_to(HERE)):sha(p) for p in (HERE/'native').iterdir()},binaries={p.name:sha(p) for p in build.glob('*.so')},cuda_built=cuda)
 (build/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cuda',action='store_true');main(p.parse_args().cuda)
