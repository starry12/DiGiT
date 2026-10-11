"""Compile private binaries with one low-priority CPU; never initialize CUDA."""
import argparse,hashlib,json,os,resource,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/large_graph_fast_20261011_v1/build'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def binary(kind,role='io'):return OUT/(kind+'_'+role+'.so')
def verify(kind,role='io'):
    p=binary(kind,role);r=json.loads(p.with_suffix('.json').read_text())
    if sha(p)!=r['sha256']:raise RuntimeError('Fast binary changed')
    for f,h in r['sources'].items():
        if sha(ROOT/f)!=h:raise RuntimeError('Fast native source changed: '+f)
    return p
def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args()
    nvcc=['/usr/local/cuda-12.4/bin/nvcc','-std=c++14','-O2','-arch=sm_89','--default-stream','per-thread','-shared','-Xcompiler','-fPIC']
    includes=subprocess.check_output(['/home/embed/miniconda3/envs/gids/bin/python','-m','pybind11','--includes'],text=True).split()
    def limits():
        os.nice(19);os.sched_setaffinity(0,{min(os.sched_getaffinity(0)-{2})});resource.setrlimit(resource.RLIMIT_AS,(8*2**30,8*2**30))
    for kind in ('ukl','cl'):
        native=HERE/('io_'+kind)
        jobs={'sampler':nvcc+(['-DCOVERED_ABSOLUTE'] if kind=='cl' else [])+[str(HERE/'native'/kind/'stream.cu')],
              'io':nvcc+includes+(['-DCL_STORAGE_ROWS_MAX=1174089720ULL'] if kind=='cl' else [])+[
                  '-I'+str(HERE),'-I'+str(native/'gids_module/include'),'-I'+str(native/'bam/include'),
                  '-I'+str(ROOT/'candidates/io_accounting_v1/native/bam/include'),'-I'+str(ROOT/'candidates/io_accounting_v1/native/bam/include/freestanding/include'),
                  str(native/'gids_module/gids_nvme.cu'),str(ROOT/'bam/build/lib/libnvm.so'),'-Xlinker','-rpath','-Xlinker',str(ROOT/'bam/build/lib'),'-Xcompiler','-pthread']}
        for role,cmd in jobs.items():
            target=binary(kind,role);cmd+=['-o',str(target)]
            if not a.execute:print(json.dumps(cmd));continue
            if target.exists():verify(kind,role);continue
            OUT.mkdir(parents=True,exist_ok=True);start=time.time()
            files=list((HERE/'native').rglob('*')) if role=='sampler' else list(native.rglob('*'))+list(HERE.glob('*.h*'))
            sources={str(f.relative_to(ROOT)):sha(f) for f in files if f.is_file()}
            with target.with_suffix('.log').open('w') as log:
                result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,preexec_fn=limits,timeout=600)
            if result.returncode:
                if target.exists():target.unlink()
                raise RuntimeError('Compile failed; see '+str(target.with_suffix('.log')))
            target.with_suffix('.json').write_text(json.dumps(dict(passed=True,sha256=sha(target),sources=sources,command=cmd,seconds=time.time()-start,gpu_initialized=False,native_accepted=False),indent=2)+'\n')
            print(kind,role,'built',flush=True)
if __name__=='__main__':main()
