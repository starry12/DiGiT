"""Build independent CL-capacity backends without changing baseline algorithms."""
import argparse,hashlib,json,os,resource,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/cl_training_20261005_v11'
def binary(arm):
    if arm not in ('gids','digit'):raise ValueError('Unknown arm')
    return OUT/'build'/arm/'BAM_Feature_Store.so'
BINARY=binary('digit')
def command(arm):
    includes=subprocess.check_output(['/home/embed/miniconda3/envs/gids/bin/python','-B','-m','pybind11','--includes'],text=True).split()
    native=HERE/('native_'+arm);limit=978408104 if arm=='gids' else 1174089720
    return ['/usr/local/cuda-12.4/bin/nvcc','-std=c++11','-O3','-arch=sm_89','--default-stream','per-thread','-shared','-Xcompiler','-fPIC',*includes,
      '-DCL_STORAGE_ROWS_MAX='+str(limit)+'ULL','-I'+str(HERE),'-I'+str(native/'gids_module/include'),'-I'+str(native/'bam/include'),
      '-I'+str(ROOT/'candidates/io_accounting_v1/native/bam/include'),'-I'+str(ROOT/'candidates/io_accounting_v1/native/bam/include/freestanding/include'),
      str(native/'gids_module/gids_nvme.cu'),str(ROOT/'bam/build/lib/libnvm.so'),'-Xlinker','-rpath','-Xlinker',str(ROOT/'bam/build/lib'),'-Xcompiler','-pthread','-o',str(binary(arm))]
def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args()
    for arm in ('gids','digit'):
        cmd=command(arm)
        if not a.execute:print(json.dumps(cmd));continue
        target=binary(arm)
        if target.exists():raise FileExistsError('Preserve prior binary')
        target.parent.mkdir(parents=True,exist_ok=True);start=time.time()
        def limits():
            os.nice(19);resource.setrlimit(resource.RLIMIT_AS,(8*2**30,8*2**30));os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
        with (OUT/('build_'+arm+'.log')).open('x') as log:
            subprocess.run(cmd,check=True,timeout=240,stdout=log,stderr=subprocess.STDOUT,preexec_fn=limits)
        (OUT/('build_'+arm+'_receipt.json')).write_text(json.dumps(dict(arm=arm,binary_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),command=cmd,seconds=time.time()-start,cuda_initialized=False,native_accepted=False),indent=2)+'\n')
if __name__=='__main__':main()
