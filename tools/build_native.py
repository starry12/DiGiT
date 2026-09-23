"""Build a new PA artifact copy. Never install a driver or launch a GPU workload."""
import argparse,hashlib,json,os,shlex,shutil,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from artifact_integrity import verify_package
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
parent=verify_package();target=a.output.expanduser().resolve()
if target.exists() or a.output.is_symlink() or target==ROOT or ROOT in target.parents:p.error('Choose a new directory outside this source tree')
python=sys.executable;nvcc=os.environ.get('DIGIT_NVCC','nvcc');cmake=os.environ.get('DIGIT_CMAKE','cmake');cc=os.environ.get('CC','cc');jobs=os.environ.get('DIGIT_BUILD_JOBS','2')
if not jobs.isdecimal() or int(jobs)<1:p.error('DIGIT_BUILD_JOBS must be positive')
if not a.dry_run:
    for cmd in (nvcc,cmake,cc):
        if not shutil.which(cmd):p.error('Missing build tool: '+cmd)
pybind='<python -m pybind11 --cmakedir>' if a.dry_run else subprocess.check_output([python,'-m','pybind11','--cmakedir'],text=True).strip()
bam=target/'third_party/bam';output=target/'training/sage/runtime/digit';build=target/'.native-build/output'
steps=[
 [cmake,'-S',bam,'-B',bam/'build','-DCMAKE_CUDA_COMPILER='+nvcc,'-DCMAKE_CUDA_ARCHITECTURES=89','-Dnvidia_archs=89','-Dno_module=ON','-Dno_smartio=ON','-Dno_fio=ON'],
 [cmake,'--build',bam/'build','--target','libnvm','--parallel',jobs],
 [cc,'-std=gnu11','-O2','-I'+str(bam/'include'),'-I'+str(bam/'build/include'),target/'ae/native/identity/module.c',target/'ae/native/identity/common.c','-L'+str(bam/'build/lib'),'-Wl,-rpath,$ORIGIN/../../../third_party/bam/build/lib','-lnvm','-o',target/'ae/native/identity/identify-module'],
 ['bash',target/'runtime/io/build.sh'],['bash',target/'training/sage/build.sh'],
 [cmake,'-S',target/'ae/native/papers/output','-B',build,'-Dpybind11_DIR='+pybind,'-DPYTHON_EXECUTABLE='+python,'-DPython_EXECUTABLE='+python,'-DCMAKE_CUDA_COMPILER='+nvcc,'-DCMAKE_CUDA_ARCHITECTURES=89','-DDIGIT_EXTENSION_OUTPUT='+str(output)],
 [cmake,'--build',build,'--parallel',jobs],
]
print(json.dumps(dict(parent_sha256=parent,target=str(target),commands=[[str(x) for x in row] for row in steps],gpu_workload=False,ssd_access=False,native_acceptance='pending'),indent=2))
if a.dry_run:sys.exit(0)
def ignore(base,names):
    excludes={'__pycache__','.git','results','data','deployment','ssd_state','build','.native-build','ARTIFACT_MANIFEST.json'} if Path(base)==ROOT else {'__pycache__','build'}
    return [n for n in names if n in excludes or n.endswith(('.so','.pyc','.o','.a')) or n=='identify-module']
shutil.copytree(str(ROOT),str(target),ignore=ignore)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',DIGIT_PYTHON=python,DIGIT_NVCC=nvcc,PYTHONDONTWRITEBYTECODE='1')
logdir=target/'results/build';logdir.mkdir(parents=True)
with (logdir/'build.log').open('x') as log:
    for step in steps:
        log.write(shlex.join([str(x) for x in step])+'\n');log.flush()
        subprocess.run([str(x) for x in step],cwd=str(target),env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
shutil.rmtree(str(target/'.native-build'))
# CMake-generated absolute build paths are not part of the final source seal.
for child in (bam/'build').iterdir():
    if child.name!='lib':
        shutil.rmtree(str(child)) if child.is_dir() else child.unlink()
(target/'provenance/native_build.json').write_text(json.dumps(dict(parent_sha256=parent,native_acceptance='pending',gpu_workload=False,ssd_access=False),indent=2)+'\n')
subprocess.run([python,'-B',str(target/'scripts/seal.py')],check=True,cwd=str(target),env=env)
print('Build complete; bind prepared data and run fresh check/smoke before use.')
