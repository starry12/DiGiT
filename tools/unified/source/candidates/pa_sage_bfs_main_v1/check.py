"""Source-bound CPU checks; no GPU allocation or raw-SSD access."""
import ast
import subprocess
from .common import *


def main():
    execution=verify()
    OUT.mkdir(parents=True,exist_ok=True)
    for path in HERE.glob('*.py'):ast.parse(path.read_text(),filename=str(path))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',
             OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    commands=[('unit_tests',[PY,'-B','-m',MODULE+'.tests'])]
    for arm in cfg()['ablation_order']:
        code='''
import sys
from candidates.pa_sage_bfs_main_v1.common import setup,backend_binary,PARENT
arm=sys.argv[1]
setup(arm)
import torch, runner
from candidates.io_accounting_v1.accounting import install_runner
install_runner(runner)
from candidates.pa_sage_bfs_main_v1 import worker,training_loop,evaluation
from digit import DiGiTSamplerCUDA
import BAM_Feature_Store
from pathlib import Path
assert not torch.cuda.is_initialized()
assert [n for n in sys.modules if n.endswith('DiGiTSamplerCUDA')]==['digit.DiGiTSamplerCUDA']
assert Path(DiGiTSamplerCUDA.__file__).resolve()==PARENT/'runtime/digit/DiGiTSamplerCUDA.so'
assert Path(sys.modules['BAM_Feature_Store.BAM_Feature_Store'].__file__).resolve()==backend_binary(arm)
assert training_loop.EvaluationSession is evaluation.EvaluationSession
print(arm, 'CPU imports PASS; correct sampler/I/O binaries; CUDA not initialized')
'''
        commands.append(('imports_'+arm,[PY,'-B','-c',code,arm]))
    records=[]
    write(OUT/'cpu_checks.json',dict(passed=False,source_sha256=execution,stage='checking'))
    for name,cmd in commands:
        r=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True,timeout=120)
        (OUT/(name+'.log')).write_text(r.stdout+r.stderr)
        records.append(dict(name=name,returncode=r.returncode))
        print(name,r.returncode,flush=True)
        require(r.returncode==0,'CPU check failed; inspect '+str(OUT/(name+'.log')))
    require(verify()==execution,'Source changed during checks')
    write(OUT/'cpu_checks.json',dict(passed=True,source_sha256=execution,checks=records,
        completed_unix=time.time(),cuda_initialized=False,scope='CPU/import only; no full-graph training acceptance'))


if __name__=='__main__':main()
