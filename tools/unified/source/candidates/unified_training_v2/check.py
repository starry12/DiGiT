"""Run CPU integration tests and retain source hashes, without native execution."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    here=Path(__file__).resolve().parent;root=here.parents[1]
    output=root/'results/unified_training_20261011_v1';output.mkdir(exist_ok=True)
    files={str(p.relative_to(here)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(here.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    result=subprocess.run([sys.executable,'-B','-m','unittest','candidates.unified_training_v2.tests','-v'],
                          cwd=root,env=env,capture_output=True,text=True)
    (output/'cpu_checks.log').write_text(result.stdout+result.stderr)
    receipt=dict(passed=result.returncode==0,returncode=result.returncode,source_files=files,
                 scope='CPU protocol and optimizer fixtures only',gpu_used=False,ssd_used=False,
                 native_adapters_integrated=False,native_migration_accepted=False,
                 migration_acceptance_mode='short1')
    (output/'cpu_checks.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(result.stdout+result.stderr,end='');print('Receipt:',output/'cpu_checks.json')
    raise SystemExit(result.returncode)


if __name__=='__main__':main()
