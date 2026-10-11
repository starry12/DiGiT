#!/usr/bin/env python3
"""Independent AE entry. --check validates inputs without GPU allocation or SSD I/O."""
import argparse,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ae.common import *

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset',choices=['papers','igb'])
    p.add_argument('--model',choices=['sage','gcn','gat'],default='sage')
    p.add_argument('--arm',choices=['gids','digit_full'],default='digit_full')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--output',type=Path)
    p.add_argument('--check',action='store_true')
    p.add_argument('--imports',action='store_true',help='Also import native/runtime modules; use with --check')
    a=p.parse_args()
    p.error('IGB extraction covers GraphSAGE, seed 0 only') if a.dataset=='igb' and (a.model!='sage' or a.seed!=0) else None
    p.error('Papers seeds must be 0, 1 or 2') if a.dataset=='papers' and a.seed not in (0,1,2) else None
    p.error('--imports requires --check') if a.imports and not a.check else None
    threads='16' if a.dataset=='papers' else '1'
    os.environ['OMP_NUM_THREADS']=threads;os.environ['MKL_NUM_THREADS']=threads;os.environ['OPENBLAS_NUM_THREADS']='1'
    os.environ.setdefault('DGLBACKEND','pytorch');os.environ['USE_DETERMINISTIC_ALG']='1'
    os.environ['DIGIT_VALIDATION_PROFILE']='fast';os.environ['DIGIT_TRAINING_PROFILE']='off'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    paths=[ROOT/'ae'/a.dataset]
    paths += [ROOT/'ae/papers/runtime'] if a.dataset=='papers' else [ROOT/'ae/igb/runtime']
    sys.path[:0]=list(map(str,paths))
    if a.check:
        from ae.check import check
        print(json.dumps(check(a.dataset,a.arm,imports=a.imports),indent=2));return
    require(a.output is not None,'Specify --output under a fresh run directory')
    a.output=a.output.expanduser().resolve()
    verify_release()
    if a.dataset=='papers':
        from worker import run
        run(a)
    else:
        check_device();os.environ['DIGIT_RUN_OUTPUT']=str(a.output)
        from engine import run
        run('full' if a.arm=='digit_full' else 'gids','formal',a.output.name,a.seed)
if __name__=='__main__':main()
