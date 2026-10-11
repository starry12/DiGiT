from ae.common import *
import numpy as np, gzip
D=ROOT/'configs/papers'
P=D
S=ROOT/'data/papers'
TEST=Path(digit_paths.locations()['papers_test_trace'])
BASE_RECEIPT=S/'ssd/gids_verify.json'

def arr(path): return np.asarray(np.load(path,mmap_mode='r',allow_pickle=False))
def prior(): return module('digit_papers_dataset',ROOT/'ae/papers/dataset.py')
def verify_execution(arm):
    require(arm in ('gids','digit_full'),'Unknown arm')
    check_payload('papers_'+('full' if arm=='digit_full' else 'gids'))
    return verify_release()
def progress(**kw):
    print(json.dumps(dict(updated=time.time(),pid=os.getpid(),**kw)),flush=True)
