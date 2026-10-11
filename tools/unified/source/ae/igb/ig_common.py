from ae.common import *
import numpy as np
D=Path(__file__).resolve().parent
L=ROOT/'data/igb'
CFG=ROOT/'configs/igb'
N=269346174; E=3996442004; K=26934617
GIB=2**30

def verify(stage=None): return verify_release()
def progress(path,stage,**kw):
    v=dict(stage=stage,time=time.time(),pid=os.getpid(),**kw);write(Path(path)/'progress.json',v);print(json.dumps(v),flush=True);return v

def observation():
    selected=os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',')[0]
    uuid=subprocess.check_output(['nvidia-smi','-i',selected,'--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).strip()
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=15)
    return dict(pids=[int(x.split(',')[1]) for x in rows.splitlines() if uuid in x and int(x.split(',')[1])!=os.getpid()])
def digest(x):
    import numpy as np,torch
    if torch.is_tensor(x):x=x.detach().cpu().contiguous().numpy()
    return hashlib.sha256(memoryview(np.ascontiguousarray(x)).cast('B')).hexdigest()

def state_hash(obj):
    import torch
    h=hashlib.sha256()
    def walk(v):
        if torch.is_tensor(v):h.update(str((v.dtype,tuple(v.shape))).encode());h.update(bytes.fromhex(digest(v)))
        elif isinstance(v,dict):
            for k in sorted(v,key=str):h.update(str(k).encode());walk(v[k])
        elif isinstance(v,(list,tuple)):
            for t in v:walk(t)
        else:h.update(repr(v).encode())
    walk(obj);return h.hexdigest()
