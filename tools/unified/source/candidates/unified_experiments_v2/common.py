import hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/unified_experiments_20261011_v2'
PY='/home/embed/miniconda3/envs/gids/bin/python'
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify():
    for name,h in read(HERE/'manifest.json')['files'].items():
        if sha(ROOT/name)!=h:raise RuntimeError('Unified entry source changed: '+name)
    return sha(HERE/'manifest.json')
