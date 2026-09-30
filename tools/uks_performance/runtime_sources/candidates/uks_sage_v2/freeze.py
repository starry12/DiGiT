"""Seal this preparation version; final native version will have a new manifest."""
import json,hashlib
from .common import HERE,ROOT,write,sha
DEPENDENCIES=['evaluation/digit/normalized_csc.py','evaluation/digit/large_preprocess.py','evaluation/digit/retiring_job.py','evaluation/digit/graph_source.py','digit_paths.py','ae/igb/models.py']
def verify():
    value=json.loads((HERE/'manifest.json').read_text())
    actual={p.name for p in HERE.iterdir() if p.is_file() and p.name!='manifest.json'}
    if actual!=set(value['files']):raise RuntimeError('UKS source file set differs')
    for rel,digest in value['files'].items():
        if sha(HERE/rel)!=digest:raise RuntimeError('UKS source changed: '+rel)
    for rel,digest in value['dependencies'].items():
        if sha(ROOT/rel)!=digest:raise RuntimeError('UKS preparation dependency changed: '+rel)
    return sha(HERE/'manifest.json')
def seal():
    if (HERE/'manifest.json').exists():raise RuntimeError('Existing frozen version; create a new candidate')
    write(HERE/'manifest.json',dict(schema='digit-uks-preparation-manifest-v1',scope='source preparation and bounded-window timing helper only; no native acceptance',files={p.name:sha(p) for p in sorted(HERE.iterdir()) if p.is_file()},dependencies={s:sha(ROOT/s) for s in DEPENDENCIES}))
    return verify()
if __name__=='__main__':print(seal())
