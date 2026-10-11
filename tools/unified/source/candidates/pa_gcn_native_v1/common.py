"""PA/GCN version bindings; reuse frozen PA native backend and immutable data."""
import os,sys,json,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];P=HERE/'protocol.json'
sys.path.insert(0,str(ROOT))
from ae.common import read,write,sha,require,host
from candidates.pa_sage_bidir_native_v2.common import setup,source_config,splits,progress,array_sha
from submission.v2.common import environment,output_path,POLICY
PARENT=ROOT/'candidates/pa_sage_bidir_native_v2'
def cfg():return read(P)
def verify():
    from submission.v2.common import verify as verify_parent
    m=read(HERE/'manifest.json');require(verify_parent()==m['parent_submission_sha256'],'Frozen PA/SAGE submission changed')
    for rel,digest in m['files'].items():require(sha(HERE/rel)==digest,'GCN candidate changed: '+rel)
    for rel,digest in m['dependencies'].items():require(sha(ROOT/rel)==digest,'GCN dependency changed: '+rel)
    return sha(HERE/'manifest.json')
verify_worker=verify
