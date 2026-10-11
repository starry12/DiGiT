"""Selected immutable point protocol; inherited native runtime remains untouched."""
import json,os,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PARENT=ROOT/'candidates/pa_sage_bidir_native_v2'
P=Path(os.environ.get('DIGIT_LAYOUT_PROTOCOL',str(HERE/'UNSELECTED.json'))).resolve()
from ae.common import sha,write,require,host
from ae.pa_sage.common import source_config,splits

def read(path):return json.loads(Path(path).read_text())
def cfg():
    from .protocol import validate
    p=read(P);validate(p);return p
def setup():
    from candidates.pa_sage_bidir_native_v2.common import setup as parent_setup
    parent_setup()
def progress(output,stage,**kw):
    v=dict(stage=stage,pid=os.getpid(),time_unix=time.time(),**kw);write(Path(output)/'progress.json',v);print(json.dumps(v),flush=True)
def verify():
    from candidates.pa_sage_bidir_native_v2.common import verify as parent_verify
    m=read(HERE/'manifest.json')
    require(parent_verify()==m['parent_candidate_sha256'],'Parent native runtime changed')
    for name,digest in m['files'].items():require(sha(HERE/name)==digest,'Training adapter changed: '+name)
    return sha(HERE/'manifest.json')
