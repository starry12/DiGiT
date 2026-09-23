"""PA/GAT version bindings; reuse frozen PA native backend and immutable data."""
import os,sys,json,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];P=HERE/'protocol.json'
sys.path.insert(0,str(ROOT))
from ae.common import read,write,sha,require,host
from training.sage.common import setup,source_config,splits,progress,array_sha
from evaluation.sage.common import environment,output_path,POLICY
PARENT=ROOT/'training/sage'
def cfg():return read(P)
def verify():
    from artifact_integrity import verify_component
    return verify_component(HERE)
verify_worker=verify
