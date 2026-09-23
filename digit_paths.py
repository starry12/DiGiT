"""Portable paths for the standalone DiGiT artifact."""
import json, os, re
from pathlib import Path

def project_root(): return Path(__file__).resolve().parent

def locations():
    path=project_root()/'configs/external_paths.json'
    values=json.loads(path.read_text()) if path.exists() else {}
    deployment=project_root()/'deployment/local.json'
    if deployment.exists(): values.update(json.loads(deployment.read_text()).get('locations',{}))
    return {key:os.environ.get('DIGIT_'+key.upper(),value) for key,value in values.items() if isinstance(value,str)}

def storage_root(): return Path(locations().get('storage_root','/mnt/n3/gids_all')).expanduser()

def expand_paths(value):
    if not isinstance(value,str): return value
    root=str(project_root())
    protected='__DIGIT_CURRENT_PROJECT_ROOT__'
    # Protect the package root because it may be nested below a historical root.
    value=re.sub(re.escape(root)+r'(?![A-Za-z0-9_])',lambda _:protected,value)
    for old,new in [('/home/embed/digit',protected),('/home/embed/gids_all',protected),('/mnt/n3/gids_all',str(storage_root()))]:
        value=re.sub(re.escape(old)+r'(?![A-Za-z0-9_])',lambda _:new,value)
    values=dict(locations(),project_root=root)
    for key,path in values.items(): value=value.replace('${'+key.upper()+'}',os.path.expanduser(path))
    value=value.replace(protected,root)
    for old,new in {'candidates/pa_sage_bidir_native_v2': 'training/sage', 'candidates/pa_gcn_native_v1': 'training/gcn', 'candidates/pa_gat_native_v1': 'training/gat', 'candidates/io_accounting_v1': 'runtime/io', 'candidates/pa_sage_direction_v2/graph': 'ae/papers/graph_checks', 'candidates/pa_sage_selected_v1/config.json': 'configs/sage_selected.json', 'candidates/pa_sage_selected_v1/selected_training_report.json': 'reference/sage_correctness.json', 'submission/v2': 'evaluation/sage', 'submission/v3': 'evaluation/gcn', 'submission/v4': 'evaluation/gat'}.items():
        value=value.replace(old,new)
    return value

def relocate(value):
    if isinstance(value,dict): return {expand_paths(k):relocate(v) for k,v in value.items()}
    if isinstance(value,list): return [relocate(v) for v in value]
    return expand_paths(value)

def json_loads(value,*args,**kw): return relocate(json.loads(value,*args,**kw))
def json_load(stream,*args,**kw): return relocate(json.load(stream,*args,**kw))
