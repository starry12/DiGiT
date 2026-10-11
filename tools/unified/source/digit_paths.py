"""Portable paths for the standalone DiGiT artifact."""
import json, os, re
from pathlib import Path

def project_root(): return Path(__file__).resolve().parent

def locations():
    path=project_root()/'configs/external_paths.json'
    values=json.loads(path.read_text()) if path.exists() else {}
    return {key:os.environ.get('DIGIT_'+key.upper(),value) for key,value in values.items() if isinstance(value,str)}

def storage_root(): return Path(locations().get('storage_root','/mnt/n3/gids_all')).expanduser()

def expand_paths(value):
    if not isinstance(value,str): return value
    values=dict(locations(),project_root=str(project_root()))
    for key,path in values.items(): value=value.replace('${'+key.upper()+'}',os.path.expanduser(path))
    # Compatibility for the generic library's preexisting data-source options.
    for old,new in [('/home/embed/gids_all',str(project_root())),('/mnt/n3/gids_all',str(storage_root()))]:
        value=re.sub(re.escape(old)+r'(?![A-Za-z0-9_])',lambda _:new,value)
    return value

def relocate(value):
    if isinstance(value,dict): return {expand_paths(k):relocate(v) for k,v in value.items()}
    if isinstance(value,list): return [relocate(v) for v in value]
    return expand_paths(value)

def json_loads(value,*args,**kw): return relocate(json.loads(value,*args,**kw))
def json_load(stream,*args,**kw): return relocate(json.load(stream,*args,**kw))
