#!/usr/bin/env python3
"""Seal a new source/build snapshot; refuse to replace an existing manifest."""
import json,os,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from artifact_integrity import MUTABLE_ROOTS,sha,verify_package

def inventory():
    result={}
    for base,dirs,names in os.walk(str(ROOT),followlinks=False):
        if Path(base)==ROOT:dirs[:]=[n for n in dirs if n not in MUTABLE_ROOTS]
        for name in dirs+names:
            p=Path(base)/name
            if p.is_symlink():raise RuntimeError('Static symlink: '+str(p))
        for name in names:
            p=Path(base)/name;key=p.relative_to(ROOT).as_posix()
            if key=='ARTIFACT_MANIFEST.json':continue
            if p.suffix in ('.pyc','.pyo') or '__pycache__' in p.parts:raise RuntimeError('Remove bytecode before sealing')
            result[key]=sha(p)
    return result

def main():
    if (ROOT/'ARTIFACT_MANIFEST.json').exists():raise RuntimeError('Already sealed; create a new copy for changes')
    if (ROOT/'deployment/local.json').exists():raise RuntimeError('Cannot seal a deployed checkout')
    before=inventory()
    for component in json.loads((ROOT/'provenance/components.json').read_text()):
        prefix=component+'/'
        value=dict(component=component,files={k[len(prefix):]:v for k,v in before.items() if k.startswith(prefix) and k!=prefix+'manifest.json'})
        (ROOT/component/'manifest.json').write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    value=dict(schema='digit-ae-artifact-v1',files=inventory(),data_included=False,native_acceptance='pending for this refactored snapshot',scope={'dataset':'PA','models':['sage','gcn','gat']})
    (ROOT/'ARTIFACT_MANIFEST.json').write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(passed=True,sha256=verify_package(),files=len(value['files']))))
if __name__=='__main__':main()
