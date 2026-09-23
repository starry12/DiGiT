"""Bind the selected prepared PA graph, layout, source features and evaluation traces."""
import argparse,fcntl,signal,subprocess,sys,os,time
from training.sage.common import *
from ae.common import check_device,check_payload
from ae.pa_sage.monitor_control import ExternalMonitor
from training.sage.validation import report_check,pair_check

def identity(path):
    st=Path(path).stat();return dict(device=st.st_dev,inode=st.st_ino,bytes=st.st_size,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns)
def input_binding(output,execution):
    p=cfg();data=ROOT/p['data'];base=ROOT/p['base_layout'];ready=read(data/'prepared.json')
    require(ready['passed'] and ready['protocol_sha256']==sha(P) and ready['base_manifest_sha256']==sha(base/'final/bundle/manifest.json'),'Prepared overlay differs')
    require(ready['preparation_code_sha256']==read(ROOT/'provenance/preparation_hashes.json'),'Preparation provenance changed')
    files={}
    def add(path,expected=None):
        path=Path(path);before=identity(path);digest=sha(path);require(identity(path)==before,'Input changed while hashing')
        require(expected is None or digest==expected,'Input hash mismatch: '+str(path))
        try:key=str(path.relative_to(ROOT))
        except ValueError:key=str(path)
        files[key]=dict(sha256=digest,identity=before)
    for name,digest in ready['bindings'].items():add(data/name,digest)
    add(data/'prepared.json');old=read(base/'prepared.json')
    for name in ('orders.json','gids_cpu_rows.npy','full_cpu_rows.npy','final/bundle/manifest.json'):add(base/name,old['bindings'][name])
    add(base/'ssd_ready.json');full=read(base/'ssd_ready.json');add(full['state'],full['state_sha256']);add(full['verify_receipt'],full['verify_receipt_sha256'])
    plain=check_payload('papers_gids');add(plain['state'],plain['state_sha256'])
    source=source_config()
    for desc in (source['source_features'],source['label_identity'],source['source_contract']['original_edges']):add(desc['path'],desc['sha256'])
    for name in ('validation_trace','test_trace'):add(ROOT/p[name]/'manifest.json')
    value=dict(candidate_sha256=execution,protocol_sha256=sha(P),prepared_sha256=sha(data/'prepared.json'),files=files)
    write(output/'inputs.json',value);return value
