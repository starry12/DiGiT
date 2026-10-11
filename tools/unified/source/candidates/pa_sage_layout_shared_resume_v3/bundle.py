"""Load verified graph/address metadata without a materialized feature file.

Feature correctness is supplied by a separate shared-pool readback receipt;
this is deliberately not represented as a full real-feature artifact.
"""
import numpy as np
from .common import read,sha,require,Path,setup

def load_metadata(base):
    setup()
    from digit.artifacts import ArtifactBundle
    base=Path(base);root=base/'final/bundle';m=read(root/'manifest.json')
    build=read(base/'build_receipt.json');validation=read(base/'final/validation.json')
    strict=build['point']['id']=='g2_r20'
    require(build['passed'] and build['manifest_sha256']==sha(root/'manifest.json'),'Layout receipt mismatch')
    require((validation.get('status')=='skipped_by_user' and validation['passed'] is None) or
            validation['passed'] is True,'Wrong preparation status')
    if strict:require(validation['passed'] and validation['expanded_adjacency_multiset_exact'] and validation['storage_inverse_exact'] and validation['group_padding_checked'] and validation['hot_excluded'],'Pilot needs full metadata validation')
    descriptor=base/'final/descriptor.json'
    require(validation['descriptor_sha256']==sha(descriptor),'Metadata validation changed')
    d=read(descriptor)
    for key in ('dataset','grouping','feature','io'):
        require(m[key]==d[key],'Descriptor geometry changed: '+key)
    require(m.get('io_geometry')==d.get('io_geometry'),'I/O geometry changed')
    arrays={}
    for name,entry in m['files'].items():
        if name=='reordered_features':continue
        require(entry==d['files'][name],'Array descriptor changed')
        path=root/entry['path']
        if strict:require(sha(path)==entry['sha256'],'Pilot metadata checksum mismatch')
        a=np.load(path,mmap_mode='r',allow_pickle=False)
        require(a.dtype==np.int64 and list(a.shape)==entry['shape'],'Wrong metadata shape/type')
        arrays[name]=a
    return ArtifactBundle(root,m,arrays)
