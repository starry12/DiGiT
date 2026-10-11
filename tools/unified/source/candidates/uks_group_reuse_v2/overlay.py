"""Only two adjacency arrays change; all physical feature metadata is reused."""
import copy
import numpy as np
from .common import ARRAYS,require,check_prepared

def apply(base,ptr,idx,proof):
    from digit.artifacts import ArtifactBundle
    n=int(base.manifest['dataset']['num_nodes']);g=base.num_groups
    require(base.group_size==2 and proof['passed'] and proof['expanded_multigraph_exact'],'Unverified g2 overlay')
    require(proof['nodes']==n and proof['groups']==g and proof['edges']==base.manifest['dataset']['num_edges'],'Wrong graph/group identity')
    require(ptr.dtype==idx.dtype==np.int64 and ptr.shape==(n+g+1,) and idx.shape==(proof['units'],),'Wrong overlay extent')
    require(ptr[0]==0 and ptr[-1]==ptr[n]==len(idx),'Wrong overlay boundaries')
    arrays=dict(base.arrays);manifest=copy.deepcopy(base.manifest)
    for name,a in zip(ARRAYS,(ptr,idx)):
        arrays[name]=a;manifest['files'][name]['shape']=list(a.shape)
    # Existing group_owner describes construction provenance only. It is never
    # changed into a claim that a reused group has just one referencing owner.
    manifest.setdefault('metadata',{})['group_reuse_overlay']=dict(group_size=2,group_owner_is_creation_provenance=True,validation=proof)
    return ArtifactBundle(base.root,manifest,arrays)

def load(base,folder):
    from pathlib import Path
    receipt=check_prepared(folder)
    arrays=[np.load(Path(folder)/(name+'.npy'),mmap_mode='r') for name in ARRAYS]
    return apply(base,*arrays,receipt['validation'])
