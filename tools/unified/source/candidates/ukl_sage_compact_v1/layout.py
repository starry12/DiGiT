"""Host compact group metadata; storage rows are never narrowed to int32."""
import numpy as np
from .common import require,bounded


def node_ids(values):
    values=np.asarray(values)
    require(values.dtype.kind=='i' and (not values.size or (values.min()>=0 and values.max()<2**31)),'Node/group ID exceeds int32')
    return values.astype(np.int32)


def storage_ids(values,extent):
    values=np.asarray(values)
    require(type(extent) is int and 0<extent<2**63 and values.dtype==np.int64 and
            (not values.size or (values.min()>=0 and values.max()<extent)),'Storage IDs must remain valid int64')
    return values


def address(row,storage_rows,offset=0):
    require(type(row) is int and type(storage_rows) is int and 0<=row<storage_rows and storage_rows%8==0,'Invalid 512B row/page extent')
    require(type(offset) is int and offset>=0 and offset%4096==0 and offset+storage_rows*512<2**63,'Invalid byte extent')
    page,lane=divmod(row,8)
    return dict(storage_row=row,page=page,byte_in_page=lane*512,row_offset_bytes=offset+row*512,
                nvme_read_offset_bytes=offset+page*4096,read_bytes=4096,useful_bytes=512)


class Grouped:
    def __init__(self,bundle,*,reference=False):
        bounded(bundle.num_nodes,len(bundle.arrays['reordered_indices']),True)
        self.nodes=bundle.num_nodes;self.groups=bundle.num_groups;self.size=bundle.group_size
        require(self.size==2 and bundle.io_geometry.feature_row_bytes==512,'UKL requires g2 and 128D')
        a=bundle.arrays;self.ptr=np.asarray(a['reordered_indptr'])
        require(self.ptr.dtype==np.int64 and self.ptr.shape==(self.nodes+self.groups+1,) and self.ptr[0]==0 and
                np.all(self.ptr[1:]>=self.ptr[:-1]) and self.ptr[-1]==len(a['reordered_indices']),'Invalid group offsets')
        cast=lambda x:node_ids(x).astype(np.int64) if reference else node_ids(x)
        self.units=cast(a['reordered_indices']);self.members=cast(a['group_members']);self.super_to_group=cast(a['supernode_to_group'])
        require(self.members.shape==(self.groups,2) and self.super_to_group.shape==(self.groups,) and
                (not self.units.size or self.units.max()<self.nodes+self.groups) and
                (not self.members.size or self.members.max()<self.nodes) and
                (not self.super_to_group.size or self.super_to_group.max()<self.groups),'Bad group IDs')
        extent=len(a['storage_to_node']);self.base=storage_ids(np.asarray(a['group_storage_base']),extent)
        self.primary=storage_ids(np.asarray(a['node_to_primary_row']),extent)
        require(self.base.shape==(self.groups,) and self.primary.shape==(self.nodes,) and
                np.all(self.base%8==0) and (not len(self.base) or self.base.max()+1<extent),'Bad group storage extent')
        self.array_bytes=sum(x.nbytes for x in (self.ptr,self.units,self.members,self.super_to_group,self.base,self.primary))

    def row(self,owner):
        require(0<=int(owner)<self.nodes,'Bad logical owner')
        lo,hi=map(int,self.ptr[int(owner):int(owner)+2]);return self.units[lo:hi]

    def expand(self,unit):
        unit=int(unit)
        if unit<self.nodes:return [unit],{}
        group=int(self.super_to_group[unit-self.nodes]);members=self.members[group].astype(np.int64).tolist()
        return members,{s:int(self.base[group])+i for i,s in enumerate(members)}
