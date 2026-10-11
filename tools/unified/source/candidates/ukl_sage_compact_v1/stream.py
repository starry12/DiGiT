"""Bounded framed adjacency interface. Input direction is part of identity."""
import hashlib
import json
import struct
import numpy as np
from .common import require, bounded

MAGIC=b'DIGUKL01'
END=b'DIGEND01'
MAX_HEADER=16384


def validate_header(h):
    require(set(h)=={'schema','nodes','edges','orientation','source_sha256','node_map_sha256'}, 'Wrong stream header fields')
    require(h['schema']=='digit-ukl-incoming-stream-v1' and type(h['nodes']) is int and 0<h['nodes']<2**31 and
            type(h['edges']) is int and 0<=h['edges']<2**63, 'Wrong stream dimensions')
    require(h['orientation'] in ('incoming','outgoing'), 'Unknown orientation')
    for name in ('source_sha256','node_map_sha256'):
        v=h[name];require(isinstance(v,str) and len(v)==64 and all(c in '0123456789abcdef' for c in v),'Unbound source/map identity')
    return h


def encode_header(h):
    data=json.dumps(validate_header(h),sort_keys=True,separators=(',',':')).encode()
    require(len(data)<=MAX_HEADER,'Oversized header')
    return MAGIC+struct.pack('>I',len(data))+data


class Reader:
    def __init__(self, source, expected, *, fixture=False, chunk_items=65536):
        validate_header(expected);bounded(expected['nodes'],expected['edges'],fixture)
        require(type(chunk_items) is int and 0<chunk_items<=65536,'Invalid stream buffer')
        self.source=source;self.chunk_items=chunk_items;self.digest=hashlib.sha256()
        self.owner=0;self.edges=0;self.pending=False;self.complete=False
        require(self.take(8)==MAGIC,'Wrong stream magic')
        length=struct.unpack('>I',self.take(4))[0];require(0<length<=MAX_HEADER,'Oversized stream header')
        self.header=validate_header(json.loads(self.take(length)))
        require(self.header==expected,'Stream binding/direction changed')

    def take(self,n):
        # Read requests and temporary buffers remain bounded, including pipes
        # that return fewer bytes than requested.
        require(0<=n<=max(MAX_HEADER,65536*4),'Unbounded stream read')
        parts=[];left=n
        while left:
            part=self.source.read(left)
            require(bool(part) and len(part)<=left,'Truncated/malformed stream')
            parts.append(part);left-=len(part)
        data=b''.join(parts);self.digest.update(data);return data

    def next_owner(self):
        require(not self.pending and not self.complete and self.owner<self.header['nodes'],'Unconsumed or excess owner')
        owner,degree=struct.unpack('>iq',self.take(12))
        require(owner==self.owner and 0<=degree<=self.header['edges']-self.edges,'Owner order or degree mismatch')
        self.pending=True
        def chunks():
            left=degree
            while left:
                count=min(left,self.chunk_items)
                values=np.frombuffer(self.take(count*4),dtype='>i4').astype('<i4')
                require(not values.size or (values.min()>=0 and values.max()<self.header['nodes']),'Neighbor outside logical range')
                left-=count;self.edges+=count
                yield values
            self.pending=False;self.owner+=1
        return owner,degree,chunks()

    def finish(self):
        require(not self.pending and self.owner==self.header['nodes'] and not self.complete,'Incomplete stream')
        require(self.take(8)==END and struct.unpack('>q',self.take(8))[0]==self.edges==self.header['edges'],'Bad edge footer')
        require(self.source.read(1)==b'','Trailing stream bytes')
        self.complete=True
        return self.digest.hexdigest()


def write_fixture(target, header, rows):
    validate_header(header);bounded(header['nodes'],header['edges'],True)
    target.write(encode_header(header));total=0;owners=0
    for owner,values in enumerate(rows):
        values=np.asarray(values)
        require(values.ndim==1 and values.dtype.kind in 'iu' and (not len(values) or (values.min()>=0 and values.max()<header['nodes'])),'Invalid fixture neighbor')
        require(owner<header['nodes'],'Extra owner');target.write(struct.pack('>iq',owner,len(values)))
        target.write(values.astype('>i4').tobytes());total+=len(values);owners+=1
    require(owners==header['nodes'] and total==header['edges'],'Incomplete fixture source')
    target.write(END+struct.pack('>q',total))
