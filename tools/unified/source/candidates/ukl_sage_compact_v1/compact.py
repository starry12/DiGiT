"""Failure-atomic incoming stream -> compact CSC; EIDs exist only on demand."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import struct
import numpy as np
from .common import require, bounded, read, write_new, sha, heavy_lock, cpu_gate
from .stream import Reader, validate_header


def _checkpoint(path,value):
    temp=path.with_suffix('.next')
    with temp.open('w') as f:
        json.dump(value,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def _prefix(path,length):
    h=hashlib.sha256()
    with path.open('rb') as f:
        left=length
        while left:
            data=f.read(min(left,1048576));require(bool(data),'Checkpoint file was truncated')
            h.update(data);left-=len(data)
    return h


def build(source,output,expected,*,fixture=False,resume=False,checkpoint_owners=256,stop_after=None):
    validate_header(expected);bounded(expected['nodes'],expected['edges'],fixture)
    require(expected['orientation']=='incoming','Outgoing adjacency needs a bound transpose, not a direction relabel')
    require(type(checkpoint_owners) is int and checkpoint_owners>0,'Invalid checkpoint interval')
    require(stop_after is None or (type(stop_after) is int and 0<stop_after<=expected['nodes']),'Invalid stop owner')
    with (contextlib.nullcontext() if fixture else heavy_lock()):
        return _build(Path(source),Path(output),expected,fixture,resume,checkpoint_owners,stop_after)


def _build(source,output,expected,fixture,resume,checkpoint_owners,stop_after):
    partial=output.with_name(output.name+'.partial')
    require(not output.exists(),'Preserve completed output')
    def source_identity():
        s=source.stat();return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns)
    identity=source_identity()
    # Binding/direction checks occur before creating output files.
    with source.open('rb') as stream:
        reader=Reader(stream,expected,fixture=fixture)
        if resume:
            state=read(partial/'checkpoint.json')
            require(state['binding']==expected,'Checkpoint input identity changed')
            require(state['source_identity']==identity,'Source file changed since checkpoint')
            require(0<=state['owners']<=expected['nodes'],'Bad checkpoint cursor')
            for _ in range(state['owners']):
                owner,degree,chunks=reader.next_owner()
                for _ in chunks:pass
            require(reader.digest.hexdigest()==state['input_prefix_sha256'] and reader.edges==state['input_edges'],
                    'Input prefix changed since checkpoint')
            hashes={}
            for name,length in (('indptr.i64',(state['owners']+1)*8),('indices.i32',state['normalized_edges']*4)):
                path=partial/name;require(not path.is_symlink(),'Checkpoint file is a symlink')
                h=_prefix(path,length);require(h.hexdigest()==state['output_sha256'][name],'Output prefix changed')
                hashes[name]=h
            edges,removed=state['normalized_edges'],state['removed_self_edges']
            files={n:(partial/n).open('r+b') for n in hashes}
            for name,f in files.items():
                length=(state['owners']+1)*8 if name=='indptr.i64' else edges*4
                f.truncate(length);f.seek(length)
        else:
            partial.mkdir(parents=True,exist_ok=False)
            files={n:(partial/n).open('xb') for n in ('indptr.i64','indices.i32')}
            hashes={n:hashlib.sha256() for n in files};edges=removed=0
            initial=struct.pack('<q',0);files['indptr.i64'].write(initial);hashes['indptr.i64'].update(initial)
        with contextlib.ExitStack() as stack:
            for f in files.values():stack.enter_context(f)
            def save():
                for f in files.values():f.flush();os.fsync(f.fileno())
                value=dict(binding=expected,source_identity=identity,owners=reader.owner,input_edges=reader.edges,normalized_edges=edges,
                    removed_self_edges=removed,input_prefix_sha256=reader.digest.hexdigest(),
                    output_sha256={n:h.hexdigest() for n,h in hashes.items()})
                _checkpoint(partial/'checkpoint.json',value);return value
            if not resume:save()
            while reader.owner<expected['nodes']:
                if fixture:cpu_gate()
                owner,degree,chunks=reader.next_owner()
                for values in chunks:
                    keep=values!=owner;removed+=int((~keep).sum());values=values[keep]
                    data=values.astype('<i4',copy=False).tobytes();files['indices.i32'].write(data)
                    hashes['indices.i32'].update(data);edges+=len(values)
                data=struct.pack('<i',owner);files['indices.i32'].write(data);hashes['indices.i32'].update(data);edges+=1
                require(edges<2**63,'CSC offset overflow')
                data=struct.pack('<q',edges);files['indptr.i64'].write(data);hashes['indptr.i64'].update(data)
                if reader.owner%checkpoint_owners==0 or reader.owner==expected['nodes'] or reader.owner==stop_after:
                    save()
                if reader.owner==stop_after:
                    return dict(complete=False,checkpoint_owners=reader.owner,partial=str(partial))
            input_sha=reader.finish()
            require(source_identity()==identity,'Source changed during build')
            require(edges==expected['edges']-removed+expected['nodes'],'Normalized edge count mismatch')
            # Keep the checkpoint at the last owner boundary, before the
            # footer. Resume can replay it even if final publication crashes.
        manifest=dict(schema='digit-ukl-compact-csc-v1',complete=True,fixture=fixture,nodes=expected['nodes'],edges=edges,
            binding=expected,input_stream_sha256=input_sha,removed_self_edges=removed,
            offsets_dtype='<i8',indices_dtype='<i4',eid_storage='implicit CSC position',
            files={name:dict(bytes=(partial/name).stat().st_size,sha256=h.hexdigest()) for name,h in hashes.items()},
            resident_array_bytes=8*(expected['nodes']+1)+4*edges,full_eid_array_bytes=0,
            resume_policy='verify source/output prefixes; replay source to committed owner; discard only uncommitted own-file tails')
        if (partial/'manifest.json').exists():
            require(read(partial/'manifest.json')==manifest,'Existing final receipt differs; preserve failed output')
        else:
            write_new(partial/'manifest.json',manifest)
        require(not output.exists(),'Output appeared during build')
        partial.rename(output)
        return manifest


def positions64(start,slots,end):
    require(type(start) is int and type(end) is int and 0<=start<=end<2**63,'Bad edge interval')
    slots=np.asarray(slots)
    require(slots.dtype==np.int64 and slots.ndim==1 and (not len(slots) or (slots.min()>=0 and slots.max()<end-start)),
            'Sample slot outside owner degree')
    return slots+np.int64(start)


class CSC:
    """Read-only host arrays. No full DGL graph, GPU upload or explicit EID array."""
    def __init__(self,ptr,indices,*,fixture=False,reference=False):
        n=len(ptr)-1;bounded(n,len(indices),fixture)
        require(ptr.dtype==np.int64 and ptr.ndim==1 and ptr[0]==0 and ptr[-1]==len(indices),'Invalid offsets')
        require(indices.ndim==1 and indices.dtype==(np.int64 if reference else np.int32),'Invalid neighbor width')
        for lo in range(0,len(ptr)-1,65536):
            hi=min(lo+65536,len(ptr)-1)
            require(np.all(ptr[lo+1:hi+1]>=ptr[lo:hi]),'Nonmonotone offsets')
        for lo in range(0,len(indices),65536):
            x=indices[lo:lo+65536];require(not len(x) or (x.min()>=0 and x.max()<n),'Invalid neighbor ID')
        self.ptr=ptr;self.indices=indices;self.nodes=n;self.edges=len(indices);self.fixture=fixture

    @classmethod
    def open(cls,path,*,fixture=False):
        path=Path(path);m=read(path/'manifest.json');bounded(m['nodes'],m['edges'],fixture)
        require(m['complete'] is True and m['schema']=='digit-ukl-compact-csc-v1' and
                m['offsets_dtype']=='<i8' and m['indices_dtype']=='<i4' and m['eid_storage']=='implicit CSC position','Invalid compact receipt')
        for n,item in m['files'].items():
            require(n in ('indptr.i64','indices.i32') and (path/n).stat().st_size==item['bytes'] and sha(path/n)==item['sha256'],'Compact file integrity mismatch')
        require(set(m['files'])=={'indptr.i64','indices.i32'},'Unexpected compact arrays')
        return cls(np.memmap(path/'indptr.i64',mode='r',dtype='<i8',shape=(m['nodes']+1,)),
                   np.memmap(path/'indices.i32',mode='r',dtype='<i4',shape=(m['edges'],)),fixture=fixture)

    def row(self,owner):
        require(0<=int(owner)<self.nodes,'Invalid owner')
        lo,hi=map(int,self.ptr[int(owner):int(owner)+2]);return self.indices[lo:hi]

    def eids(self,owner,slots):
        require(0<=int(owner)<self.nodes,'Invalid owner')
        return positions64(int(self.ptr[owner]),slots,int(self.ptr[owner+1]))

    def resolve(self,owner,sources,chunk=65536):
        """Only selected duplicate occurrences receive IDs; scan with bounded scratch."""
        require(0<len(sources)<=1024 and 0<chunk<=65536,'Bounded selected frontier required')
        wanted={int(s):[] for s in sources};needed={s:int(np.count_nonzero(np.asarray(sources)==s)) for s in wanted}
        row=self.row(owner)
        for lo in range(0,len(row),chunk):
            part=row[lo:lo+chunk]
            for s,hits in wanted.items():
                if len(hits)<needed[s]:
                    slots=np.flatnonzero(part==s)[:needed[s]-len(hits)]
                    hits.extend(int(self.ptr[owner])+lo+int(i) for i in slots)
            if all(len(wanted[s])==needed[s] for s in wanted):break
        require(all(len(wanted[s])==needed[s] for s in wanted),'Selected multiedge absent')
        used={s:0 for s in wanted};answer=[]
        for s in sources:
            s=int(s);answer.append(wanted[s][used[s]]);used[s]+=1
        return np.asarray(answer,dtype=np.int64)
