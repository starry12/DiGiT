"""Tiny CPU-only original CSC/sparse graph views over owned anonymous pages.

Full CL loading and GPU registration are deliberately not exposed. A future
admitted controller must supply its own actual limits and independent guards.
"""
import numpy as np
from candidates.ukl_real_multi_v9.memory import Arena, assert_anonymous
from candidates.ukl_native_sampling_v10r4 import fork_guard as H, sampling as S

MAX_BYTES=2**20
MAX_NODES=4096
MAX_EDGES=131072


def prepare_cpu():
    if H._ACTIVE:raise RuntimeError('CPU imports must precede anonymous ownership')
    # numpy.testing lazily asks platform.uname(), which can spawn uname -p on
    # this Python version. Resolve that before any graph arena is alive.
    import platform
    import numpy.testing
    platform.uname()
    np.testing.assert_array_equal(np.array([0],np.int64),np.array([0],np.int64))


def require_memory(remaining,available=None):
    if type(remaining) is not int or not 0<=remaining<=MAX_BYTES:
        raise ValueError('Remaining fixture byte cap')
    if available is None:
        with open('/proc/meminfo') as stream:
            available=next((int(line.split()[1])*1024 for line in stream if line.startswith('MemAvailable:')),None)
    required=remaining+76*2**30  # same 64 GiB reserve plus 12 GiB headroom
    if type(available) is not int or available<required:
        raise RuntimeError('Host remaining-allocation admission refused')
    return required


class CSC:
    def __init__(self,arena,nodes,edges,origin=0):
        if (type(arena) is not Arena or not arena.loaded or arena.closed
                or set(arena.offsets)!={'indptr','indices'}):
            raise ValueError('Loaded exact owned CSC arena required')
        if any(type(v) is not int for v in (nodes,edges,origin)) or not (0<nodes<2**31 and 0<edges<2**63 and 0<=origin<2**63-edges):
            raise ValueError('CSC dimensions/origin out of range')
        if arena.lengths!={'indptr':8*(nodes+1),'indices':4*edges}:
            raise ValueError('CSC payload width mismatch')
        H.assert_dontfork(arena);assert_anonymous(arena.address,arena.size)
        self.arena=arena;self.nodes=nodes;self.edges=edges;self.origin=origin
        self.views={};self.closed=False
        try:
            for key,dtype in [('indptr',np.int64),('indices',np.int32)]:
                view=memoryview(arena.mm).toreadonly()
                part=view[arena.offsets[key]:arena.offsets[key]+arena.lengths[key]]
                try:self.views[key]=np.frombuffer(part,dtype=dtype)
                finally:part.release();view.release()
            if int(self.views['indptr'][0])!=origin or int(self.views['indptr'][-1])!=origin+edges:
                raise ValueError('CSC pointer endpoints mismatch')
        except BaseException:
            self.close();raise

    def gather(self,owner,slots):
        if self.closed or self.arena.closed:raise RuntimeError('CSC is closed')
        if type(owner) is not int or not 0<=owner<self.nodes:raise ValueError('Owner outside graph')
        slots=np.asarray(slots)
        if slots.dtype!=np.int64 or slots.ndim!=1 or len(slots)>1024:
            raise ValueError('At most 1024 int64 selected positions')
        begin,end=map(int,self.views['indptr'][owner:owner+2])
        if not self.origin<=begin<=end<=self.origin+self.edges:
            raise ValueError('Referenced pointer pair outside CSC')
        if np.any(slots<0) or np.any(slots>=end-begin):raise ValueError('Slot outside adjacency')
        eids=slots+np.int64(begin)
        values=self.views['indices'][eids-self.origin].copy()
        if np.any(values<0) or np.any(values>=self.nodes):raise ValueError('Neighbor outside graph')
        return values,eids

    def close(self):
        self.views.clear();self.closed=True


class Session:
    """Reject production sizes BEFORE allocations, imports or file opens.

    check() is invoked between bounded reads; cooperative cancellation leaves
    no GPU registration. This class opens only CPU fixture regular files.
    Caller must prewarm framework dependencies before entering this session.
    """
    def __init__(self,specs,metadata,mode='csc',check=lambda:None):
        n,e=metadata['nodes'],metadata['edges']
        if (type(n) is not int or type(e) is not int or not 0<n<=MAX_NODES or not 0<e<=MAX_EDGES
                or mode not in ('csc','sparse')):
            raise ValueError('Small CL fixture only; full graph loading disabled')
        total=0
        for spec in specs.values():
            length=spec['length']
            if type(length) is not int or length<=0 or spec.get('offset',0)!=0:
                raise ValueError('Fixture requires complete bounded payloads')
            h=spec.get('sha256','')
            if len(h)!=64 or any(c not in '0123456789abcdef' for c in h):
                raise ValueError('Fixture payload digest required')
            total+=(length+4095)//4096*4096
        if total>MAX_BYTES:raise ValueError('CL fixture anonymous byte cap')
        if set(specs)!=({'indptr','indices'} if mode=='csc' else set(S._NAMES.values())):
            raise ValueError('Exact CSC/overlay payload names required')
        prepare_cpu()
        self.arena=self.graph=self.native=None
        self.guard=H.OwnershipGuard()
        self.check=check;check();self.guard.enter()
        try:
            def admission(remaining):
                check()
                require_memory(remaining)
            self.arena=Arena(specs,admission_check=admission,max_bytes=MAX_BYTES)
            self.guard.track(self.arena);H.dontfork_arena(self.arena)
            self.arena.load(lambda *args:check())
            H.assert_dontfork(self.arena);check()
            if mode=='csc':self.graph=CSC(self.arena,n,e,metadata.get('origin',0))
            else:
                self.graph=S.Graph(self.arena,metadata)
                self.native=S.Native(self.graph)
        except BaseException:
            self.close();raise

    def close(self):
        if self.native is not None:self.native.close()
        if self.graph is not None:self.graph.close()
        if self.arena is not None:
            self.arena.close()
            self.guard.confirm_released(self.arena)
        if self.guard.active:self.guard.release()

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
