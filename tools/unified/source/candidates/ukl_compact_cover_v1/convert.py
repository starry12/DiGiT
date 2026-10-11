"""Lossless, bounded conversion of global covered EIDs to row-local uint32.

Only a new derived file is written. Every source payload is SHA-256 checked;
every output value is reconstructed to its original global int64 EID. Pointer
blocks and covered chunks are bounded independently, including high-degree rows.
This is CPU file processing, never a CUDA registration route.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np

NODE_BLOCK = 65536
COVER_BLOCK = 1048576

def identity(path):
    s = Path(path).stat()
    return dict(device=s.st_dev, inode=s.st_ino, bytes=s.st_size,
                mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)

def require(ok, message):
    if not ok: raise ValueError(message)

def normalized(value):
    names=('device','inode','bytes','mtime_ns','ctime_ns')
    return value if isinstance(value,dict) else dict(zip(names,value))

class Source:
    def __init__(self, spec):
        self.spec = spec; self.path = Path(spec['path'])
        require(identity(self.path) == normalized(spec['identity']), 'Source identity changed')
        self.f = self.path.open('rb', buffering=0); self.f.seek(spec['offset'])
        self.h = hashlib.sha256(); self.remaining = spec['length']
        self.advised=spec['offset']//4096*4096
    def take(self, count):
        size = count * 8
        require(0 <= size <= self.remaining, 'Source range exceeded')
        b = self.f.read(size)
        require(len(b) == size, 'Short source read')
        self.remaining -= size; self.h.update(b)
        end=self.f.tell()//4096*4096
        if end>self.advised:
            os.posix_fadvise(self.f.fileno(),self.advised,end-self.advised,os.POSIX_FADV_DONTNEED)
            self.advised=end
        return np.frombuffer(b, dtype='<i8')
    def finish(self):
        require(self.remaining == 0 and self.h.hexdigest() == self.spec['sha256'], 'Source payload digest mismatch')
        require(identity(self.path) == normalized(self.spec['identity']), 'Source changed during conversion')
    def close(self): self.f.close()

def convert(specs, nodes, output, event=lambda **kw: None):
    output = Path(output); partial = output.with_name(output.name + '.partial')
    require(not output.exists() and not partial.exists(), 'Preserve existing output/partial evidence')
    require(nodes > 0 and all(specs[k]['length'] == (nodes+1)*8 for k in ('ptr','gptr')), 'Pointer extent')
    output.parent.mkdir(parents=True, exist_ok=True)
    space=os.statvfs(output.parent)
    require(space.f_bavail*space.f_frsize >= specs['covered']['length']//2 + 8*2**30,
            'Insufficient free space for independent compact output and reserve')
    sources = {}; h = hashlib.sha256(); count = maximum = 0; started = time.monotonic()
    try:
        for k in ('ptr','gptr','covered'):sources[k]=Source(specs[k])
        p0 = int(sources['ptr'].take(1)[0]); g0 = int(sources['gptr'].take(1)[0])
        require(p0 >= 0 and g0 == 0, 'Pointer origin')
        with partial.open('xb', buffering=0) as out:
            advised=0
            for lo in range(0, nodes, NODE_BLOCK):
                n = min(NODE_BLOCK, nodes-lo)
                p = np.concatenate(([p0], sources['ptr'].take(n)))
                g = np.concatenate(([g0], sources['gptr'].take(n)))
                require(np.all(p[1:] >= p[:-1]) and np.all(g[1:] >= g[:-1]), 'Nonmonotone pointers')
                require(np.all(g[1:]-g[:-1] <= (p[1:]-p[:-1])//2), 'Covered extent exceeds owner degree')
                require(int(g[-1]) <= specs['covered']['length']//16, 'Group pointer exceeds covered payload')
                left, right = 2*int(g[0]), 2*int(g[-1]); previous_owner = previous_eid = -1
                for at in range(left, right, COVER_BLOCK):
                    hi = min(right, at+COVER_BLOCK)
                    eids = sources['covered'].take(hi-at)
                    ordinals = np.arange(at, hi, dtype=np.int64)
                    owners = np.searchsorted(g*2, ordinals, side='right')-1
                    require(np.all(owners >= 0) and np.all(owners < n), 'Missing covered owner')
                    offsets = eids-p[owners]
                    require(np.all(offsets >= 0) and np.all(eids < p[owners+1]), 'Covered EID outside owner')
                    require(np.all(offsets <= np.iinfo(np.uint32).max), 'Row-relative offset overflows uint32')
                    require(not (int(owners[0]) == previous_owner and int(eids[0]) <= previous_eid), 'Unsorted covered boundary')
                    require(np.all((owners[1:] != owners[:-1]) | (eids[1:] > eids[:-1])), 'Unsorted/duplicate covered EID')
                    small = offsets.astype('<u4')
                    require(np.array_equal(small.astype(np.int64)+p[owners], eids), 'Lossy covered conversion')
                    blob = small.tobytes(); out.write(blob); h.update(blob); count += len(small)
                    if count//(16*1048576) != (count-len(small))//(16*1048576):
                        os.fdatasync(out.fileno())
                        end=out.tell()//4096*4096
                        os.posix_fadvise(out.fileno(),advised,end-advised,os.POSIX_FADV_DONTNEED)
                        advised=end
                    maximum = max(maximum, int(small.max()))
                    previous_owner, previous_eid = int(owners[-1]), int(eids[-1])
                p0, g0 = int(p[-1]), int(g[-1])
                event(stage='convert', owners=min(nodes, lo+n), nodes=nodes, covered=count,
                      seconds=time.monotonic()-started)
            require(count*8 == specs['covered']['length'] and count == 2*g0, 'Incomplete covered conversion')
            for source in sources.values(): source.finish()
            out.flush(); os.fsync(out.fileno())
        # Independent output-file hash check (not just the bytes submitted to write).
        check = hashlib.sha256()
        with partial.open('rb') as f:
            for chunk in iter(lambda: f.read(8*1024**2), b''):
                check.update(chunk)
                os.posix_fadvise(f.fileno(),f.tell()-len(chunk),len(chunk),os.POSIX_FADV_DONTNEED)
        require(check.hexdigest() == h.hexdigest(), 'Written output digest mismatch')
        os.rename(partial, output)
        fd = os.open(output.parent, os.O_DIRECTORY); os.fsync(fd); os.close(fd)
        return dict(passed=True, encoding='owner_relative_u32', global_eids='int64_unchanged',
                    sources=specs, nodes=nodes, covered=count, max_relative_offset=maximum,
                    reconstructed_all_values=True, all_source_payloads_verified=True,
                    output=dict(path=str(output.resolve()),offset=0,length=count*4,
                                identity=identity(output),sha256=h.hexdigest()),
                    bytes_saved=count*4, seconds=time.monotonic()-started)
    finally:
        for source in sources.values(): source.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    r=json.loads(a.prepared.read_text());base=r['base']['base_specs']
    specs=dict(ptr=base['indptr'],gptr=r['files']['ptr.i64'],covered=r['files']['covered.i64'])
    specs={k:{f:v[f] for f in ('path','offset','length','identity','sha256')} for k,v in specs.items()}
    if not a.execute:
        print(json.dumps(dict(action='new_lossless_derived_file',bytes_saved=specs['covered']['length']//2,output=str(a.output))));return
    last=[0.]
    def event(**kw):
        if time.monotonic()-last[0]>=30:
            print(json.dumps(kw),flush=True);last[0]=time.monotonic()
    receipt=convert(specs,r['base']['base_metadata']['nodes'],a.output,event)
    target=a.output.with_name(a.output.name+'.json')
    with target.open('x') as f:json.dump(receipt,f,indent=2);f.flush();os.fsync(f.fileno())
    print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
