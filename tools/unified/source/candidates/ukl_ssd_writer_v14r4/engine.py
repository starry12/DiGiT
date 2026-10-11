"""Transactional write algorithm shared by the native transport and CPU tests."""
import hashlib
import mmap
import os
from pathlib import Path
import time
from candidates.ukl_feature_binding_v13 import plan as F
PAGE=4096;CHUNK=8*2**20
CHECKPOINT_BYTES=2**30

class Pace:
    def __init__(self,rate,check):
        if type(rate)is not int or rate<=0:raise ValueError('positive rate')
        self.rate,self.check=rate,check;self.when=time.monotonic()
    def account(self,n):
        # No accumulated idle-time credit: at most one bounded chunk may burst.
        self.when=max(self.when,time.monotonic())+n/self.rate
        while True:
            self.check();delay=self.when-time.monotonic()
            if delay<=0:return
            time.sleep(min(delay,.1))

class DirectSource:
    def __init__(self,receipt):
        self.receipt=receipt;self.path=Path(receipt['path'])
        if self.path.resolve(strict=True)!=self.path or self.path.is_symlink():raise RuntimeError('Source symlink')
        self.fd=os.open(str(self.path),os.O_RDONLY|os.O_DIRECT|os.O_NOFOLLOW|os.O_CLOEXEC)
        self.buf=mmap.mmap(-1,CHUNK,flags=mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS)
        try:
            self.buf.madvise(mmap.MADV_DONTFORK);self.check()
        except BaseException:
            self.close();raise
    def check(self):
        if F.ident(os.fstat(self.fd))!=self.receipt['identity'] or F.ident(self.path.stat())!=self.receipt['identity']:
            raise RuntimeError('Source identity changed')
    def read(self,offset,length):
        if offset%PAGE or not 0<length<=CHUNK or offset+length>self.receipt['bytes']:raise ValueError('Source slice bound')
        self.check();rounded=F.align(length,PAGE);v=memoryview(self.buf)
        try:
            got=os.preadv(self.fd,[v[:rounded]],offset)
            if got!=length:raise RuntimeError('Short or overlong source read')
            return bytes(v[:length])
        finally:v.release()
    def close(self):self.buf.close();os.close(self.fd)


def validate_regions(binding,authorization):
    if (not authorization or authorization.get('schema')!='ukl-explicit-overwrite-v14r2'
            or authorization.get('blank_scan_required') is not False
            or set(binding['arms'])!={'gids','digit'}
            or {k:a['region'] for k,a in binding['arms'].items()}!=authorization.get('allowed_regions')):
        raise RuntimeError('Explicit overwrite authorization does not match regions')
    spans=[];capacity=authorization['device']['capacity_bytes']
    for a in binding['arms'].values():
        r=a['region'];lo=r['device_offset_bytes'];hi=r['end_bytes_exclusive']
        if (lo%PAGE or hi!=lo+r['payload_bytes'] or r['payload_bytes']!=F.align(r['logical_bytes'],PAGE)
                or r['guard_before']!=[lo-PAGE,lo] or r['guard_after']!=[hi,hi+PAGE]
                or lo<PAGE or hi+PAGE>capacity or a['source']['bytes']!=r['logical_bytes']):
            raise ValueError('Invalid authorized region geometry')
        if any(lo-PAGE<end and begin<hi+PAGE for begin,end in spans):raise ValueError('Authorized regions overlap')
        spans.append((lo-PAGE,hi+PAGE))
    cursors=authorization.get('resume_offsets')
    if not isinstance(cursors,dict) or set(cursors)!=set(binding['arms']):raise ValueError('Missing resume offsets')
    for arm,a in binding['arms'].items():
        n=cursors[arm];total=a['region']['payload_bytes']
        if type(n)is not int or n<0 or n>total or (n!=total and n%CHUNK):raise ValueError('Unaligned resume offset')


def run(device,binding,record,check=lambda:None,event=lambda **kw:None,source_factory=DirectSource,rate=512*2**20,authorization=None):
    """Verify ALL retained content before writing only the authorized suffix.

    Old GIDS receipt was lost by the previous failure path. Full device content
    is rehashed against the frozen accepted source SHA; partial DiGiT is compared
    exactly to source before reuse. Any mismatch stops without a device write.
    """
    validate_regions(binding,authorization)
    auth_sha=F.digest(authorization);cursors=authorization['resume_offsets']
    pace=Pace(rate,check);guards={};results={};hashes={};prefix_hashes={}
    def read_at(off,n):
        check();v=device.read(off,n)
        if len(v)!=n:raise RuntimeError('Short device read')
        pace.account(n);return v
    def guard_hashes(r):
        return {label:hashlib.sha256(read_at(r[label][0],r[label][1]-r[label][0])).hexdigest() for label in ('guard_before','guard_after')}
    try:
        for arm,a in binding['arms'].items():guards[arm]=guard_hashes(a['region'])
        # Both retained prefixes must be verified before the first write.
        for arm,a in binding['arms'].items():
            r=a['region'];cursor=cursors[arm];source=source_factory(a['source']);h=hashlib.sha256()
            try:
                for part in F.write_schedule(r):
                    at=part['source_offset']
                    if at>=cursor:break
                    actual=read_at(part['device_offset'],part['write_bytes']);valid=part['source_bytes']
                    if cursor==r['payload_bytes']:
                        if actual[valid:]!=bytes(part['zero_tail_bytes']):raise RuntimeError('Retained tail mismatch')
                        h.update(actual[:valid])
                    else:
                        expected=source.read(at,valid)
                        if actual!=expected+bytes(part['zero_tail_bytes']):raise RuntimeError('Retained prefix mismatch: '+arm)
                        h.update(expected)
                    event(stage='verify_retained',arm=arm,bytes_done=at+part['write_bytes'],bytes_total=cursor)
                source.check()
                if cursor==r['payload_bytes'] and h.hexdigest()!=a['source']['sha256']:
                    raise RuntimeError('Retained full SHA256 mismatch: '+arm)
                hashes[arm]=h;prefix_hashes[arm]=h.hexdigest()
                if guard_hashes(r)!=guards[arm]:raise RuntimeError('Boundary guard changed during retained verification')
                record(arm,'retained_verified',dict(verified_retained_bytes=cursor,retained_sha256=h.hexdigest(),
                    full_retained_readback=cursor==r['payload_bytes'],overwrite_authorization_sha256=auth_sha,guards_before=guards[arm]))
            finally:source.close()
        for arm,a in binding['arms'].items():
            r=a['region'];cursor=cursors[arm];source=source_factory(a['source']);h=hashes[arm];done=cursor;last_checkpoint=cursor
            record(arm,'write_in_progress' if cursor<r['payload_bytes'] else 'readback_in_progress',
                   dict(overwrite_authorization_sha256=auth_sha,blank_scan_performed=False,reused_bytes=cursor,guards_before=guards[arm]))
            try:
                for part in F.write_schedule(r):
                    if part['source_offset']<cursor:continue
                    check();data=source.read(part['source_offset'],part['source_bytes']);h.update(data)
                    data+=bytes(part['zero_tail_bytes'])
                    if len(data)!=part['write_bytes']:raise RuntimeError('Writer schedule coverage')
                    device.write(part['device_offset'],data);done+=len(data);pace.account(len(data))
                    event(stage='write',arm=arm,bytes_done=done,bytes_total=r['payload_bytes'],reused_bytes=cursor)
                    if done-last_checkpoint>=CHECKPOINT_BYTES:
                        device.flush();source.check()
                        record(arm,'checkpoint',dict(durable_prefix_bytes=done,prefix_source_sha256=h.hexdigest(),
                            guards_before=guards[arm],overwrite_authorization_sha256=auth_sha))
                        last_checkpoint=done
                source.check()
                if h.hexdigest()!=a['source']['sha256']:raise RuntimeError('Written source SHA256 mismatch')
                device.flush();record(arm,'written_unverified',dict(written_bytes=done,new_written_bytes=done-cursor,source_sha256=h.hexdigest()))
                samples=[]
                for sample in F.sample_pages(r):
                    actual=read_at(sample['device_offset'],PAGE);valid=sample['source_bytes']
                    expected=source.read(sample['relative_offset'],valid) if valid else b''
                    if actual!=expected+bytes(PAGE-valid):raise RuntimeError('Device readback/tail mismatch')
                    samples.append(dict(sample,sha256=hashlib.sha256(actual).hexdigest()))
                after=guard_hashes(r)
                if after!=guards[arm]:raise RuntimeError('Boundary guard changed')
                source.check();result=dict(passed=True,written_bytes=done,new_written_bytes=done-cursor,reused_bytes=cursor,
                    source_sha256=h.hexdigest(),retained_sha256=prefix_hashes[arm],retained_verified_bytes=cursor,
                    blank_scan_performed=False,overwrite_authorization_sha256=auth_sha,
                    guards_before=guards[arm],guards_after=after,samples=samples,sample_readback_passed=True,
                    full_readback_passed=cursor==r['payload_bytes'],tail_zero_verified=True,raw_ssd_bound=True)
                record(arm,'verified',result);results[arm]=result
            finally:source.close()
        return dict(passed=True,arms=results,raw_ssd_written=any(v['new_written_bytes'] for v in results.values()),raw_ssd_bound=True,full_readback_passed=False)
    except BaseException as error:
        for arm in binding['arms']:record(arm,'failed_occupied',dict(error=repr(error),automatic_reuse=False))
        raise
