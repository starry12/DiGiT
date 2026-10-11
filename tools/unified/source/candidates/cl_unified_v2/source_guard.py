import json
import os
import stat
import time
from pathlib import Path


def identity(path):
    s=path.lstat()
    target=None
    if stat.S_ISLNK(s.st_mode):
        t=path.stat()
        target=(t.st_dev,t.st_ino,t.st_size,t.st_mtime_ns,t.st_ctime_ns)
    return (s.st_dev,s.st_ino,s.st_mode,s.st_size,s.st_mtime_ns,s.st_ctime_ns,target)


class SourceGuard:
    def __init__(self,root,manifests,verify_full,expected):
        self.pid=os.getpid();self.checks=0;self.last_check=0.0;self.check_seconds=0.0
        manifests=[Path(p) for p in manifests]
        initial={p:identity(p) for p in manifests}
        paths=set(manifests)
        for p in manifests:
            paths.update(Path(root)/name for name in json.loads(p.read_text())['files'])
        self.identities={p:identity(p) for p in paths}
        if any(self.identities[p]!=v for p,v in initial.items()):
            raise RuntimeError('Source manifest changed during inventory')
        self.digest=verify_full()
        if self.digest!=expected:raise RuntimeError('Source differs from controller admission')
        self.check()

    def check(self):
        started=time.monotonic()
        if os.getpid()!=self.pid:raise RuntimeError('Source guard cannot cross worker processes')
        for p,previous in self.identities.items():
            try:current=identity(p)
            except OSError as error:raise RuntimeError('Source unavailable: '+str(p)) from error
            if current!=previous:raise RuntimeError('Source identity changed: '+str(p))
        self.checks+=1;self.last_check=time.monotonic()
        self.check_seconds+=self.last_check-started
        return self.digest

    def check_due(self):
        if time.monotonic()-self.last_check>=1:self.check()

    def receipt(self):
        return dict(manifest_sha256=self.digest,files=len(self.identities),
                    identity_checks=self.checks,identity_check_seconds=self.check_seconds,full_hash_phase='init',
                    final_full_hash_required_in_controller=True)


def manifests(root):
    return [root/'candidates'/name/'manifest.json' for name in
            ('group_cache_migration_v1','cl_unified_v2')]
