"""Use existing experiment locks; never wait while holding one of several locks."""
import contextlib,fcntl,json,os
from pathlib import Path
from .common import ROOT,cfg,read
LOCKS=(Path('/tmp/digit-pa-bidir-controller.lock'),Path('/tmp/digit-pa-sage-libnvm0.lock'))
class Busy(RuntimeError):pass
@contextlib.contextmanager
def exclusive_idle(paths=LOCKS):
    opened=[]
    try:
        for path in paths:
            # Existing shared locks must be real files; missing/unknown is not idle.
            if path.is_symlink() or not path.is_file():raise Busy('Shared experiment lock is unavailable: '+str(path))
            stream=path.open('rb');opened.append(stream)
            try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise Busy('An experiment is using GPU/host/SSD resources: '+str(path))
        yield tuple(stream.fileno() for stream in opened)
    finally:
        for stream in reversed(opened):stream.close()
def active_ig():
    path=ROOT/cfg()['paths']['active_ig']
    if not path.is_file():return dict(known=False,reason='IG status unavailable')
    state=read(path);workers=[]
    for item in [state]+state.get('workers',[]):
        pid=item.get('pid')
        if not isinstance(pid,int) or pid<=0:continue
        try:
            cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            live='candidates.ig_perf_' in cmd
        except PermissionError:live=None
        except FileNotFoundError:live=False
        workers.append(dict(pid=pid,live_matching_ig=live))
    return dict(known=True,stage=state.get('stage'),complete=state.get('complete'),passed=state.get('passed'),processes=workers)
def ensure_ig_closed():
    value=active_ig()
    if not value['known'] or any(w['live_matching_ig'] is not False for w in value['processes']):raise Busy('IG is active or its liveness is unknown; defer UKS bulk work')
    if value['stage'] not in ('complete','failed','complete_with_monitoring_gaps'):
        raise Busy('IG controller has not reached a terminal state')
    return value
