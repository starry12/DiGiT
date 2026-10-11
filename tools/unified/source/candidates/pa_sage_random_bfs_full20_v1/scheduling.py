"""Do not mistake a failed-predecessor watcher for an executing experiment."""
import os
import subprocess
from pathlib import Path
from .common import ROOT,read
REASON='UKL/CL service unsuccessful or unknown; IG remains blocked'

def idle_failed_watcher(plan,state,pid,args,children,reason):
    return (state.get('stage')=='waiting' and state.get('reason')==REASON and reason==REASON
            and pid>0 and state.get('pid')==pid and not children.strip()
            and args==plan.get('command') and state.get('output')==plan.get('output'))

def exclude_self(busy,unit,main_pid,pid):
    if not unit.startswith('digit-pa-random-bfs-full20-') or main_pid!=pid or pid<=0:return busy
    return [item for item in busy if item.split(':',1)[0]!=unit]

def blockers():
    from candidates.uks_pa_fast_v1.start import blockers as previous
    busy=previous()
    # The inherited broad digit-pa* query also lists this controller itself.
    cgroup=next((s[3:] for s in Path('/proc/self/cgroup').read_text().splitlines() if s.startswith('0::')), '')
    own=Path(cgroup).name
    if own.startswith('digit-pa-random-bfs-full20-'):
        main=int(subprocess.check_output(['systemctl','show',own,'-p','MainPID','--value'],text=True,timeout=15).strip())
        busy=exclude_self(busy,own,main,os.getpid())
    latest=ROOT/'results/ig_after_cl_v1/latest.json'
    if not latest.exists():return busy
    try:
        plan=read(latest);unit=plan['unit'].removesuffix('.service')+'.service' if hasattr(str,'removesuffix') else plan['unit'].replace('.service','')+'.service'
        if unit not in busy:return busy
        folder=Path(plan['output']).resolve()
        if folder.parent!=ROOT/'results/ig_after_cl_v1/runs':return busy
        pid=int(subprocess.check_output(['systemctl','show',unit,'-p','MainPID','--value'],text=True,timeout=15).strip())
        if pid<=0:return busy
        args=Path('/proc/%d/cmdline'%pid).read_bytes().rstrip(b'\0').decode().split('\0')
        children=Path('/proc/%d/task/%d/children'%(pid,pid)).read_text()
        from candidates.ig_after_cl_v1.start import current_reason
        if idle_failed_watcher(plan,read(folder/'status.json'),pid,args,children,current_reason(plan['dependency'])):
            busy.remove(unit)
    except (OSError,ValueError,KeyError,subprocess.SubprocessError):
        pass  # Unknown state keeps the conservative blocker.
    return busy
