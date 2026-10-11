"""Temporarily borrow the exclusion provided by a frozen UKL preparation unit."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from candidates.uks_window_diag_v1.common import LOCKS,require,write
UNIT='digit-group-reuse-ukl-migration.service'

def unit_state():
    text=subprocess.check_output(['systemctl','show',UNIT,'-p','MainPID','-p','ControlGroup','-p','ActiveState','-p','FreezerState'],text=True,timeout=20)
    return dict(line.split('=',1) for line in text.splitlines())

def identity(s):
    pid=int(s['MainPID']);require(pid>1 and s['ActiveState']=='active','UKL preparation is not active')
    base=Path('/proc')/str(pid)
    argv=(base/'cmdline').read_bytes().split(b'\0')
    require(b'candidates.group_cache_migration_v1.prepare' in argv and b'UKL' in argv and b'--execute' in argv,'Unexpected UKL process')
    require((base/'cwd').resolve()==ROOT,'Unexpected UKL cwd')
    require(base.stat().st_uid==0,'Unexpected UKL owner')
    require(s['ControlGroup']=='/system.slice/'+UNIT,'Unexpected cgroup')
    return dict(pid=pid,start_ticks=(base/'stat').read_text().rsplit(')',1)[1].split()[19],cgroup=s['ControlGroup'])

def locks_held(pid):
    rows=Path('/proc/locks').read_text().splitlines()
    for path in LOCKS:
        p=Path(path);require(p.is_file() and not p.is_symlink(),'Unsafe lock path')
        st=p.stat();matched=False
        for line in rows:
            f=line.split()
            if len(f)<8 or f[1:4]!=['FLOCK','ADVISORY','WRITE'] or f[4]!=str(pid):continue
            major,minor,inode=f[5].split(':')
            if (int(major,16),int(minor,16),int(inode))==(os.major(st.st_dev),os.minor(st.st_dev),st.st_ino):matched=True
        require(matched,'UKL no longer owns shared lock: '+path)

def validate(path):
    lease=json.loads(Path(path).read_text());s=unit_state()
    require(lease['owned'] and not lease.get('restored',False),'Inactive pause lease')
    require(identity(s)==lease['identity'],'UKL identity changed')
    require(s['FreezerState']=='frozen','UKL is no longer frozen')
    cg=Path('/sys/fs/cgroup')/s['ControlGroup'].lstrip('/')
    require('frozen 1' in (cg/'cgroup.events').read_text(),'Cgroup not completely frozen')
    locks_held(lease['identity']['pid'])
    return lease

def begin(path):
    s=unit_state();ident=identity(s)
    require(s['FreezerState']=='running','Refuse to take over an existing pause')
    state=json.loads((ROOT/'results/group_cache_migration_v1/UKL/preparation_status.json').read_text())
    require(state['stage']=='validate','Only pause the existing CPU validation phase')
    locks_held(ident['pid'])
    cg=Path('/sys/fs/cgroup')/ident['cgroup'].lstrip('/')
    for pid in (cg/'cgroup.procs').read_text().split():
        if int(pid)==ident['pid']:continue
        cmd=(Path('/proc')/pid/'cmdline').read_bytes().split(b'\0')
        require(cmd[:2]==[str(ROOT/'results/group_cache_migration_v1/build/reuse').encode(),b'validate'],'Unexpected UKL child')
    active=subprocess.check_output(['nvidia-smi','-i','2','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=20).strip()
    require(not active,'GPU2 has an active compute process')
    # Persist recovery ownership before freezing: ExecStopPost can recover even
    # if the controller is killed between freeze and the verification below.
    write(path,dict(owned=True,restored=False,identity=ident,unit=UNIT,started_unix=time.time()))
    subprocess.run(['systemctl','freeze',UNIT],check=True,timeout=30)
    return validate(path)

def restore(path):
    path=Path(path)
    if not path.exists():return
    lease=json.loads(path.read_text())
    if not lease['owned'] or lease.get('restored'):return
    s=unit_state();require(identity(s)==lease['identity'],'Refuse to thaw a different UKL process')
    subprocess.run(['systemctl','thaw',UNIT],check=True,timeout=30)
    require(unit_state()['FreezerState']=='running','UKL resume failed')
    lease.update(restored=True,restored_unix=time.time());write(path,lease)

if __name__=='__main__':
    require(os.geteuid()==0 and len(sys.argv)==3 and sys.argv[1]=='--restore','Recovery invocation required')
    restore(Path(sys.argv[2]))
