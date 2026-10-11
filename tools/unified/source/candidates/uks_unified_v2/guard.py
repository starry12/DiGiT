import os
import subprocess
from pathlib import Path
from candidates.uks_pa_fast_v1.common import validate_inherited_locks,require


def device_identity():
    s=Path('/dev/libnvm0').stat()
    return s.st_dev,s.st_ino,s.st_rdev


class RuntimeGuard:
    def __init__(self,lock_fds):
        from ae.common import check_device
        self.lock_fds=lock_fds
        validate_inherited_locks(lock_fds)
        before=device_identity()
        check_device()
        require(device_identity()==before,'SSD identity changed during admission')
        self.identity=before

    def check(self):
        validate_inherited_locks(self.lock_fds)
        require(device_identity()==self.identity,'SSD device changed during training')
        p=subprocess.run(['fuser','/dev/libnvm0'],capture_output=True,text=True,timeout=15)
        require(p.returncode in (0,1),'SSD ownership query failed')
        owners={int(v) for v in p.stdout.split()}
        require((p.returncode==0)==bool(owners),'Inconsistent SSD ownership query')
        require(owners<={os.getpid()},'Foreign SSD user during training')
        from candidates.uks_native_v1.binding import check
        check()
