"""Linux parent-death cleanup for this controller's children only."""
import ctypes
import os
import signal


def bind_parent(parent_pid):
    if os.getppid()!=parent_pid:raise RuntimeError('Owning controller already exited')
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(1,signal.SIGTERM,0,0,0)!=0:
        raise OSError(ctypes.get_errno(),'PR_SET_PDEATHSIG failed')
    if os.getppid()!=parent_pid:raise RuntimeError('Owning controller exited during startup')
