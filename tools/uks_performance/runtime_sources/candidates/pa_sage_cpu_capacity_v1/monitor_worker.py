"""Standalone nvidia-smi monitor with parent-death cleanup; stdlib imports only."""
import argparse
import ctypes
import importlib.util
import os
from pathlib import Path
import signal


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--gpu',required=True)
    p.add_argument('--parent',required=True,type=int);a=p.parse_args()
    if os.getppid()!=a.parent:raise RuntimeError('Monitor owner exited')
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(1,signal.SIGTERM,0,0,0)!=0:raise OSError(ctypes.get_errno(),'Monitor parent-death setup failed')
    if os.getppid()!=a.parent:raise RuntimeError('Monitor owner changed')
    source=Path(__file__).resolve().parents[2]/'ae/pa_sage/gpu_monitor.py'
    spec=importlib.util.spec_from_file_location('_cache_nvidia_smi_monitor',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.main(a.output,a.gpu,.5)


if __name__=='__main__':main()
