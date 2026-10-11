"""Load only pure NumPy artifact modules, without digit's DGL/CUDA __init__."""
import importlib
import sys
import types
from .common import ROOT

NAMESPACE='_digit_layout_sweep_cpu'
RUNTIME=ROOT/'candidates/pa_sage_bidir_native_v2/runtime/digit'

def module(name):
    if NAMESPACE not in sys.modules:
        package=types.ModuleType(NAMESPACE)
        package.__path__=[str(RUNTIME)]
        package.__package__=NAMESPACE
        sys.modules[NAMESPACE]=package
    return importlib.import_module(NAMESPACE+'.'+name)

def artifacts():
    return module('artifacts')
