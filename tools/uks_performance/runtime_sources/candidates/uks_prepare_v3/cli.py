"""Source/CPU preparation only. No automatic workload launch."""
import argparse,json
from pathlib import Path
from .common import OUT,HERE,LARGE,GRID,read,write_new,heavy_gate,require,verify
from .protocol import compile_plan


def plan():
    p=compile_plan();n,e=p['nodes'],p['source_edges'];width=1024
    return dict(protocol=p,budget=dict(formal_admission=False,host_required_bytes=None,gpu_required_bytes=None,
        original_feature_bytes=n*width,cpu_cache_feature_bytes=p['cpu_cache_rows']*width,
        original_csc_plus_eids_upper_bytes=8*(n+1+2*(e+n)),
        remaining='Actual group count, padding, row maps, model/sampler peak and workspaces must be bound before native admission'),
        native_ready=False,automatic_start=False,large_output_root=str(LARGE))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('plan','cpu-check','native-preview','run','verify'),nargs='?',default='native-preview')
    parser.add_argument('--output',type=Path);parser.add_argument('--execute',action='store_true');a=parser.parse_args()
    if a.execute:
        heavy_gate() # before output creation, framework import or native command
        raise ValueError('UKS source adapter only: bind large data/profile/budgets and native controller before execution')
    if a.action=='plan':
        out=a.output or OUT/'plan';out.mkdir(parents=True,exist_ok=False);value=plan()
        write_new(out/'protocol.json',value.pop('protocol'));write_new(out/'readiness.json',value)
        print('UKS plan only; no large generation, compiler, CUDA/SSD or background service.')
    elif a.action=='cpu-check':
        import unittest
        from . import test_prepare
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_prepare))
        require(result.wasSuccessful(),'Preparation checks failed')
    elif a.action=='verify':print(verify())
    else:print(json.dumps(plan(),indent=2))


if __name__=='__main__':main()
