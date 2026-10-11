"""Plan/source/CPU entry only; no background workload or device initialization."""
import argparse
import json
from pathlib import Path
from .common import OUT,heavy_gate,write_new,verify
from .protocol import compile_plan,budget
from .decoder import plan as decoder_plan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('plan','cpu-check','preview','verify','run'),nargs='?',default='preview')
    p.add_argument('--output',type=Path);p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.execute:
        heavy_gate()
        raise ValueError('UKL native execution awaits decoder receipts, scalable grouping, sampler kernels, budgets and acceptance')
    if a.action=='cpu-check':
        from .tests import run_checks
        run_checks(a.output or OUT/'cpu_checks')
    elif a.action=='verify':print(verify())
    elif a.action=='plan':
        out=a.output or OUT/'plan';out.mkdir(parents=True,exist_ok=False)
        write_new(out/'protocol.json',compile_plan());write_new(out/'budget.json',budget());write_new(out/'decoder.json',decoder_plan())
        print('Small-file plan only; no data download, decoding, compilation or native execution.')
    else:print(json.dumps(dict(protocol=compile_plan(),budget=budget(),decoder=decoder_plan()),indent=2))


if __name__=='__main__':main()
