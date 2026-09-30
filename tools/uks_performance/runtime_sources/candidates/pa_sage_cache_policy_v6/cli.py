"""Preview or explicitly run the revised cache-policy comparison."""
import argparse
import json
from pathlib import Path
from .common import OUT,read,verify,heavy_gate
from .protocol import compile_plan,validate,schedule


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('run','verify'),nargs='?',default='run')
    p.add_argument('--protocol',type=Path,default=OUT/'plan/protocol.json')
    p.add_argument('--output',type=Path,default=OUT/'native')
    p.add_argument('--execute',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if a.action=='verify':print(verify());return
    if a.resume and not a.execute:p.error('--resume requires --execute')
    if a.execute:
        heavy_gate()
        from .controller import run
        run(a.protocol.resolve(),a.output.resolve(),a.resume)
    else:
        config=validate(read(a.protocol)) if a.protocol.exists() else compile_plan()
        print(json.dumps(dict(execute=False,continuous_gpu_monitoring=False,
            startup_gpu_admission=True,schedule=schedule(config)),indent=2))


if __name__=='__main__':main()
