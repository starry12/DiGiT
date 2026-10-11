"""Tiny GPU ABI/atomic/merge check; raw NVMe is denied by the parent unit."""
import argparse,hashlib,importlib.util,json,os
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=('gids','digit'),required=True);p.add_argument('--out',required=True);a=p.parse_args()
    if os.environ.get('CL_MICRO_WORKER')!='1':raise RuntimeError('Use bounded CL micro controller')
    from .build import BINARY as compact
    from candidates.ukl_mixed512_v21.build import BINARY as original
    binary=original if a.arm=='gids' else compact
    spec=importlib.util.spec_from_file_location('BAM_Feature_Store',str(binary));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    abi=list(module.ukl_layout_abi())
    if abi!=([2,32,512,512] if a.arm=='gids' else [3,24,512,512]):raise RuntimeError('Native ABI differs')
    from .gpu_micro import run
    result=run(module)
    resident=list(module.full_resident_probe_test())
    result.update(arm=a.arm,abi=abi,resident=resident,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),raw_ssd_access=False)
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
