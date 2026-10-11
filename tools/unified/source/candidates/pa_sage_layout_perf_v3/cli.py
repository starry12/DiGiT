"""Preview by default. Native execution is explicit and uses the global lock."""
import argparse,json,os,sys
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path);parser.add_argument('--execute',action='store_true');a=parser.parse_args()
    os.environ['DIGIT_LAYOUT_PROTOCOL']=str(a.protocol.resolve())
    from .common import cfg
    p=cfg();base=Path(p['base_layout'])
    required=[base/'build_receipt.json',base/'final/bundle/manifest.json',base/'full_cpu_rows.npy',
              Path(p['overlay'])/'overlay_receipt.json',Path(p['ssd_ready'])]
    missing=[str(path) for path in required if not path.is_file()]
    print(json.dumps(dict(point=p['point'],execute=a.execute,missing_inputs=missing,
          native_ready=False,readiness_note='Existence preview only; hashes, SSD receipts and live admission are checked during execution'),indent=2),flush=True)
    if not a.execute:return
    if missing:raise RuntimeError('Point is not prepared; no worker started')
    if os.geteuid()!=0:raise RuntimeError('Native NVMe mapping requires an administrator launch')
    from . import run
    sys.argv=[sys.argv[0]]+(['--output',str(a.output)] if a.output else [])
    run.main()
if __name__=='__main__':main()
