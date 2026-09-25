"""Preview by default. Native execution is explicit and uses the global lock."""
import argparse,json,os,sys
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path);parser.add_argument('--execute',action='store_true');parser.add_argument('--budget-output',type=Path);parser.add_argument('--bind-only',type=Path);a=parser.parse_args()
    os.environ['DIGIT_LAYOUT_PROTOCOL']=str(a.protocol.resolve())
    from .common import cfg
    p=cfg();base=Path(p['base_layout'])
    if a.bind_only:
        if a.execute or a.budget_output:raise RuntimeError('Read-only binding is a separate preflight')
        from .common import verify
        from .binding import input_binding
        a.bind_only.mkdir(parents=True,exist_ok=False)
        input_binding(a.bind_only,verify());return
    if a.budget_output:
        if a.execute:raise RuntimeError('Budget-only and native execution are separate')
        from .common import verify,write
        from .admission import estimate
        verify();write(a.budget_output,estimate());return
    required=[base/'build_receipt.json',base/'final/bundle/manifest.json',base/'full_cpu_rows.npy',
              Path(p['overlay'])/'overlay_receipt.json']
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
