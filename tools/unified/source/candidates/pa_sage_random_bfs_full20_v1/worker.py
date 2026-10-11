import argparse
from .common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['gids','digit']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--fixtures',action='store_true');p.add_argument('--lock-fds',required=True);a=p.parse_args()
    heavy_gate();source=verify();validate_inherited_locks(a.lock_fds)
    require(a.output.resolve().parent.parent==OUT/'runs' and not a.output.exists(),'Unexpected/existing output')
    a.output.mkdir();setup()
    if a.fixtures:
        from .checks import gpu_check
        result=gpu_check()
    else:
        require(a.arm is not None,'Missing arm');prepared()
        from .engine import run
        result=run(a.arm,a.output,a.smoke);prepared()
        from .dataset import check_report_roots
        check_report_roots(result,a.arm)
        if not a.smoke:
            from .dataset import validate_epochs
            validate_epochs(result)
    require(verify()==source,'Source changed during worker')
    write(a.output/'report.json',dict(result,source_sha256=source))
    write(a.output/'progress.json',dict(stage='complete',passed=True))

if __name__=='__main__':main()
