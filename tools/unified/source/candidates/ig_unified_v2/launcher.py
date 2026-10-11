import argparse,os,signal,subprocess,sys
from pathlib import Path
from contextlib import contextmanager
from .common import verify,environment,host
from candidates.ig_demand_pair_v1.launcher import run,admission as prior_admission
from .admission import estimate
@contextmanager
def admission():
    with prior_admission() as reason:
        if reason is None and host()<estimate()['host_required_bytes']:reason='IG unified host memory budget'
        yield reason
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['short1','short5','full20'],default='short1');a=p.parse_args()
    if os.geteuid()!=0 or not __debug__:raise RuntimeError('Use local sudo')
    os.environ.update(environment(2))
    from .controller import execute
    def stopped(signum,frame):raise InterruptedError('IG unified controller signaled')
    signal.signal(signal.SIGTERM,stopped)
    def experiment(child):
        # CUDA fixture runs in a fresh child and exits before a native worker.
        subprocess.run([sys.executable,'-B','-m','candidates.ig_unified_v2.gpu_check',
                        '--output',str(a.output/'gpu_check.json')],check=True)
        return execute(a.mode,child)
    run(a.output,verify(),admission,experiment,wait=True)
if __name__=='__main__':main()
