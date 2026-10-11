"""Reserve a status directory once; give the experiment a fresh child directory."""
import argparse
from contextlib import contextmanager,ExitStack
import fcntl
import os
from pathlib import Path
import signal
import time
from ae.common import write
from submission.v2.common import output_path
from .common import ROOT,verify


def run(output,launcher_sha,admit,experiment,wait=False,sleep=time.sleep):
    output=output_path(output)
    output.mkdir(parents=True,exist_ok=False)
    child=output/'experiment'
    base=dict(pid=os.getpid(),output=str(output),experiment_output=str(child),
              launcher_sha256=launcher_sha,started_unix=time.time(),poll_seconds=60)
    def save(**fields):
        write(output/'status.json',dict(base,updated_unix=time.time(),**fields))
    save(stage='queued',passed=False,complete=False)
    try:
        while True:
            with admit() as blocked:
                if blocked is None:
                    save(stage='running',passed=False,complete=False)
                    # Do not mkdir(child): the frozen experiment validates that
                    # its output does not exist, then creates it itself.
                    result=experiment(child)
                    save(stage=result['stage'],passed=result['passed'],complete=result['complete'],
                         experiment_result=result,finished_unix=time.time())
                    return result
                if not wait:raise RuntimeError(blocked)
                save(stage='waiting',reason=blocked,passed=False,complete=False)
            sleep(60)
    except BaseException as exc:
        save(stage='failed',passed=False,complete=False,error=type(exc).__name__+': '+str(exc),
             finished_unix=time.time())
        raise


@contextmanager
def admission():
    from candidates.uks_group_first_cpu2_v1.resources import target_gpu_busy
    from candidates.ig_demand_pair_v1.common import host,cfg
    with ExitStack() as stack:
        reason=None
        try:
            for path in ('/tmp/digit-pa-bidir-controller.lock','/tmp/digit-pa-sage-experiment.lock',
                         str(ROOT/'ssd_state/libnvm0.prepare.lock'),'/run/digit-ae-selfservice/exclusive.lock'):
                lock=stack.enter_context(open(path,'r+'))
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:reason='shared experiment lock busy'
        if reason is None and target_gpu_busy():reason='GPU2 occupied'
        if reason is None and host()<cfg()['host_required_bytes']:reason='host memory admission'
        yield reason


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wait',action='store_true');a=p.parse_args()
    if os.geteuid()!=0 or not __debug__:raise RuntimeError('Use local sudo with assertions enabled')
    from candidates.ig_demand_pair_v1.common import environment
    os.environ.update(environment(2))
    from candidates.ig_demand_pair_v1.gpu_check import require_accepted
    from candidates.ig_demand_pair_v1.controller import execute
    source=verify();require_accepted()
    def stopped(signum,frame):raise InterruptedError('Controller signal '+str(signum))
    signal.signal(signal.SIGTERM,stopped)
    run(a.output,source,admission,lambda child:execute('representative',child,2,'sage'),wait=a.wait)

if __name__=='__main__':main()
