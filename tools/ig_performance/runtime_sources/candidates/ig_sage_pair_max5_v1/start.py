"""Launch five paired GIDS/DiGiT rounds; display the maximum same-round speedup."""
import argparse
import datetime
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from candidates.ig_sage_pair_max5_v1.common import *
from candidates.ig_sage_host_telemetry_v1 import start as inherited

SCRIPT=Path(__file__).resolve()


def identities():
    identity=verify()
    parent_out=ROOT/'results/ig_sage_host_telemetry_20260927_v1'
    binding=read(parent_out/'inputs.json')
    require(binding['candidate_sha256']==PARENT_SHA,'Wrong parent input binding')
    check_inputs(binding)
    pre=ROOT/'results/ig_perf_preflight_20260922_v5/check'
    receipt=read(pre/'status.json')
    require(receipt['passed'] and receipt['complete'],'Inherited native preflight incomplete')
    for name,digest in receipt['evidence_sha256'].items():require(sha(pre/name)==digest,'Native preflight evidence changed')
    return dict(wrapper_sha256=identity,native_parent_sha256=PARENT_SHA,
        native_inputs_sha256=sha(parent_out/'inputs.json'),inherited_preflight_sha256=sha(pre/'status.json'))


def worker(folder):
    require(os.geteuid()==0 and os.environ.get('TMUX'),'Native worker requires sudo/tmux')
    require(folder.resolve().parent==OUT/'native','Unexpected output directory')
    child=None;code=1;receipt=dict(passed=False,complete=False,started_unix=time.time())
    try:
        require(read(folder/'identity.json')==identities(),'Source changed after launch')
        command=[PY,'-B','-u','-m','candidates.ig_sage_pair_max5_v1.controller','--output',str(folder/'experiment')]
        write(folder/'experiment_command.json',command)
        with (folder/'native.log').open('x') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=environment(2),stdout=log,stderr=subprocess.STDOUT)
            receipt['controller_pid']=child.pid;write(folder/'worker_status.json',receipt)
            code=child.wait()
        require(code==0,'Native controller failed; inspect native.log and experiment/status.json')
        from candidates.ig_sage_pair_max5_v1.controller import review
        result=review(folder/'experiment')
        require(result==read(folder/'experiment/summary.json'),'Independent summary rebuild differs')
        require(identities()==read(folder/'identity.json'),'Source changed during repetitions')
        write(folder/'completion_review.json',result)
        receipt.update(passed=True,complete=True,diagnostic_complete=result['diagnostic_complete'],selected_round=result['statistics']['selected_round'],max_observed_speedup=result['statistics']['max_observed_speedup']);code=0
    except BaseException as exc:
        receipt['error']=type(exc).__name__+': '+str(exc);code=130 if isinstance(exc,KeyboardInterrupt) else 1
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=60)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        write(folder/'worker_exit.json',dict(receipt,returncode=code,finished_unix=time.time()))
    return code


def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group()
    g.add_argument('--execute',action='store_true');g.add_argument('--supervise',type=Path);g.add_argument('--worker',type=Path)
    a=p.parse_args();signal.signal(signal.SIGTERM,inherited.stop);signal.signal(signal.SIGINT,inherited.stop)
    if a.worker:return worker(a.worker)
    if a.supervise:
        # Reuse validated detached lifecycle/admission and unique tmux socket.
        inherited.OUT=OUT;inherited.SCRIPT=SCRIPT;inherited.identities=identities
        return inherited.supervise(a.supervise)
    identity=identities();stamp=datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+str(os.getpid())
    folder=OUT/'native'/stamp;unit='digit-ig-sage-pair-max5-20260928-v1-'+stamp
    command=['/usr/bin/systemd-run','--unit='+unit,'--property=Type=exec','--property=WorkingDirectory='+str(ROOT),
        '--property=UMask=0022','--property=Restart=no','--property=KillMode=control-group','--property=TimeoutStopSec=90',
        '--property=LimitMEMLOCK=infinity','--setenv=CUDA_VISIBLE_DEVICES=2','--setenv=PYTHONDONTWRITEBYTECODE=1',
        '--setenv=LD_LIBRARY_PATH='+str(ROOT/'bam/build/lib'),PY,'-B','-u',str(SCRIPT),'--supervise',str(folder)]
    value=dict(command=command,unit=unit+'.service',output=str(folder),identity=identity,execute=a.execute,
        workers=10,repetitions_per_arm=5,paired_rounds=5,arms=['gids','digit_full'],schedule=[j['mode'] for j in read(HERE/'protocol.json')['schedule']],warmup=20,measured_batches=300,perf=False,new_native_smokes=0,automatic_retries=0,
        primary_statistic='max over five rounds of same-round GIDS time / DiGiT time; retain all ten runs',
        monitoring='Original host-stage timing, CPU/GPU telemetry and NVML',interference_free_proven=False,
        gpu=2,cpu_binding={'gids':'unchanged default','digit_full':'original CPU2 binding after imports'},raw_ssd_writes=False,host_admission_gib=320)
    print(__import__('json').dumps(value,indent=2),flush=True)
    if a.execute:
        require(os.geteuid()==0,'Use local sudo for native BaM access')
        subprocess.run(command,check=True);write(OUT/'launch.json',dict(value,started_unix=time.time()))
    return 0


if __name__=='__main__':sys.exit(main())
