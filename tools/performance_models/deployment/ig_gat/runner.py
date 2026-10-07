"""Fixed AE transport for the immutable five-pair IG controller; no training changes."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import model_context
model_context.activate()
import sys
sys.path.insert(0,'/srv/digit-ae/admin/identity_v1')
import stable_identity
_identity_registry=stable_identity.install('IG')
import argparse,fcntl,json,os,signal,subprocess,sys,time,uuid,hashlib
from pathlib import Path
import contextlib
sys.path.insert(0,'/srv/digit-ae/admin/gpu_selection_v1')
import gpu_selection as auto_gpu

ROOT=Path('/home/embed/digit');CONTROL=Path('/srv/digit-ae/admin/multimodel_v2/ig_gat')
OUT=Path('/srv/digit-ae/ig-gat-performance-results');PY='/srv/digit-ae/env/bin/python'
sys.path.insert(0,str(ROOT))
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def require(v,msg):
    if not v:raise RuntimeError(msg)
def identities():
    require(sha(ROOT/'snapshot_identity.json')==sha(CONTROL/'snapshot_identity.json'),'Wrong private namespace')
    for n,d in read(CONTROL/'snapshot_manifest.json')['files'].items():
        require(not (ROOT/n).is_symlink() and sha(ROOT/n)==d,'Snapshot drift: '+n)
    from candidates.ig_sage_pair_max5_v1.common import verify
    return dict(model_identity=model_context.verify(),snapshot_sha256=sha(CONTROL/'snapshot_manifest.json'),candidate_sha256=verify(),python=PY,gpu_transport_sha256=auto_gpu.transport())
def controller():
    sys.path.insert(0,str(CONTROL))
    import ig_controller as c
    from candidates.ig_sage_host_telemetry_v1 import telemetry_review
    import ig_telemetry_review
    telemetry_review.review=ig_telemetry_review.review
    c.PY=PY # Fixed trusted interpreter only; source/model/monitor/protocol remain identical.
    return c

def worker(folder):
    require(folder.resolve().parent==OUT/'native' and os.geteuid()==0 and os.environ.get('TMUX'),'Invalid worker context')
    receipt=dict(passed=False,complete=False);child=None;code=1
    try:
        require(identities()==read(folder/'identity.json'),'Launch identity changed')
        with (folder/'native.log').open('x') as log:
            child=subprocess.Popen([PY,'-I','-B','-u',str(CONTROL/'runner.py'),'--native',str(folder)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            code=child.wait()
        require(code==0,'Native controller failed')
        result=controller().review(folder/'experiment')
        require(result==read(folder/'experiment/summary.json'),'Summary rebuild differs')
        require(identities()==read(folder/'identity.json'),'Completion identity changed')
        write(folder/'completion_review.json',result)
        receipt.update(passed=True,complete=True,diagnostic_complete=result['diagnostic_complete']);code=0
    except BaseException as e:receipt['error']=type(e).__name__+': '+str(e);code=1
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=60)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        write(folder/'worker_exit.json',dict(receipt,returncode=code))
    return code

def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group();g.add_argument('--worker',type=Path);g.add_argument('--native',type=Path);g.add_argument('--selftest',action='store_true');a=p.parse_args()
    require(os.geteuid()==0 and __debug__,'Fixed root service required')
    from candidates.ig_sage_host_telemetry_v1 import start as inherited
    signal.signal(signal.SIGTERM,inherited.stop);signal.signal(signal.SIGINT,inherited.stop)
    if a.worker:return worker(a.worker)
    if a.native:
        require(a.native.resolve().parent==OUT/'native','Invalid native output')
        sys.argv=[sys.argv[0],'--output',str(a.native/'experiment')];controller().main();return 0
    with open('/run/digit-ae-selfservice/exclusive.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.selftest:
            identities()
            subprocess.run([PY,'-I','-B',str(CONTROL/'check_imports.py')],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True,timeout=300)
            write(CONTROL/'state/selftest.json',dict(passed=True,complete=True,native_acceptance=False));return 0
        folder=OUT/'native'/(time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:12])
        write(CONTROL/'state/latest.json',dict(output=str(folder),invocation_id=os.environ.get('INVOCATION_ID')))
        inherited.OUT=OUT;inherited.PY=PY;inherited.SCRIPT=CONTROL/'runner.py';inherited.identities=identities
        original_environment=inherited.environment
        inherited.environment=lambda gpu=None:original_environment(auto_gpu.assignment()['index'])
        inherited.admission=auto_gpu.ig_admission
        with contextlib.ExitStack() as gpu_stack:
            try:
                selection=auto_gpu.activate(gpu_stack)
            except BaseException as exc:
                folder.mkdir(parents=True)
                write(folder/'status.json',dict(passed=False,complete=False,stage='failed',error=str(exc)))
                raise
            write(CONTROL/'state/latest.json',dict(output=str(folder),invocation_id=os.environ.get('INVOCATION_ID'),gpu_selection=selection))
            rc=inherited.supervise(folder)
            write(folder/'gpu_selection.json',selection)
        if rc==0:
            result=read(folder/'completion_review.json')
            display=dict(passed=True,complete=True,speedup=result['statistics']['max_observed_speedup'],
                selection_rule='maximum_same_round_speedup_out_of_five',native_acceptance=True,
                invocation_id=os.environ.get('INVOCATION_ID'),summary_sha256=sha(folder/'experiment/summary.json'),
                review_sha256=sha(folder/'completion_review.json'))
            write(folder/'result.json',display)
            (folder/'README.md').write_text('# IG/GAT\n\nFinal speedup: **%.2f×**, maximum observed paired speedup across five rounds.\n'%display['speedup'])
        return rc
if __name__=='__main__':sys.exit(main())
