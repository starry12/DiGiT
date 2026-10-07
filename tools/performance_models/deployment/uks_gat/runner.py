"""AE transport for the frozen UKS Freq+BFS controller; no training changes."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import model_context
model_context.activate()
import sys
sys.path.insert(0,'/srv/digit-ae/admin/identity_v1')
import stable_identity
_identity_registry=stable_identity.install('UKS')
import argparse,json,os,subprocess,sys,time,uuid
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from review import read,sha,require,review
ROOT=Path('/home/embed/digit');C=Path('/srv/digit-ae/admin/multimodel_v2/uks_gat')
OUT=Path('/srv/digit-ae/uks-gat-performance-results');PY='/srv/digit-ae/env/bin/python'
sys.path.insert(0,str(ROOT))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def identities():
    require(sha(ROOT/'snapshot_identity.json')==sha(C/'snapshot_identity.json'),'Wrong namespace')
    manifest=read(C/'snapshot_manifest.json')
    for name,digest in dict(manifest['files'],**manifest['live_ssd_states']).items():
        p=ROOT/name;require(not p.is_symlink() and sha(p)==digest,'Snapshot changed: '+name)
    from candidates.uks_freq_bfs_retry_v1.common import verify
    for name,digest in read(C/'transport_manifest.json')['files'].items():
        require(sha(C/name)==digest,'Transport changed: '+name)
    return dict(model_identity=model_context.verify(),snapshot_sha256=sha(C/'snapshot_manifest.json'),source_sha256=verify(),transport_sha256=sha(C/'transport_manifest.json'))
def main():
    p=argparse.ArgumentParser();p.add_argument('--selftest',action='store_true');args=p.parse_args()
    require(os.geteuid()==0 and __debug__,'Fixed root service required')
    identity=identities()
    if args.selftest:
        subprocess.run([PY,'-I','-B',str(C/'check_imports.py')],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True,timeout=300)
        write(C/'state/selftest.json',dict(passed=True,native_acceptance=False,identity=identity));return
    folder=OUT/(time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:12]);folder.mkdir(parents=True)
    invocation=os.environ['INVOCATION_ID'];write(C/'state/latest.json',dict(output=str(folder),invocation_id=invocation))
    write(folder/'identity.json',identity)
    try:
        import controller
        require(Path(controller.__file__).resolve()==C/'controller.py','Wrong controller module')
        controller.PYTHON=PY
        # run() retains original resource locks, admission and worker protocol.
        with (folder/'controller.log').open('w') as log:
            oldout,olderr=os.dup(1),os.dup(2)
            try:
                os.dup2(log.fileno(),1);os.dup2(log.fileno(),2);controller.run(folder/'experiment')
            finally:os.dup2(oldout,1);os.dup2(olderr,2);os.close(oldout);os.close(olderr)
        gpu=read(folder/'experiment/gpu_assignment.json')['selected']
        result=review(folder/'experiment',expected_gpu=gpu,transport_sha=identity['transport_sha256']);require(identities()==identity,'Completion identity drift')
        write(folder/'completion_review.json',result)
        write(folder/'result.json',dict(result,native_acceptance=True,invocation_id=invocation,review_sha256=sha(folder/'completion_review.json')))
    except BaseException as e:write(folder/'error.json',dict(error=repr(e)));raise
if __name__=='__main__':main()
