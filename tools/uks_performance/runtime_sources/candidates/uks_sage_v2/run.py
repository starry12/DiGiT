"""Nonintrusive UKS source preparation; full native execution is a later version."""
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path
from .common import ROOT,HERE,cfg,write,read,sha,output_path,available_host,source_root,identity,data_root,storage_snapshot,receipt_path
from .inventory import audit,capacity
from .safety import Busy,exclusive_idle,ensure_ig_closed,active_ig
from .freeze import verify

def csc_command():
    data=data_root()
    return [sys.executable,str(ROOT/'evaluation/digit/normalized_csc.py'),'--edges',str(source_root()/'edge_index.npy'),'--features',str(data/'synthetic/node_feat.npy'),'--workspace',str(data/'csc'),'--chunk-edges','262144','--fan-in','16','--memory-mib','256','--address-space-mib','2048']
def planning():
    p=cfg();return dict(schema='digit-uks-sage-plan-v1',candidate_sha256=verify(),protocol_sha256=sha(HERE/'protocol.json'),protocol=p,capacity=capacity(),storage=storage_snapshot(),ig=active_ig(),
        actions_ready=['audit (header/stat/tiny samples)','check (small CPU fixtures only)','prepare-source (guarded bulk synthetic files)','prepare-csc (guarded full source/hash/external sort)'],
        csc_command=csc_command(),remaining=['Full source scan and actual normalized CSC/edge count','Independent cache profile/selection and g2 layout/payload generation','256D native I/O/sampler/model admission and correct effective-I/O counting','Non-overlapping SSD range plan and actual preparation/readback','Native short correctness pair then 20 warmup + 3x30 measured training batches per arm'],
        ready_to_train=False,raw_ssd_access=False,bulk_operations_started=False,concurrent_ig_policy='Only small code/metadata/fixture work now; guarded bulk stages refuse while native experiment locks or IG processes are active')

def bulk(action,output):
    output=output_path(output);output.mkdir(parents=True);state=dict(schema='digit-uks-stage-v1',stage='guard',passed=False,complete=False,candidate_sha256=verify(),protocol_sha256=sha(HERE/'protocol.json'),action=action,raw_ssd_access=False,started_unix=time.time(),bulk_started=False,data_root=str(data_root()))
    def mark(**kw):state.update(kw,updated_unix=time.time());write(output/'status.json',state)
    mark()
    try:
        # Keep both locks for the entire stage, including any child build process.
        with exclusive_idle() as lock_fds:
            state['ig_at_admission']=ensure_ig_closed()
            if available_host()<cfg()['host_required_bytes']:raise Busy('Need 320 GiB MemAvailable before bulk work')
            if storage_snapshot()['free_bytes']<cfg()['heavy_stage_free_disk_bytes']:raise Busy('Need 600 GiB free on selected data filesystem before bulk work')
            if 'TMUX' not in os.environ:raise Busy('Bulk work requires a tmux terminal')
            source=audit();write(output/'source_audit.json',source);data=data_root();data.mkdir(parents=True,exist_ok=True)
            if action=='prepare-source':
                from .synthetic import generate
                last=[0.0]
                def notify(**kw):
                    if time.monotonic()-last[0]>10 or kw['rows_done']==kw['total_rows']:mark(**kw);last[0]=time.monotonic()
                mark(stage='synthetic_sources',bulk_started=True)
                generate(data/'synthetic',cfg(),notify)
                write(data/'source_audit.json',source)
                receipt=data/'synthetic/ready.json'
            else:
                ready=read(data/'synthetic/ready.json')
                if not ready['passed'] or ready['protocol']!=cfg():raise RuntimeError('Synthetic data contract differs')
                for name,item in ready['files'].items():
                    if identity(data/'synthetic'/name)!=item['identity']:raise RuntimeError('Prepared source identity changed: '+name)
                if read(data/'source_audit.json')['edges']['identity']!=source['edges']['identity']:raise RuntimeError('Original graph source changed')
                mark(stage='normalized_csc',bulk_started=True,command=csc_command())
                with (output/'csc.log').open('x') as log:
                    result=subprocess.run(csc_command(),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,pass_fds=lock_fds,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
                if result.returncode:raise RuntimeError('CSC build failed; inspect retained csc.log/workspace')
                receipt=data/'csc/graph_source.json';state['graph']=read(receipt)
                if state['graph']['num_nodes']!=cfg()['nodes']:raise RuntimeError('Normalized graph size differs')
            if identity(source_root()/'edge_index.npy')!=source['edges']['identity']:raise RuntimeError('Source changed during preparation')
            if verify()!=state['candidate_sha256']:raise RuntimeError('UKS preparation code changed')
            mark(stage='complete',passed=True,complete=True,receipt=receipt_path(receipt),receipt_sha256=sha(receipt),ready_to_train=False,finished_unix=time.time())
            return 0
    except Busy as exc:mark(stage='deferred',reason=str(exc),ready_to_train=False);print(str(exc),file=sys.stderr);return 75
    except BaseException as exc:mark(stage='failed',error=type(exc).__name__+': '+str(exc),ready_to_train=False);raise

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['plan','audit','check','prepare-source','prepare-csc']);parser.add_argument('--output',type=Path);parser.add_argument('--dry-run',action='store_true');a=parser.parse_args();verify()
    if a.action=='plan' or a.dry_run:print(json.dumps(planning(),indent=2));return 0
    if not a.output:parser.error('--output required')
    if a.action.startswith('prepare-'):return bulk(a.action,a.output)
    out=output_path(a.output);out.mkdir(parents=True)
    if a.action=='audit':
        value=planning();value['source_audit']=audit();value['project_free_bytes']=shutil.disk_usage(ROOT).free;write(out/'audit.json',value)
        write(out/'status.json',dict(passed=True,complete=True,stage='bounded_inventory_complete',candidate_sha256=verify(),bulk_scan=False,native_ready=False));return 0
    with (out/'tests.log').open('x') as log:ret=subprocess.run([sys.executable,'-m','candidates.uks_sage_v2.tests'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    write(out/'status.json',dict(passed=ret.returncode==0,complete=ret.returncode==0,stage='small_fixture_checks',exit_code=ret.returncode,candidate_sha256=verify(),native_ready=False,bulk_scan=False,raw_ssd_access=False))
    return ret.returncode
if __name__=='__main__':
    try:sys.exit(main())
    except (ValueError,RuntimeError,OSError,KeyError) as exc:print('ERROR: '+str(exc),file=sys.stderr);sys.exit(1)
