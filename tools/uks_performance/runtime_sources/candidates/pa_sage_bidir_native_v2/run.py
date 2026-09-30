"""Wait for preparation, accept a short pair, then run fresh 20-epoch workers."""
import argparse,fcntl,signal,subprocess,sys,os,time
from candidates.pa_sage_bidir_native_v2.common import *
from ae.common import check_device,check_payload
from ae.pa_sage.monitor_control import ExternalMonitor
from ae.pa_sage.monitor_validation import monitor_evidence
from candidates.pa_sage_bidir_native_v2.validation import report_check,pair_check

def identity(path):
    st=Path(path).stat();return dict(device=st.st_dev,inode=st.st_ino,bytes=st.st_size,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns)
def input_binding(output,execution):
    p=cfg();data=ROOT/p['data'];base=ROOT/p['base_layout'];ready=read(data/'prepared.json')
    require(ready['passed'] and ready['protocol_sha256']==sha(P) and ready['base_manifest_sha256']==sha(base/'final/bundle/manifest.json'),'Prepared overlay differs')
    for rel,digest in ready['preparation_code_sha256'].items():require(sha(ROOT/rel)==digest,'Preparation code changed')
    files={}
    def add(path,expected=None):
        path=Path(path);before=identity(path);digest=sha(path);require(identity(path)==before,'Input changed while hashing')
        require(expected is None or digest==expected,'Input hash mismatch: '+str(path))
        try:key=str(path.relative_to(ROOT))
        except ValueError:key=str(path)
        files[key]=dict(sha256=digest,identity=before)
    for name,digest in ready['bindings'].items():add(data/name,digest)
    add(data/'prepared.json');old=read(base/'prepared.json')
    for name in ('orders.json','gids_cpu_rows.npy','full_cpu_rows.npy','final/bundle/manifest.json'):add(base/name,old['bindings'][name])
    add(base/'ssd_ready.json');full=read(base/'ssd_ready.json');add(full['state'],full['state_sha256']);add(full['verify_receipt'],full['verify_receipt_sha256'])
    plain=check_payload('papers_gids');add(plain['state'],plain['state_sha256'])
    source=source_config()
    for desc in (source['source_features'],source['label_identity'],source['source_contract']['original_edges']):add(desc['path'],desc['sha256'])
    for name in ('validation_trace','test_trace'):add(ROOT/p[name]/'manifest.json')
    value=dict(candidate_sha256=execution,protocol_sha256=sha(P),prepared_sha256=sha(data/'prepared.json'),files=files)
    write(output/'inputs.json',value);return value

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'results/pa_sage_bidir_native_full_20260921_v2');a=parser.parse_args()
    require(os.geteuid()==0 and os.environ.get('TMUX'),'Run with local sudo inside tmux')
    lock=open('/tmp/digit-pa-bidir-controller.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    execution=verify();a.output.mkdir(parents=True,exist_ok=False);state=dict(schema='digit-bidir-native-run-v1',passed=False,complete=False,pid=os.getpid(),candidate_sha256=execution,stage='waiting_preparation',workers=[],started_unix=time.time(),raw_ssd_writes=False)
    monitor=None;child=None
    def save(**kw):state.update(kw,updated_unix=time.time());write(a.output/'status.json',state)
    def stop(signum,frame):raise KeyboardInterrupt('Controller signaled')
    signal.signal(signal.SIGTERM,stop);save()
    try:
        data=ROOT/cfg()['data'];prep=ROOT/'results/pa_sage_bidir_native_20260921_v1/prepare/status.json'
        while not (data/'prepared.json').exists():
            if prep.exists():require(read(prep)['stage']!='failed','Preparation failed; inspect prepare.log')
            time.sleep(5)
        save(stage='binding_inputs');binding=input_binding(a.output,execution)
        for mode in ('smoke','full'):
            reports={}
            for arm in ('gids','digit_full'):
                check_device();require(verify()==execution,'Candidate changed');save(stage=mode+'_'+arm)
                folder=a.output/mode/arm;folder.parent.mkdir(exist_ok=True)
                mon_dir=a.output/(mode+'_monitor_'+arm);mon_dir.mkdir()
                monitor=ExternalMonitor(mon_dir/'external_gpu');mon_ready=monitor.start()
                command=[sys.executable,'-u','-m','candidates.pa_sage_bidir_native_v2.worker','--arm',arm,'--output',str(folder),'--binding',str(a.output/'inputs.json')]
                if mode=='smoke':command.append('--smoke')
                w=dict(arm=arm,mode=mode,command=command,started_unix=time.time(),status='running');state['workers'].append(w)
                with (a.output/(mode+'_'+arm+'.log')).open('x') as logfile:
                    child=subprocess.Popen(command,cwd=ROOT,stdout=logfile,stderr=subprocess.STDOUT);w['pid']=child.pid;save();released=False;deadline=None
                    while child.poll() is None:
                        receipt=folder/'worker_ready.json'
                        if receipt.exists() and not released:
                            require(read(receipt)['passed'],'Worker acceptance failed');mon_code=monitor.stop();monitor=None
                            write(folder/'release_worker.json',dict(monitor_stopped=True,monitor_returncode=mon_code));released=True;deadline=time.monotonic()+180
                        if deadline is not None:require(time.monotonic()<deadline,'Worker teardown timeout')
                        time.sleep(.5)
                    code=child.wait();child=None
                w.update(returncode=code,finished_unix=time.time(),status='complete' if code==0 else 'failed');save()
                require(code==0 and released,'Worker failed: '+mode+' '+arm);require(mon_code==0,'External monitor failed')
                path=folder/'report.json';require(sha(path)==read(folder/'worker_ready.json')['report_sha256'],'Report changed')
                r=read(path);ms=dict(pid=state['pid'],workers=[w],external_monitor=mon_ready,external_monitor_returncode=mon_code)
                require(read(mon_dir/'external_gpu/summary.json')['errors']==[],'Monitor has errors')
                r['external_monitor']=monitor_evidence(mon_dir,ms,arm,read(folder/'resources.json'))
                report_check(r,arm,mode=='smoke',binding,folder)
                write(a.output/(mode+'_'+arm+'_accepted.json'),r);reports[arm]=r
            pair=pair_check(reports,mode=='smoke');pair.update(candidate_sha256=execution,report_sha256={arm:sha(a.output/(mode+'_'+arm+'_accepted.json')) for arm in reports})
            write(a.output/(mode+'_summary.json'),pair);save(stage=mode+'_accepted')
        require(verify()==execution,'Candidate changed');check_device()
        for path,desc in binding['files'].items():require(identity(ROOT/path)==desc['identity'],'Input changed during experiment')
        save(stage='complete',passed=True,complete=True,finished_unix=time.time())
    except BaseException as exc:save(stage='failed',error=type(exc).__name__+': '+str(exc));raise
    finally:
        if monitor is not None:monitor.stop()
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=60)
            except subprocess.TimeoutExpired:child.kill();child.wait()
if __name__=='__main__':main()
