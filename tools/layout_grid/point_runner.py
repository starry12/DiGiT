"""One independently bound layout: smoke gate, then a fresh one-epoch worker."""
import argparse,fcntl,signal,subprocess,sys,os,time
sys.path.insert(0,'/home/embed/digit')
from candidates.pa_sage_layout_shared_resume_v3.common import *
from ae.common import check_device,check_payload
from ae.pa_sage.monitor_control import ExternalMonitor
from ae.pa_sage.monitor_validation import monitor_evidence
from candidates.pa_sage_layout_shared_resume_v3.validation import report_check,point_summary,mode_pair_check

def identity(path):
    st=Path(path).stat();return dict(device=st.st_dev,inode=st.st_ino,bytes=st.st_size,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns)
from candidates.pa_sage_layout_shared_resume_v3.binding import input_binding

def worker_modes(p):return ('smoke','full') if p['verification_policy']['independent_native_smoke'] else ('full',)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'results/pa_sage_layout_shared_resume_20260925_v3/native'/cfg()['point']['id']);a=parser.parse_args()
    require(__debug__ and os.geteuid()==0,'Native controller requires root without Python optimization')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='2','Only physical GPU 2 is authorized')
    require(str(os.getppid())==os.environ.get('DIGIT_LAYOUT_CONTROLLER_PID'),'Grid controller must own the shared locks')
    execution=verify();a.output.mkdir(parents=True,exist_ok=False);state=dict(schema='digit-pa-sage-layout-point-run-v1',point=cfg()['point'],passed=False,complete=False,pid=os.getpid(),candidate_sha256=execution,stage='waiting_preparation',workers=[],started_unix=time.time(),raw_ssd_writes=False)
    monitor=None;child=None
    def save(**kw):state.update(kw,updated_unix=time.time());write(a.output/'status.json',state)
    def stop(signum,frame):raise KeyboardInterrupt('Controller signaled')
    signal.signal(signal.SIGTERM,stop);save()
    try:
        data=ROOT/cfg()['data']
        require((data/'prepared.json').exists(),'Prepared data missing; no automatic preprocessing')
        save(stage='binding_inputs');binding=input_binding(a.output,execution)
        for mode in worker_modes(cfg()):
            reports={}
            for arm in ('digit_full',):
                check_device();require(verify()==execution,'Candidate changed');save(stage=mode+'_'+arm)
                folder=a.output/mode/arm;folder.parent.mkdir(exist_ok=True)
                mon_dir=a.output/(mode+'_monitor_'+arm);mon_dir.mkdir()
                monitor=ExternalMonitor(mon_dir/'external_gpu');mon_ready=monitor.start()
                command=[sys.executable,'-u','-m','candidates.pa_sage_layout_shared_resume_v3.worker','--arm',arm,'--output',str(folder),'--binding',str(a.output/'inputs.json')]
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
            if mode=='full' and cfg()['verification_policy']['independent_native_smoke']:mode_pair_check(read(a.output/'smoke_digit_full_accepted.json'),reports['digit_full'])
            pair=point_summary(reports,mode=='smoke');pair.update(candidate_sha256=execution,report_sha256={arm:sha(a.output/(mode+'_'+arm+'_accepted.json')) for arm in reports})
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
