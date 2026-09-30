"""Controller for two fresh-process, read-only SSD training smoke workers."""
import argparse,json,os,subprocess,sys,time,signal
from pathlib import Path
from candidates.io_accounting_v1.common import HERE,ROOT,sha,verify_release
from ae.common import write,check_device
from ae.pa_sage.monitor_control import ExternalMonitor
from ae.pa_sage.monitor_validation import monitor_evidence
from ae.pa_sage.native_gate import current_binding,validate_reports
from candidates.io_accounting_v1.accounting import validate_region,summarize,require

def read(p):return json.loads(Path(p).read_text())
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--data',type=Path,default=ROOT/'data/papers_g2_random_v2');a=parser.parse_args()
    require(os.geteuid()==0,'Native SSD gate requires local sudo')
    require(os.environ.get('TMUX'),'Run inside tmux')
    execution=verify_release();check_device();a.output=a.output.resolve();a.data=a.data.resolve()
    a.output.mkdir(parents=True,exist_ok=False)
    state=dict(schema='digit-useful-io-gate-v1',pid=os.getpid(),passed=False,complete=False,stage='starting',
               execution_sha256=execution,raw_ssd_writes=False,full_training=False,workers=[],started_unix=time.time())
    monitor=None;child=None;reports={}
    def save():state['updated_unix']=time.time();write(a.output/'status.json',state)
    def stop(signum,frame):raise KeyboardInterrupt('Controller signaled')
    signal.signal(signal.SIGTERM,stop);save()
    try:
        binding=current_binding(a.data);binding['execution_sha256']=execution
        for arm in ('gids','digit_full'):
            check_device();state['stage']=arm
            mon_dir=a.output/('monitor_'+arm);mon_dir.mkdir()
            monitor=ExternalMonitor(mon_dir/'external_gpu');ready=monitor.start()
            command=[sys.executable,'-u','-m','candidates.io_accounting_v1.worker','--arm',arm,
                     '--data',str(a.data),'--output',str(a.output/arm/'report.json')]
            worker=dict(arm=arm,command=command,started_unix=time.time(),status='running')
            state['workers'].append(worker)
            with (a.output/(arm+'.log')).open('x') as log:
                child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                worker['pid']=child.pid;save();released=False;release_deadline=None
                while child.poll() is None:
                    receipt=a.output/arm/'worker_ready.json'
                    if receipt.exists() and not released:
                        require(read(receipt)['passed'],'Worker did not pass')
                        code=monitor.stop();monitor=None
                        write(a.output/arm/'release_worker.json',dict(monitor_stopped=True,monitor_returncode=code))
                        released=True;release_deadline=time.monotonic()+180
                    if release_deadline is not None:require(time.monotonic()<release_deadline,'Worker teardown timed out')
                    time.sleep(.5)
                returncode=child.wait();child=None
            worker.update(returncode=returncode,finished_unix=time.time(),status='complete' if returncode==0 else 'failed');save()
            require(returncode==0 and released,'Native worker failed; inspect '+arm+'.log')
            require(code==0,'External monitor failed')
            path=a.output/arm/'report.json';report=read(path)
            require(sha(path)==read(a.output/arm/'worker_ready.json')['report_sha256'],'Worker report changed')
            per_monitor=dict(pid=os.getpid(),workers=[worker],external_monitor=ready,external_monitor_returncode=code)
            report['external_monitor']=monitor_evidence(mon_dir,per_monitor,arm,read(a.output/arm/'resources.json'))
            for e in report['epochs']:
                validate_region(e['training']);validate_region(e['validation'])
            require(report['io_accounting_training']==summarize([(e['training'],e['train_seconds']) for e in report['epochs']]),'Summary mismatch')
            reports[arm]=report
            # Keep the original worker report unchanged; augmented copy has its own hash.
            write(a.output/(arm+'_accepted.json'),report)
        validate_reports(reports,binding)
        require(verify_release()==execution,'Candidate changed')
        now=current_binding(a.data);now['execution_sha256']=execution
        require(now==binding,'Data or SSD binding changed');check_device()
        summary=dict(passed=True,scope='short native counter gate only; directed graph; no accuracy/performance claim',
                     raw_ssd_writes=False,test_calls=0,execution_sha256=execution,
                     arms={arm:r['io_accounting_training'] for arm,r in reports.items()},
                     report_sha256={arm:sha(a.output/(arm+'_accepted.json')) for arm in reports},binding=binding)
        write(a.output/'summary.json',summary)
        state.update(passed=True,complete=True,stage='complete',finished_unix=time.time());save()
    except BaseException as exc:
        state.update(stage='failed',error=type(exc).__name__+': '+str(exc));save();raise
    finally:
        if monitor is not None:monitor.stop()
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();child.wait()
if __name__=='__main__':main()
