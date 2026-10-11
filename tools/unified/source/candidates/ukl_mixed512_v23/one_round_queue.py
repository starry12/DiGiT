"""Run host -> buffer -> gpu, exactly one accepted pair per variant."""
import argparse,fcntl,hashlib,json,os,subprocess,time
from pathlib import Path
from . import protocol as P
from .series import write

VARIANTS=('host','buffer','gpu')
PREVIOUS=P.ROOT/'results/ukl_mixed512_20261004_v22/stop_v21/request.json'


def stopped_receipt():
    request=json.loads(PREVIOUS.read_text())
    lease=request['controller'];proc=Path('/proc')/str(lease['pid'])/'stat'
    if proc.exists() and proc.read_text().rsplit(')',1)[1].split()[19]==lease['start_ticks']:
        raise RuntimeError('Previous controller is still stopping or checking the system')
    path=Path(request['stage_path'])/'acceptance.json'
    a=json.loads(path.read_text());state=a['worker_state']
    if not (state['MainPID']=='0' and a['gpu_released'] and a['post_guard_passed']
            and a['kernel_monitor_ok'] and len(a['post_probes'])==2
            and all(v['passed'] for v in a['post_probes'])):
        raise RuntimeError('Previous worker release/system checks did not pass')
    if 'Controller requested cooperative stop' not in str(a.get('worker_failure',{})):
        raise RuntimeError('Previous worker exit was not the requested cooperative stop')
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),gpu_released=True)


def accepted_variant(report,mode,started):
    if not (report.get('passed') is True and report.get('state')=='complete'
            and report.get('rounds_complete')==1 and report.get('runs_complete')==2
            and report.get('time',0)>=started and len(report.get('pairs',[]))==1):
        raise RuntimeError('Variant did not complete exactly one fresh pair: '+mode)
    pair=report['pairs'][0]
    if pair['gids']['pair_id']!=pair['digit']['pair_id']:
        raise RuntimeError('Different pairs')
    for arm in ('gids','digit'):
        r=pair[arm];path=Path(r['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=r['sha256']:
            raise RuntimeError('Changed variant receipt')
        a=json.loads(path.read_text())
        if not (a['passed'] and a['worker']['sampler_variant']==('baseline_v21' if arm=='gids' else mode)
                and a['worker_state']['Result']=='success' and a['worker_state']['ExecMainStatus']=='0'
                and a['gpu_released'] and a['kernel_monitor_ok'] and a['post_guard_passed']):
            raise RuntimeError('Unaccepted variant worker')
    return dict(variant=mode,gids_seconds=pair['gids']['seconds'],digit_seconds=pair['digit']['seconds'],
                speedup=pair['speedup'],gpu=report['selected_gpu'],pair=pair)


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');args=p.parse_args()
    digest=P.verify_manifest()
    commands={mode:['/usr/bin/python3','-B','-u','-m','candidates.ukl_mixed512_v23.cli','--execute'] for mode in VARIANTS}
    if not args.execute:
        print(json.dumps(dict(execute=False,variants=list(VARIANTS),rounds_per_variant=P.ROUNDS,
                             arms=['gids','digit'],warmup=20,measured=300,commands=commands,
                             variant_environment='UKL_V23_VARIANT',manifest_sha256=digest),indent=2));return
    if os.geteuid()!=0:raise RuntimeError('sudo required for bounded GPU/SSD workers')
    if P.ROUNDS!=1:raise RuntimeError('This queue requires one round per variant')
    queue_lock=(P.BASE_OUT/'one_round/queue.lock').open('a+')
    fcntl.flock(queue_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prior=stopped_receipt()
    run_id=time.strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid())
    path=P.BASE_OUT/'one_round'/('queue_'+run_id+'.json');records=[]
    def record(state,**kw):
        value=dict(state=state,time=time.time(),variants=list(VARIANTS),rounds_per_variant=1,
                   previous_stop=prior,records=records,manifest_sha256=digest,**kw)
        write(path,value);write(P.BASE_OUT/'one_round/latest.json',value)
    record('starting')
    try:
        for mode in VARIANTS:
            record('running',variant=mode)
            print('Starting '+mode+': one GIDS -> DiGiT pair',flush=True)
            started=time.time()
            subprocess.run(commands[mode],env=dict(os.environ,UKL_V23_VARIANT=mode,PYTHONDONTWRITEBYTECODE='1'),check=True)
            report=json.loads((P.BASE_OUT/mode/'latest_status.json').read_text())
            records.append(accepted_variant(report,mode,started))
            record('running',variant=mode)
        record('complete',passed=True,same_gpu=len({tuple(r['gpu']) for r in records})==1)
        print(json.dumps(dict(passed=True,report=str(path),results=[{k:v for k,v in r.items() if k!='pair'} for r in records])),flush=True)
    except BaseException as error:
        record('failed',passed=False,error=repr(error));raise


if __name__=='__main__':main()
