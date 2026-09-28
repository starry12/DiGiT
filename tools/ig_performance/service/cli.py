"""Fixed IG performance command; fresh acceptance is separate from author reference."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
C=Path('/srv/digit-ae/admin/ig_performance_v1');O=Path('/srv/digit-ae/ig-performance-results/native')
UNIT='digit-ae-ig-sage-performance.service'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def output(value):
    p=Path(value)
    if p.parent!=O or p.resolve().parent!=O.resolve() or p.is_symlink():raise ValueError('Invalid output')
    return p

def view():
    svc=subprocess.check_output(['/usr/bin/systemctl','show',UNIT,'--property=ActiveState,Result,ExecMainStatus,InvocationID'],text=True)
    service=dict(s.split('=',1) for s in svc.splitlines() if '=' in s)
    active=service.get('ActiveState') in ('active','activating','deactivating')
    if not (C/'state/latest.json').exists():return dict(state='STARTING' if active else 'NOT_STARTED')
    request=read(C/'state/latest.json');out=output(request['output']);v=dict(state='RUNNING' if active else 'INCOMPLETE',output=str(out))
    if not (out/'status.json').exists():return v
    state=read(out/'status.json');v.update(stage=state.get('stage'),error=state.get('error'))
    if state.get('stage') in ('failed','interrupted'):v['state']='FAILED'
    if not active and state.get('passed') and state.get('complete'):
        if service.get('Result')!='success' or service.get('ExecMainStatus')!='0' or (service.get('InvocationID') and service['InvocationID']!=request['invocation_id']):return dict(v,state='FAILED')
        final=read(out/'result.json');review=read(out/'completion_review.json')
        if not (final['native_acceptance'] and final['invocation_id']==request['invocation_id'] and sha(out/'completion_review.json')==final['review_sha256'] and sha(out/'experiment/summary.json')==final['summary_sha256'] and review==read(out/'experiment/summary.json') and review['passed'] and review['complete'] and len(review['rows'])==10 and review['strict_resource_acceptance'] and review['diagnostic_complete']):raise ValueError('Invalid completion evidence')
        if final['speedup']!=review['statistics']['max_observed_speedup']:raise ValueError('Speedup mismatch')
        v.update(state='PASS',summary=final)
    if (out/'experiment/progress_summary.json').exists():v['runs_complete']=read(out/'experiment/progress_summary.json')['runs_complete']
    return v

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('performance','status','results','logs','stop'));p.add_argument('dataset',choices=('IG',));p.add_argument('model',choices=('sage',));p.add_argument('--action',dest='selector',choices=('performance',));p.add_argument('--reference',action='store_true');p.add_argument('--json',action='store_true');a=p.parse_args()
    if (a.action=='performance')==bool(a.selector):p.error('Use performance IG sage to start; --action performance for inspection/stop')
    if a.reference and a.action!='results':p.error('--reference requires results')
    if a.json and a.action not in ('status','results'):p.error('--json requires status/results')
    if a.action in ('performance','stop'):
        subprocess.run(['/usr/bin/sudo','-n','/usr/bin/systemctl','--no-block','start' if a.action=='performance' else 'stop',UNIT],check=True)
        print('Request sent; inspect digit-ae status IG sage --action performance.');return 0
    v=dict(state='AE_REFERENCE',summary=read(C/'accepted_reference.json')) if a.reference else view()
    if a.json:print(json.dumps(v,indent=2))
    elif a.action=='logs' and v.get('output'):
        f=output(v['output'])/'native.log'
        if f.exists():subprocess.run(['/usr/bin/tail','-n','50','--',str(f)],check=True)
    else:
        print('IG / SAGE | '+v['state'])
        if a.action=='results' and 'summary' in v:print('Final speedup: %.2fx (maximum observed paired speedup across five rounds)'%v['summary']['speedup'])
        for k in ('stage','runs_complete','output','error'):
            if v.get(k) is not None:print(k+': '+str(v[k]))
    return (0 if v['state'] in ('PASS','AE_REFERENCE') else 1 if v['state']=='FAILED' else 3) if a.action=='results' else 0
if __name__=='__main__':
    try:sys.exit(main())
    except (OSError,ValueError,KeyError,RuntimeError,subprocess.CalledProcessError) as e:print(str(e),file=sys.stderr);sys.exit(1)
