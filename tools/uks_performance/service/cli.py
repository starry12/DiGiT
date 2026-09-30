"""Fixed UKS reviewer commands with explicit author-reference separation."""
import argparse,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from review import read,sha,require,review
C=Path('/srv/digit-ae/admin/uks_performance_v2');OUT=Path('/srv/digit-ae/uks-performance-results')
UNIT='digit-ae-uks-sage-performance.service'
def output(s):
    p=Path(s);require(not p.is_symlink() and p.parent==OUT and p.resolve().parent==OUT.resolve(),'Invalid output');return p
def view():
    text=subprocess.check_output(['/usr/bin/systemctl','show',UNIT,'--property=ActiveState,Result,ExecMainStatus,InvocationID'],text=True)
    service=dict(x.split('=',1) for x in text.splitlines() if '=' in x)
    active=service.get('ActiveState') in ('active','activating','deactivating')
    if not (C/'state/latest.json').exists():return dict(state='STARTING' if active else 'NOT_STARTED')
    latest=read(C/'state/latest.json');p=output(latest['output']);v=dict(state='RUNNING' if active else 'INCOMPLETE',output=str(p))
    if (p/'experiment/status.json').exists():
        s=read(p/'experiment/status.json');v.update(stage=s.get('stage'),runs_complete=sum(x.startswith('round') for x in s.get('completed',[])),error=s.get('error'),gpu=s.get('gpu_assignment',{}).get('index'))
    if (p/'error.json').exists():v.update(state='FAILED',error=read(p/'error.json')['error'])
    if not active and (p/'result.json').exists():
        require(service.get('Result')=='success' and service.get('ExecMainStatus')=='0','Service failed')
        require(not service.get('InvocationID') or service['InvocationID']==latest['invocation_id'],'Stale invocation')
        final=read(p/'result.json');gpu=read(p/'experiment/gpu_assignment.json')['selected'];r=review(p/'experiment',expected_gpu=gpu,transport_sha=sha(C/'transport_manifest.json'))
        require(final['native_acceptance'] and final['invocation_id']==latest['invocation_id'],'Stale result')
        require(read(p/'completion_review.json')==r and sha(p/'completion_review.json')==final['review_sha256'] and final['speedup']==r['speedup'],'Completion drift')
        v.update(state='PASS',summary=dict(speedup=r['speedup'],selection=r['selection'],node_sets_differ=True))
    elif not active and service.get('Result')!='success':v['state']='FAILED'
    return v
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('performance','status','results','logs','stop'));p.add_argument('dataset',choices=('UKS',));p.add_argument('model',choices=('sage',));p.add_argument('--action',dest='selector',choices=('performance',));p.add_argument('--reference',action='store_true');p.add_argument('--json',action='store_true');a=p.parse_args()
    if (a.action=='performance')==bool(a.selector):p.error('Start: performance UKS sage; inspect/stop: --action performance')
    if a.reference and a.action!='results':p.error('--reference requires results')
    if a.json and a.action not in ('status','results'):p.error('--json requires status/results')
    if a.action in ('performance','stop'):
        subprocess.run(['/usr/bin/sudo','-n','/usr/bin/systemctl','--no-block','start' if a.action=='performance' else 'stop',UNIT],check=True)
        print('Request sent; inspect digit-ae status UKS sage --action performance.');return 0
    v=dict(state='AUTHOR_REFERENCE',summary=read(C/'author_reference.json')) if a.reference else view()
    if a.json:print(json.dumps(v,indent=2))
    elif a.action=='logs' and v.get('output'):
        folder=output(v['output']);s=read(folder/'experiment/status.json') if (folder/'experiment/status.json').exists() else {}
        stage=s.get('stage','');log=folder/'experiment'/stage/'worker.log'
        if stage not in (['smoke_digit']+['round%d_%s'%(i,x) for i in range(1,6) for x in ('gids','digit')]) or not log.exists():log=folder/'controller.log'
        if log.exists():subprocess.run(['/usr/bin/tail','-n','40','--',str(log)],check=True)
    else:
        print('UKS / SAGE | '+v['state'])
        if a.action=='results' and 'summary' in v:print('Final speedup: %.2fx'%v['summary']['speedup'])
        for k in ('stage','runs_complete','gpu','output','error'):
            if v.get(k) is not None:print(k+': '+str(v[k]))
    return 1 if v['state']=='FAILED' else 0
if __name__=='__main__':
    try:sys.exit(main())
    except (OSError,ValueError,KeyError,RuntimeError,subprocess.CalledProcessError) as e:print(str(e),file=sys.stderr);sys.exit(1)
