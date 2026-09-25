"""Fixed PA/SAGE prepared-layout grid commands; caller cannot choose privileged paths."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
CONTROL=Path('/srv/digit-ae/admin/layout_v1')
OUTPUTS=Path('/srv/digit-ae/layout-results')
UNIT='digit-ae-pa-sage-layout.service'
POINTS={'g%d_r%02d'%(g,r) for g in (1,2,4) for r in (0,10,20,40,80)}
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def parse(argv):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('layout','status','results','logs','stop'))
    p.add_argument('dataset',choices=('PA',));p.add_argument('model',choices=('sage',))
    p.add_argument('--action',dest='selector',choices=('layout',))
    p.add_argument('--json',action='store_true');p.add_argument('--reference',action='store_true')
    a=p.parse_args(argv)
    if (a.action=='layout')==bool(a.selector):p.error('Use layout PA sage to start; --action layout for other commands')
    if a.json and a.action not in ('status','results'):p.error('--json is for status/results')
    if a.reference and a.action!='results':p.error('--reference is for results only')
    return a

def checked_output(value):
    p=Path(value)
    if not p.is_absolute() or p.parent!=OUTPUTS or p.is_symlink() or p.resolve().parent!=OUTPUTS.resolve():
        raise ValueError('Invalid result directory')
    return p

def checked_file(root,relative):
    p=Path(relative)
    if p.is_absolute() or '..' in p.parts:raise ValueError('Invalid result path')
    f=root/p
    if f.resolve().parent!=f.parent or f.is_symlink():raise ValueError('Symlink result path')
    return f

def view():
    if not (CONTROL/'state/latest.json').exists():return dict(state='NOT_STARTED')
    request=read(CONTROL/'state/latest.json');out=checked_output(request['output']);s=read(checked_file(out,'status.json'))
    proc=subprocess.run(['/usr/bin/systemctl','show',UNIT,'--property=ActiveState,SubState,Result,ExecMainStatus,InvocationID'],
        text=True,capture_output=True,check=True)
    service=dict(line.split('=',1) for line in proc.stdout.splitlines() if '=' in line)
    active=service.get('ActiveState') in ('active','activating','deactivating')
    kind='RUNNING' if active else 'INCOMPLETE'
    if s.get('stage')=='failed':kind='FAILED'
    elif s.get('complete') and s.get('passed'):
        same=not service.get('InvocationID') or service['InvocationID']==s.get('invocation_id')
        kind='FINALIZING' if active else ('PASS' if same and service.get('Result')=='success' and service.get('ExecMainStatus')=='0' else 'FAILED')
    v=dict(state=kind,stage=s.get('stage'),output=str(out),completed=s.get('completed',[]),
        service=service,steps=s.get('steps',[]),error=s.get('error'))
    if kind=='PASS':
        f=checked_file(out,'summary.json');summary=read(f)
        if not summary['passed'] or sha(f)!=s.get('summary_sha256') or not s.get('native_acceptance'):
            raise ValueError('Invalid final summary')
        if len(summary['points'])!=15 or {r['point']['id'] for r in summary['points']}!=POINTS:
            raise ValueError('Incomplete final grid')
        for r in summary['points']:
            f=checked_file(out,'native/'+r['point']['id']+'/full_digit_full_accepted.json')
            if sha(f)!=r['accepted_report_sha256']:raise ValueError('Accepted report changed')
        v['summary']=summary
    return v

def main(argv=None):
    a=parse(sys.argv[1:] if argv is None else argv)
    if a.action in ('layout','stop'):
        subprocess.run(['/usr/bin/sudo','-n','/usr/bin/systemctl','--no-block','start' if a.action=='layout' else 'stop',UNIT],check=True)
        print('Start requested; fresh AE acceptance pending.' if a.action=='layout' else 'Stop requested; wait for resource release.')
        print('digit-ae status PA sage --action layout');return 0
    if a.reference:
        r=read(CONTROL/'author_reference.json');v=dict(state='AUTHOR_REFERENCE',summary=r,
            note='Accepted author run; not a fresh AE-account replay')
    else:v=view()
    if a.action=='logs' and 'output' in v:
        out=checked_output(v['output']);files=['status.json']
        if v.get('steps'):
            point=v['steps'][-1]['point']
            if point not in POINTS:raise ValueError('Invalid point in state')
            files.append(point+'.log')
        for name in files:
            f=checked_file(out,name)
            if f.is_file():subprocess.run(['/usr/bin/tail','-n','50','--',str(f)],check=True)
    elif a.json:print(json.dumps(v,indent=2))
    else:
        print('PA / SAGE layout | '+v['state'])
        for key in ('stage','output','error','note'):
            if v.get(key):print(key+': '+str(v[key]))
        if 'completed' in v:print('Completed: %d/15'%len(v['completed']))
        if a.action=='results' and 'summary' in v:
            print('point       epoch(s)   speedup vs g2/r20')
            for r in v['summary']['points']:
                print('%-10s %8.2f %8.2fx'%(r['point']['id'],r['mean_training_epoch_seconds'],r['speedup_vs_fresh_g2_r20']))
    return (0 if v['state'] in ('PASS','AUTHOR_REFERENCE') else 1 if v['state']=='FAILED' else 3) if a.action=='results' else 0

if __name__=='__main__':
    try:sys.exit(main())
    except (OSError,ValueError,KeyError,RuntimeError,subprocess.CalledProcessError) as exc:
        print(str(exc),file=sys.stderr);sys.exit(1)
