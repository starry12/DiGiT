"""UKL/GAT reviewer commands; a service launch is not acceptance."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import model_context
import argparse,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
from review import review

def view():
 raw=subprocess.check_output(['/usr/bin/systemctl','show',UNIT,'--property=ActiveState,SubState,Result,ExecMainStatus,InvocationID'],text=True)
 s=dict(line.split('=',1) for line in raw.splitlines() if '=' in line)
 active=s.get('ActiveState') in ('active','activating','deactivating')
 latest=CONTROL/'state/latest.json'
 if not latest.exists():return dict(state='STARTING' if active else 'NOT_STARTED')
 record=read(latest);folder=request_path(record['output'])
 if active and s.get('InvocationID')!=record['invocation_id']:return dict(state='STARTING')
 v=dict(state='RUNNING' if active else 'INCOMPLETE',output=str(folder))
 if (folder/'experiment/latest_status.json').exists():
  z=read(folder/'experiment/latest_status.json');v.update(stage=z.get('stage'),runs_complete=z.get('runs_complete'),rounds_complete=z.get('rounds_complete'),gpu=z.get('selected_gpu',[None])[0])
 if (folder/'error.json').exists():v.update(state='FAILED',error=read(folder/'error.json')['error'])
 if not active and (folder/'result.json').exists():
  require(s.get('Result')=='success' and s.get('ExecMainStatus')=='0','Service exit failed')
  require(not s.get('InvocationID') or s['InvocationID']==record['invocation_id'],'Stale invocation')
  final=read(folder/'result.json');ident=read(folder/'identity.json')
  require(final['invocation_id']==record['invocation_id'] and ident['transport_sha256']==transport_identity(),'Stale result/transport')
  result=review(folder/'experiment',read(CONTROL/'snapshot_manifest.json')['source_manifest_sha256'])
  require(result==read(folder/'completion_review.json') and sha(folder/'completion_review.json')==final['completion_sha256'] and final['speedup']==result['speedup'],'Completion drift')
  v.update(state='PASS',summary=dict(speedup=result['speedup'],selection=result['selection']))
 elif not active and s.get('Result') not in ('success',None):v['state']='FAILED'
 return v

def parse(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('command',choices=('performance','status','results','logs','stop'));p.add_argument('dataset',choices=('UKL',));p.add_argument('model',choices=('gat',))
 p.add_argument('--action',choices=('performance',));p.add_argument('--json',action='store_true');a=p.parse_args(argv)
 if (a.command=='performance')==bool(a.action):p.error('Launch: performance UKL gat; inspect/stop requires --action performance')
 if a.json and a.command not in ('status','results'):p.error('--json requires status/results')
 return a

def main(argv=None):
 a=parse(argv)
 if a.command in ('performance','stop'):
  subprocess.run(['/usr/bin/sudo','-n','/usr/bin/systemctl','--no-block','start' if a.command=='performance' else 'stop',UNIT],check=True)
  print('Request sent; inspect digit-ae status UKL gat --action performance.');return 0
 v=view()
 if a.json:print(json.dumps(v,indent=2))
 elif a.command=='logs' and v.get('output'):
  p=request_path(v['output'])/'controller.log'
  if p.exists():subprocess.run(['/usr/bin/tail','-n','40','--',str(p)],check=True)
 else:
  print('UKL / GAT | '+v['state'])
  if a.command=='results' and 'summary' in v:print('DiGiT vs GIDS: %.2f×'%v['summary']['speedup'])
  for key in ('stage','runs_complete','rounds_complete','gpu','output','error'):
   if v.get(key) is not None:print(key+': '+str(v[key]))
 return 1 if v['state']=='FAILED' else 0
if __name__=='__main__':
 try:sys.exit(main())
 except (OSError,ValueError,KeyError,RuntimeError,subprocess.CalledProcessError) as e:print(str(e),file=sys.stderr);sys.exit(1)
