"""Service namespaces and output plumbing; frozen final guard/compute unchanged."""
import os,pwd,subprocess
from pathlib import Path
from types import SimpleNamespace
from common import *

def properties():
 return ['BindReadOnlyPaths='+str(SNAPSHOT)+':'+str(ROOT),
         'BindReadOnlyPaths='+str(ROOT/'ssd_state')+':'+str(ROOT/'ssd_state'),
         'PartOf='+UNIT,'UnsetEnvironment=PYTHONPATH PYTHONHOME PYTHONOPTIMIZE LD_PRELOAD',
         'Environment=PATH=/srv/digit-ae/env/bin:/usr/bin:/bin:/usr/sbin:/sbin']

def adapt_command(args,worker=False):
 require(args[0]=='/usr/bin/systemd-run','Only fixed transient services')
 result=list(args)
 if worker:
  i=result.index('-m');require(result[i+1]=='candidates.ukl_common_gpu_five_v27.bootstrap','Unexpected worker')
  result[i:i+2]=[str(CONTROL/'worker.py')]
  # Isolated Python still explicitly imports from the read-only snapshot.
  i=result.index(PYTHON);result.insert(i+1,'-I')
 result[1:1]=['--property='+p for p in properties()]
 return result

def run(experiment):
 from candidates.ukl_common_gpu_five_v27 import start,io_preflight
 P=configure(experiment)
 original=start.command;probe=start.legacy.probe
 start.command=lambda *a:adapt_command(original(*a),worker=True)
 # Frozen author controller chowns stage outputs to embed; AE evidence stays root-owned.
 start.pwd=SimpleNamespace(getpwnam=lambda name:pwd.getpwnam('root') if name=='embed' else pwd.getpwnam(name))
 def fs_probe(base,out,label):
  target=OUTPUT/'probes' if base==ROOT/'results' else PROBES
  require(base in (ROOT/'results',Path('/mnt/n0')),'Unexpected probe source')
  return probe(target,out,label)
 start.legacy.probe=fs_probe
 # A transient service starts in PID1's mount namespace, not its parent's.
 # The CPU I/O probe must receive the same immutable source mount explicitly.
 io_preflight.subprocess=SimpleNamespace(run=lambda args,**kw:subprocess.run(adapt_command(args),**kw),
                                        TimeoutExpired=subprocess.TimeoutExpired)
 start.execute()
