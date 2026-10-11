#!/usr/bin/env python3
"""PA/SAGE random-order reconstruction: prepare, check, or train."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ae.pa_sage.common import DEFAULT_DATA, DEFAULT_REPORT, PROTOCOL, setup_imports, sha, read, write, require


def prepare(args):
    import fcntl
    args.report.mkdir(parents=True,exist_ok=True)
    lock=(args.report/'prepare.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stages=[('sources','sources.json'),('bootstrap','bootstrap/validation.json'),('profile','hot_ready.json'),
            ('final','final/validation.json'),('traces','traces_ready.json'),('payload','final/payload_ready.json'),('complete','prepared.json')]
    sources={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'ae/pa_sage').glob('*.py') if p.name in ('common.py','prepare.py','__init__.py')}
    binding=dict(protocol_sha256=sha(PROTOCOL),preparation_sources=sources,data=str(args.data.resolve()))
    status_path=args.report/'status.json'
    if status_path.exists():
        status=read(status_path);require(status['binding']==binding,'Preparation protocol/source changed; use a new version/report')
    else:status=dict(binding=binding,stages=[],started_unix=time.time())
    for stage,receipt in stages:
        if (args.data/receipt).exists():
            item=read(args.data/receipt)
            require(item.get('passed',False),'Existing stage is not accepted: '+stage)
            if 'protocol_sha256' in item:require(item['protocol_sha256']==binding['protocol_sha256'],'Existing stage protocol mismatch')
            continue
        record=dict(stage=stage,started_unix=time.time(),status='running')
        status['stages'].append(record);status.update(stage=stage,complete=False,pid=os.getpid());write(status_path,status)
        command=[sys.executable,'-m','ae.pa_sage.prepare',stage,'--data',str(args.data),'--report',str(args.report)]
        logfile=args.report/('%02d_%s.log'%(len(status['stages']),stage))
        with logfile.open('x') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            record.update(command=command,pid=child.pid,log=str(logfile));write(status_path,status)
            code=child.wait()
        record.update(returncode=code,ended_unix=time.time(),status='complete' if code==0 else 'failed');write(status_path,status)
        require(code==0,'Stage failed: '+stage+'; inspect '+str(logfile))
    status.update(stage='completed',complete=True,ended_unix=time.time());write(status_path,status)
    print(json.dumps(status,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','check','ssd-plan','train','native-smoke','experiment','plan'])
    p.add_argument('--data',type=Path,default=DEFAULT_DATA)
    p.add_argument('--report',type=Path,default=DEFAULT_REPORT)
    p.add_argument('--arm',choices=['gids','digit_full'],default='digit_full')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--repeat',type=int,choices=[0,1],default=0)
    p.add_argument('--seeds',type=int,nargs='+',help='Experiment subset; default all frozen seeds')
    p.add_argument('--repeats',type=int,nargs='+',help='Experiment subset; default both frozen repeats')
    p.add_argument('--smoke',action='store_true',help='Two epochs, two training and validation batches each; not a performance result')
    p.add_argument('--live',action='store_true',help='Query current GPU/host admission with check')
    p.add_argument('--output',type=Path)
    a=p.parse_args();setup_imports()
    a.data=a.data.resolve();a.report=a.report.resolve()
    if a.action=='prepare':prepare(a)
    elif a.action=='ssd-plan':
        from ae.pa_sage.storage import audit
        result=audit(a.data)
        if a.output:
            require(not a.output.exists(),'Output must be fresh');write(a.output,result)
        print(json.dumps(result,indent=2))
    elif a.action=='check':
        from ae.pa_sage.train import readiness
        result=readiness(a.data,a.live)
        if a.output:
            require(not a.output.exists(),'Output must be fresh');write(a.output,result)
        print(json.dumps(result,indent=2))
    elif a.action in ('experiment','plan'):
        from ae.pa_sage.experiment import run,make_plan
        if a.action=='plan':print(json.dumps(make_plan(a.seeds,a.repeats),indent=2))
        else:
            require(a.output is not None,'Experiment requires a fresh --output directory')
            run(a.data,a.output,a.seeds,a.repeats)
    elif a.action=='native-smoke':
        require(a.output is not None,'Native smoke requires a fresh --output directory')
        from ae.pa_sage.native_smoke import run
        run(a.data, a.output)
    else:
        require(a.output is not None,'Training requires a fresh --output report path')
        from ae.pa_sage.train import run
        run(a)


if __name__=='__main__':main()
