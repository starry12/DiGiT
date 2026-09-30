"""Serial stage journal, fresh children, immutable accepted receipts and explicit resume."""
import csv
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid
from .common import ROOT,HERE,OUT,LARGE,GRID,PY,MODULE,ARMS,read,sha,write,require,heavy_gate,lock,verify
from .protocol import validate,schedule
from .monitor import run_monitored,stop_child


def stage_key(item):
    return item['stage']+('_'+item['arm'] if item['arm'] else '')


class Journal:
    def __init__(self,output,execution,protocol_sha,resume=False):
        self.output=Path(output);path=self.output/'status.json'
        if resume:
            self.state=read(path)
            require(self.state['source_sha256']==execution and self.state['protocol_sha256']==protocol_sha,
                    'Resume requires the identical source and protocol')
            require(not self.state['complete'],'This comparison is already complete')
            self.state.setdefault('resume_history',[]).append(dict(pid=os.getpid(),time_unix=time.time()))
        else:
            self.output.mkdir(parents=True,exist_ok=False)
            self.state=dict(schema='digit-cpu-capacity-controller-v1',run_id=uuid.uuid4().hex,
                source_sha256=execution,protocol_sha256=protocol_sha,passed=False,complete=False,
                started_unix=time.time(),completed={},attempts=[],raw_ssd_writes=False)
        self.state.update(pid=os.getpid(),stage='initializing');self.save()

    def save(self,**kw):
        self.state.update(kw,updated_unix=time.time());write(self.output/'status.json',self.state)

    def previous(self,key):
        item=self.state['completed'].get(key)
        if item:
            require(sha(item['path'])==item['sha256'],'Accepted stage output changed: '+key)
            return Path(item['path'])

    def attempt(self,key):
        number=sum(a['key']==key for a in self.state['attempts'])
        path=self.output/'stages'/key/('attempt_%02d'%number);path.mkdir(parents=True,exist_ok=False)
        self.state['attempts'].append(dict(key=key,number=number,path=str(path),started_unix=time.time(),status='running'))
        self.save(stage=key,error=None);return path

    def commit(self,key,report):
        require(read(report).get('passed') is True,'Cannot accept a failed stage')
        self.state['completed'][key]=dict(path=str(report),sha256=sha(report))
        self.state['attempts'][-1].update(status='accepted',finished_unix=time.time())
        self.save(stage=key+'_accepted')

    def fail(self,exc):
        if self.state['attempts'] and self.state['attempts'][-1]['status']=='running':
            self.state['attempts'][-1].update(status='failed',finished_unix=time.time(),error=type(exc).__name__+': '+str(exc))
        self.save(stage='failed',passed=False,complete=False,error=type(exc).__name__+': '+str(exc))


def run_plain(command,attempt,notify=lambda **kw:None):
    child=None
    try:
        with (attempt/'child.log').open('x') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            notify(child_pid=child.pid,command=command)
            code=child.wait()
        require(code==0,'Preparation child failed; see '+str(attempt/'child.log'))
    finally:stop_child(child)


def wait_available(plan,journal):
    from .admission import live
    while True:
        heavy_gate()
        state=live(plan)
        require(plan['required_bytes']<=state['gpu_total_bytes'],'Required envelope exceeds this GPU')
        write(journal.output/'resource_wait.json',state)
        if state['passed']:return
        journal.save(stage='waiting_resources')
        time.sleep(30)


def child_command(stage,arm,protocol,binding,attempt,prepared=None,large=None,short=None):
    command=[PY,'-B','-u','-m',MODULE+'.worker','--mode',stage,'--protocol',str(protocol),
        '--binding',str(binding),'--output',str(attempt/'worker'),'--controller-pid',str(os.getpid())]
    for flag,value in (('arm',arm),('prepared',prepared),('large',large),('short-receipt',short)):
        if value is not None:command.extend(['--'+flag,str(value)])
    return command


def run(protocol_path,output,resume=False):
    # The first executed action is a read-only gate. No waiting job is queued
    # behind the existing grid, and no directories/devices are opened first.
    heavy_gate();p=validate(read(protocol_path));execution=verify()
    with lock(GRID/'controller.lock',create=False):heavy_gate()
    from contextlib import ExitStack
    with ExitStack() as stack:
        stack.enter_context(lock('/run/digit-ae-selfservice/exclusive.lock'))
        stack.enter_context(lock('/tmp/digit-pa-bidir-controller.lock'))
        stack.enter_context(lock('/tmp/digit-pa-cpu-capacity-v1-controller.lock'))
        journal=Journal(output,execution,sha(protocol_path),resume)
        def stop(signum,frame):raise KeyboardInterrupt('Cache controller signaled')
        old_term=signal.signal(signal.SIGTERM,stop);old_int=signal.signal(signal.SIGINT,stop)
        accepted={};short_path=None
        try:
            for item in schedule(p):
                heavy_gate();require(verify()==execution,'Candidate changed during run')
                stage,arm=item['stage'],item['arm'];key=stage_key(item)
                old=journal.previous(key)
                if old is not None:
                    accepted[key]=old
                    # Revalidate accepted records below before downstream reuse.
                    if stage=='bind':
                        from .binding import check
                        check(read(old),protocol_path,execution)
                    elif stage=='build':
                        from candidates.pa_sage_cpu_capacity_v1.common import binary_receipt
                        require(read(old)['build_receipt']==binary_receipt(),'Built binary changed since this run')
                    elif stage=='prepare':
                        from .prepared import check
                        check(read(old),p,protocol_path,accepted['bind'],execution)
                    if stage=='smoke' and arm==ARMS[-1]:
                        short_path=ensure_short_matrix(journal,p,accepted,protocol_path,execution)
                    continue
                attempt=journal.attempt(key)
                large=LARGE/journal.state['run_id']/key/attempt.name
                if stage=='bind':
                    from .binding import bind
                    bind(p,protocol_path,attempt,execution);path=attempt/'inputs.json'
                elif stage=='build':
                    from candidates.pa_sage_cpu_capacity_v1.common import BINARY,binary_receipt
                    if not BINARY.exists():run_plain([PY,'-B','-m','candidates.pa_sage_cpu_capacity_v1.build','--execute'],attempt,journal.save)
                    receipt=binary_receipt();path=attempt/'build.json'
                    write(path,dict(passed=True,build_receipt=receipt))
                elif stage=='prepare':
                    from .prepared import seal
                    path=attempt/'prepared.json'
                    seal(p,protocol_path,accepted['bind'],accepted['profile'],path,execution)
                elif stage in ('profile','smoke','full'):
                    from .admission import estimate
                    bound=read(accepted['bind'])
                    from .binding import check as check_inputs
                    check_inputs(bound,protocol_path,execution)
                    wait_available(estimate(p,bound['manifest'],arm or 'cpu20',profile=stage=='profile'),journal)
                    journal.save(stage=key)
                    if stage in ('smoke','full'):
                        from .prepared import check as check_prepared
                        check_prepared(read(accepted['prepare']),p,protocol_path,accepted['bind'],execution)
                    if stage=='full':
                        short_path=ensure_short_matrix(journal,p,accepted,protocol_path,execution)
                    command=child_command(stage,arm,protocol_path,accepted['bind'],attempt,
                        prepared=accepted.get('prepare'),large=large if stage=='profile' else None,short=short_path)
                    report,row=run_monitored(command,attempt,arm or 'profile',lambda **kw:journal.save(worker=kw))
                    if stage in ('smoke','full'):
                        from candidates.pa_sage_cpu_capacity_v1.common import binary_receipt
                        backend=binary_receipt()['binary_sha256']
                        from .validation import short,full
                        if stage=='smoke':short(report,p,arm,execution,sha(protocol_path),sha(accepted['prepare']),backend)
                        else:full(report,p,arm,execution,sha(protocol_path),sha(accepted['prepare']),short_path,backend)
                    else:
                        require(report['kind']=='independent_native_presampling' and report['passed'] and
                            report['optimizer_updates']==report['evaluation_calls']==report['feature_reads']==0,
                            'Invalid profile report')
                    path=attempt/'accepted.json';write(path,report)
                else:
                    from .validation import aggregate
                    from candidates.pa_sage_cpu_capacity_v1.common import binary_receipt
                    short_path=ensure_short_matrix(journal,p,accepted,protocol_path,execution)
                    result=aggregate({a:accepted['full_'+a] for a in ARMS},p,execution,sha(protocol_path),
                        sha(accepted['prepare']),short_path,binary_receipt()['binary_sha256'])
                    path=attempt/'summary.json';write(path,result);write(journal.output/'summary.json',result)
                    with (attempt/'summary.csv').open('x',newline='') as stream:
                        fields=['arm','training_seconds','order_excluded_seconds','speedup_vs_cpu00','cpu_percent','cpu_rows','cpu_feature_bytes','gpu_feature_cache_bytes']
                        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
                        for a in ARMS:writer.writerow(dict(arm=a,**{k:result['arms'][a][k] for k in fields[1:]}))
                journal.commit(key,path);accepted[key]=path
                if stage=='smoke' and arm==ARMS[-1]:short_path=ensure_short_matrix(journal,p,accepted,protocol_path,execution)
            from .binding import check
            check(read(accepted['bind']),protocol_path,execution)
            require(verify()==execution,'Source changed before completion')
            journal.save(stage='complete',passed=True,complete=True,finished_unix=time.time())
        except BaseException as exc:journal.fail(exc);raise
        finally:
            signal.signal(signal.SIGTERM,old_term);signal.signal(signal.SIGINT,old_int)


def ensure_short_matrix(journal,p,accepted,protocol_path,execution):
    from .validation import make_smoke_matrix,smoke_matrix
    from candidates.pa_sage_cpu_capacity_v1.common import binary_receipt
    prepared_sha=sha(accepted['prepare'])
    paths={a:accepted['smoke_'+a] for a in ARMS}
    value=make_smoke_matrix(paths,p,execution,sha(protocol_path),prepared_sha,binary_receipt()['binary_sha256'])
    path=journal.output/'short_matrix.json'
    if path.exists():require(read(path)==value,'Preserved short matrix differs; no silent replacement')
    else:write(path,value)
    smoke_matrix(value,p,execution,sha(protocol_path),prepared_sha)
    return path
